import math
import re
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cifar_resnet import CIFARBasicBlock, CIFARResNet
from .outputs import ClassifModelOutput
from .resnet import Block, Bottleneck, ResNet


class ClassificationFeatureSelectionWrapper(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 lambda_coef: float = 0.01,
                 gumbel_init_mode: str = "auto",
                 criterion=nn.CrossEntropyLoss(),
                 regularization_loss=lambda x: 0):
        super().__init__()
        self.backbone = backbone
        self.criterion = criterion
        self.lambda_coef = lambda_coef
        self.gumbel_init_mode = gumbel_init_mode
        self.regularization_loss = regularization_loss
        self._initialize_gumbel_layers()

    def forward(self, X, y) -> ClassifModelOutput:
        logits = self.backbone(X)
        ce_loss = self.criterion(logits, y)
        if float(self.lambda_coef) == 0.0:
            reg_loss = logits.new_zeros(())
            loss = ce_loss
        else:
            reg_loss = self.regularization_loss(self.backbone)
            loss = ce_loss + self.lambda_coef*reg_loss

        return ClassifModelOutput(
            ce_loss=ce_loss,
            regularization_loss=reg_loss,
            loss=loss,
            logits=logits
        )

    def _resolve_gumbel_init_mode(self) -> str:
        mode = str(self.gumbel_init_mode).lower()
        if mode == "auto":
            return "fully_open" if float(self.lambda_coef) == 0.0 else "paper"
        if mode in {"paper", "fully_open"}:
            return mode
        raise ValueError(
            "gumbel_init_mode must be one of: 'auto', 'paper', 'fully_open'."
        )

    def _initialize_gumbel_layers(self) -> None:
        init_mode = self._resolve_gumbel_init_mode()
        bypass_gumbel = float(self.lambda_coef) == 0.0
        for module in get_gumbel_modules(self.backbone).values():
            module.reset_parameters(init_mode=init_mode)
            module.set_bypass(bypass_gumbel)


# LEGACY: old custom Gumbel-Softmax helper used only by legacy paths, not by the main_gumbel pipeline.
class GumbleSoftmax(torch.nn.Module):
    def __init__(self, hard=False):
        super(GumbleSoftmax, self).__init__()
        self.hard = hard
        self.gpu = False

    def cuda(self):
        self.gpu = True

    def cpu(self):
        self.gpu = False

    def sample_gumbel(self, shape, eps=1e-10):
        """Sample from Gumbel(0, 1)"""
        noise = torch.rand(shape)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        if self.gpu:
            return torch.autograd.Variable(noise).cuda()
        else:
            return torch.autograd.Variable(noise)

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = - \
            torch.log(eps - torch.log(uniform_samples_tensor + eps))
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        dim = logits.size(-1)
        gumble_samples_tensor = self.sample_gumbel_like(logits.data)
        gumble_trick_log_prob_samples = logits + \
            torch.autograd.Variable(gumble_samples_tensor)
        soft_samples = F.softmax(
            gumble_trick_log_prob_samples / temperature, dim)
        return soft_samples

    def gumbel_softmax(self, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            _, max_value_indexes = y.data.max(1, keepdim=True)
            y_hard = logits.data.clone().zero_().scatter_(1, max_value_indexes, 1)
            y = torch.autograd.Variable(y_hard - y.data) + y
        return y

    def forward(self, logits, temp=1, force_hard=False):
        samplesize = logits.size()

        if self.training and not force_hard:
            return self.gumbel_softmax(logits, temperature=temp, hard=False)
        else:
            return self.gumbel_softmax(logits, temperature=temp, hard=True)


# ACTUAL: shared Gumbel selector used by the current main_gumbel pipeline.
# Main pipeline uses compute_gates/apply_gates/regularization_loss/get_selection_probs below.
class GumbelLayer(nn.Module):
    """
    Gumbel-Softmax based feature selector.
    Uses categorical distribution over {off, on} for each feature.

    Reference: "Categorical Reparameterization with Gumbel-Softmax"
    (Jang et al., ICLR 2017)

    Fixed implementation: Properly handles batch dimension and sampling.
    """

    PAPER_INIT_OFF_LOGIT = 0.1
    PAPER_INIT_ON_LOGIT = 2.0
    PAPER_INIT_NOISE_SCALE = 0.02
    FULLY_OPEN_OFF_LOGIT = -10.0
    FULLY_OPEN_ON_LOGIT = 10.0
    FULLY_OPEN_NOISE_SCALE = 0.0

    def __init__(self, input_dim: int, temperature: float = 1.0):
        super().__init__()
        self.logits = nn.Parameter(torch.empty(input_dim, 2))
        self.temperature = temperature
        self._bypass = False
        self.reset_parameters()

    def reset_parameters(self, init_mode: str = "paper") -> None:
        mode = str(init_mode).lower()
        if mode == "paper":
            off_logit = self.PAPER_INIT_OFF_LOGIT
            on_logit = self.PAPER_INIT_ON_LOGIT
            noise_scale = self.PAPER_INIT_NOISE_SCALE
        elif mode == "fully_open":
            off_logit = self.FULLY_OPEN_OFF_LOGIT
            on_logit = self.FULLY_OPEN_ON_LOGIT
            noise_scale = self.FULLY_OPEN_NOISE_SCALE
        else:
            raise ValueError(
                "init_mode must be one of: 'paper', 'fully_open'."
            )

        with torch.no_grad():
            self.logits[:, 0].fill_(float(off_logit))
            self.logits[:, 1].fill_(float(on_logit))
            if float(noise_scale) > 0.0:
                self.logits.add_(torch.randn_like(self.logits) * float(noise_scale))

    def set_bypass(self, enabled: bool) -> None:
        self._bypass = bool(enabled)

    @property
    def bypass(self) -> bool:
        return self._bypass


    # ACTUAL: computes broadcastable Gumbel gates for the current CIFAR main pipeline.
    def compute_gates(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if self._bypass:
            gates = x.new_ones((batch_size, self.logits.shape[0]))
            while len(gates.shape) < len(x.shape):
                gates = gates.unsqueeze(-1)
            return gates
        if self.training:
            logits = self.logits.unsqueeze(0).expand(batch_size, -1, -1)
            sampled = F.gumbel_softmax(
                logits,
                tau=self.temperature,
                hard=True,
                dim=-1
            )
            gates = sampled[..., 1]
        else:
            gates = (self.logits[:, 1] > self.logits[:, 0]).float().unsqueeze(0)
            gates = gates.expand(batch_size, -1)
        while len(gates.shape) < len(x.shape):
            gates = gates.unsqueeze(-1)
        return gates

    # ACTUAL: applies Gumbel gates to activations in the current CIFAR main pipeline.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.compute_gates(x)

    # ACTUAL: regularizer used by the current main_gumbel training loss.
    def regularization_loss(self) -> torch.Tensor:
        """Compute regularization: sum of "on" state probabilities."""
        if self._bypass:
            return self.logits.new_zeros(())
        probs = F.softmax(self.logits, dim=1)[:, 1]
        return torch.mean(probs)

    # ACTUAL: probability readout used by current metrics/logging in the main_gumbel pipeline.
    def get_selection_probs(self) -> torch.Tensor:
        """Get selection probabilities for each feature."""
        if self._bypass:
            return torch.ones(self.logits.shape[0], device=self.logits.device)
        return F.softmax(self.logits, dim=1)[:, 1].detach()

    # ACTUAL: temperature control hook for the current Gumbel layer implementation.
    def set_temperature(self, temperature: float):
        """Update temperature for annealing schedule."""
        self.temperature = temperature


class STGChannelLayer(nn.Module):
    """
    Stochastic Gates (STG) selector for channel-wise masking.

    Reference: Yamada et al., "Feature Selection using Stochastic Gates".
    """

    def __init__(self, input_dim: int, sigma: float = 0.5, init_mu: float | str = 1.0):
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be positive for STGChannelLayer.")

        if isinstance(init_mu, str):
            if init_mu.lower() != "random":
                raise ValueError("init_mu must be numeric or the literal string 'random'.")
            mu_init = 0.01 * torch.randn(input_dim)
        else:
            mu_init = 0.01 * torch.randn(input_dim) + float(init_mu)

        self.mu = nn.Parameter(mu_init)
        self.sigma = sigma
        self._select_features_count = 0.0
        self._num_forwards = 0

    def compute_gates(self, x: torch.Tensor) -> torch.Tensor:
        self._num_forwards += 1

        if self.training:
            z = self.mu + torch.randn_like(self.mu) * self.sigma
            self._select_features_count += float((z > 0).sum().item())
        else:
            z = self.mu
            self._select_features_count += float((self.mu > 0).sum().item())

        gates = torch.clamp(z, 0.0, 1.0)
        reshape_dims = (1, -1) + (1,) * max(0, x.dim() - 2)
        return gates.view(*reshape_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.compute_gates(x)

    def reset_counter(self):
        self._select_features_count = 0.0
        self._num_forwards = 0

    def get_select_features_mean(self) -> float:
        if self._num_forwards == 0:
            return 0.0
        return self._select_features_count / self._num_forwards

    def regularization_loss(self) -> torch.Tensor:
        probs = 0.5 * (1 + torch.erf(self.mu / (self.sigma * math.sqrt(2))))
        return torch.mean(probs)

    def get_selection_probs(self) -> torch.Tensor:
        return (
            0.5 * (1 + torch.erf(self.mu / (self.sigma * math.sqrt(2))))
        ).detach()


class STGBasicBlock(Block):
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, sigma: float = 0.5):
        super().__init__(in_channels, out_channels, i_downsample=i_downsample, stride=stride)
        self.stg1 = STGChannelLayer(out_channels, sigma=sigma)
        self.stg2 = STGChannelLayer(out_channels * self.expansion, sigma=sigma)

    def forward(self, x):
        identity = x

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.stg1(x)
        x = self.batch_norm2(self.conv2(x))
        x = self.stg2(x)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class STGBottleneckLayer(Bottleneck):
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, sigma: float = 0.5):
        super().__init__(in_channels, out_channels, i_downsample=i_downsample, stride=stride)
        self.stg1 = STGChannelLayer(out_channels, sigma=sigma)
        self.stg2 = STGChannelLayer(out_channels, sigma=sigma)
        self.stg3 = STGChannelLayer(out_channels * self.expansion, sigma=sigma)

    def forward(self, x):
        identity = x

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.stg1(x)

        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.stg2(x)

        x = self.batch_norm3(self.conv3(x))
        x = self.stg3(x)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class GumbelBottleneckLayer(Bottleneck):
    def __init__(
        self,
        in_channels,
        out_channels,
        i_downsample=None,
        stride=1,
        temperature: float = 1.0,
    ):
        super().__init__(in_channels, out_channels, i_downsample=i_downsample, stride=stride)
        self.gumbel_layer = GumbelLayer(
            input_dim=out_channels * self.expansion,
            temperature=temperature,
        )

    def forward(self, x):
        identity = x

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))
        x = self.gumbel_layer(x)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


# ACTUAL: current Gumbel block used by main_gumbel on CIFAR.
class CIFARGumbelBasicBlock(CIFARBasicBlock):
    def __init__(self, in_planes, planes, stride=1, option: str = "A", temperature: float = 1):
        super().__init__(in_planes, planes, stride=stride, option=option)
        self.gumbel_layer = GumbelLayer(
            input_dim=planes * self.expansion,
            temperature=temperature,
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.gumbel_layer(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CIFARSTGBasicBlock(CIFARBasicBlock):
    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        option: str = "A",
        sigma: float = 0.5,
        init_mu: float = 1.0,
    ):
        super().__init__(in_planes, planes, stride=stride, option=option)
        self.stg_layer = STGChannelLayer(
            input_dim=planes * self.expansion,
            sigma=sigma,
            init_mu=init_mu,
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.stg_layer(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AIGBottleneckLayer(Bottleneck):
    def __init__(self, in_planes, planes, i_downsample=None, stride=1, temperature: float = 1, hard: bool = False):
        super().__init__(in_planes, planes, i_downsample=i_downsample, stride=stride)
        self.gumbel = GumbleSoftmax(hard=hard)
        self.temperature = temperature
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels=in_planes,
                      out_channels=16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=1)
        )
        # initial opening rate of the gate of about 85%
        self.adapter[3].bias.data[0] = 0.1
        self.adapter[3].bias.data[1] = 2

    def to(self, device, *args, **kwargs):
        super().to(device, *args, **kwargs)
        if device != 'cpu':
            self.gumbel.cuda()
        return self

    def forward(self, x):
        w = self.adapter(F.avg_pool2d(x, x.size(2)))
        w = self.gumbel(w, temp=self.temperature, force_hard=True)
        # todo: mb to register buffer?
        self.activations = w[:, 1].unsqueeze(1)

        identity = x
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x = identity + x*w[:, 1].unsqueeze(1)
        x = self.relu(x)

        return x


def parse_AIG_activations(model: nn.Module, buff=None, prefix: str = None):
    if buff is None:
        buff = {}
    for name, module in _get_aig_modules(model, buff={}, prefix=prefix).items():
        activations = getattr(module, "activations", None)
        if activations is not None:
            buff[f'g_prob_{name}'] = activations
    return buff


def get_AIG_regularization_loss(model: nn.Module):
    aig_modules = _get_aig_modules(model)
    activations = [
        module.activations
        for module in aig_modules.values()
        if getattr(module, "activations", None) is not None
    ]
    if len(activations) == 0:
        return 0.0

    reg_loss = 0.0
    for act in activations:
        reg_loss += act.mean()**2
    reg_loss /= len(activations)
    return reg_loss


def _collect_modules_by_type(model: nn.Module, module_types, buff=None, prefix: str = None):
    if buff is None:
        buff = {}
    for name, module in model.named_children():
        full_name = f'{prefix}.{name}' if prefix is not None else name
        if isinstance(module, module_types):
            buff[full_name] = module
        _collect_modules_by_type(module, module_types, buff, full_name)
    return buff


def _get_cached_modules_by_type(model: nn.Module, cache_key: str, module_types):
    cache_attr = "_net_complexity_module_cache"
    cache = getattr(model, cache_attr, None)
    if cache is None:
        cache = {}
        setattr(model, cache_attr, cache)

    modules = cache.get(cache_key)
    if modules is None:
        modules = _collect_modules_by_type(model, module_types)
        cache[cache_key] = modules
    return modules


# ACTUAL: helper used by the current main_gumbel pipeline to collect all Gumbel layers.
def _get_gumbel_modules(model: nn.Module, buff=None, prefix: str = None):
    if buff is not None or prefix is not None:
        return _collect_modules_by_type(model, GumbelLayer, buff=buff, prefix=prefix)
    return _get_cached_modules_by_type(model, "gumbel", GumbelLayer)


def _get_aig_modules(model: nn.Module, buff=None, prefix: str = None):
    if buff is not None or prefix is not None:
        return _collect_modules_by_type(model, AIGBottleneckLayer, buff=buff, prefix=prefix)
    return _get_cached_modules_by_type(model, "aig", AIGBottleneckLayer)


def get_AIG_modules(model: nn.Module):
    return _get_aig_modules(model)


# ACTUAL: public wrapper used by the training/metrics pipeline after the recursive helper was internalized.
def get_gumbel_modules(model: nn.Module):
    return _get_gumbel_modules(model)


# ACTUAL: regularization entry point used by the current main_gumbel training loss.
def get_gumbel_loss(model: nn.Module):
    gumbel_modules = _get_gumbel_modules(model)
    if len(gumbel_modules) == 0:
        return 0.0
    loss = 0.0
    for _, module in gumbel_modules.items():
        loss += module.regularization_loss()
    loss /= len(gumbel_modules)
    return loss


def _get_stg_modules(model: nn.Module, buff=None, prefix: str = None):
    if buff is not None or prefix is not None:
        return _collect_modules_by_type(model, STGChannelLayer, buff=buff, prefix=prefix)
    return _get_cached_modules_by_type(model, "stg", STGChannelLayer)


def get_stg_modules(model: nn.Module):
    return _get_stg_modules(model)


def get_stg_loss(model: nn.Module):
    stg_modules = _get_stg_modules(model)
    if len(stg_modules) == 0:
        return 0.0

    loss = 0.0
    for _, module in stg_modules.items():
        loss += module.regularization_loss()
    loss /= len(stg_modules)
    return loss


def replace_layers_by_regex(model, pattern, new_layer_factory):
    """
    Replace layers whose prefix matches a regex pattern with new layers.

    Args:
        model (nn.Module): The PyTorch model
        pattern (str): Regular expression pattern to match layer prefixes
        new_layer_factory (callable or nn.Module): Factory function that returns a new layer,
                                                    or a layer instance to use for all matches
    Returns:
        nn.Module: Modified model
    """
    regex = re.compile(pattern)

    def _replace_module(module, prefix=""):
        for name, child in list(module.named_children()):
            child_prefix = f"{prefix}.{name}" if prefix else name

            if regex.match(child_prefix):
                if callable(new_layer_factory):
                    new_layer = new_layer_factory(child)
                else:
                    new_layer = new_layer_factory

                setattr(module, name, new_layer)
                print(
                    f"Replaced layer '{child_prefix}' with {type(new_layer).__name__}")
            else:
                _replace_module(child, child_prefix)

    _replace_module(model)
    return model


def ResNet50(
    num_classes,
    in_channels=3,
    resnet_block=Bottleneck,
    stem_kernel_size: int = 7,
    stem_stride: int = 2,
    stem_padding: int = 3,
    use_maxpool: bool = True,
):
    return ResNet(
        resnet_block,
        [3, 4, 6, 3],
        num_classes,
        in_channels,
        stem_kernel_size=stem_kernel_size,
        stem_stride=stem_stride,
        stem_padding=stem_padding,
        use_maxpool=use_maxpool,
    )


def ResNet101(
    num_classes,
    in_channels=3,
    resnet_block=Bottleneck,
    stem_kernel_size: int = 7,
    stem_stride: int = 2,
    stem_padding: int = 3,
    use_maxpool: bool = True,
):
    return ResNet(
        resnet_block,
        [3, 4, 23, 3],
        num_classes,
        in_channels,
        stem_kernel_size=stem_kernel_size,
        stem_stride=stem_stride,
        stem_padding=stem_padding,
        use_maxpool=use_maxpool,
    )


def ResNet152(
    num_classes,
    in_channels=3,
    resnet_block=Bottleneck,
    stem_kernel_size: int = 7,
    stem_stride: int = 2,
    stem_padding: int = 3,
    use_maxpool: bool = True,
):
    return ResNet(
        resnet_block,
        [3, 8, 36, 3],
        num_classes,
        in_channels,
        stem_kernel_size=stem_kernel_size,
        stem_stride=stem_stride,
        stem_padding=stem_padding,
        use_maxpool=use_maxpool,
    )


def STGResNet50(
    num_classes,
    in_channels=3,
    sigma: float = 0.5,
    resnet_block=STGBottleneckLayer,
    stem_kernel_size: int = 7,
    stem_stride: int = 2,
    stem_padding: int = 3,
    use_maxpool: bool = True,
):
    return ResNet(
        partial(resnet_block, sigma=sigma),
        [3, 4, 6, 3],
        num_classes,
        in_channels,
        stem_feature_selector_factory=partial(STGChannelLayer, sigma=sigma),
        stem_kernel_size=stem_kernel_size,
        stem_stride=stem_stride,
        stem_padding=stem_padding,
        use_maxpool=use_maxpool,
    )


def STGResNet101(
    num_classes,
    in_channels=3,
    sigma: float = 0.5,
    resnet_block=STGBottleneckLayer,
    stem_kernel_size: int = 7,
    stem_stride: int = 2,
    stem_padding: int = 3,
    use_maxpool: bool = True,
):
    return ResNet(
        partial(resnet_block, sigma=sigma),
        [3, 4, 23, 3],
        num_classes,
        in_channels,
        stem_feature_selector_factory=partial(STGChannelLayer, sigma=sigma),
        stem_kernel_size=stem_kernel_size,
        stem_stride=stem_stride,
        stem_padding=stem_padding,
        use_maxpool=use_maxpool,
    )


def STGResNet152(
    num_classes,
    in_channels=3,
    sigma: float = 0.5,
    resnet_block=STGBottleneckLayer,
    stem_kernel_size: int = 7,
    stem_stride: int = 2,
    stem_padding: int = 3,
    use_maxpool: bool = True,
):
    return ResNet(
        partial(resnet_block, sigma=sigma),
        [3, 8, 36, 3],
        num_classes,
        in_channels,
        stem_feature_selector_factory=partial(STGChannelLayer, sigma=sigma),
        stem_kernel_size=stem_kernel_size,
        stem_stride=stem_stride,
        stem_padding=stem_padding,
        use_maxpool=use_maxpool,
    )


def CIFARResNet20(num_classes, in_channels=3, resnet_block=CIFARBasicBlock, shortcut_option: str = "A"):
    return CIFARResNet(
        resnet_block,
        [3, 3, 3],
        num_classes=num_classes,
        in_channels=in_channels,
        shortcut_option=shortcut_option,
    )


def CIFARResNet32(num_classes, in_channels=3, resnet_block=CIFARBasicBlock, shortcut_option: str = "A"):
    return CIFARResNet(
        resnet_block,
        [5, 5, 5],
        num_classes=num_classes,
        in_channels=in_channels,
        shortcut_option=shortcut_option,
    )


def CIFARResNet44(num_classes, in_channels=3, resnet_block=CIFARBasicBlock, shortcut_option: str = "A"):
    return CIFARResNet(
        resnet_block,
        [7, 7, 7],
        num_classes=num_classes,
        in_channels=in_channels,
        shortcut_option=shortcut_option,
    )


def CIFARResNet56(num_classes, in_channels=3, resnet_block=CIFARBasicBlock, shortcut_option: str = "A"):
    return CIFARResNet(
        resnet_block,
        [9, 9, 9],
        num_classes=num_classes,
        in_channels=in_channels,
        shortcut_option=shortcut_option,
    )


def CIFARResNet110(num_classes, in_channels=3, resnet_block=CIFARBasicBlock, shortcut_option: str = "A"):
    return CIFARResNet(
        resnet_block,
        [18, 18, 18],
        num_classes=num_classes,
        in_channels=in_channels,
        shortcut_option=shortcut_option,
    )


def CIFARResNet1202(num_classes, in_channels=3, resnet_block=CIFARBasicBlock, shortcut_option: str = "A"):
    return CIFARResNet(
        resnet_block,
        [200, 200, 200],
        num_classes=num_classes,
        in_channels=in_channels,
        shortcut_option=shortcut_option,
    )
