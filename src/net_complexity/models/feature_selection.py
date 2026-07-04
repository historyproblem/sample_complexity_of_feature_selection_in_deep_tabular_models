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
                 bypass_on_zero_lambda: bool = True,
                 criterion=nn.CrossEntropyLoss(),
                 regularization_loss=lambda x: 0):
        super().__init__()
        self.backbone = backbone
        self.criterion = criterion
        self.lambda_coef = lambda_coef
        self.gumbel_init_mode = gumbel_init_mode
        self.bypass_on_zero_lambda = bool(bypass_on_zero_lambda)
        self.regularization_loss = regularization_loss
        self._initialize_gumbel_layers()
        self.set_aig_bypass(self._should_bypass_gumbel())

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

    def _should_bypass_gumbel(self) -> bool:
        return self.bypass_on_zero_lambda and float(self.lambda_coef) == 0.0

    def _resolve_gumbel_init_mode(self) -> str:
        mode = str(self.gumbel_init_mode).lower()
        if mode == "auto":
            return "fully_open" if self._should_bypass_gumbel() else "paper"
        if mode in {"paper", "paper_resnet50", "fully_open"}:
            return mode
        raise ValueError(
            "gumbel_init_mode must be one of: 'auto', 'paper', 'paper_resnet50', 'fully_open'."
        )

    def _initialize_gumbel_layers(self) -> None:
        init_mode = self._resolve_gumbel_init_mode()
        bypass_gumbel = self._should_bypass_gumbel()
        for module in get_gumbel_modules(self.backbone).values():
            module.reset_parameters(init_mode=init_mode)
            module.set_bypass(bypass_gumbel)

    def set_gumbel_bypass(self, enabled: bool) -> None:
        for module in get_gumbel_modules(self.backbone).values():
            module.set_bypass(enabled)

    def set_aig_bypass(self, enabled: bool) -> None:
        for module in get_AIG_modules(self.backbone).values():
            module.set_bypass(enabled)

    def set_gumbel_gate_modes(
        self,
        *,
        train_gate_mode: str | None = None,
        eval_gate_mode: str | None = None,
    ) -> None:
        for module in get_gumbel_modules(self.backbone).values():
            module.set_gate_modes(
                train_gate_mode=train_gate_mode,
                eval_gate_mode=eval_gate_mode,
            )

    def set_gumbel_open_bias(self, value: float, p_min: float = 0.02, p_max: float = 0.50) -> None:
        for module in get_gumbel_modules(self.backbone).values():
            module.set_open_bias(value, p_min=p_min, p_max=p_max)

    def set_lambda_coef(self, lambda_coef: float, *, bypass_gumbel: bool | None = None) -> None:
        self.lambda_coef = float(lambda_coef)
        if bypass_gumbel is None:
            bypass_gumbel = self._should_bypass_gumbel()
        self.set_gumbel_bypass(bool(bypass_gumbel))
        self.set_aig_bypass(bool(bypass_gumbel))


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
    PAPER_RESNET50_INIT_OFF_LOGIT = 0.1
    PAPER_RESNET50_INIT_ON_LOGIT = 4.0
    PAPER_RESNET50_INIT_NOISE_SCALE = 0.02
    FULLY_OPEN_OFF_LOGIT = -10.0
    FULLY_OPEN_ON_LOGIT = 10.0
    FULLY_OPEN_NOISE_SCALE = 0.0
    VALID_GATE_MODES = {
        "gumbel_hard",
        "deterministic_soft",
        "deterministic_hard",
        "ste_hard",
        "ones",
    }

    def __init__(
        self,
        input_dim: int,
        temperature: float = 1.0,
        beta: float = 1.0,
        force_ones_mask: bool = False,
        deterministic_soft_mask: bool = False,
        deterministic_hard_mask: bool = False,
        train_gate_mode: str | None = None,
        eval_gate_mode: str | None = None,
        gate_threshold: float = 0.5,
    ):
        super().__init__()
        if beta < 0:
            raise ValueError("beta must be non-negative for GumbelLayer.")
        self.logits = nn.Parameter(torch.empty(input_dim, 2))
        self.temperature = temperature
        self.beta = float(beta)
        self.gate_threshold = float(gate_threshold)
        if not 0.0 <= self.gate_threshold <= 1.0:
            raise ValueError("gate_threshold must be within [0.0, 1.0].")
        (
            self.force_ones_mask,
            self.deterministic_soft_mask,
            self.deterministic_hard_mask,
            self.train_gate_mode,
            self.eval_gate_mode,
        ) = self._resolve_gate_mode_configuration(
            force_ones_mask=force_ones_mask,
            deterministic_soft_mask=deterministic_soft_mask,
            deterministic_hard_mask=deterministic_hard_mask,
            train_gate_mode=train_gate_mode,
            eval_gate_mode=eval_gate_mode,
        )
        self._bypass = False
        self._open_bias = 0.0
        self._open_bias_p_min = 0.02
        self._open_bias_p_max = 0.50
        self.reset_parameters()

    @classmethod
    def _normalize_gate_mode(cls, mode: str | None) -> str | None:
        if mode is None:
            return None
        normalized = str(mode).strip().lower()
        if normalized not in cls.VALID_GATE_MODES:
            allowed = ", ".join(sorted(cls.VALID_GATE_MODES))
            raise ValueError(f"gate mode must be one of: {allowed}. Got: {mode!r}")
        return normalized

    @classmethod
    def _resolve_gate_mode_configuration(
        cls,
        *,
        force_ones_mask: bool,
        deterministic_soft_mask: bool,
        deterministic_hard_mask: bool,
        train_gate_mode: str | None,
        eval_gate_mode: str | None,
    ) -> tuple[bool, bool, bool, str, str]:
        legacy_flags = {
            "ones": bool(force_ones_mask),
            "deterministic_soft": bool(deterministic_soft_mask),
            "deterministic_hard": bool(deterministic_hard_mask),
        }
        enabled_legacy_modes = [mode for mode, enabled in legacy_flags.items() if enabled]
        normalized_train = cls._normalize_gate_mode(train_gate_mode)
        normalized_eval = cls._normalize_gate_mode(eval_gate_mode)

        if enabled_legacy_modes and (normalized_train is not None or normalized_eval is not None):
            raise ValueError(
                "Legacy mask flags and explicit train/eval gate modes are mutually exclusive."
            )
        if len(enabled_legacy_modes) > 1:
            raise ValueError(
                "force_ones_mask, deterministic_soft_mask, and deterministic_hard_mask are mutually exclusive."
            )

        if enabled_legacy_modes:
            legacy_mode = enabled_legacy_modes[0]
            normalized_train = legacy_mode
            normalized_eval = legacy_mode

        if normalized_train is None:
            normalized_train = "gumbel_hard"
        if normalized_eval is None:
            normalized_eval = "deterministic_hard"

        return (
            bool(force_ones_mask),
            bool(deterministic_soft_mask),
            bool(deterministic_hard_mask),
            normalized_train,
            normalized_eval,
        )

    def reset_parameters(self, init_mode: str = "paper") -> None:
        mode = str(init_mode).lower()
        if mode == "paper":
            off_logit = self.PAPER_INIT_OFF_LOGIT
            on_logit = self.PAPER_INIT_ON_LOGIT
            noise_scale = self.PAPER_INIT_NOISE_SCALE
        elif mode == "paper_resnet50":
            off_logit = self.PAPER_RESNET50_INIT_OFF_LOGIT
            on_logit = self.PAPER_RESNET50_INIT_ON_LOGIT
            noise_scale = self.PAPER_RESNET50_INIT_NOISE_SCALE
        elif mode == "fully_open":
            off_logit = self.FULLY_OPEN_OFF_LOGIT
            on_logit = self.FULLY_OPEN_ON_LOGIT
            noise_scale = self.FULLY_OPEN_NOISE_SCALE
        else:
            raise ValueError(
                "init_mode must be one of: 'paper', 'paper_resnet50', 'fully_open'."
            )

        with torch.no_grad():
            self.logits[:, 0].fill_(float(off_logit))
            self.logits[:, 1].fill_(float(on_logit))
            if float(noise_scale) > 0.0:
                self.logits.add_(torch.randn_like(self.logits) * float(noise_scale))

    def set_bypass(self, enabled: bool) -> None:
        self._bypass = bool(enabled)

    def set_gate_modes(
        self,
        *,
        train_gate_mode: str | None = None,
        eval_gate_mode: str | None = None,
    ) -> None:
        if train_gate_mode is not None:
            self.train_gate_mode = self._normalize_gate_mode(train_gate_mode)
        if eval_gate_mode is not None:
            self.eval_gate_mode = self._normalize_gate_mode(eval_gate_mode)
        self.force_ones_mask = self.train_gate_mode == "ones" and self.eval_gate_mode == "ones"
        self.deterministic_soft_mask = (
            self.train_gate_mode == "deterministic_soft"
            and self.eval_gate_mode == "deterministic_soft"
        )
        self.deterministic_hard_mask = (
            self.train_gate_mode == "deterministic_hard"
            and self.eval_gate_mode == "deterministic_hard"
        )

    @property
    def bypass(self) -> bool:
        return self._bypass

    @property
    def open_bias(self) -> float:
        return self._open_bias

    def set_open_bias(self, value: float, p_min: float = 0.02, p_max: float = 0.50) -> None:
        value = float(value)
        p_min = float(p_min)
        p_max = float(p_max)
        if value < 0.0:
            raise ValueError("open bias must be non-negative.")
        if not 0.0 <= p_min < p_max <= 1.0:
            raise ValueError(
                "open bias probability bounds must satisfy 0.0 <= p_min < p_max <= 1.0."
            )
        self._open_bias = value
        self._open_bias_p_min = p_min
        self._open_bias_p_max = p_max

    def _raw_selection_probs(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=1)[:, 1]

    def _revive_mask(self, probs_on: torch.Tensor) -> torch.Tensor:
        return (probs_on > self._open_bias_p_min) & (probs_on < self._open_bias_p_max)

    def _effective_logits(self) -> torch.Tensor:
        if self._open_bias <= 0.0:
            return self.logits
        raw_probs_on = self._raw_selection_probs()
        revive_mask = self._revive_mask(raw_probs_on)
        if not bool(revive_mask.any().item()):
            return self.logits
        logits = self.logits.clone()
        logits[:, 1] = logits[:, 1] + float(self._open_bias) * revive_mask.to(
            dtype=logits.dtype
        )
        return logits

    def get_open_bias_candidate_stats(
        self,
        p_min: float | None = None,
        p_max: float | None = None,
    ) -> dict[str, float | int]:
        p_min = self._open_bias_p_min if p_min is None else float(p_min)
        p_max = self._open_bias_p_max if p_max is None else float(p_max)
        with torch.no_grad():
            probs_on = self._raw_selection_probs().detach()
            revive_mask = (probs_on > p_min) & (probs_on < p_max)
            candidate_probs = probs_on[revive_mask]
            return {
                "num_channels": int(probs_on.numel()),
                "revive_candidates": int(revive_mask.sum().item()),
                "sum_p_open_candidates": (
                    float(candidate_probs.sum().item()) if candidate_probs.numel() else 0.0
                ),
                "sum_p_open_all": float(probs_on.sum().item()),
            }

    def _selection_probs(self, batch_size: int) -> torch.Tensor:
        probs_on = F.softmax(self._effective_logits(), dim=1)[:, 1].unsqueeze(0)
        return probs_on.expand(batch_size, -1)

    def _hard_threshold_mask(self, batch_size: int) -> torch.Tensor:
        probs_on = self._selection_probs(batch_size)
        return (probs_on > self.gate_threshold).float()

    def _sample_gumbel_like(self, template_tensor: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        uniform_samples = torch.rand_like(template_tensor)
        uniform_samples = uniform_samples.clamp_(min=eps, max=1.0 - eps)
        return -torch.log(-torch.log(uniform_samples))

    def _sample_gumbel_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        gumbel_noise = self._sample_gumbel_like(logits) * self.beta
        return F.softmax((logits + gumbel_noise) / self.temperature, dim=-1)

    def _straight_through_hard_sample(self, soft_samples: torch.Tensor) -> torch.Tensor:
        max_value_indexes = soft_samples.argmax(dim=-1, keepdim=True)
        hard_samples = torch.zeros_like(soft_samples).scatter_(-1, max_value_indexes, 1.0)
        return hard_samples - soft_samples.detach() + soft_samples

    def _compute_mode_gates(self, mode: str, batch_size: int) -> torch.Tensor:
        if mode == "ones":
            return self.logits.new_ones((batch_size, self.logits.shape[0]))
        if mode == "deterministic_soft":
            return self._selection_probs(batch_size)
        if mode == "deterministic_hard":
            return self._hard_threshold_mask(batch_size)
        if mode == "ste_hard":
            soft_mask = self._selection_probs(batch_size)
            hard_mask = self._hard_threshold_mask(batch_size)
            return hard_mask.detach() - soft_mask.detach() + soft_mask
        if mode == "gumbel_hard":
            logits = self._effective_logits().unsqueeze(0).expand(batch_size, -1, -1)
            soft_samples = self._sample_gumbel_softmax(logits)
            sampled = self._straight_through_hard_sample(soft_samples)
            return sampled[..., 1]
        raise AssertionError(f"Unhandled gate mode: {mode}")

    # ACTUAL: computes broadcastable Gumbel gates for the current CIFAR main pipeline.
    def compute_gates(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if self._bypass:
            gates = x.new_ones((batch_size, self.logits.shape[0]))
            while len(gates.shape) < len(x.shape):
                gates = gates.unsqueeze(-1)
            return gates
        active_mode = self.train_gate_mode if self.training else self.eval_gate_mode
        gates = self._compute_mode_gates(active_mode, batch_size)
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
        return F.softmax(self._effective_logits(), dim=1)[:, 1].detach()

    # ACTUAL: temperature control hook for the current Gumbel layer implementation.
    def set_temperature(self, temperature: float):
        """Update temperature for annealing schedule."""
        self.temperature = temperature

    def set_beta(self, beta: float):
        """Update the Gumbel noise scale."""
        if beta < 0:
            raise ValueError("beta must be non-negative for GumbelLayer.")
        self.beta = float(beta)


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
        beta: float = 1.0,
        force_ones_mask: bool = False,
        deterministic_soft_mask: bool = False,
        deterministic_hard_mask: bool = False,
        train_gate_mode: str | None = None,
        eval_gate_mode: str | None = None,
        gate_threshold: float = 0.5,
    ):
        super().__init__(in_channels, out_channels, i_downsample=i_downsample, stride=stride)
        self.gumbel_layer = GumbelLayer(
            input_dim=out_channels * self.expansion,
            temperature=temperature,
            beta=beta,
            force_ones_mask=force_ones_mask,
            deterministic_soft_mask=deterministic_soft_mask,
            deterministic_hard_mask=deterministic_hard_mask,
            train_gate_mode=train_gate_mode,
            eval_gate_mode=eval_gate_mode,
            gate_threshold=gate_threshold,
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
    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        option: str = "A",
        temperature: float = 1,
        beta: float = 1.0,
        force_ones_mask: bool = False,
        deterministic_soft_mask: bool = False,
        deterministic_hard_mask: bool = False,
        train_gate_mode: str | None = None,
        eval_gate_mode: str | None = None,
        gate_threshold: float = 0.5,
    ):
        super().__init__(in_planes, planes, stride=stride, option=option)
        self.gumbel_layer = GumbelLayer(
            input_dim=planes * self.expansion,
            temperature=temperature,
            beta=beta,
            force_ones_mask=force_ones_mask,
            deterministic_soft_mask=deterministic_soft_mask,
            deterministic_hard_mask=deterministic_hard_mask,
            train_gate_mode=train_gate_mode,
            eval_gate_mode=eval_gate_mode,
            gate_threshold=gate_threshold,
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.gumbel_layer(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MaskedGumbelLayer(GumbelLayer):
    """GumbelLayer that permanently disables a subset of channels.

    Disabled channels: gate is always 0 regardless of learned logits.
    They are excluded from the regularisation mean and reported as
    zero in get_selection_probs().  Logits of disabled channels receive
    no gradient and are never updated.

    Two ways to disable channels:
      1. Pass disabled_channels at construction time (same list for every
         block when set via the YAML resnet_block config).
      2. Call apply_channel_mask() from channel_pruning after construction
         to set per-layer channel sets loaded from a previous run.
    """

    def __init__(
        self,
        input_dim: int,
        temperature: float = 1.0,
        disabled_channels: list[int] | None = None,
    ):
        super().__init__(input_dim, temperature)
        channel_mask = torch.ones(input_dim)
        if disabled_channels:
            for ch in disabled_channels:
                if 0 <= ch < input_dim:
                    channel_mask[ch] = 0.0
        self.register_buffer("channel_mask", channel_mask)

    def compute_gates(self, x: torch.Tensor) -> torch.Tensor:
        gates = super().compute_gates(x)
        # channel_mask: [C] → reshape to [1, C, 1, 1, ...] for broadcast
        mask = self.channel_mask.view(1, -1, *([1] * (len(x.shape) - 2)))
        return gates * mask

    def regularization_loss(self) -> torch.Tensor:
        probs = F.softmax(self.logits, dim=1)[:, 1]
        total_enabled = self.channel_mask.sum()
        if total_enabled == 0:
            return torch.tensor(0.0, device=self.logits.device)
        return (probs * self.channel_mask).sum() / total_enabled

    def get_selection_probs(self) -> torch.Tensor:
        probs = super().get_selection_probs()
        return probs * self.channel_mask


class CIFARMaskedGumbelBasicBlock(CIFARBasicBlock):
    """CIFAR BasicBlock with a MaskedGumbelLayer for per-channel gating.

    Functionally identical to CIFARGumbelBasicBlock but uses
    MaskedGumbelLayer, which supports permanently disabling channels
    identified by a prior Gumbel training run.
    """

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        option: str = "A",
        temperature: float = 1.0,
        disabled_channels: list[int] | None = None,
    ):
        super().__init__(in_planes, planes, stride=stride, option=option)
        self.gumbel_layer = MaskedGumbelLayer(
            input_dim=planes * self.expansion,
            temperature=temperature,
            disabled_channels=disabled_channels,
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.gumbel_layer(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SkippedCIFARBasicBlock(CIFARBasicBlock):
    """CIFARBasicBlock with its residual branch permanently disabled.

    Forward pass returns only the shortcut output (identity or spatially
    downsampled), making this block a no-op residual.  All conv/BN weights
    are retained for checkpoint compatibility but never executed.

    Use this for ablation studies where specific blocks are excluded from
    the residual computation while preserving the overall network depth.
    """

    def forward(self, x):
        return F.relu(self.shortcut(x))


class SkippedBottleneck(Bottleneck):
    """Bottleneck with its residual branch permanently disabled.

    Forward pass returns only the shortcut output (identity or downsampled).
    All conv/BN weights are retained for checkpoint compatibility but never executed.
    """

    def forward(self, x):
        if self.i_downsample is not None:
            x = self.i_downsample(x)
        return self.relu(x)


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
        self.bypass = False
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

    def set_bypass(self, enabled: bool) -> None:
        self.bypass = bool(enabled)

    def forward(self, x):
        if self.bypass:
            self.activations = x.new_ones((x.shape[0], 1, 1, 1))
        else:
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
        x = identity + x*self.activations
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


class AIGRegularizationLoss:
    """Callable regularization loss over AIG gate activations.

    Supports L1 (linear penalty on mean gate probability, encourages sparser gates)
    and L2 (quadratic penalty, matches the legacy ``get_AIG_regularization_loss`` behaviour).

    Args:
        norm: ``"l1"`` or ``"l2"`` (default).

    Example YAML usage::

        model:
          regularization_loss:
            _target_: net_complexity.wrappers.AIGRegularizationLoss
            norm: l1
    """

    _SUPPORTED_NORMS = frozenset({"l1", "l2"})

    def __init__(self, norm: str = "l2") -> None:
        if norm not in self._SUPPORTED_NORMS:
            raise ValueError(
                f"AIGRegularizationLoss: norm must be one of "
                f"{sorted(self._SUPPORTED_NORMS)}, got {norm!r}"
            )
        self.norm = norm

    def __call__(self, model: nn.Module):
        aig_modules = _get_aig_modules(model)
        activations = [
            module.activations
            for module in aig_modules.values()
            if getattr(module, "activations", None) is not None
        ]
        if not activations:
            return 0.0

        reg_loss = 0.0
        for act in activations:
            mean_act = act.mean()
            reg_loss = reg_loss + (mean_act.abs() if self.norm == "l1" else mean_act ** 2)
        return reg_loss / len(activations)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(norm={self.norm!r})"


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
