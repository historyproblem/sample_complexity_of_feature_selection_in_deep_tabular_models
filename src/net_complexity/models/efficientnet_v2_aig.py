from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn

from .aig import (
    AIGBlockGate,
    bernoulli_kl_from_closed_open_log_odds,
    entropy_regularization_sign,
    normalize_posterior_kl_reduction,
)
from .outputs import ClassifModelOutput


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        norm_layer: type[nn.Module],
        act_layer: type[nn.Module],
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_channels),
            act_layer(),
        )


class SEUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 4,
        act_layer: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        hidden_channels = max(1, in_channels // int(reduction_ratio))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
        self.act1 = act_layer()
        self.act2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.act1(scale)
        scale = self.fc2(scale)
        scale = self.act2(scale)
        return x * scale


class StochasticDepth(nn.Module):
    def __init__(self, prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= prob < 1.0:
            raise ValueError("stochastic depth probability must be in [0, 1).")
        self.prob = float(prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prob == 0.0 or not self.training:
            return x

        survival = 1.0 - self.prob
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(survival)
        return x * mask.div_(survival)


@dataclass
class MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    in_channels: int
    out_channels: int
    num_layers: int
    use_se: bool
    fused: bool

    @staticmethod
    def adjust_channels(channels: int, factor: float, divisible: int = 8) -> int:
        scaled = channels * factor
        rounded = max(divisible, int(scaled + divisible / 2) // divisible * divisible)
        if rounded < 0.9 * scaled:
            rounded += divisible
        return int(rounded)


class EfficientNetV2AIGBlock(nn.Module):
    def __init__(
        self,
        config: MBConvConfig,
        *,
        sd_prob: float = 0.0,
        gate_hidden_channels: int = 16,
        keep_prob_init: float = 0.9,
        gate_threshold: float = 0.5,
        gate_temperature: float = 1.0,
        gate_regularization: str = "l2_gate",
        act_layer: type[nn.Module] = nn.SiLU,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        inter_channels = config.adjust_channels(config.in_channels, config.expand_ratio)
        layers: list[tuple[str, nn.Module]] = []

        if config.expand_ratio == 1:
            layers.append(
                (
                    "fused",
                    ConvBNAct(
                        config.in_channels,
                        inter_channels,
                        config.kernel,
                        config.stride,
                        groups=1,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                    ),
                )
            )
        elif config.fused:
            layers.append(
                (
                    "fused",
                    ConvBNAct(
                        config.in_channels,
                        inter_channels,
                        config.kernel,
                        config.stride,
                        groups=1,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                    ),
                )
            )
            layers.append(
                (
                    "fused_point_wise",
                    ConvBNAct(
                        inter_channels,
                        config.out_channels,
                        1,
                        1,
                        groups=1,
                        norm_layer=norm_layer,
                        act_layer=nn.Identity,
                    ),
                )
            )
        else:
            layers.append(
                (
                    "linear_bottleneck",
                    ConvBNAct(
                        config.in_channels,
                        inter_channels,
                        1,
                        1,
                        groups=1,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                    ),
                )
            )
            layers.append(
                (
                    "depth_wise",
                    ConvBNAct(
                        inter_channels,
                        inter_channels,
                        config.kernel,
                        config.stride,
                        groups=inter_channels,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                    ),
                )
            )
            if config.use_se:
                layers.append(("se", SEUnit(inter_channels, int(4 * config.expand_ratio), act_layer)))
            layers.append(
                (
                    "point_wise",
                    ConvBNAct(
                        inter_channels,
                        config.out_channels,
                        1,
                        1,
                        groups=1,
                        norm_layer=norm_layer,
                        act_layer=nn.Identity,
                    ),
                )
            )

        self.branch = nn.Sequential(OrderedDict(layers))
        self.use_skip_connection = (
            config.stride == 1 and config.in_channels == config.out_channels
        )
        self.stochastic_path = StochasticDepth(sd_prob)
        self.gate = (
            AIGBlockGate(
                config.in_channels,
                hidden_channels=gate_hidden_channels,
                keep_prob_init=keep_prob_init,
                threshold=gate_threshold,
                temperature=gate_temperature,
                regularization=gate_regularization,
            )
            if self.use_skip_connection
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.branch(x)
        if not self.use_skip_connection:
            return residual

        residual = self.stochastic_path(residual)
        if self.gate is None:
            return x + residual
        return x + self.gate(x) * residual


def efficientnet_v2_structure(
    variant: str,
) -> list[tuple[float, int, int, int, int, int, bool, bool]]:
    variant = str(variant).lower()
    if variant in {"s", "efficientnet_v2_s", "efficientnetv2_s"}:
        return [
            # expand, kernel, stride, in, out, layers, se, fused
            (1, 3, 1, 24, 24, 2, False, True),
            (4, 3, 2, 24, 48, 4, False, True),
            (4, 3, 2, 48, 64, 4, False, True),
            (4, 3, 2, 64, 128, 6, True, False),
            (6, 3, 1, 128, 160, 9, True, False),
            (6, 3, 2, 160, 256, 15, True, False),
        ]
    if variant in {"m", "efficientnet_v2_m", "efficientnetv2_m"}:
        return [
            # expand, kernel, stride, in, out, layers, se, fused
            (1, 3, 1, 24, 24, 3, False, True),
            (4, 3, 2, 24, 48, 5, False, True),
            (4, 3, 2, 48, 80, 5, False, True),
            (4, 3, 2, 80, 160, 7, True, False),
            (6, 3, 1, 160, 176, 14, True, False),
            (6, 3, 2, 176, 304, 18, True, False),
            (6, 3, 1, 304, 512, 5, True, False),
        ]
    raise ValueError("EfficientNetV2 AIG variant must be one of: 's', 'm'.")


def efficientnet_v2_s_structure() -> list[tuple[float, int, int, int, int, int, bool, bool]]:
    return efficientnet_v2_structure("s")


def efficientnet_v2_m_structure() -> list[tuple[float, int, int, int, int, int, bool, bool]]:
    return efficientnet_v2_structure("m")


class AIGEfficientNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        variant: str = "s",
        lambda_coef: float = 0.0,
        bypass_on_zero_lambda: bool = False,
        dropout: float = 0.1,
        stochastic_depth: float = 0.0,
        gate_hidden_channels: int = 16,
        keep_prob_init: float = 0.9,
        gate_threshold: float = 0.5,
        gate_temperature: float = 1.0,
        gate_regularization: str = "l2_gate",
        entropy_regularization: str = "disabled",
        posterior_kl_reduction: str = "mean",
        stem_stride: int = 1,
        criterion: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.variant = str(variant).lower()
        self.lambda_coef = float(lambda_coef)
        self.bypass_on_zero_lambda = bool(bypass_on_zero_lambda)
        self.entropy_regularization = str(entropy_regularization).strip().lower()
        self.entropy_regularization_sign = entropy_regularization_sign(
            self.entropy_regularization
        )
        self.posterior_kl_reduction = normalize_posterior_kl_reduction(
            posterior_kl_reduction
        )
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

        layer_infos = [
            MBConvConfig(*layer_config)
            for layer_config in efficientnet_v2_structure(self.variant)
        ]
        self.in_channel = layer_infos[0].in_channels
        self.final_stage_channel = layer_infos[-1].out_channels
        self.out_channels = 1280
        self.num_blocks = sum(stage.num_layers for stage in layer_infos)
        self.cur_block = 0
        self.stochastic_depth = float(stochastic_depth)

        self.stem = ConvBNAct(
            in_channels,
            self.in_channel,
            3,
            stem_stride,
            groups=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.SiLU,
        )
        self.blocks = nn.Sequential(
            *self._make_stages(
                layer_infos,
                gate_hidden_channels=gate_hidden_channels,
                keep_prob_init=keep_prob_init,
                gate_threshold=gate_threshold,
                gate_temperature=gate_temperature,
                gate_regularization=gate_regularization,
            )
        )
        self.head = nn.Sequential(
            OrderedDict(
                [
                    (
                        "bottleneck",
                        ConvBNAct(
                            self.final_stage_channel,
                            self.out_channels,
                            1,
                            1,
                            groups=1,
                            norm_layer=nn.BatchNorm2d,
                            act_layer=nn.SiLU,
                        ),
                    ),
                    ("avgpool", nn.AdaptiveAvgPool2d((1, 1))),
                    ("flatten", nn.Flatten()),
                    ("dropout", nn.Dropout(p=dropout, inplace=True)),
                    ("classifier", nn.Linear(self.out_channels, num_classes)),
                ]
            )
        )

        self._init_weights()
        for gate in self._iter_gates():
            gate.reset_parameters()
        self.set_aig_bypass(self._should_bypass_aig())

    def _make_stages(
        self,
        layer_infos: list[MBConvConfig],
        *,
        gate_hidden_channels: int,
        keep_prob_init: float,
        gate_threshold: float,
        gate_temperature: float,
        gate_regularization: str,
    ) -> list[nn.Module]:
        return [
            layer
            for layer_info in layer_infos
            for layer in self._make_layers(
                copy.copy(layer_info),
                gate_hidden_channels=gate_hidden_channels,
                keep_prob_init=keep_prob_init,
                gate_threshold=gate_threshold,
                gate_temperature=gate_temperature,
                gate_regularization=gate_regularization,
            )
        ]

    def _make_layers(
        self,
        layer_info: MBConvConfig,
        *,
        gate_hidden_channels: int,
        keep_prob_init: float,
        gate_threshold: float,
        gate_temperature: float,
        gate_regularization: str,
    ) -> list[nn.Module]:
        layers = []
        for _ in range(layer_info.num_layers):
            layers.append(
                EfficientNetV2AIGBlock(
                    layer_info,
                    sd_prob=self._next_sd_prob(),
                    gate_hidden_channels=gate_hidden_channels,
                    keep_prob_init=keep_prob_init,
                    gate_threshold=gate_threshold,
                    gate_temperature=gate_temperature,
                    gate_regularization=gate_regularization,
                )
            )
            layer_info.in_channels = layer_info.out_channels
            layer_info.stride = 1
        return layers

    def _next_sd_prob(self) -> float:
        sd_prob = self.stochastic_depth * (self.cur_block / self.num_blocks)
        self.cur_block += 1
        return float(sd_prob)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _should_bypass_aig(self) -> bool:
        return (
            self.bypass_on_zero_lambda
            and float(self.lambda_coef) == 0.0
            and self.entropy_regularization == "disabled"
        )

    def _iter_gates(self):
        for module in self.modules():
            if isinstance(module, AIGBlockGate):
                yield module

    def set_aig_bypass(self, enabled: bool) -> None:
        for gate in self._iter_gates():
            gate.set_bypass(enabled)

    def set_lambda_coef(
        self,
        lambda_coef: float,
        *,
        bypass_gumbel: bool | None = None,
    ) -> None:
        self.lambda_coef = float(lambda_coef)
        if bypass_gumbel is None:
            bypass = self._should_bypass_aig()
        elif self.entropy_regularization != "disabled":
            bypass = False
        else:
            bypass = bool(bypass_gumbel)
        self.set_aig_bypass(bypass)

    def _collect_aux(
        self,
        logits: torch.Tensor,
    ) -> dict[str, torch.Tensor | int | None]:
        probabilities = []
        values = []
        gate_losses = []
        posterior_terms = []

        for gate in self._iter_gates():
            if gate.keep_probabilities is None or gate.activations is None:
                continue
            probabilities.append(gate.keep_probabilities.flatten(1))
            values.append(gate.activations.flatten(1))
            gate_losses.append(gate.regularization_loss())
            if gate.regularization == "l1_probability":
                posterior_terms.append(gate.posterior_regularization_terms())

        if probabilities:
            gate_probabilities = torch.cat(probabilities, dim=1)
            gate_values = torch.cat(values, dim=1)
            gate_loss = torch.stack(gate_losses).mean()
            mean_active_ratio = gate_values.mean()
        else:
            batch_size = logits.shape[0]
            gate_probabilities = logits.new_empty((batch_size, 0))
            gate_values = logits.new_empty((batch_size, 0))
            gate_loss = logits.new_zeros(())
            mean_active_ratio = logits.new_ones(())

        if posterior_terms and len(posterior_terms) == len(gate_losses):
            mean_p_open = torch.stack([term[0] for term in posterior_terms]).mean()
            negative_entropy = torch.stack([term[1] for term in posterior_terms]).mean()
        else:
            mean_p_open = None
            negative_entropy = None

        return {
            "gate_probabilities": gate_probabilities,
            "gate_values": gate_values,
            "mean_active_ratio": mean_active_ratio,
            "gate_loss": gate_loss,
            "mean_p_open": mean_p_open,
            "negative_entropy": negative_entropy,
            "posterior_gate_count": len(posterior_terms),
        }

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | int | None]] | ClassifModelOutput:
        x = self.stem(x)
        x = self.blocks(x)
        logits = self.head(x)
        aux = self._collect_aux(logits)

        if y is None:
            return logits, aux

        ce_loss = self.criterion(logits, y)
        mean_p_open = aux["mean_p_open"]
        negative_entropy = aux["negative_entropy"]
        if mean_p_open is not None and negative_entropy is not None:
            if self.entropy_regularization == "bernoulli_kl":
                gate_loss = bernoulli_kl_from_closed_open_log_odds(
                    mean_p_open,
                    negative_entropy,
                    self.lambda_coef,
                )
                if self.posterior_kl_reduction == "sum":
                    gate_loss = gate_loss * int(aux["posterior_gate_count"])
            else:
                gate_loss = (
                    float(self.lambda_coef) * mean_p_open
                    + self.entropy_regularization_sign * negative_entropy
                )
        elif self.entropy_regularization != "disabled":
            raise ValueError(
                "AIG entropy regularization requires every gate to use "
                "gate_regularization='l1_probability'."
            )
        else:
            gate_loss = float(self.lambda_coef) * aux["gate_loss"]
        loss = ce_loss + gate_loss
        return ClassifModelOutput(
            ce_loss=ce_loss,
            regularization_loss=aux["gate_loss"],
            reg_loss=gate_loss,
            mean_p_open=mean_p_open,
            negative_entropy=negative_entropy,
            loss=loss,
            logits=logits,
            mean_activations=[aux["mean_active_ratio"]],
        )


class AIGEfficientNetV2S(AIGEfficientNetV2):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("variant", "s")
        super().__init__(*args, **kwargs)


class AIGEfficientNetV2M(AIGEfficientNetV2):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("variant", "m")
        super().__init__(*args, **kwargs)
