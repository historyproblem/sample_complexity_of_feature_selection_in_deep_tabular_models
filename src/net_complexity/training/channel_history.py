from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from net_complexity.models.feature_selection import get_gumbel_modules, get_stg_modules


CHANNEL_HISTORY_FIELDNAMES = [
    "epoch",
    "stage_name",
    "stage_index",
    "block_index",
    "layer_name",
    "channel_index",
    "selection_prob",
    "zero_prob",
    "mu",
    "sigma",
    "logit_off",
    "logit_on",
    "logit_margin",
    "temperature",
    "beta",
]

_CIFAR_SELECTOR_RE = re.compile(
    r"^backbone\.(?P<stage_name>layer(?P<stage_index>\d+))\.(?P<block_index>\d+)\.(?P<selector_name>[A-Za-z0-9_]+)$"
)


def _to_float_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _parse_cifar_selector_name(layer_name: str) -> dict[str, Any]:
    match = _CIFAR_SELECTOR_RE.match(layer_name)
    if match is None:
        raise ValueError(
            "Unsupported selector module name for CIFAR ResNet channel history: "
            f"'{layer_name}'. Expected names like 'backbone.layer1.0.stg_layer'."
        )
    return {
        "stage_name": str(match.group("stage_name")),
        "stage_index": int(match.group("stage_index")),
        "block_index": int(match.group("block_index")),
    }


def _sorted_modules(modules: dict[str, nn.Module]) -> list[tuple[str, nn.Module]]:
    def _sort_key(item: tuple[str, nn.Module]) -> tuple[int, int, str]:
        layer_name, _ = item
        parsed = _parse_cifar_selector_name(layer_name)
        return parsed["stage_index"], parsed["block_index"], layer_name

    return sorted(modules.items(), key=_sort_key)


@dataclass(frozen=True)
class _CollectorSpec:
    backbone_target: str
    block_target: str


class BaseChannelHistoryCollector:
    def collect(self, model: nn.Module, epoch: int) -> list[dict[str, Any]]:
        raise NotImplementedError


class CifarResNet20STGCollector(BaseChannelHistoryCollector):
    def collect(self, model: nn.Module, epoch: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer_name, module in _sorted_modules(get_stg_modules(model)):
            parsed = _parse_cifar_selector_name(layer_name)
            selection_probs = _to_float_list(module.get_selection_probs())
            mu_values = _to_float_list(module.mu)
            sigma = float(module.sigma)

            for channel_index, selection_prob in enumerate(selection_probs):
                rows.append({
                    "epoch": int(epoch),
                    "stage_name": parsed["stage_name"],
                    "stage_index": parsed["stage_index"],
                    "block_index": parsed["block_index"],
                    "layer_name": layer_name,
                    "channel_index": int(channel_index),
                    "selection_prob": float(selection_prob),
                    "zero_prob": float(1.0 - selection_prob),
                    "mu": float(mu_values[channel_index]),
                    "sigma": sigma,
                    "logit_off": None,
                    "logit_on": None,
                    "logit_margin": None,
                    "temperature": None,
                    "beta": None,
                })
        return rows


class CifarResNet20GumbelCollector(BaseChannelHistoryCollector):
    def collect(self, model: nn.Module, epoch: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer_name, module in _sorted_modules(get_gumbel_modules(model)):
            parsed = _parse_cifar_selector_name(layer_name)
            selection_probs = _to_float_list(module.get_selection_probs())
            logits = module.logits.detach().cpu()
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError(
                    "Gumbel channel history expects selector logits with shape [channels, 2], "
                    f"got {tuple(logits.shape)} for '{layer_name}'."
                )
            logit_pairs = logits.tolist()
            temperature = float(module.temperature)
            beta = float(module.beta)

            for channel_index, selection_prob in enumerate(selection_probs):
                logit_off = float(logit_pairs[channel_index][0])
                logit_on = float(logit_pairs[channel_index][1])
                rows.append({
                    "epoch": int(epoch),
                    "stage_name": parsed["stage_name"],
                    "stage_index": parsed["stage_index"],
                    "block_index": parsed["block_index"],
                    "layer_name": layer_name,
                    "channel_index": int(channel_index),
                    "selection_prob": float(selection_prob),
                    "zero_prob": float(1.0 - selection_prob),
                    "mu": None,
                    "sigma": None,
                    "logit_off": logit_off,
                    "logit_on": logit_on,
                    "logit_margin": float(logit_on - logit_off),
                    "temperature": temperature,
                    "beta": beta,
                })
        return rows


_COLLECTORS: dict[_CollectorSpec, BaseChannelHistoryCollector] = {
    _CollectorSpec(
        backbone_target="net_complexity.wrappers.CIFARResNet20",
        block_target="net_complexity.wrappers.CIFARSTGBasicBlock",
    ): CifarResNet20STGCollector(),
    _CollectorSpec(
        backbone_target="net_complexity.wrappers.CIFARResNet20",
        block_target="net_complexity.wrappers.CIFARGumbelBasicBlock",
    ): CifarResNet20GumbelCollector(),
    _CollectorSpec(
        backbone_target="net_complexity.wrappers.CIFARResNet20",
        block_target="net_complexity.wrappers.CIFARMaskedGumbelBasicBlock",
    ): CifarResNet20GumbelCollector(),
}


def resolve_channel_history_collector(config: DictConfig) -> BaseChannelHistoryCollector:
    backbone_target = str(OmegaConf.select(config, "model.backbone._target_") or "")
    block_target = str(OmegaConf.select(config, "model.backbone.resnet_block._target_") or "")
    collector = _COLLECTORS.get(
        _CollectorSpec(
            backbone_target=backbone_target,
            block_target=block_target,
        )
    )
    if collector is None:
        raise ValueError(
            "No channel history collector is registered for this model configuration: "
            f"model.backbone._target_='{backbone_target}', "
            f"model.backbone.resnet_block._target_='{block_target}'."
        )
    return collector
