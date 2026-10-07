from __future__ import annotations

import torch
import torch.nn as nn

from ..models.autopruner import (
    AutoPrunerLayer,
    export_pruned_autopruner_backbone,
    get_autopruner_modules,
)
from ._channel_prob import ChannelZeroProbMetric
from .base import BaseMetric
from .inspection import ComputeCollector


class AutoPrunerProbMetric(ChannelZeroProbMetric):
    """Selection statistics for the deterministic AutoPruner channel codes."""

    def __init__(self, log_channel_zero_probs: bool = True):
        super().__init__(
            get_autopruner_modules,
            log_channel_zero_probs=log_channel_zero_probs,
        )


def _num_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class AutoPrunerComplexityMetric(BaseMetric):
    """Compare the physically exported network with its dense deployment form.

    AutoPruner coding convolutions exist only while pruning. They are therefore
    reported separately and excluded from both deployment parameter/FLOP totals.
    """

    def __init__(self, *, count_bias: bool = True, sample_size: int = 1) -> None:
        self.collector = ComputeCollector(count_bias=count_bias)
        self.sample_size = int(sample_size)
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive.")
        self.reset()

    def update(self, input, output, targets, model=None):
        if model is not None:
            self.model = model
        if self.input_sample is None:
            self.input_sample = input[: self.sample_size].detach().cpu()

    def compute(self):
        if self.model is None or self.input_sample is None:
            return {}

        modules = get_autopruner_modules(self.model)
        dense = export_pruned_autopruner_backbone(
            self.model,
            use_binary_masks=False,
        )
        pruned = export_pruned_autopruner_backbone(
            self.model,
            use_binary_masks=True,
        )
        dense_compute, _ = self.collector.collect_model(dense, self.input_sample)
        pruned_compute, _ = self.collector.collect_model(pruned, self.input_sample)

        total_channels = sum(module.channels for module in modules.values())
        active_channels = sum(
            int((module.get_binary_mask() > 0.5).sum().item())
            for module in modules.values()
        )
        coder_parameters = sum(
            _num_parameters(module.coder)
            for module in modules.values()
            if isinstance(module, AutoPrunerLayer)
        )
        dense_parameters = _num_parameters(dense)
        pruned_parameters = _num_parameters(pruned)
        dense_macs = float(dense_compute.macs_per_sample)
        pruned_macs = float(pruned_compute.macs_per_sample)

        return {
            "autopruner_num_selectors": len(modules),
            "autopruner_total_gated_channels": total_channels,
            "autopruner_active_channels": active_channels,
            "autopruner_pruned_channels": total_channels - active_channels,
            "autopruner_channel_keep_ratio": active_channels / max(total_channels, 1),
            "autopruner_temporary_coder_params": coder_parameters,
            "autopruner_dense_deployment_params": dense_parameters,
            "autopruner_pruned_deployment_params": pruned_parameters,
            "autopruner_parameter_reduction": (
                1.0 - pruned_parameters / max(dense_parameters, 1)
            ),
            "autopruner_dense_macs_per_image": dense_macs,
            "autopruner_pruned_macs_per_image": pruned_macs,
            "autopruner_mac_reduction": 1.0 - pruned_macs / max(dense_macs, 1.0),
            "autopruner_dense_gmac_per_image": dense_macs / 1e9,
            "autopruner_pruned_gmac_per_image": pruned_macs / 1e9,
            "autopruner_dense_gflop_per_image": 2.0 * dense_macs / 1e9,
            "autopruner_pruned_gflop_per_image": 2.0 * pruned_macs / 1e9,
        }

    def reset(self):
        self.model: nn.Module | None = None
        self.input_sample: torch.Tensor | None = None
