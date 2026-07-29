from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from net_complexity.models.stochastic_depth import get_stochastic_depth_blocks

from .base import BaseMetric
from .inspection import ComputeCollector


def _unwrap_model(model: nn.Module | None) -> nn.Module | None:
    if model is None:
        return None
    backbone = getattr(model, "backbone", None)
    return backbone if isinstance(backbone, nn.Module) else model


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


class StochasticDepthActiveBlocksMetric(BaseMetric):
    """Expected and observed active-block counts for Huang stochastic depth."""

    def __init__(self) -> None:
        self.reset()

    def update(self, input, output, targets, model=None):
        self.model = _unwrap_model(model)
        if self.model is None:
            return

        blocks = get_stochastic_depth_blocks(self.model)
        if not blocks:
            return

        active_blocks = sum(
            float(block.last_survival_mask.detach().float().item())
            for block in blocks.values()
        )
        self.forward_active_blocks.append(active_blocks)
        if self.model.training:
            self.train_active_blocks.append(active_blocks)

    def compute(self):
        if self.model is None:
            return {}

        blocks = get_stochastic_depth_blocks(self.model)
        if not blocks:
            return {}

        num_blocks = len(blocks)
        expected_active_blocks = sum(
            float(block.survival_probability)
            for block in blocks.values()
        )
        metrics = {
            "stochastic_depth_num_blocks": float(num_blocks),
            "stochastic_depth_expected_active_train_blocks": float(expected_active_blocks),
            "stochastic_depth_expected_active_train_block_ratio": (
                float(expected_active_blocks / num_blocks)
                if num_blocks > 0
                else 0.0
            ),
            "stochastic_depth_average_active_blocks": _mean(self.forward_active_blocks),
        }
        if self.train_active_blocks:
            metrics["stochastic_depth_actual_active_train_blocks"] = _mean(
                self.train_active_blocks
            )
            metrics["stochastic_depth_actual_active_train_block_ratio"] = (
                metrics["stochastic_depth_actual_active_train_blocks"] / num_blocks
            )
        return metrics

    def reset(self):
        self.model = None
        self.forward_active_blocks: list[float] = []
        self.train_active_blocks: list[float] = []


class StochasticDepthFLOPsMetric(BaseMetric):
    """Full inference FLOPs and Huang expected/actual train FLOPs."""

    def __init__(
        self,
        count_bias: bool = True,
        sample_size: int = 1,
    ) -> None:
        self.collector = ComputeCollector(count_bias=count_bias)
        self.sample_size = int(sample_size)
        if self.sample_size <= 0:
            raise ValueError("sample_size must be >= 1.")
        self.reset()

    def update(self, input, output, targets, model=None):
        self.model = _unwrap_model(model)
        if self.input_sample is None and isinstance(input, torch.Tensor):
            self.input_sample = input[: self.sample_size].detach().cpu()

        if self.model is None or not self.model.training:
            return

        blocks = get_stochastic_depth_blocks(self.model)
        if not blocks:
            return

        self.train_gate_snapshots.append(
            {
                name: float(block.last_survival_mask.detach().float().item())
                for name, block in blocks.items()
            }
        )

    def compute(self):
        if self.model is None or self.input_sample is None:
            return {}

        model_metrics, per_layer = self.collector.collect_model(
            model=self.model,
            input_sample=self.input_sample,
        )
        blocks = get_stochastic_depth_blocks(self.model)
        if not blocks:
            return {}

        full_inference_flops = float(model_metrics.flops)
        batch_size = max(int(model_metrics.batch_size), 1)
        branch_flops_by_name = {
            name: float(self._branch_flops(name, per_layer))
            for name in blocks
        }
        stochastic_branch_flops = sum(branch_flops_by_name.values())
        always_computed_flops = max(full_inference_flops - stochastic_branch_flops, 0.0)

        expected_active_branch_flops = sum(
            branch_flops_by_name[name] * float(blocks[name].survival_probability)
            for name in blocks
        )
        expected_train_flops = always_computed_flops + expected_active_branch_flops

        metrics = {
            "stochastic_depth_full_inference_flops": full_inference_flops,
            "stochastic_depth_full_inference_flops_per_sample": (
                full_inference_flops / batch_size
            ),
            "stochastic_depth_expected_train_flops": expected_train_flops,
            "stochastic_depth_expected_train_flops_per_sample": (
                expected_train_flops / batch_size
            ),
            "stochastic_depth_stochastic_branch_flops": stochastic_branch_flops,
            "stochastic_depth_stochastic_branch_flops_per_sample": (
                stochastic_branch_flops / batch_size
            ),
            "stochastic_depth_expected_flops_skip_ratio": (
                (full_inference_flops - expected_train_flops) / full_inference_flops
                if full_inference_flops > 0.0
                else 0.0
            ),
        }

        if self.train_gate_snapshots:
            actual_train_flops = always_computed_flops + _mean([
                sum(
                    branch_flops_by_name.get(name, 0.0) * gate
                    for name, gate in snapshot.items()
                )
                for snapshot in self.train_gate_snapshots
            ])
            metrics.update({
                "stochastic_depth_actual_train_flops": actual_train_flops,
                "stochastic_depth_actual_train_flops_per_sample": (
                    actual_train_flops / batch_size
                ),
                "stochastic_depth_actual_flops_skip_ratio": (
                    (full_inference_flops - actual_train_flops) / full_inference_flops
                    if full_inference_flops > 0.0
                    else 0.0
                ),
            })

        return metrics

    def reset(self):
        self.model = None
        self.input_sample = None
        self.train_gate_snapshots: list[dict[str, float]] = []

    @staticmethod
    def _branch_flops(name: str, per_layer: Mapping[str, object]) -> int:
        branch_prefixes = (
            f"{name}.conv1",
            f"{name}.conv2",
            f"{name}.conv3",
        )
        return sum(
            int(getattr(layer_metrics, "flops", 0))
            for layer_name, layer_metrics in per_layer.items()
            if layer_name.startswith(branch_prefixes)
        )
