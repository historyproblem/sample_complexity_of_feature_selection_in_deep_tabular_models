from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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


def _uses_stochastic_eval(blocks: Mapping[str, nn.Module]) -> bool:
    return any(
        getattr(block, "eval_mode", "expected") == "stochastic"
        for block in blocks.values()
    )


@dataclass(frozen=True)
class _DenseFLOPsProfile:
    """FLOPs measured with every stochastic residual branch enabled."""

    input_shape: tuple[int, ...]
    batch_size: int
    forward_flops: float
    branch_flops_by_name: dict[str, float]


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
        else:
            self.inference_active_blocks.append(active_blocks)

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
        expected_inference_active_blocks = (
            expected_active_blocks if _uses_stochastic_eval(blocks) else float(num_blocks)
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
            "stochastic_depth_expected_inference_active_blocks": (
                expected_inference_active_blocks
            ),
            "stochastic_depth_expected_inference_active_block_ratio": (
                float(expected_inference_active_blocks / num_blocks)
                if num_blocks > 0
                else 0.0
            ),
        }
        if self.train_active_blocks:
            metrics["stochastic_depth_actual_active_train_blocks"] = _mean(
                self.train_active_blocks
            )
            metrics["stochastic_depth_actual_active_train_block_ratio"] = (
                metrics["stochastic_depth_actual_active_train_blocks"] / num_blocks
            )
        if self.inference_active_blocks:
            metrics["stochastic_depth_actual_inference_active_blocks"] = _mean(
                self.inference_active_blocks
            )
            metrics["stochastic_depth_actual_inference_active_block_ratio"] = (
                metrics["stochastic_depth_actual_inference_active_blocks"] / num_blocks
            )
        return metrics

    def reset(self):
        self.model = None
        self.forward_active_blocks: list[float] = []
        self.train_active_blocks: list[float] = []
        self.inference_active_blocks: list[float] = []


class StochasticDepthFLOPsMetric(BaseMetric):
    """Dense reference, stochastic-depth training, and inference FLOPs.

    The static reference is measured by running the model once with all survival
    probabilities set to one. In ``eval_mode=expected``, Huang stochastic depth
    uses that full graph at inference. In ``eval_mode=stochastic``, inference
    executes only residual branches whose Bernoulli masks are one, matching the
    train-time depth sampling while the surrounding model remains in eval mode.

    ``ComputeCollector`` counts Conv/Linear forward FLOPs using ``FLOP = 2 *
    MAC``.  The forward+backward values are therefore explicit estimates:
    forward FLOPs multiplied by ``1 + backward_flops_multiplier``.  Optimizer
    updates and non-Conv/Linear operations are not included.

    Legacy ``*_train_flops`` keys retain their original train-forward meaning.
    New keys use ``*_train_forward_flops`` or
    ``*_train_forward_backward_flops`` to make the convention unambiguous.
    """

    def __init__(
        self,
        count_bias: bool = True,
        sample_size: int = 1,
        backward_flops_multiplier: float = 2.0,
    ) -> None:
        self.collector = ComputeCollector(count_bias=count_bias)
        self.sample_size = int(sample_size)
        if self.sample_size <= 0:
            raise ValueError("sample_size must be >= 1.")
        self.backward_flops_multiplier = float(backward_flops_multiplier)
        if self.backward_flops_multiplier < 0.0:
            raise ValueError("backward_flops_multiplier must be >= 0.")
        self._dense_profile: _DenseFLOPsProfile | None = None
        self._dense_profile_model_id: int | None = None
        self.reset()

    def update(self, input, output, targets, model=None):
        self.model = _unwrap_model(model)
        if self.input_sample is None and isinstance(input, torch.Tensor):
            self.input_sample = input[: self.sample_size].detach().cpu()

        if self.model is None or not isinstance(input, torch.Tensor) or input.ndim == 0:
            return

        blocks = get_stochastic_depth_blocks(self.model)
        if not blocks:
            return

        batch_size = int(input.shape[0])
        if not self.model.training:
            self.inference_samples += batch_size
            if _uses_stochastic_eval(blocks):
                for name, block in blocks.items():
                    gate = float(block.last_survival_mask.detach().float().item())
                    self.inference_gate_sample_sums[name] = (
                        self.inference_gate_sample_sums.get(name, 0.0)
                        + gate * batch_size
                    )
            return

        for name, block in blocks.items():
            gate = float(block.last_survival_mask.detach().float().item())
            self.train_gate_sample_sums[name] = (
                self.train_gate_sample_sums.get(name, 0.0) + gate * batch_size
            )
        self.train_samples += batch_size

    def compute(self):
        if self.model is None or self.input_sample is None:
            return {}

        blocks = get_stochastic_depth_blocks(self.model)
        if not blocks:
            return {}

        profile = self._get_dense_profile(self.model, self.input_sample, blocks)
        full_inference_flops = profile.forward_flops
        batch_size = profile.batch_size
        full_inference_flops_per_sample = full_inference_flops / batch_size
        branch_flops_by_name = profile.branch_flops_by_name
        stochastic_branch_flops = sum(branch_flops_by_name.values())
        always_computed_flops = max(full_inference_flops - stochastic_branch_flops, 0.0)

        expected_active_branch_flops = sum(
            branch_flops_by_name[name] * float(blocks[name].survival_probability)
            for name in blocks
        )
        expected_train_flops = always_computed_flops + expected_active_branch_flops
        expected_train_flops_per_sample = expected_train_flops / batch_size
        stochastic_eval = _uses_stochastic_eval(blocks)
        expected_inference_flops_per_sample = (
            expected_train_flops_per_sample
            if stochastic_eval
            else full_inference_flops_per_sample
        )
        actual_inference_flops_per_sample = expected_inference_flops_per_sample
        if self.inference_samples > 0 and stochastic_eval:
            actual_active_branch_flops_per_sample = sum(
                (branch_flops_by_name.get(name, 0.0) / batch_size)
                * (
                    self.inference_gate_sample_sums.get(name, 0.0)
                    / self.inference_samples
                )
                for name in blocks
            )
            actual_inference_flops_per_sample = (
                always_computed_flops / batch_size
            ) + actual_active_branch_flops_per_sample
        train_forward_backward_multiplier = 1.0 + self.backward_flops_multiplier

        metrics = {
            # Backward-compatible names. These are all forward-pass FLOPs.
            "stochastic_depth_full_inference_flops": full_inference_flops,
            "stochastic_depth_full_inference_flops_per_sample": full_inference_flops_per_sample,
            "stochastic_depth_expected_train_flops": expected_train_flops,
            "stochastic_depth_expected_train_flops_per_sample": expected_train_flops_per_sample,
            "stochastic_depth_stochastic_branch_flops": stochastic_branch_flops,
            "stochastic_depth_stochastic_branch_flops_per_sample": (
                stochastic_branch_flops / batch_size
            ),
            "stochastic_depth_expected_flops_skip_ratio": (
                (full_inference_flops - expected_train_flops) / full_inference_flops
                if full_inference_flops > 0.0
                else 0.0
            ),
            # Explicit reporting convention.
            "stochastic_depth_dense_reference_forward_flops_per_sample": (
                full_inference_flops_per_sample
            ),
            "stochastic_depth_inference_forward_flops_per_sample": (
                actual_inference_flops_per_sample
            ),
            "stochastic_depth_expected_inference_forward_flops_per_sample": (
                expected_inference_flops_per_sample
            ),
            "stochastic_depth_expected_train_forward_flops_per_sample": (
                expected_train_flops_per_sample
            ),
            "stochastic_depth_always_computed_forward_flops_per_sample": (
                always_computed_flops / batch_size
            ),
            "stochastic_depth_backward_flops_multiplier": (
                self.backward_flops_multiplier
            ),
            "stochastic_depth_dense_reference_train_forward_backward_flops_per_sample": (
                full_inference_flops_per_sample * train_forward_backward_multiplier
            ),
            "stochastic_depth_expected_train_forward_backward_flops_per_sample": (
                expected_train_flops_per_sample * train_forward_backward_multiplier
            ),
        }

        if self.inference_samples > 0:
            metrics.update({
                "stochastic_depth_inference_samples": float(self.inference_samples),
                "stochastic_depth_actual_inference_forward_flops_per_sample": (
                    actual_inference_flops_per_sample
                ),
                "stochastic_depth_actual_inference_flops_skip_ratio": (
                    (
                        full_inference_flops_per_sample
                        - actual_inference_flops_per_sample
                    )
                    / full_inference_flops_per_sample
                    if full_inference_flops_per_sample > 0.0
                    else 0.0
                ),
                "stochastic_depth_inference_forward_flops_total": (
                    actual_inference_flops_per_sample * self.inference_samples
                ),
            })

        if self.train_samples > 0:
            actual_active_branch_flops_per_sample = sum(
                (branch_flops_by_name.get(name, 0.0) / batch_size)
                * (self.train_gate_sample_sums.get(name, 0.0) / self.train_samples)
                for name in blocks
            )
            actual_train_flops_per_sample = (
                always_computed_flops / batch_size
            ) + actual_active_branch_flops_per_sample
            actual_train_flops = actual_train_flops_per_sample * batch_size
            dense_train_forward_backward_per_sample = (
                full_inference_flops_per_sample * train_forward_backward_multiplier
            )
            expected_train_forward_backward_per_sample = (
                expected_train_flops_per_sample * train_forward_backward_multiplier
            )
            actual_train_forward_backward_per_sample = (
                actual_train_flops_per_sample * train_forward_backward_multiplier
            )
            metrics.update({
                "stochastic_depth_actual_train_flops": actual_train_flops,
                "stochastic_depth_actual_train_flops_per_sample": actual_train_flops_per_sample,
                "stochastic_depth_actual_flops_skip_ratio": (
                    (
                        full_inference_flops_per_sample
                        - actual_train_flops_per_sample
                    )
                    / full_inference_flops_per_sample
                    if full_inference_flops_per_sample > 0.0
                    else 0.0
                ),
                "stochastic_depth_train_samples": float(self.train_samples),
                "stochastic_depth_actual_train_forward_flops_per_sample": (
                    actual_train_flops_per_sample
                ),
                "stochastic_depth_actual_train_forward_backward_flops_per_sample": (
                    actual_train_forward_backward_per_sample
                ),
                "stochastic_depth_dense_reference_train_forward_flops_epoch": (
                    full_inference_flops_per_sample * self.train_samples
                ),
                "stochastic_depth_expected_train_forward_flops_epoch": (
                    expected_train_flops_per_sample * self.train_samples
                ),
                "stochastic_depth_actual_train_forward_flops_epoch": (
                    actual_train_flops_per_sample * self.train_samples
                ),
                "stochastic_depth_dense_reference_train_forward_backward_flops_epoch": (
                    dense_train_forward_backward_per_sample * self.train_samples
                ),
                "stochastic_depth_expected_train_forward_backward_flops_epoch": (
                    expected_train_forward_backward_per_sample * self.train_samples
                ),
                "stochastic_depth_actual_train_forward_backward_flops_epoch": (
                    actual_train_forward_backward_per_sample * self.train_samples
                ),
            })

        return metrics

    def reset(self):
        self.model = None
        self.input_sample = None
        self.train_gate_sample_sums: dict[str, float] = {}
        self.inference_gate_sample_sums: dict[str, float] = {}
        self.train_samples = 0
        self.inference_samples = 0

    def _get_dense_profile(
        self,
        model: nn.Module,
        input_sample: torch.Tensor,
        blocks: Mapping[str, nn.Module],
    ) -> _DenseFLOPsProfile:
        input_shape = tuple(input_sample.shape)
        if (
            self._dense_profile is not None
            and self._dense_profile_model_id == id(model)
            and self._dense_profile.input_shape == input_shape
        ):
            return self._dense_profile

        survival_probabilities = {
            name: float(getattr(block, "survival_probability"))
            for name, block in blocks.items()
        }
        try:
            # This is the explicit p_L=1 dense reference requested for both
            # normalization and full-graph inference FLOPs.
            for block in blocks.values():
                block.survival_probability = 1.0
            model_metrics, per_layer = self.collector.collect_model(
                model=model,
                input_sample=input_sample,
            )
        finally:
            for name, block in blocks.items():
                block.survival_probability = survival_probabilities[name]

        profile = _DenseFLOPsProfile(
            input_shape=input_shape,
            batch_size=max(int(model_metrics.batch_size), 1),
            forward_flops=float(model_metrics.flops),
            branch_flops_by_name={
                name: float(self._branch_flops(name, per_layer))
                for name in blocks
            },
        )
        self._dense_profile = profile
        self._dense_profile_model_id = id(model)
        return profile

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
