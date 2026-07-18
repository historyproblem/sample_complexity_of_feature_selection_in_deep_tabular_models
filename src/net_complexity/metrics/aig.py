from collections import defaultdict

import torch

from ..models.feature_selection import get_AIG_modules
from .base import BaseMetric
from .inspection import ComputeCollector


class AIGActivationsMetric(BaseMetric):
    def __init__(self):
        self._activation_sums = defaultdict(float)
        self._activation_counts = defaultdict(int)

    def update(self, input, output, targets, model=None):
        for name, module in get_AIG_modules(model).items():
            value = getattr(module, "activations", None)
            if value is None:
                continue
            metric_name = f"g_prob_{name}"
            detached_value = value.detach()
            self._activation_sums[metric_name] += float(detached_value.sum().item())
            self._activation_counts[metric_name] += detached_value.numel()

    def compute(self):
        activations_dict = {}
        means = []
        for name, total in self._activation_sums.items():
            count = self._activation_counts[name]
            if count == 0:
                continue
            mean = total / count
            activations_dict[name] = mean
            means.append(mean)
        if len(means) == 0:
            activations_dict['average_prob'] = 0.0
            activations_dict['max_prob'] = 0.0
            activations_dict['min_prob'] = 0.0
            activations_dict['average_zero_prob'] = 1.0
            activations_dict['max_zero_prob'] = 1.0
            activations_dict['min_zero_prob'] = 1.0
            return activations_dict

        activations_dict['average_prob'] = float(sum(means) / len(means))
        activations_dict['max_prob'] = float(max(means))
        activations_dict['min_prob'] = float(min(means))
        activations_dict['average_zero_prob'] = float(1.0 - activations_dict['average_prob'])
        activations_dict['max_zero_prob'] = float(1.0 - activations_dict['min_prob'])
        activations_dict['min_zero_prob'] = float(1.0 - activations_dict['max_prob'])
        return activations_dict

    def reset(self):
        self._activation_sums = defaultdict(float)
        self._activation_counts = defaultdict(int)


class AIGFLOPsMetric(BaseMetric):
    """
    Static and AIG-adjusted FLOPs estimate.

    `ComputeCollector` gives the full forward cost. For shape-safe AIG blocks we
    additionally estimate skipped residual-branch FLOPs from the latest hard gate
    values and report the adjusted cost.
    """

    def __init__(
        self,
        count_bias: bool = True,
        sample_size: int = 1,
    ):
        self.collector = ComputeCollector(count_bias=count_bias)
        self.sample_size = int(sample_size)
        if self.sample_size <= 0:
            raise ValueError("sample_size must be >= 1.")
        self.reset()

    def update(self, input, output, targets, model=None):
        self.model = self._unwrap_model(model)
        if self.input_sample is None and isinstance(input, torch.Tensor):
            self.input_sample = input[: self.sample_size].detach().cpu()

    def compute(self):
        if self.model is None or self.input_sample is None:
            return {}

        model_metrics, per_layer = self.collector.collect_model(
            model=self.model,
            input_sample=self.input_sample,
        )
        aig_modules = get_AIG_modules(self.model)

        gated_branch_flops = 0.0
        active_branch_flops = 0.0
        gate_means = []

        for name, module in aig_modules.items():
            branch_flops = float(self._branch_flops(name, per_layer))
            if branch_flops <= 0.0:
                continue

            gate_mean = self._gate_mean(module)
            gated_branch_flops += branch_flops
            active_branch_flops += branch_flops * gate_mean
            gate_means.append(gate_mean)

        skipped_branch_flops = max(gated_branch_flops - active_branch_flops, 0.0)
        active_flops = max(float(model_metrics.flops) - skipped_branch_flops, 0.0)
        batch_size = max(int(model_metrics.batch_size), 1)
        static_flops = float(model_metrics.flops)

        return {
            "aig_static_flops": static_flops,
            "aig_static_flops_per_sample": static_flops / batch_size,
            "aig_active_flops": active_flops,
            "aig_active_flops_per_sample": active_flops / batch_size,
            "aig_skipped_flops": skipped_branch_flops,
            "aig_skipped_flops_per_sample": skipped_branch_flops / batch_size,
            "aig_gated_branch_flops": gated_branch_flops,
            "aig_gated_branch_flops_per_sample": gated_branch_flops / batch_size,
            "aig_flops_skip_ratio": (
                skipped_branch_flops / static_flops
                if static_flops > 0.0
                else 0.0
            ),
            "aig_flops_active_ratio": (
                active_flops / static_flops
                if static_flops > 0.0
                else 1.0
            ),
            "aig_num_gated_blocks": len(gate_means),
            "aig_mean_gate_for_flops": (
                float(sum(gate_means) / len(gate_means))
                if gate_means
                else 1.0
            ),
        }

    def reset(self):
        self.model = None
        self.input_sample = None

    @staticmethod
    def _unwrap_model(model):
        if model is None:
            return None
        backbone = getattr(model, "backbone", None)
        return backbone if backbone is not None else model

    @staticmethod
    def _gate_mean(module) -> float:
        activations = getattr(module, "activations", None)
        if isinstance(activations, torch.Tensor):
            return float(activations.detach().float().mean().item())

        keep_probabilities = getattr(module, "keep_probabilities", None)
        if isinstance(keep_probabilities, torch.Tensor):
            return float(keep_probabilities.detach().float().mean().item())

        return 1.0

    @staticmethod
    def _branch_flops(name: str, per_layer) -> int:
        if name.endswith(".gate"):
            block_prefix = name[: -len(".gate")]
            branch_prefix = f"{block_prefix}.branch."
            return sum(
                layer_metrics.flops
                for layer_name, layer_metrics in per_layer.items()
                if layer_name.startswith(branch_prefix)
            )

        # Legacy AIGBottleneckLayer: only conv1/conv2/conv3 are gated; adapter
        # and downsample are always evaluated.
        legacy_prefixes = (
            f"{name}.conv1",
            f"{name}.conv2",
            f"{name}.conv3",
        )
        return sum(
            layer_metrics.flops
            for layer_name, layer_metrics in per_layer.items()
            if layer_name.startswith(legacy_prefixes)
        )
