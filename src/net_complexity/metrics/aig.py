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
    Static and AIG-adjusted FLOPs/parameter-count estimate.

    `ComputeCollector` gives the full forward cost (and, per layer, the
    parameter count). For shape-safe AIG blocks we additionally estimate
    skipped residual-branch FLOPs *and parameters* from the latest hard gate
    values and report the adjusted cost.

    The parameter numbers answer a different question than
    ``aig_static_flops``/params reported elsewhere (e.g.
    ``WeightStatisticsMetric``): classic AIG never removes a weight tensor
    from the model — ``aig_static_params`` is therefore constant across a
    whole t-sweep, same as the raw parameter count. ``aig_active_params`` is
    instead the *expected* number of parameters actually exercised by a
    single inference, averaging over the dataset's realized per-block gate
    decisions — i.e. an always-on block's weights count fully, a block open
    for e.g. 30% of validation images contributes 30% of its branch weights
    to the expectation. It is an average over inputs, not a property of any
    single forward pass (a real input either fully uses or fully skips a
    given block's branch).
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
        for name, module in get_AIG_modules(self.model).items():
            activations = getattr(module, "activations", None)
            if isinstance(activations, torch.Tensor):
                detached = activations.detach().float()
                self._gate_sums[name] += float(detached.sum().item())
                self._gate_counts[name] += int(detached.numel())

    def compute(self):
        if self.model is None or self.input_sample is None:
            return {}

        model_metrics, per_layer = self.collector.collect_model(
            model=self.model,
            input_sample=self.input_sample,
        )
        aig_modules = get_AIG_modules(self.model)

        gated_branch_macs = 0.0
        active_branch_macs = 0.0
        gated_branch_params = 0.0
        active_branch_params = 0.0
        gate_means = []

        for name, module in aig_modules.items():
            branch_macs = float(self._branch_metric_sum(name, per_layer, "macs"))
            branch_params = float(self._branch_metric_sum(name, per_layer, "num_params"))
            if branch_macs <= 0.0 and branch_params <= 0.0:
                continue

            gate_mean = self._dataset_gate_mean(name, module)
            gated_branch_macs += branch_macs
            active_branch_macs += branch_macs * gate_mean
            gated_branch_params += branch_params
            active_branch_params += branch_params * gate_mean
            gate_means.append(gate_mean)

        skipped_branch_macs = max(gated_branch_macs - active_branch_macs, 0.0)
        skipped_branch_params = max(gated_branch_params - active_branch_params, 0.0)
        batch_size = max(int(model_metrics.batch_size), 1)
        executed_macs_per_image = float(model_metrics.macs) / batch_size
        ideal_routed_macs_per_image = max(
            float(model_metrics.macs) - skipped_branch_macs,
            0.0,
        ) / batch_size
        executed_gmac_per_image = executed_macs_per_image / 1e9
        ideal_routed_gmac_per_image = ideal_routed_macs_per_image / 1e9
        skipped_gmac_per_image = (skipped_branch_macs / batch_size) / 1e9

        # Canonical reporting uses multiply-adds (GMAC/image). The optional
        # GFLOP columns follow the explicit convention FLOP = 2 * MAC.
        canonical = {
            "executed_gmac_per_image": executed_gmac_per_image,
            "ideal_routed_gmac_per_image": ideal_routed_gmac_per_image,
            "executed_gflop_per_image": 2.0 * executed_gmac_per_image,
            "ideal_routed_gflop_per_image": 2.0 * ideal_routed_gmac_per_image,
            "aig_executed_gmac_per_image": executed_gmac_per_image,
            "aig_ideal_routed_gmac_per_image": ideal_routed_gmac_per_image,
            "aig_ideal_skipped_gmac_per_image": skipped_gmac_per_image,
            "aig_executed_gflop_per_image": 2.0 * executed_gmac_per_image,
            "aig_ideal_routed_gflop_per_image": 2.0 * ideal_routed_gmac_per_image,
        }

        static_flops = float(model_metrics.flops)
        skipped_branch_flops = 2.0 * skipped_branch_macs
        active_flops = max(static_flops - skipped_branch_flops, 0.0)

        static_params = float(sum(layer_metrics.num_params for layer_metrics in per_layer.values()))
        active_params = max(static_params - skipped_branch_params, 0.0)

        return {
            **canonical,
            "aig_static_flops": static_flops,
            "aig_static_flops_per_sample": static_flops / batch_size,
            "aig_active_flops": active_flops,
            "aig_active_flops_per_sample": active_flops / batch_size,
            "aig_skipped_flops": skipped_branch_flops,
            "aig_skipped_flops_per_sample": skipped_branch_flops / batch_size,
            "aig_gated_branch_flops": 2.0 * gated_branch_macs,
            "aig_gated_branch_flops_per_sample": 2.0 * gated_branch_macs / batch_size,
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
            # Params, unlike FLOPs, don't scale with batch_size — a single
            # architecture trace already gives the whole-model count, so
            # there's no "_per_sample" variant here.
            "aig_static_params": static_params,
            "aig_active_params": active_params,
            "aig_skipped_params": skipped_branch_params,
            "aig_gated_branch_params": gated_branch_params,
            "aig_params_skip_ratio": (
                skipped_branch_params / static_params
                if static_params > 0.0
                else 0.0
            ),
            "aig_params_active_ratio": (
                active_params / static_params
                if static_params > 0.0
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
        self._gate_sums = defaultdict(float)
        self._gate_counts = defaultdict(int)

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

    def _dataset_gate_mean(self, name: str, module) -> float:
        count = self._gate_counts[name]
        if count > 0:
            return self._gate_sums[name] / count
        return self._gate_mean(module)

    @staticmethod
    def _branch_metric_sum(name: str, per_layer, attr: str) -> int:
        """Sum ``getattr(layer_metrics, attr)`` (e.g. "macs"/"num_params")
        over the sub-modules belonging to the gated residual branch of block
        ``name`` — everything the AIG gate can skip, excluding the always-on
        stem/downsample/fc.
        """
        if name.endswith(".gate"):
            block_prefix = name[: -len(".gate")]
            branch_prefix = f"{block_prefix}.branch."
            return sum(
                getattr(layer_metrics, attr)
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
            getattr(layer_metrics, attr)
            for layer_name, layer_metrics in per_layer.items()
            if layer_name.startswith(legacy_prefixes)
        )
