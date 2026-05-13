from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .base import BaseMetric


ModuleGetter = Callable[[Any], Mapping[str, Any]]


class ChannelZeroProbMetric(BaseMetric):
    """Shared metric for selectors that expose per-channel selection probabilities."""

    def __init__(
        self,
        module_getter: ModuleGetter,
        *,
        log_channel_zero_probs: bool = True,
    ):
        self._module_getter = module_getter
        self.log_channel_zero_probs = log_channel_zero_probs
        self._channel_probs: defaultdict[str, list[np.ndarray]] = defaultdict(list)

    def update(self, input, output, targets, model=None):
        if model is None:
            return

        modules_dict = self._module_getter(model)
        for name, module in modules_dict.items():
            value = module.get_selection_probs().detach().cpu().numpy()
            self._channel_probs[name].append(np.asarray(value, dtype=np.float64))

    def compute(self):
        if not self._channel_probs:
            return {}

        results: dict[str, float] = {}
        real_means: list[float] = []
        estim_means: list[float] = []
        zero_means: list[float] = []
        total_channels = 0
        total_real_active_channels = 0.0
        total_estim_active_channels = 0.0
        total_real_zero_channels = 0.0
        total_estim_zero_channels = 0.0

        for name, values in self._channel_probs.items():
            stacked = np.stack(values, axis=0)
            mean_selection_probs = stacked.mean(axis=0)
            mean_zero_probs = 1.0 - mean_selection_probs
            hard_active = (stacked > 0.5).astype(np.float64)
            num_channels = int(mean_selection_probs.size)

            avg_estim_prob = float(mean_selection_probs.mean())
            avg_real_prob = float(hard_active.mean())
            avg_zero_prob = float(mean_zero_probs.mean())
            estim_active_channels = float(mean_selection_probs.sum())
            estim_zero_channels = float(mean_zero_probs.sum())
            real_active_channels = float(hard_active.sum(axis=1).mean())
            real_zero_channels = float(num_channels - real_active_channels)

            results[f"{name}_avg_estim_prob"] = avg_estim_prob
            results[f"{name}_avg_real_prob"] = avg_real_prob
            results[f"{name}_avg_zero_prob"] = avg_zero_prob
            results[f"{name}_num_channels"] = num_channels
            results[f"{name}_estim_active_channels"] = estim_active_channels
            results[f"{name}_estim_zero_channels"] = estim_zero_channels
            results[f"{name}_real_active_channels"] = real_active_channels
            results[f"{name}_real_zero_channels"] = real_zero_channels

            if self.log_channel_zero_probs:
                channel_index_width = max(3, len(str(len(mean_zero_probs) - 1)))
                for channel_index, zero_prob in enumerate(mean_zero_probs):
                    results[
                        f"{name}.channel_{channel_index:0{channel_index_width}d}_zero_prob"
                    ] = float(zero_prob)

            real_means.append(avg_real_prob)
            estim_means.append(avg_estim_prob)
            zero_means.append(avg_zero_prob)
            total_channels += num_channels
            total_real_active_channels += real_active_channels
            total_estim_active_channels += estim_active_channels
            total_real_zero_channels += real_zero_channels
            total_estim_zero_channels += estim_zero_channels

        results["average_layer_real_prob"] = float(np.mean(real_means))
        results["average_real_prob"] = float(total_real_active_channels / total_channels)
        results["max_real_prob"] = float(np.max(real_means))
        results["min_real_prob"] = float(np.min(real_means))

        results["average_layer_estim_prob"] = float(np.mean(estim_means))
        results["average_estim_prob"] = float(total_estim_active_channels / total_channels)
        results["max_estim_prob"] = float(np.max(estim_means))
        # Keep the legacy typo for backward compatibility with existing notebooks/parsers.
        results["min_estim_prob"] = float(np.min(estim_means))
        results["min_estimm_prob"] = results["min_estim_prob"]

        results["average_layer_zero_prob"] = float(np.mean(zero_means))
        results["average_zero_prob"] = float(total_estim_zero_channels / total_channels)
        results["max_zero_prob"] = float(np.max(zero_means))
        results["min_zero_prob"] = float(np.min(zero_means))
        results["total_channels"] = total_channels
        results["real_active_channels"] = total_real_active_channels
        results["real_zero_channels"] = total_real_zero_channels
        results["estim_active_channels"] = total_estim_active_channels
        results["estim_zero_channels"] = total_estim_zero_channels
        return results

    def reset(self):
        self._channel_probs.clear()
