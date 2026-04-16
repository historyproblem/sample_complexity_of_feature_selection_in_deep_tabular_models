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

        for name, values in self._channel_probs.items():
            stacked = np.stack(values, axis=0)
            mean_selection_probs = stacked.mean(axis=0)
            mean_zero_probs = 1.0 - mean_selection_probs

            avg_estim_prob = float(mean_selection_probs.mean())
            avg_real_prob = float((stacked > 0.5).mean())
            avg_zero_prob = float(mean_zero_probs.mean())

            results[f"{name}_avg_estim_prob"] = avg_estim_prob
            results[f"{name}_avg_real_prob"] = avg_real_prob
            results[f"{name}_avg_zero_prob"] = avg_zero_prob

            if self.log_channel_zero_probs:
                channel_index_width = max(3, len(str(len(mean_zero_probs) - 1)))
                for channel_index, zero_prob in enumerate(mean_zero_probs):
                    results[
                        f"{name}.channel_{channel_index:0{channel_index_width}d}_zero_prob"
                    ] = float(zero_prob)

            real_means.append(avg_real_prob)
            estim_means.append(avg_estim_prob)
            zero_means.append(avg_zero_prob)

        results["average_real_prob"] = float(np.mean(real_means))
        results["max_real_prob"] = float(np.max(real_means))
        results["min_real_prob"] = float(np.min(real_means))

        results["average_estim_prob"] = float(np.mean(estim_means))
        results["max_estim_prob"] = float(np.max(estim_means))
        results["min_estimm_prob"] = float(np.min(estim_means))

        results["average_zero_prob"] = float(np.mean(zero_means))
        results["max_zero_prob"] = float(np.max(zero_means))
        results["min_zero_prob"] = float(np.min(zero_means))
        return results

    def reset(self):
        self._channel_probs.clear()
