from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .base import BaseMetric


ModuleGetter = Callable[[Any], Mapping[str, Any]]


def survivor_channel_metrics(model, module_getter=None) -> dict:
    """Channel-weighted RAW/EFFECTIVE metrics, independent of legacy zero mass.

    Empty survivor sets produce JSON null plus an explicit status. The hard
    readout describes the deterministic eval predictor; a soft/stochastic eval
    mode has no deterministic hard decision and is reported as unavailable.
    """
    if module_getter is None:
        from net_complexity.models.feature_selection import get_gumbel_modules
        module_getter = get_gumbel_modules
    result = {}
    original = surviving = 0
    raw_off = effective_off = effective_on = hard_closed = 0.0
    hard_available = True
    raw_means, effective_means = [], []
    for name, gate in module_getter(model).items():
        if not hasattr(gate, "get_raw_selection_probs"):
            continue
        raw = gate.get_raw_selection_probs()
        effective = gate.get_effective_selection_probs()
        mask = gate.get_permanent_survivor_mask()
        n0, nt = gate.initial_channels, int(mask.sum().item())
        raw_mass = float((1 - raw[mask]).sum().item())
        effective_mass = float((1 - effective[mask]).sum().item())
        on_mass = float(effective[mask].sum().item())
        try:
            closed = float((~gate.get_hard_gate_decisions(apply_permanent_mask=False)[mask]).sum().item())
        except ValueError:
            closed = None
            hard_available = False
        values = {
            "original_channels": n0, "surviving_channels": nt,
            "physical_pruned_fraction": 1 - nt / n0 if n0 else None,
            "survivor_zero_prob_raw": raw_mass / nt if nt else None,
            "survivor_zero_prob_effective": effective_mass / nt if nt else None,
            "survivor_hard_closed_fraction": closed / nt if nt and closed is not None else None,
            "original_coordinate_zero_mass": 1 - on_mass / n0 if n0 else None,
            "survivor_status": "ok" if nt else "no_surviving_channels",
        }
        result.update({f"{name}_{key}": value for key, value in values.items()})
        original += n0
        surviving += nt
        raw_off += raw_mass
        effective_off += effective_mass
        effective_on += on_mass
        hard_closed += closed or 0
        if nt:
            raw_means.append(raw_mass / nt)
            effective_means.append(effective_mass / nt)
    result.update({
        "original_channels": original, "surviving_channels": surviving,
        "permanently_disabled_channels": original - surviving,
        "physical_pruned_fraction": 1 - surviving / original if original else None,
        "survivor_zero_prob_raw": raw_off / surviving if surviving else None,
        "survivor_zero_prob_effective": effective_off / surviving if surviving else None,
        "survivor_hard_closed_fraction": hard_closed / surviving if surviving and hard_available else None,
        "survivor_hard_closed_channels": hard_closed if hard_available else None,
        "original_coordinate_zero_mass": 1 - effective_on / original if original else None,
        "survivor_status": "ok" if surviving else "no_surviving_channels",
        "hard_decision_status": "deterministic_eval" if hard_available else "unavailable_eval_mode",
        "layer_macro_survivor_zero_prob_raw": float(np.mean(raw_means)) if raw_means else None,
        "layer_macro_survivor_zero_prob_effective": float(np.mean(effective_means)) if effective_means else None,
    })
    return result


class ChannelZeroProbMetric(BaseMetric):
    """Legacy original-coordinate metrics plus explicitly named survivor metrics.

    ``average_zero_prob`` retains the historical masked EFFECTIVE zero mass.
    It must not be interpreted as newly learned closure among survivors.
    """

    def __init__(
        self,
        module_getter: ModuleGetter,
        *,
        log_channel_zero_probs: bool = True,
    ):
        self._module_getter = module_getter
        self.log_channel_zero_probs = log_channel_zero_probs
        self._channel_probs: defaultdict[str, list[np.ndarray]] = defaultdict(list)
        self._hard_probs: defaultdict[str, list[np.ndarray]] = defaultdict(list)
        self._survivor_observations: list[dict] = []

    def update(self, input, output, targets, model=None):
        if model is None:
            return

        modules_dict = self._module_getter(model)
        for name, module in modules_dict.items():
            value = module.get_selection_probs().detach().cpu().numpy()
            self._channel_probs[name].append(np.asarray(value, dtype=np.float64))
            # Legacy STG has no threshold field. Gumbel uses its own configured
            # threshold, preserving the forward tie rule.
            threshold = getattr(module, "gate_threshold", 0.5)
            self._hard_probs[name].append(np.asarray(value > threshold, dtype=np.float64))
        if any(hasattr(module, "get_raw_selection_probs") for module in modules_dict.values()):
            self._survivor_observations.append(survivor_channel_metrics(model, self._module_getter))

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
            hard_active = np.stack(self._hard_probs[name], axis=0)
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
        if self._survivor_observations:
            for key in self._survivor_observations[0]:
                values = [item[key] for item in self._survivor_observations]
                if any(value is None for value in values):
                    results[key] = None
                elif isinstance(values[0], str):
                    results[key] = values[0] if len(set(values)) == 1 else "mixed"
                else:
                    results[key] = float(np.mean(values))
        return results

    def reset(self):
        self._channel_probs.clear()
        self._hard_probs.clear()
        self._survivor_observations.clear()
