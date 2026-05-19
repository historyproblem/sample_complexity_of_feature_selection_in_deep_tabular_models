from collections import defaultdict

from ..models.feature_selection import get_AIG_modules
from .base import BaseMetric


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
            self._activation_sums[metric_name] += float(value.detach().mean().item())
            self._activation_counts[metric_name] += 1

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
            return activations_dict

        activations_dict['average_prob'] = float(sum(means) / len(means))
        activations_dict['max_prob'] = float(max(means))
        activations_dict['min_prob'] = float(min(means))
        return activations_dict

    def reset(self):
        self._activation_sums = defaultdict(float)
        self._activation_counts = defaultdict(int)
