import numpy as np
import torch

from collections import defaultdict

from ..wrappers import get_gumbel_modules
from .base import BaseMetric


class GumbelProbMetric(BaseMetric):
    def __init__(self):
        self.all_probs_dict = defaultdict(list)

    def update(self, input, output, targets, model=None):
        modules_dict = get_gumbel_modules(model)
        for name, module in modules_dict.items():
            value = module.get_selection_probs()
            self.all_probs_dict[f'{name}_avg_estim_prob'].append(
                value.detach().cpu().numpy().mean())
            self.all_probs_dict[f'{name}_avg_real_prob'].append(
                (value.detach().cpu().numpy() > 0.5).sum() / len(value))

    def compute(self):
        real_means = []
        estim_means = []
        for name, value in self.all_probs_dict.items():
            if not isinstance(value, list):
                break
            self.all_probs_dict[name] = float(np.mean(value))
            if 'real' in name:
                real_means.append(self.all_probs_dict[name])
            else:
                estim_means.append(self.all_probs_dict[name])

        if not real_means and not estim_means:
            return {}

        self.all_probs_dict['average_real_prob'] = float(
            np.mean(real_means))
        self.all_probs_dict['max_real_prob'] = float(np.max(real_means))
        self.all_probs_dict['min_real_prob'] = float(np.min(real_means))

        self.all_probs_dict['average_estim_prob'] = float(
            np.mean(estim_means))
        self.all_probs_dict['max_estim_prob'] = float(
            np.max(estim_means))
        self.all_probs_dict['min_estimm_prob'] = float(
            np.min(estim_means))
        return self.all_probs_dict

    def reset(self):
        del self.all_probs_dict
        self.all_probs_dict = defaultdict(list)
