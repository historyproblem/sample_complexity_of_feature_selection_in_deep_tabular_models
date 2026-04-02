import numpy as np
import torch

from collections import defaultdict

from ..wrappers import parse_AIG_activations
from .base import BaseMetric


class AIGActivationsMetric(BaseMetric):
    def __init__(self):
        self.all_activations_dict = defaultdict(list)

    def update(self, input, output, targets, model=None):
        activations_dict = parse_AIG_activations(model)
        for name, value in activations_dict.items():
            self.all_activations_dict[name].append(
                value.detach().cpu().numpy().mean())

    def compute(self):
        means = []
        for name, value in self.all_activations_dict.items():
            if not isinstance(value, list):
                break
            self.all_activations_dict[name] = float(np.mean(value))
            means.append(self.all_activations_dict[name])
        self.all_activations_dict['average_prob'] = float(np.mean(means))
        self.all_activations_dict['max_prob'] = float(np.max(means))
        self.all_activations_dict['min_prob'] = float(np.min(means))
        return self.all_activations_dict

    def reset(self):
        del self.all_activations_dict
        self.all_activations_dict = defaultdict(list)
