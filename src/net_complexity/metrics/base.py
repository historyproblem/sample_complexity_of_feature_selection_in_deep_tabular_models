from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
import torch


class BaseMetric(ABC):

    def __init__(self):
        pass

    def update(self, input, output, targets, model=None):
        pass

    def compute(self):
        pass

    def reset(self):
        pass


class Multimetric(BaseMetric):
    def __init__(self, metrics_list: list[BaseMetric], prefix: str = ""):
        self.prefix = prefix
        self.metrics = metrics_list

    def update(self, input, output, targets, model=None):
        for metric in self.metrics:
            metric.update(input, output, targets, model)

    def compute(self):
        res_dict = {}
        for metric in self.metrics:
            metric_dict = metric.compute()
            for key, value in metric_dict.items():
                res_dict[f'{self.prefix}_{key}'] = value
        return res_dict

    def reset(self):
        for metric in self.metrics:
            metric.reset()


class MultiLossMetric(BaseMetric):
    """Sample-weighted means; compute() never mutates the accumulated state.

    Assumes CE/loss fields are batch means (as in the bundled wrappers).
    """

    def __init__(self):
        self.reset()

    def update(self, input, output, targets, model=None):
        n = int(targets.shape[0])
        regularization = getattr(output, "regularization_loss", None)
        if regularization is None:
            regularization = 0.0
        reg_loss = getattr(output, "reg_loss", None)
        values = {
            "ce_loss": output.ce_loss,
            "regularization_loss": regularization,
            "reg_loss": regularization if reg_loss is None else reg_loss,
            "loss": output.loss,
            "mean_p_open": getattr(output, "mean_p_open", None),
            "negative_entropy": getattr(output, "negative_entropy", None),
        }
        for name, value in values.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().item()
            self._sums[name] += float(value) * n
            self._counts[name] += n

    def compute(self):
        return {name: value / self._counts[name]
                for name, value in self._sums.items() if self._counts[name]}

    def reset(self):
        self._sums = defaultdict(float)
        self._counts = defaultdict(int)
