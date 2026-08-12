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
    def __init__(self):
        self.losses_dict = defaultdict(list)

    def update(self, input, output, targets, model=None):
        self.losses_dict['ce_loss'].append(float(output.ce_loss.detach().item()))

        regularization_loss = output.regularization_loss
        if isinstance(regularization_loss, torch.Tensor):
            regularization_loss = regularization_loss.detach().item()
        elif regularization_loss is None:
            regularization_loss = 0.0
        self.losses_dict['regularization_loss'].append(float(regularization_loss))
        reg_loss = getattr(output, "reg_loss", None)
        if isinstance(reg_loss, torch.Tensor):
            reg_loss = reg_loss.detach().item()
        elif reg_loss is None:
            reg_loss = regularization_loss
        self.losses_dict['reg_loss'].append(float(reg_loss))

        for name in ("mean_p_open", "negative_entropy"):
            value = getattr(output, name, None)
            if value is not None:
                if isinstance(value, torch.Tensor):
                    value = value.detach().item()
                self.losses_dict[name].append(float(value))

        self.losses_dict['loss'].append(float(output.loss.detach().item()))

    def compute(self):
        for name, metric_list in self.losses_dict.items():
            self.losses_dict[name] = float(np.mean(metric_list))
        return self.losses_dict

    def reset(self):
        del self.losses_dict
        self.losses_dict = defaultdict(list)
