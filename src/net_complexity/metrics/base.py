from abc import ABC, abstractmethod


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
