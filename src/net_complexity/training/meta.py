from dataclasses import dataclass


class Metrics:
    def __init__(self, train_metrics=None, valid_metrics=None, test_metrics=None):
        self.train_metrics = train_metrics if train_metrics is not None else []
        self.valid_metrics = valid_metrics if valid_metrics is not None else []
        self.test_metrics = test_metrics if test_metrics is not None else []
