import numpy as np

from .base import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self):
        self.buff = []

    def update(self, input, output, targets, model=None):
        accuracy = (output.logits.argmax(dim=-1) == targets).float().mean().item()
        self.buff.append(float(accuracy))

    def compute(self):
        return {'accuracy': float(np.mean(self.buff))}

    def reset(self):
        self.buff = []
