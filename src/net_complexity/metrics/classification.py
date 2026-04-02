import numpy as np

from .base import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self):
        self.buff = []

    def update(self, input, output, targets, model=None):
        length = output.logits.shape[0]
        self.buff.append((output.logits.argmax(dim=-1) ==
                         targets).sum().detach().cpu().numpy()/length)

    def compute(self):
        return {'accuracy': float(np.mean(self.buff))}

    def reset(self):
        self.buff = []
