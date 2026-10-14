from .base import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self):
        self.reset()

    def update(self, input, output, targets, model=None):
        self.correct += int((output.logits.argmax(dim=-1) == targets).sum().item())
        self.total += int(targets.numel())

    def compute(self):
        return {"accuracy": self.correct / self.total if self.total else 0.0}

    def reset(self):
        self.correct = 0
        self.total = 0
