from .base import BaseMetric


class Accuracy(BaseMetric):
    """Dataset-level accuracy, invariant to evaluation batch partitioning."""

    def __init__(self, return_counts=False):
        self.return_counts = bool(return_counts)
        self.reset()

    def update(self, input, output, targets, model=None):
        predicted = output.logits.detach().argmax(dim=1)
        self.correct += int((predicted == targets).sum().item())
        self.total += int(targets.numel())

    def compute(self):
        if self.total == 0:
            raise ValueError("Cannot compute accuracy on an empty dataset.")
        result = {"accuracy": self.correct / self.total}
        if self.return_counts:
            result.update(correct_count=self.correct, example_count=self.total)
        return result

    def reset(self):
        self.correct = 0
        self.total = 0
