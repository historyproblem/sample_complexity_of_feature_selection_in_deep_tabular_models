from types import SimpleNamespace

import pytest
import torch

from net_complexity.metrics.classification import Accuracy


def test_accuracy_weights_a_short_final_batch_by_its_sample_count():
    metric = Accuracy()
    for count, correct in ((128, True), (16, False)):
        logits = torch.zeros(count, 2)
        logits[:, 0 if correct else 1] = 1
        metric.update(None, SimpleNamespace(logits=logits), torch.zeros(count).long())
    assert metric.compute() == {"accuracy": pytest.approx(128 / 144)}


def test_accuracy_reset_does_not_mix_evaluation_passes():
    metric = Accuracy()
    output = SimpleNamespace(logits=torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    metric.update(None, output, torch.tensor([0, 1]))
    assert metric.compute() == {"accuracy": 1.0}
    metric.reset()
    metric.update(None, output, torch.tensor([1, 0]))
    assert metric.compute() == {"accuracy": 0.0}
