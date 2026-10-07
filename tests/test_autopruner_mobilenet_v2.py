import math

import torch
import torch.nn as nn

from net_complexity.metrics.autopruner import AutoPrunerControllerMetric
from net_complexity.models.autopruner import (
    AutoPrunerLayer,
    AutoPrunerWrapper,
    export_pruned_autopruner_backbone,
    get_autopruner_modules,
)
from net_complexity.models.autopruner_mobilenet_v2 import AutoPrunerMobileNetV2
from net_complexity.models.mobilenet_v2 import MobileNetV2


def test_project_checkpoint_loads_and_mask_exports_same_function(tmp_path):
    torch.manual_seed(7)
    plain = MobileNetV2(num_classes=200)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {"model_state_dict": {f"backbone.{k}": v for k, v in plain.state_dict().items()}},
        checkpoint_path,
    )

    model = AutoPrunerWrapper(
        AutoPrunerMobileNetV2(num_classes=200, input_size=32),
        pretrained_checkpoint=str(checkpoint_path),
        pruning_epochs_per_stage=15,
        final_fine_tune_epochs=30,
    )
    assert model.expected_num_epochs == 45
    assert model.pretrained_load_info["missing_parameter_keys"] == []
    assert len(get_autopruner_modules(model)) == 16
    assert all(
        selector.alpha_start == 0.1 and selector.alpha_stop == 100.0
        for selector in get_autopruner_modules(model).values()
    )

    for selector in get_autopruner_modules(model).values():
        selector.binary_mask[1::2] = 0
        selector.phase.fill_(selector.PHASE_HARD)
    model.eval()
    deployment = export_pruned_autopruner_backbone(model).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        expected = model.backbone(x)
        actual = deployment(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    artifact_info = model.on_best_checkpoint_loaded(run_dir=tmp_path)
    assert (tmp_path / "checkpoints" / "autopruner_pruned.pt").is_file()
    assert artifact_info["autopruner_pruned_checkpoint"].endswith("autopruner_pruned.pt")
    with torch.inference_mode():
        test_logits = model(x, torch.zeros(1, dtype=torch.long)).logits
    torch.testing.assert_close(test_logits, actual, rtol=1e-5, atol=1e-5)
    assert sum(p.numel() for p in deployment.parameters()) < sum(
        p.numel() for p in plain.parameters()
    )


def test_search_phase_trains_mobile_selectors():
    model = AutoPrunerWrapper(
        AutoPrunerMobileNetV2(num_classes=10, input_size=32),
        pruning_epochs_per_stage=15,
        final_fine_tune_epochs=30,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    model.on_train_epoch_start(epoch=1, optimizer=optimizer, batches_per_epoch=1)
    model.train()
    output = model(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
    output.loss.backward()
    assert any(
        selector.coder.weight.grad is not None
        for selector in get_autopruner_modules(model).values()
    )


def test_auto_lambda_step_and_entropy_are_active_during_search():
    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.pruner = AutoPrunerLayer(4, 2, stage_index=0, code_window_size=2)
            self.head = nn.Linear(4, 2)

        def forward(self, x):
            return self.head(self.pruner(x).mean((2, 3)))

    model = AutoPrunerWrapper(
        TinyBackbone(),
        pruning_epochs_per_stage=15,
        final_fine_tune_epochs=30,
        lambda_log_step_init="auto",
        entropy_regularization="plus_negative_entropy",
        entropy_regularization_coef=0.3,
        pruning_lr=0.01,
        pruning_lr_policy="constant",
        fine_tune_lr_policy="cosine",
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.on_train_epoch_start(epoch=1, optimizer=optimizer, batches_per_epoch=4)
    selector = model.backbone.pruner
    expected_step = math.log(100.0 / 10.0) / (2 * 5)
    assert math.isclose(float(selector.lambda_log_step), expected_step, rel_tol=1e-6)
    assert optimizer.param_groups[0]["lr"] == 0.01

    model.train()
    output = model(torch.randn(2, 4, 2, 2), torch.tensor([0, 1]))
    metric = AutoPrunerControllerMetric()
    metric.update(None, output, None, model=model)
    assert math.isclose(
        metric.compute()["autopruner_lambda_log_step_mean"],
        expected_step,
        rel_tol=1e-6,
    )
    assert output.negative_entropy.item() < 0
    torch.testing.assert_close(
        output.reg_loss,
        selector.weighted_regularization() + 0.3 * output.negative_entropy,
    )
    output.loss.backward()
    assert selector.coder.weight.grad is not None

    selector._update_consensus(torch.ones(2, 4), allow_convergence_boost=False)
    assert math.isclose(
        float(selector.adaptive_regularization),
        10.0 * math.exp(expected_step),
        rel_tol=1e-6,
    )

    model.on_validation_epoch_end(
        epoch=15, valid_metrics={"valid_accuracy": 0.5}, optimizer=optimizer
    )
    model.on_train_epoch_start(epoch=16, optimizer=optimizer, batches_per_epoch=4)
    assert optimizer.param_groups[0]["lr"] == 0.01
    model.on_train_epoch_start(epoch=17, optimizer=optimizer, batches_per_epoch=4)
    assert math.isclose(
        optimizer.param_groups[0]["lr"],
        0.01 * 0.5 * (1.0 + math.cos(math.pi / 30)),
        rel_tol=1e-6,
    )
