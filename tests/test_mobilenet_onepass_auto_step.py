import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from net_complexity.data.dataloaders import Dataloaders
from net_complexity.training import engine
from net_complexity.training.adaptive_lambda import AdaptiveLambdaController
from net_complexity.training.meta import Metrics


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


class MetricState:
    def __init__(self):
        self.values = {}

    def compute(self):
        return dict(self.values)

    def reset(self):
        self.values = {}


@pytest.mark.parametrize(
    ("gap_suffix", "soft_drop", "hard_drop"),
    [("7_8", 0.07, 0.08), ("8_12", 0.08, 0.12)],
    ids=["7_8pp", "8_12pp"],
)
@pytest.mark.parametrize(
    ("accuracy_position", "expected_action", "direction"),
    [
        ("above_soft", "increase_lambda", 1),
        ("between_gaps", "hold", 0),
        ("below_hard", "decrease_lambda", -1),
    ],
)
def test_onepass_recipe_uses_unboosted_auto_step_with_adaptive_lambda(
    monkeypatch, gap_suffix, soft_drop, hard_drop, accuracy_position, expected_action, direction
):
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(
            config_name=f"mobilenet_onepass_thr05_gap{gap_suffix}pp",
            overrides=[
                "training_arguments.adaptive_lambda.baseline_history_dir="
                "/test/imported_dense_reference_20261002",
                "+training_arguments.evaluate_test=false",
            ],
        )

    adaptive = cfg.training_arguments.adaptive_lambda
    assert adaptive.enabled is True
    assert (adaptive.soft_drop, adaptive.hard_drop) == (soft_drop, hard_drop)
    assert adaptive.log_step_init == "auto"
    assert adaptive.adaptive_log_step_enabled is False
    assert cfg.training_arguments.num_epochs == 30
    assert adaptive.warmup_epochs == 3
    assert adaptive.update_every_epochs == 1
    assert cfg.model.lambda_coef == pytest.approx(1e-4)
    assert adaptive.lambda_max == 100
    accuracy = 0.8 - {
        "above_soft": soft_drop / 2,
        "between_gaps": (soft_drop + hard_drop) / 2,
        "below_hard": hard_drop + 0.01,
    }[accuracy_position]

    # Production auto targets lambda=10 by the first third of the search,
    # giving seven updates after warmup. The clamp at 100 is a separate bound.
    expected_step = math.log(10 / 1e-4) / 7
    assert expected_step == pytest.approx(1.6447036378528899)
    assert math.exp(expected_step) == pytest.approx(5.179474679231211)

    model = torch.nn.Linear(1, 1)
    model.lambda_coef = cfg.model.lambda_coef
    metrics = Metrics(
        train_metrics=MetricState(),
        valid_metrics=MetricState(),
        test_metrics=MetricState(),
    )
    controllers = []
    observations = []
    lambda_used = {}

    def capture_initialized_controller(model, controller):
        assert isinstance(controller, AdaptiveLambdaController)
        assert controller.adaptive_log_step_enabled is False
        assert controller.step == pytest.approx(expected_step)
        controllers.append(controller)

    def fake_train_epoch(model, *args, epoch, **kwargs):
        lambda_used[epoch] = model.lambda_coef
        metrics.train_metrics.values = {"train_loss": 1.0}

    def fake_evaluate(model, loader, metric_state, device, *, stage, **kwargs):
        assert stage == "valid"
        metric_state.values = {
            "valid_accuracy": accuracy,
            # No increase in closure would trigger boost in the old recipe.
            "valid_average_zero_prob": 0.05,
        }

    def record_epoch(epoch, *args):
        observations.append({"epoch": epoch, **controllers[0].summary_state()})

    monkeypatch.setattr(engine, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(engine, "evaluate", fake_evaluate)
    result = engine.train(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler_state=None,
        dataloaders=Dataloaders(),
        training_arguments=cfg.training_arguments,
        metrics=metrics,
        device="cpu",
        adaptive_reference_by_epoch={epoch: 0.8 for epoch in range(1, 31)},
        runtime_initialized_callback=capture_initialized_controller,
        epoch_end_callback=record_epoch,
    )

    assert result["num_epochs_executed"] == 30
    assert result["test_metrics"] == {}
    assert len(observations) == 30
    previous_lambda = cfg.model.lambda_coef
    for state in observations:
        epoch = state["epoch"]
        assert lambda_used[epoch] == pytest.approx(previous_lambda)
        assert state["enabled"] is True
        assert state["adaptive_lambda_step"] == pytest.approx(expected_step)
        assert state["adaptive_lambda_effective_log_step"] == pytest.approx(expected_step)
        assert state["adaptive_lambda_log_step_boost_level"] == 0
        assert state["adaptive_lambda_step_action"] == "step_disabled"
        if epoch <= 3:
            assert state["adaptive_lambda_action"] == "warmup"
            assert state["lambda_coef"] == pytest.approx(cfg.model.lambda_coef)
        else:
            assert state["adaptive_lambda_action"] == expected_action
            # A bound can shorten the actual change, but the requested log step
            # must remain the exact auto value throughout the entire search.
            expected_lambda = min(
                adaptive.lambda_max,
                max(adaptive.lambda_min, previous_lambda * math.exp(direction * expected_step)),
            )
            assert state["lambda_coef"] == pytest.approx(expected_lambda)
        previous_lambda = state["lambda_coef"]

    if direction == 1:
        assert observations[9]["lambda_coef"] == pytest.approx(10)
        assert observations[-1]["lambda_coef"] == pytest.approx(adaptive.lambda_max)
    elif direction == -1:
        assert observations[-1]["lambda_coef"] == pytest.approx(adaptive.lambda_min)
