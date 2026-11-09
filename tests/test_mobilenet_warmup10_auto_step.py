import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from net_complexity.data.dataloaders import Dataloaders
from net_complexity.training import engine
from net_complexity.training.meta import Metrics


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


class MetricState:
    def __init__(self):
        self.values = {}

    def compute(self):
        return dict(self.values)

    def reset(self):
        self.values = {}


def warmup10_config():
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        return compose(
            config_name="mobilenet_onepass_thr05_gap8_12pp",
            overrides=[
                "training_arguments.adaptive_lambda.baseline_history_dir=/test/dense_reference",
                "training_arguments.adaptive_lambda.warmup_epochs=10",
                "+training_arguments.adaptive_lambda.auto_log_step_epoch_fraction=1.0",
                "+training_arguments.evaluate_test=false",
            ],
        )


def run_feedback(monkeypatch, cfg, accuracy):
    model = torch.nn.Linear(1, 1)
    model.lambda_coef = cfg.model.lambda_coef
    metrics = Metrics(train_metrics=MetricState(), valid_metrics=MetricState())
    controllers = []
    states = []
    used = {}

    def fake_train_epoch(model, *args, epoch, **kwargs):
        used[epoch] = model.lambda_coef
        metrics.train_metrics.values = {"train_loss": 1.0}

    def fake_evaluate(model, loader, metric_state, device, *, stage, **kwargs):
        assert stage == "valid"
        metric_state.values = {"valid_accuracy": accuracy, "valid_average_zero_prob": 0.05}

    def capture_epoch(epoch, *args):
        states.append(controllers[0].summary_state())

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
        runtime_initialized_callback=lambda model, controller: controllers.append(controller),
        epoch_end_callback=capture_epoch,
    )
    return result, states, used


@pytest.mark.parametrize(
    ("accuracy", "action", "direction"),
    [(0.75, "increase_lambda", 1), (0.70, "hold", 0), (0.65, "decrease_lambda", -1)],
)
def test_thirty_epoch_search_uses_ten_epoch_warmup_and_remaining_twenty_auto_updates(
    monkeypatch, accuracy, action, direction
):
    cfg = warmup10_config()
    adaptive = cfg.training_arguments.adaptive_lambda
    assert cfg.training_arguments.num_epochs == 30
    assert adaptive.enabled is True
    assert adaptive.log_step_init == "auto"
    assert adaptive.adaptive_log_step_enabled is False
    expected_step = math.log(10 / cfg.model.lambda_coef) / 20
    assert expected_step == pytest.approx(0.5756462732485115)
    assert math.exp(expected_step) == pytest.approx(1.778279410038923)

    result, states, used = run_feedback(monkeypatch, cfg, accuracy)
    assert result["num_epochs_executed"] == 30
    assert len(states) == 30
    previous_lambda = cfg.model.lambda_coef
    for epoch, state in enumerate(states, start=1):
        assert state["enabled"] is True
        assert state["adaptive_lambda_step"] == pytest.approx(expected_step)
        assert state["adaptive_lambda_effective_log_step"] == pytest.approx(expected_step)
        assert state["adaptive_lambda_log_step_boost_level"] == 0
        assert state["adaptive_lambda_step_action"] == "step_disabled"
        assert used[epoch] == pytest.approx(previous_lambda)
        if epoch <= 10:
            assert state["adaptive_lambda_action"] == "warmup"
            assert state["lambda_coef"] == pytest.approx(cfg.model.lambda_coef)
        else:
            assert state["adaptive_lambda_action"] == action
            expected_lambda = min(
                adaptive.lambda_max,
                max(adaptive.lambda_min, previous_lambda * math.exp(direction * expected_step)),
            )
            assert state["lambda_coef"] == pytest.approx(expected_lambda)
        previous_lambda = state["lambda_coef"]

    assert used[11] == pytest.approx(cfg.model.lambda_coef)
    assert used[12] == pytest.approx(cfg.model.lambda_coef * math.exp(direction * expected_step))
    if direction == 1:
        # The final decision calibrates lambda=10 after epoch 30; that value
        # is not used to train an extra epoch.
        assert states[-1]["lambda_coef"] == pytest.approx(10)
        assert used[30] == pytest.approx(10 / math.exp(expected_step))
    elif direction == -1:
        assert states[-1]["lambda_coef"] == pytest.approx(adaptive.lambda_min)


@pytest.mark.parametrize("fraction", [0, -0.1, 1.01, float("nan"), float("inf"), None, "invalid"])
def test_auto_target_fraction_rejects_invalid_values_before_training(monkeypatch, fraction):
    cfg = warmup10_config()
    cfg.training_arguments.adaptive_lambda.auto_log_step_epoch_fraction = fraction
    with pytest.raises(ValueError, match="auto_log_step_epoch_fraction must be finite"):
        run_feedback(monkeypatch, cfg, 0.75)


def test_default_fraction_still_requires_an_update_before_the_first_third(monkeypatch):
    cfg = warmup10_config()
    del cfg.training_arguments.adaptive_lambda.auto_log_step_epoch_fraction
    with pytest.raises(ValueError, match="at least one lambda update"):
        run_feedback(monkeypatch, cfg, 0.75)


def test_explicit_numeric_step_does_not_use_the_auto_fraction(monkeypatch):
    cfg = warmup10_config()
    OmegaConf.update(cfg, "training_arguments.adaptive_lambda.log_step_init", 0.25)
    cfg.training_arguments.adaptive_lambda.auto_log_step_epoch_fraction = "unused"
    _, states, _ = run_feedback(monkeypatch, cfg, 0.75)
    assert all(state["adaptive_lambda_effective_log_step"] == pytest.approx(0.25) for state in states)
