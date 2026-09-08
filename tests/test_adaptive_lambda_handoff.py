from copy import deepcopy
import math

import pytest
import torch
from omegaconf import OmegaConf

from net_complexity.data.dataloaders import Dataloaders
from net_complexity.training import engine
from net_complexity.training.adaptive_lambda import AdaptiveLambdaController
from net_complexity.training.meta import Metrics
from net_complexity.training.run_history import RunHistory


class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.register_buffer("open_bias", torch.zeros(1))
        self.lambda_coef = 0.001

    def set_lambda_coef(self, value, *, bypass_gumbel=False):
        self.lambda_coef = float(value)

    def set_gumbel_open_bias(self, value, *, p_min, p_max):
        self.open_bias.fill_(value)


def apply_lambda(model, value):
    model.set_lambda_coef(value)


def controller(**overrides):
    kwargs = dict(initial_lambda_coef=0.001, warmup_epochs=0, update_every_epochs=1,
                  acc_window=1, log_step_init=math.log(2),
                  reference_accuracy_by_epoch={epoch: 0.99 for epoch in range(1, 31)},
                  recovery_config=dict(enabled=True, min_epoch=1, patience=1,
                                       require_slow_recovery=False, use_zero_prob_filter=False,
                                       recovery_epochs=4, drop_min=0.005, target_acc_margin=0.001))
    kwargs.update(overrides)
    return AdaptiveLambdaController(**kwargs)


def step(control, model, epoch, accuracy, zero_prob=0.4):
    return control.on_epoch_end(epoch=epoch, model=model,
                               valid_metrics={"valid_accuracy": accuracy,
                                              "valid_average_zero_prob": zero_prob},
                               apply_lambda=apply_lambda)


def test_controller_roundtrip_preserves_boost_recovery_and_next_actions(tmp_path):
    original, first = controller(), Probe()
    original.apply_initial_state(first, apply_lambda=apply_lambda)
    step(original, first, 1, 0.99)
    step(original, first, 2, 0.99)
    assert original.log_step_boost_level > 0
    # Accuracy stays in the soft band but enters the narrow recovery band,
    # producing nonzero boost and a partially decayed active recovery together.
    step(original, first, 3, 0.98)
    step(original, first, 4, 0.98)
    assert original.recovery_active and original.log_step_boost_level > 0
    path = tmp_path / "state.pt"
    torch.save(original.state_dict(), path)
    saved = torch.load(path, weights_only=True)
    restored, second = controller(initial_lambda_coef=0.7), Probe()
    restored.load_state_dict(saved)
    restored.apply_initial_state(second, apply_lambda=apply_lambda)
    assert restored.state_dict() == original.state_dict()
    assert second.open_bias.item() == pytest.approx(0.135)
    for epoch, accuracy in [(5, 0.98), (6, 0.99), (7, 0.99), (8, 0.99)]:
        a, b = step(original, first, epoch, accuracy), step(restored, second, epoch, accuracy)
        assert a.metrics == b.metrics
        assert restored.state_dict() == original.state_dict()
        assert second.lambda_coef == first.lambda_coef
        torch.testing.assert_close(second.open_bias, first.open_bias)
    # Returned state is detached from controller internals.
    saved["runtime"]["observed_accuracy_by_epoch"][1] = 0.0
    assert restored.observed_accuracy_by_epoch[1] == 0.99


@pytest.mark.parametrize("bad", ["config", "reference", "missing", "version", "nonfinite", "window"])
def test_controller_rejects_incompatible_or_partial_state(bad):
    control = controller()
    before = control.state_dict()
    value = deepcopy(before)
    if bad == "config":
        value["config"]["warmup_epochs"] += 1
    elif bad == "reference":
        value["config"]["reference_accuracy_by_epoch"][1] = 0.8
    elif bad == "missing":
        del value["runtime"]["recovery_epochs_left"]
    elif bad == "version":
        value["version"] = 2
    elif bad == "nonfinite":
        value["runtime"]["log_lambda"] = float("nan")
    else:
        value["runtime"]["acc_history"] = [0.9, 0.8]
    with pytest.raises(ValueError):
        control.load_state_dict(value)
    assert control.state_dict() == before


class MetricState:
    def __init__(self):
        self.values = {}

    def compute(self):
        return dict(self.values)

    def reset(self):
        self.values = {}


def test_engine_handoff_uses_global_epoch_and_selected_state_without_mutating_best(tmp_path, monkeypatch):
    model = Probe()
    adaptive = controller(warmup_epochs=10, adaptive_log_step_enabled=False)
    step(adaptive, model, 10, 0.99)
    config = OmegaConf.create({
        "seed": 42,
        "training_arguments": {
            "num_epochs": 2, "global_epoch_offset": 10, "evaluate_test": False,
            "adaptive_lambda": {
                "enabled": True, "warmup_epochs": 10, "update_every_epochs": 1,
                "acc_window": 1, "log_step_init": math.log(2),
                "adaptive_log_step_enabled": False,
                "recovery": adaptive.recovery_config.as_dict(),
            },
        },
        "cyclic_channel_pruning": {"audit_protocol": True},
        "run_history": {"root_dir": str(tmp_path), "run_name": "handoff",
                        "monitor": "valid_accuracy", "mode": "max"},
    })
    history = RunHistory(config)
    metric_state = Metrics(train_metrics=MetricState(), valid_metrics=MetricState(), test_metrics=MetricState())

    def fake_train(model, *args, epoch, **kwargs):
        with torch.no_grad():
            model.weight.fill_(epoch)
        metric_state.train_metrics.values = {"train_loss": float(epoch)}

    def fake_evaluate(model, loader, metrics, device, *, stage, epoch, **kwargs):
        assert stage == "valid", "test must not influence training"
        metrics.values = {"valid_accuracy": 0.96 if epoch == 1 else 0.8,
                          "valid_average_zero_prob": 0.4}

    monkeypatch.setattr(engine, "train_epoch", fake_train)
    monkeypatch.setattr(engine, "evaluate", fake_evaluate)
    callbacks = []
    engine.train(model, torch.optim.SGD(model.parameters(), lr=0.1), None, Dataloaders(),
                 config.training_arguments, metric_state, "cpu", run_history=history,
                 adaptive_lambda_state=adaptive.state_dict(), adaptive_epoch_offset=10,
                 adaptive_reference_by_epoch=adaptive.reference_accuracy_by_epoch,
                 epoch_end_callback=lambda *args: callbacks.append(deepcopy(args[-1].history_records[-1])))
    best = torch.load(history.checkpoints_dir / "best.pt", weights_only=True)
    last = torch.load(history.checkpoints_dir / "last.pt", weights_only=True)
    best_state = best["extra_state"]["adaptive_lambda_state"]
    last_state = last["extra_state"]["adaptive_lambda_state"]
    assert best["epoch"] == 1 and last["epoch"] == 2
    assert best_state["runtime"]["last_epoch"] == 11
    assert last_state["runtime"]["last_epoch"] == 12
    assert best_state["runtime"]["recovery_active"] is True
    assert best_state["runtime"]["recovery_open_bias"] == pytest.approx(0.15)
    assert best["model_state_dict"]["open_bias"].item() == 0
    assert last["model_state_dict"]["open_bias"].item() == pytest.approx(0.135)
    assert best["model_state_dict"]["weight"].item() == 1
    assert last["model_state_dict"]["weight"].item() == 2
    assert best["controller_state"]["enabled"] and last["controller_state"]["enabled"]
    assert best["controller_state"]["state"] == best_state
    assert callbacks[0]["adaptive_lambda_action"] == "hold"
    assert callbacks[1]["adaptive_lambda_action"] == "decrease_lambda"
    assert callbacks[1]["lambda_next"] < callbacks[1]["lambda_used"]


def test_external_reference_bypasses_baseline_generation(monkeypatch):
    class ReachedModelSetup(Exception):
        pass

    def fail_baseline(*args, **kwargs):
        pytest.fail("external reference must not train a hidden baseline")

    def stop_at_setup(*args, **kwargs):
        raise ReachedModelSetup

    monkeypatch.setattr(engine, "_ensure_adaptive_baseline_reference", fail_baseline)
    monkeypatch.setattr(engine, "set_random_seed", stop_at_setup)
    config = OmegaConf.create({"training_arguments": {"adaptive_lambda": {"enabled": True}}})
    with pytest.raises(ReachedModelSetup):
        engine.run_training(config, adaptive_reference_by_epoch={1: 0.9})
    config.training_arguments.adaptive_lambda.baseline_history_dir = "already-configured"
    with pytest.raises(ValueError, match="conflicts"):
        engine.run_training(config, adaptive_reference_by_epoch={1: 0.9})


@pytest.mark.parametrize("reference", [{}, {0: 0.9}, {1: 90}, {1: float("nan")}])
def test_external_reference_requires_real_fractional_epoch_metrics(reference):
    config = OmegaConf.create({"training_arguments": {"adaptive_lambda": {"enabled": True}}})
    with pytest.raises(ValueError):
        engine.run_training(config, adaptive_reference_by_epoch=reference)


@pytest.mark.parametrize("enabled, expected_mode", [
    (True, "adaptive_state_pending"), (False, "structural_recovery_no_gates"),
])
def test_adaptive_audit_never_labels_checkpoints_as_fixed_pilot(tmp_path, enabled, expected_mode):
    config = OmegaConf.create({
        "training_arguments": {"adaptive_lambda": {"enabled": enabled}},
        "cyclic_channel_pruning": {"audit_protocol": "adaptive_lambda_v1"},
        "run_history": {"root_dir": str(tmp_path), "run_name": "metadata"},
    })
    history = RunHistory(config)
    model = Probe()
    checkpoint = history.save_checkpoint("best.pt", model, torch.optim.SGD(model.parameters(), lr=0.1), 1, {})
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["controller_state"] == {
        "enabled": enabled, "mode": expected_mode, "held_in_parent": not enabled,
    }
