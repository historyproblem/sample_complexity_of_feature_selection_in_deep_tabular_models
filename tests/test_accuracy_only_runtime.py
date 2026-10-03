from copy import deepcopy
import math

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from net_complexity.models.feature_selection import ClassificationFeatureSelectionWrapper, GumbelLayer
from net_complexity.training.adaptive_lambda import AccuracyOnlyLambdaController, AdaptiveLambdaController
from net_complexity.training import engine
from net_complexity.training.meta import Metrics
from net_complexity.training.run_history import RunHistory
from net_complexity.training.pruning_resume import (
    capture_eval_runtime, restore_eval_runtime, checkpoint_runtime_status,
    migrate_legacy_controller_to_accuracy_only,
)


def controller(**overrides):
    args = dict(initial_lambda_coef=0.001, reference_accuracy_by_epoch={i: .8 for i in range(1, 151)})
    args.update(overrides)
    return AccuracyOnlyLambdaController(**args)


def step(c, epoch, accuracy=.8, **noise):
    return c.on_epoch_end(epoch=epoch, model=torch.nn.Identity(),
        valid_metrics={"valid_accuracy": accuracy, **noise}, apply_lambda=lambda *_: None)


def test_cadence_gap_pairs_recovery_rebase_and_resume_are_sparsity_independent():
    a, b = controller(), controller()
    updates = []
    for epoch in range(1, 21):
        x = step(a, epoch, valid_zero_prob=0.0, removed_count=0)
        y = step(b, epoch, valid_zero_prob=1.0, removed_count=epoch * 50)
        assert x == y
        if x.lambda_changed:
            updates.append(epoch)
    assert updates == [13, 16, 19]
    alpha, window = a.lambda_coef, deepcopy(a.state_dict()["runtime"]["gap_history"])
    a.hold_recovery(global_training_epoch=35)
    assert a.lambda_coef == alpha
    assert a.state_dict()["runtime"]["search_epochs_consumed"] == 20
    assert a.state_dict()["runtime"]["gap_history"] == window
    args = dict(transition_id="R0->S1", phase_id="S1", previous_mask_hash="old", new_mask_hash="new",
                reason="physical_recovery_completed", global_training_epoch=35, search_epochs_consumed=20)
    assert a.rebase(**args)
    assert not a.rebase(**args)
    assert a.lambda_coef == alpha and a.state_dict()["runtime"]["gap_history"] == []
    assert step(a, 36).reason == "reentry_window"
    assert step(a, 37).reason == "reentry_window"
    saved = a.state_dict()
    resumed = controller()
    resumed.load_state_dict(saved)
    assert step(a, 38) == step(resumed, 38)
    assert a.lambda_coef == pytest.approx(alpha * 2)
    with pytest.raises(ValueError, match="already consumed"):
        step(resumed, 38)


@pytest.mark.parametrize("gap, factor", [(.02, 2), (.0201, 1), (.05, 1), (.0501, .5)])
def test_exact_percentage_point_boundaries_and_bounds(gap, factor):
    c = controller(initial_search_warmup=0, gap_window=1, update_every_search_epochs=1)
    event = step(c, 1, .8-gap)
    assert c.lambda_coef == pytest.approx(.001 * factor)
    if factor == 1:
        assert c.lambda_coef == .001 and not event.lambda_changed
    bounded = controller(initial_lambda_coef=.001, alpha_max=.001,
                         initial_search_warmup=0, gap_window=1, update_every_search_epochs=1)
    result = step(bounded, 1)
    assert bounded.lambda_coef == .001 and "alpha_bound" in result.reason


def test_controller_averages_aligned_gaps_and_requires_complete_feedback():
    c = controller(reference_accuracy_by_epoch={1: .1, 2: .5, 3: .9},
                   initial_search_warmup=0)
    for epoch, accuracy in enumerate([.1, .5, .9], 1):
        event = step(c, epoch, accuracy)
    assert event.metrics["quality_mean_gap"] == 0
    assert c.lambda_coef == pytest.approx(.002)
    for value in [float("nan"), float("inf"), -1, 1.1, None]:
        c = controller()
        with pytest.raises(ValueError, match="feedback/reference"):
            step(c, 1, value)
        assert c.lambda_coef == .001
    c = controller(reference_accuracy_by_epoch={1: .9})
    with pytest.raises(ValueError, match="feedback/reference"):
        step(c, 2)


def test_strict_controller_rejects_rate_options():
    model = make_model()
    cfg = OmegaConf.create({"adaptive_lambda": {"enabled": True, "control_mode": "accuracy_only",
                                              "adaptive_log_step_enabled": True}})
    with pytest.raises(ValueError, match="unknown/legacy"):
        engine._build_adaptive_lambda(cfg, model, baseline_accuracy_by_epoch={1: .8})


def make_model():
    return ClassificationFeatureSelectionWrapper(
        torch.nn.Sequential(GumbelLayer(2, train_gate_mode="deterministic_soft",
                                       eval_gate_mode="deterministic_hard"),
                            torch.nn.Dropout(.25), torch.nn.Linear(2, 2)),
        lambda_coef=.001)


class ActualMetrics:
    def __init__(self, prefix):
        self.prefix = prefix
        self.reset()

    def update(self, X, output, y, model):
        self.correct += int((output.logits.argmax(dim=1) == y).sum())
        self.count += len(y)
        self.loss += float(output.ce_loss.detach()) * len(y)

    def compute(self):
        return {f"{self.prefix}_accuracy": self.correct / self.count,
                f"{self.prefix}_ce_loss": self.loss / self.count}

    def reset(self):
        self.correct = self.count = 0
        self.loss = 0.


def make_run(tmp_path, model, ledger, *, resume=None, initialize_callback=None):
    cfg = OmegaConf.create({"seed": 7, "training_arguments": {
        "num_epochs": 3, "evaluate_test": False,
        "batchnorm_recalibration": {"enabled": False},
        "accuracy_guided_stage": {"id": "S0", "config_hash": "config", "reference_hash": "reference"},
        "adaptive_lambda": {"enabled": True, "control_mode": "accuracy_only",
            "initial_search_warmup": 0, "gap_window": 1, "update_every_search_epochs": 1}},
        "run_history": {"root_dir": str(tmp_path), "run_name": "actual", "monitor": "valid_accuracy", "mode": "max"}})
    data = TensorDataset(torch.arange(10, dtype=torch.float32).reshape(5, 2)/10,
                         torch.tensor([0, 1, 0, 1, 0]))
    train_loader = DataLoader(data, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(19))
    from net_complexity.data.dataloaders import Dataloaders
    loaders = Dataloaders()
    loaders.train_dataloader = train_loader
    loaders.valid_dataloader = DataLoader(data, batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = engine.SchedulerState(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3))
    history = RunHistory(cfg)
    history.train_dataloader_generator = train_loader.generator
    result = engine.train(model, optimizer, scheduler, loaders, cfg.training_arguments,
        Metrics(ActualMetrics("train"), ActualMetrics("valid"), None), "cpu", run_history=history,
        adaptive_reference_by_epoch={i: 0. for i in range(1, 4)}, training_ledger=ledger,
        exact_resume_checkpoint=resume, runtime_initialized_callback=initialize_callback)
    return result, history


def empty_ledger():
    return dict(global_training_epoch=0, search_epochs_consumed=0, optimizer_updates=0,
                consumed_training_examples=0)


def test_actual_penalty_logging_preserves_tiny_alpha_below_ce_roundoff(tmp_path):
    from net_complexity.models.feature_selection import get_gumbel_loss
    model = make_model()
    model.regularization_loss = get_gumbel_loss
    model.set_lambda_coef(1e-8, bypass_gumbel=False)
    _, history = make_run(tmp_path, model, empty_ledger())
    event = torch.load(history.checkpoints_dir / "epoch_0001.pt", weights_only=True)["epoch_event"]
    assert event["alpha_used"] == 1e-8
    assert event["train_L_gate_mean"] > 0


def test_real_engine_snapshots_actual_ledger_and_exact_epoch_resume(tmp_path):
    torch.manual_seed(77)
    model, ledger = make_model(), empty_ledger()
    result, history = make_run(tmp_path / "full", model, ledger)
    assert ledger == dict(global_training_epoch=3, search_epochs_consumed=3,
                          optimizer_updates=9, consumed_training_examples=15)
    first_path = history.checkpoints_dir / "epoch_0001.pt"
    first = torch.load(first_path, weights_only=True)
    event = first["epoch_event"]
    assert event["alpha_used"] == .001 and event["alpha_next"] == pytest.approx(.002)
    assert event["consumed_ledger"]["optimizer_updates"] == 3
    assert event["controller_after_feedback"]["runtime"]["next_update_epoch"] == 2
    assert checkpoint_runtime_status(first)["exact_resume"]
    resumed, resumed_ledger = make_model(), empty_ledger()
    result2, history2 = make_run(tmp_path / "resumed", resumed, resumed_ledger, resume=first_path)
    assert resumed_ledger == ledger
    assert result2["adaptive_lambda_state"] == result["adaptive_lambda_state"]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    last = torch.load(history.checkpoints_dir / "last.pt", weights_only=True)
    last2 = torch.load(history2.checkpoints_dir / "last.pt", weights_only=True)
    assert last["scheduler_state_dict"] == last2["scheduler_state_dict"]
    # Early selection does not mutate the independent consumed ledger.
    engine._load_model_checkpoint_for_evaluation(model, first_path, device="cpu")
    assert model.lambda_coef == .001 and ledger["optimizer_updates"] == 9


def test_eval_runtime_preserves_nonzero_legacy_bias_and_marks_incomplete_legacy():
    model = make_model()
    gate = model.backbone[0]
    gate.set_open_bias(.25, p_min=.01, p_max=.7)
    gate.gate_threshold = .6
    model.eval()
    x = torch.ones(3, 2)
    with torch.no_grad():
        expected = model(x, torch.zeros(3, dtype=torch.long)).logits.clone()
    runtime = capture_eval_runtime(model)
    gate.set_open_bias(0)
    gate.gate_threshold = .1
    restore_eval_runtime(model, runtime)
    with torch.no_grad():
        torch.testing.assert_close(model(x, torch.zeros(3, dtype=torch.long)).logits, expected, rtol=0, atol=0)
    assert gate.open_bias == .25 and gate.gate_threshold == .6
    assert checkpoint_runtime_status({})["status"] == "legacy_incomplete_runtime"
    with pytest.raises(ValueError, match="Incomplete"):
        checkpoint_runtime_status({"epoch_event": {"version": 1}})


def test_explicit_legacy_migration_preserves_alpha_without_exact_resume_claim():
    old = AdaptiveLambdaController(initial_lambda_coef=.002, recovery_config={"enabled": False})
    new = controller()
    result = migrate_legacy_controller_to_accuracy_only(old.state_dict(), new,
        transition_id="legacy->v3", global_training_epoch=35, search_epochs_consumed=20,
        previous_mask_hash="x", new_mask_hash="x", phase_id="S1")
    assert result["status"] == "migrated_handoff_not_exact_resume"
    assert new.lambda_coef == pytest.approx(.002)
    bad = old.state_dict()
    bad["runtime"]["recovery_active"] = True
    with pytest.raises(ValueError, match="Active legacy"):
        migrate_legacy_controller_to_accuracy_only(bad, controller(), transition_id="x",
            global_training_epoch=1, search_epochs_consumed=1, previous_mask_hash="x",
            new_mask_hash="x", phase_id="S1")


def test_partial_epoch_records_actual_updates_and_examples_without_fabricated_epoch(tmp_path, monkeypatch):
    from net_complexity.training import interruption
    ledger = empty_ledger()

    def stop_after_one(*, epoch=0, batches=0):
        if batches == 1:
            raise interruption.TrainingInterrupted(epoch=epoch, batches=batches)

    monkeypatch.setattr(interruption, "check_stop", stop_after_one)
    with pytest.raises(interruption.TrainingInterrupted):
        make_run(tmp_path, make_model(), ledger)
    assert ledger == dict(global_training_epoch=0, search_epochs_consumed=0,
                          optimizer_updates=1, consumed_training_examples=2)


def test_epoch_event_reproduces_legacy_used_bias_separately_from_next(tmp_path, monkeypatch):
    model = make_model()
    cfg = OmegaConf.create({"training_arguments": {
        "num_epochs": 2, "evaluate_test": False, "accuracy_guided_stage": {"id": "legacy_runtime_probe"},
        "adaptive_lambda": {"enabled": True, "warmup_epochs": 0, "update_every_epochs": 1,
            "acc_window": 1, "log_step_init": math.log(2), "adaptive_log_step_enabled": False,
            "recovery": {"enabled": True, "min_epoch": 1, "patience": 1,
                "require_slow_recovery": False, "use_zero_prob_filter": False,
                "recovery_epochs": 3, "drop_min": .005}}},
        "run_history": {"root_dir": str(tmp_path), "run_name": "bias"}})
    history = RunHistory(cfg)
    metrics = Metrics(ActualMetrics("train"), ActualMetrics("valid"), None)

    def fill_metric(target, accuracy):
        target.count = 100
        target.correct = int(accuracy * 100)
        target.loss = 1

    def fake_train(*args, **kwargs):
        fill_metric(metrics.train_metrics, .99)

    def fake_eval(*args, epoch, **kwargs):
        fill_metric(metrics.valid_metrics, .99 if epoch == 1 else .97)

    monkeypatch.setattr(engine, "train_epoch", fake_train)
    monkeypatch.setattr(engine, "evaluate", fake_eval)
    from net_complexity.data.dataloaders import Dataloaders
    engine.train(model, torch.optim.SGD(model.parameters(), lr=.1), None, Dataloaders(),
        cfg.training_arguments, metrics, "cpu", run_history=history,
        adaptive_reference_by_epoch={1: .99, 2: .99})
    path = history.checkpoints_dir / "epoch_0002.pt"
    saved = torch.load(path, weights_only=True)
    event = saved["epoch_event"]
    assert event["eval_runtime"]["backbone.0"]["_open_bias"] == 0
    assert event["continuation_runtime"]["backbone.0"]["_open_bias"] == .15
    assert event["controller_after_feedback"]["runtime"]["recovery_active"]
    assert model.backbone[0].open_bias == .15
    engine._load_model_checkpoint_for_evaluation(model, path, device="cpu")
    assert model.backbone[0].open_bias == 0


def test_exact_resume_refuses_tampered_consumed_ledger_before_training(tmp_path, monkeypatch):
    _, history = make_run(tmp_path / "source", make_model(), empty_ledger())
    original = torch.load(history.checkpoints_dir / "epoch_0001.pt", weights_only=True)
    original["epoch_event"]["clocks"]["global_training_epoch"] += 1
    path = tmp_path / "invalid.pt"
    torch.save(original, path)

    def forbidden_train(*args, **kwargs):
        pytest.fail("Invalid resume must fail before training")

    monkeypatch.setattr(engine, "train_epoch", forbidden_train)
    with pytest.raises(ValueError, match="consumed clocks/ledger"):
        make_run(tmp_path / "refused", make_model(), empty_ledger(), resume=path)
