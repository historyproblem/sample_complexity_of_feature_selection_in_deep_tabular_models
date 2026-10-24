from copy import deepcopy
import json
import math
import random

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from net_complexity.data.dataloaders import Dataloaders
from net_complexity.metrics import Accuracy, MultiLossMetric
from net_complexity.models.feature_selection import ClassificationFeatureSelectionWrapper, GumbelLayer
from net_complexity.training import engine
from net_complexity.training.meta import Metrics
from net_complexity.training.randomness import set_random_seed
from net_complexity.training.run_history import RunHistory
from net_complexity.training.search_reproducibility import SearchReproducibilityTrace, runtime_environment


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def run_search(root, *, horizon=4, stop=None, resume=None, trace=False, warmup=1, log_step=math.log(2)):
    set_random_seed(77)
    model = ClassificationFeatureSelectionWrapper(
        torch.nn.Sequential(
            GumbelLayer(2, train_gate_mode="ste_hard", eval_gate_mode="deterministic_hard"),
            torch.nn.BatchNorm1d(2), torch.nn.Dropout(.25), torch.nn.Linear(2, 2),
        ), lambda_coef=.001,
    )
    cfg = OmegaConf.create({"seed": 77, "training_arguments": {
        "num_epochs": horizon, "evaluate_test": False, "audit_data_seed": 42,
        "batchnorm_recalibration": {"enabled": False},
        "accuracy_guided_stage": {"id": "S0", "config_hash": "fixed", "reference_hash": "fixed"},
        "adaptive_lambda": {"enabled": True, "control_mode": "accuracy_only",
            "initial_search_warmup": warmup, "gap_window": 1,
            "update_every_search_epochs": 1, "log_step": log_step}},
        "run_history": {"root_dir": str(root), "run_name": "search", "monitor": "valid_accuracy", "mode": "max"}})
    dataset = TensorDataset(torch.arange(16, dtype=torch.float32).reshape(8, 2) / 16,
                            torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]))
    loaders = Dataloaders()
    loaders.train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True,
        generator=torch.Generator().manual_seed(42))
    loaders.valid_dataloader = DataLoader(dataset, batch_size=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = engine.SchedulerState(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=horizon))
    history = RunHistory(cfg)
    history.train_dataloader_generator = loaders.train_dataloader.generator
    metrics = engine.prepare_metrics(Metrics(
        [MultiLossMetric(), Accuracy()], [MultiLossMetric(), Accuracy(return_counts=True)], [],
    ))
    ledger = dict(global_training_epoch=0, search_epochs_consumed=0, optimizer_updates=0,
                  consumed_training_examples=0)
    observer = SearchReproducibilityTrace(root / "diagnostics") if trace else None
    callbacks = []
    result = engine.train(model, optimizer, scheduler, loaders, cfg.training_arguments, metrics, "cpu",
        run_history=history, training_ledger=ledger,
        adaptive_reference_by_epoch={i: 0. for i in range(1, horizon + 1)},
        exact_resume_checkpoint=resume, stop_after_epoch=stop, diagnostic_observer=observer,
        epoch_end_callback=lambda epoch, *args: callbacks.append(epoch))
    saved = torch.load(history.checkpoints_dir / "last.pt", weights_only=True)
    return result, history, saved, callbacks, metrics


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_operational_pause_and_resume_match_uninterrupted_search(tmp_path):
    full, _, full_checkpoint, _, _ = run_search(tmp_path / "full")
    paused, history, checkpoint, callbacks, metrics = run_search(tmp_path / "prefix", stop=2)
    assert paused["status"] == "operational_pause"
    assert paused["configured_num_epochs"] == 4 and paused["next_epoch"] == 3
    assert paused["num_epochs_executed"] == 2 and callbacks == [1, 2]
    assert not paused["test_metrics"]
    assert metrics.train_metrics.metrics[1].total == metrics.valid_metrics.metrics[1].total == 0
    assert checkpoint["scheduler_state_dict"]["T_max"] == 4
    assert checkpoint["epoch_event"]["controller_after_feedback"]["runtime"]["lambda_coef"] == pytest.approx(.002)
    resumed, _, resumed_checkpoint, callbacks, _ = run_search(tmp_path / "resumed",
        resume=history.checkpoints_dir / "epoch_0002.pt")
    assert callbacks == [3, 4]
    assert resumed["training_ledger"] == full["training_ledger"]
    assert resumed["adaptive_lambda_state"] == full["adaptive_lambda_state"]
    for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "rng_state"):
        assert_nested_equal(full_checkpoint[key], resumed_checkpoint[key])


def test_twenty_epoch_prefix_preserves_sixty_epoch_horizon_and_auto_step(tmp_path):
    result, history, checkpoint, callbacks, _ = run_search(
        tmp_path, horizon=60, stop=20, warmup=10, log_step="auto",
    )
    assert result["status"] == "operational_pause" and callbacks == list(range(1, 21))
    assert result["configured_num_epochs"] == checkpoint["scheduler_state_dict"]["T_max"] == 60
    assert result["adaptive_lambda"]["adaptive_lambda_step"] == pytest.approx(.9210340371976183)
    events = [torch.load(history.checkpoints_dir / f"epoch_{epoch:04d}.pt", weights_only=True)["epoch_event"]
              for epoch in (10, 11, 12)]
    assert events[0]["alpha_used"] == events[0]["alpha_next"] == .001
    assert events[1]["alpha_used"] == .001 and events[1]["alpha_next"] > .001
    assert events[2]["alpha_used"] == events[1]["alpha_next"]


def test_trace_is_observational_and_records_actual_batches(tmp_path):
    plain, _, plain_checkpoint, _, _ = run_search(tmp_path / "plain")
    observed, _, observed_checkpoint, _, _ = run_search(tmp_path / "observed", trace=True)
    assert plain["last_train_metrics"] == observed["last_train_metrics"]
    assert plain["last_valid_metrics"] == observed["last_valid_metrics"]
    for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "rng_state"):
        assert_nested_equal(plain_checkpoint[key], observed_checkpoint[key])
    events = [json.loads(line) for line in (tmp_path / "observed/diagnostics/trace.jsonl").read_text().splitlines()]
    inputs = [event for event in events if event["kind"] == "batch_input"]
    assert len(inputs) == 8 and all(event["inputs"]["shape"] == [4, 2] for event in inputs)
    assert len([event for event in inputs if "rng" in event]) == 4
    assert [event["epoch"] for event in events if event["kind"] == "epoch_end"] == [1, 2, 3, 4]
    assert len([event for event in events if event["kind"] == "forward"]) == 2
    assert all(event["bn_state"] for event in events if event["kind"] == "epoch_end")


def test_repeat_trace_is_identical_and_environment_does_not_change_policy(tmp_path):
    before = deepcopy(runtime_environment()["backend_policy"])
    set_random_seed(42)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    SearchReproducibilityTrace(tmp_path / "environment_only")
    assert random.getstate() == rng[0]
    np.testing.assert_equal(np.random.get_state(), rng[1])
    assert torch.equal(torch.get_rng_state(), rng[2])
    assert runtime_environment()["backend_policy"] == before
    for name in ("one", "two"):
        run_search(tmp_path / name, trace=True)
    assert (tmp_path / "one/diagnostics/trace.jsonl").read_bytes() == (tmp_path / "two/diagnostics/trace.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        SearchReproducibilityTrace(tmp_path / "environment_only")


@pytest.mark.parametrize("stop", [0, -1, 5, True, 1.5])
def test_invalid_operational_boundary_is_rejected(tmp_path, stop):
    with pytest.raises(ValueError, match="unchanged training horizon"):
        run_search(tmp_path, stop=stop)
