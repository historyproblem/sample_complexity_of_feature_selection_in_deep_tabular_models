from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pruning_pilot_common import config_for, make_initializer
from net_complexity.training import pruning_audit as audit
from net_complexity.training.pruning_measurement import mask_hash, state_hash, write_json
from net_complexity.training.pruning_resume import load_completed_search, validate_reused_dense


class LogitProbe(nn.Module):
    def __init__(self, fp32_offset=0.0, always_offset=0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([always_offset, 0.0]))
        self.fp32_offset = fp32_offset

    def forward(self, x, y):
        offset = self.fp32_offset if x.dtype == torch.float32 else 0.0
        return SimpleNamespace(logits=x + self.bias + offset)


@pytest.mark.parametrize("offset,status", [(0.0, "passed_fp32"), (1.92e-5, "passed_fp64_fallback")])
def test_equivalence_accepts_only_verified_roundoff(tmp_path, offset, status):
    carrier, structural = LogitProbe(), LogitProbe(fp32_offset=offset)
    original_hash = state_hash(structural.state_dict())
    sample = torch.tensor([[0.02, -0.1]]).repeat(17, 1), torch.zeros(17, dtype=torch.long)
    result = audit.committed_equivalence(carrier, structural, {}, sample, "cpu",
                                         report_path=tmp_path / "check.json")
    assert result["status"] == status
    assert json.loads((tmp_path / "check.json").read_text()) == result
    assert state_hash(structural.state_dict()) == original_hash
    assert structural.bias.dtype == torch.float32
    if offset:
        assert result["fp32"]["mismatched_logits"] > 0
        assert result["fp64"]["logit_count"] == 34  # Entire batch, including the final chunk.
        assert result["fp64"]["max_abs_error"] == 0


@pytest.mark.parametrize("fp32,always,error", [
    (0.0, 1.92e-5, AssertionError),  # Small but REAL transfer error must fail FP64.
    (0.01, 0.0, AssertionError),  # Large FP32-only errors still exceed the hard ceiling.
    (float("inf"), 0.0, FloatingPointError),
    (float("nan"), 0.0, FloatingPointError),
])
def test_equivalence_rejects_real_errors_and_nonfinite(tmp_path, fp32, always, error):
    sample = torch.tensor([[0.02, -0.1]]), torch.zeros(1, dtype=torch.long)
    with pytest.raises(error):
        audit.committed_equivalence(LogitProbe(), LogitProbe(fp32, always), {}, sample, "cpu",
                                    report_path=tmp_path / "check.json")
    assert json.loads((tmp_path / "check.json").read_text())["status"] == "failed"


@pytest.fixture
def stopped_search_files(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_INIT_CHECKPOINT", "unused")
    cfg = config_for("D2_internal_fixed")
    cfg.cyclic_channel_pruning.gumbel_epochs = 2
    cfg.cyclic_channel_pruning.final_epochs = 1
    source = tmp_path / "source"
    run_dir = source / "cycle_0_search" / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    OmegaConf.save(cfg, source / "resolved_config.yaml", resolve=True)
    rows = [{"global_epoch": epoch, "local_epoch": epoch, "stage": "cycle_0_search",
             "valid_accuracy": accuracy, "valid_ce_loss": 0.5, "mask_hash": mask_hash({}),
             "optimizer_steps_total": epoch * 2, "lambda_used": 0.001}
            for epoch, accuracy in ((1, 0.9), (2, 0.8))]
    with (source / "global_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    state = {"status": "failed", "global_epochs_completed": 2, "total_epochs_allocated": 3,
             "decisions": [], "accepted_mask": {}, "test_evaluated": False, "seed": 42,
             "common_init_hash": "init", "split_indices_hash": "split",
             "stages": [{"name": "cycle_0_search", "epochs": 2, "run_dir": str(run_dir),
                         "best_epoch": 1, "best_metric": 0.9}]}
    write_json(source / "pilot_state.json", state)
    write_json(run_dir / "summary.json", {"timing": {"num_epochs_executed": 2}, "test": {},
                                          "best_valid": {"epoch": 1}})
    for name, epoch in (("best", 1), ("last", 2)):
        weights = {"weight": torch.tensor([float(epoch)])}
        torch.save({"epoch": epoch, "global_epochs_completed": epoch, "pruning_mask": {},
                    "mask_hash": mask_hash({}), "model_state_dict": weights,
                    "model_state_hash": state_hash(weights),
                    "metrics": {"valid_accuracy": rows[epoch - 1]["valid_accuracy"], "valid_ce_loss": 0.5}},
                   run_dir / "checkpoints" / f"{name}.pt")
    return cfg, source, run_dir


def _load(cfg, source):
    return load_completed_search(cfg, source, common_hash="init", split_hash="split", steps_per_epoch=2)


def test_resume_uses_best_but_charges_all_completed_epochs(stopped_search_files):
    cfg, source, _ = stopped_search_files
    result = _load(cfg, source)
    assert result["stage"]["best_epoch"] == 1
    assert result["metadata"]["epochs_reused"] == 2
    assert result["metadata"]["selected_checkpoint_state_hash"] != result["metadata"]["last_checkpoint_state_hash"]


@pytest.mark.parametrize("mutation", ["partial", "committed", "config", "split", "hash", "steps", "recovery", "summary"])
def test_resume_rejects_unsafe_sources(stopped_search_files, mutation):
    cfg, source, run_dir = stopped_search_files
    state = json.loads((source / "pilot_state.json").read_text())
    if mutation == "partial":
        state["global_epochs_completed"] = 1
    elif mutation == "committed":
        state["decisions"] = [{"status": "accepted"}]
    elif mutation == "config":
        cfg.optimizer.lr = 0.02
    elif mutation == "split":
        state["split_indices_hash"] = "different"
    elif mutation == "hash":
        path = run_dir / "checkpoints" / "best.pt"
        payload = torch.load(path, weights_only=True)
        payload["model_state_dict"]["weight"].add_(1)
        torch.save(payload, path)
    elif mutation == "steps":
        path = source / "global_history.csv"
        path.write_text(path.read_text().replace(",4,", ",5,"))
    elif mutation == "recovery":
        (source / "cycle_0_recovery").mkdir()
    else:
        write_json(run_dir / "summary.json", {"timing": {"num_epochs_executed": 1}, "test": {},
                                              "best_valid": {"epoch": 1}})
    write_json(source / "pilot_state.json", state)
    with pytest.raises(ValueError, match="Unsafe pruning resume"):
        _load(cfg, source)


@pytest.mark.parametrize("job,epochs", [("D1_dense_control", 25), ("J1_dense_control", 150)])
@pytest.mark.parametrize("mutation", [None, "partial", "weights", "initializer", "config"])
def test_reused_dense_validates_original_weights_and_config(tmp_path, monkeypatch, mutation, job, epochs):
    initial_path = tmp_path / "shared.pt"
    monkeypatch.setenv("AUDIT_INIT_CHECKPOINT", str(initial_path))
    cfg = config_for(job)
    initial = {"weight": torch.tensor([1.0])}
    weights = {"weight": torch.tensor([2.0])}
    torch.save({"trained_epochs": 0, "seed": 42, "model_state_dict": initial}, initial_path)
    source = tmp_path / "dense"
    source.mkdir()
    OmegaConf.save(cfg, source / "resolved_config.yaml", resolve=True)
    state = {"status": "completed", "global_epochs_completed": epochs, "test_evaluated": False,
             "accepted_mask": {}, "seed": 42, "common_init_hash": state_hash(initial),
             "validation": {"accuracy": 0.9198}}
    checkpoint = {"model_state_dict": weights, "model_state_hash": state_hash(weights),
                  "common_init_hash": state["common_init_hash"], "global_epochs_consumed": epochs,
                  "validation": state["validation"], "pruning_mask": {}}
    if mutation == "partial":
        state["global_epochs_completed"] = epochs - 1
    elif mutation == "weights":
        weights["weight"].add_(1)
    elif mutation == "initializer":
        state["common_init_hash"] = "different"
    elif mutation == "config":
        cfg.seed = 7
    write_json(source / "pilot_state.json", state)
    torch.save(checkpoint, source / "deployment.pt")
    if mutation:
        with pytest.raises(ValueError, match="Unsafe pruning resume"):
            validate_reused_dense(cfg, source, expected_epochs=epochs)
    else:
        assert validate_reused_dense(cfg, source, expected_epochs=epochs)["reused_from"] == str(source.resolve())


@pytest.mark.parametrize("extra", [
    ["--profile", "nightly"], ["--with-random-control"], ["--preflight-only"], ["--job", "D1_dense_control"],
])
def test_launcher_rejects_unsupported_resume_modes(tmp_path, monkeypatch, extra):
    import launch_pruning_pilot as launcher
    monkeypatch.setattr(sys, "argv", ["launcher", "--profile", "daytime", "--output", str(tmp_path / "new"),
                                      "--resume-from", str(tmp_path / "old"), *extra])
    with pytest.raises(SystemExit):
        launcher.main()
    assert not (tmp_path / "new").exists()


def test_real_resume_skips_search_preserves_source_and_matches_uninterrupted(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv("AUDIT_INIT_CHECKPOINT", str(tmp_path / "shared.pt"))
    cfg = config_for("D2_internal_fixed")
    cfg.device = "cpu"
    cfg.dataloaders = {"_target_": "smoke_pruning_pilot.SmokeDataloaders", "loader_seed": 42}
    cfg.cyclic_channel_pruning.gumbel_epochs = 1
    cfg.cyclic_channel_pruning.final_epochs = 1
    cfg.cyclic_channel_pruning.commit_guard.train_bn_calibration_batches = 1
    cfg.cyclic_channel_pruning.commit_guard.max_immediate_accuracy_drop = 1.0
    cfg.cyclic_channel_pruning.commit_guard.max_recovered_accuracy_drop = 1.0
    make_initializer(cfg, tmp_path / "shared.pt")
    original_check, original_train = audit.committed_equivalence, audit.run_training
    def stop_at_boundary(*args, **kwargs):
        raise AssertionError("simulated boundary failure")
    monkeypatch.setattr(audit, "committed_equivalence", stop_at_boundary)
    source = tmp_path / "failed"
    with pytest.raises(AssertionError, match="simulated"):
        audit.run_fixed_pruning_pilot(cfg, source)
    import hashlib
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source.rglob("*") if path.is_file()}
    monkeypatch.setattr(audit, "committed_equivalence", original_check)
    original_stats = audit._equivalence_stats
    def force_double_check(actual, expected, **tolerances):
        result = original_stats(actual, expected, **tolerances)
        if tolerances == {"rtol": 1e-4, "atol": 1e-5}:
            result["mismatched_logits"] = max(1, result["mismatched_logits"])
        return result
    monkeypatch.setattr(audit, "_equivalence_stats", force_double_check)
    calls = []
    def record_training(config, **kwargs):
        calls.append((config.run_history.run_name, config.training_arguments.global_epoch_offset))
        return original_train(config, **kwargs)
    monkeypatch.setattr(audit, "run_training", record_training)
    result = audit.run_fixed_pruning_pilot(cfg, tmp_path / "resumed", resume_search_from=source)
    assert calls == [("cycle_0_recovery", 1)]
    assert result["global_epochs_completed"] == 2
    assert result["optimizer_steps_total"] == 4
    assert result["resume"]["epochs_reused"] == 1
    assert result["decisions"][0]["equivalence"]["status"] == "passed_fp64_fallback"
    assert source_hashes == {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in source.rglob("*") if path.is_file()}
    # Fresh launches must seed the initial gates, independently of caller RNG.
    torch.manual_seed(999)
    baseline = audit.run_fixed_pruning_pilot(cfg, tmp_path / "baseline")
    assert result["accepted_mask"] == baseline["accepted_mask"]
    assert result["validation"] == baseline["validation"]
    resumed = torch.load(tmp_path / "resumed" / "deployment.pt", weights_only=True)
    uninterrupted = torch.load(tmp_path / "baseline" / "deployment.pt", weights_only=True)
    assert resumed["model_state_hash"] == uninterrupted["model_state_hash"]


def test_launcher_resume_reuses_dense_and_runs_only_d2(tmp_path, monkeypatch):
    import launch_pruning_pilot as launcher
    source, output = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    write_json(source / "provenance.json", {"original": True})
    monkeypatch.setattr(sys, "argv", ["launcher", "--profile", "daytime", "--output", str(output),
                                      "--resume-from", str(source)])
    monkeypatch.setattr(launcher, "sys", SimpleNamespace(version_info=(3, 12), executable=sys.executable, version=sys.version))
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(launcher, "provenance", lambda: {"new": True})
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    from test_pruning_pilot_launcher import _completed_result
    dense = {**_completed_result(), "reused_from": str(source / "D1_dense_control")}
    monkeypatch.setattr(launcher, "validate_reused_dense", lambda *a, **k: dense)
    monkeypatch.setattr(launcher, "load_completed_search", lambda *a, **k: {"metadata": {"epochs_reused": 15}})
    monkeypatch.setattr(launcher, "make_initializer", lambda *a: pytest.fail("initializer recreated"))
    calls = []
    def child(command, log, deadline):
        calls.append(command)
        if "--job" in command:
            assert command[command.index("--job") + 1] == "D2_internal_fixed"
            assert command[command.index("--resume-from") + 1] == str(source)
            write_json(output / "D2_internal_fixed" / "pilot_state.json", _completed_result())
    monkeypatch.setattr(launcher, "run_child", child)
    launcher.main()
    assert len(calls) == 3  # Unit tests, GPU smoke, D2 only.
    assert json.loads((output / "nightly_status.json").read_text())["status"] == "completed"
    assert json.loads((output / "comparison.json").read_text())["runs"][0]["reused_from"] == dense["reused_from"]
