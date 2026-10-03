"""Real tiny Bottleneck training, export, recovery and transactional integration."""
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from net_complexity.training.accuracy_guided_pruning import (
    prepare_carry, run_accuracy_guided_pruning, select_checkpoint_records,
)
from net_complexity.training.pruning_synthetic import make_synthetic_config
from net_complexity.training.pruning_measurement import compare_predictors, state_hash


@pytest.fixture(autouse=True)
def tiny_cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_selection_uses_common_end_reference_quality_then_cost_without_mutation():
    records = [dict(epoch=1, accuracy=.80, ce_loss=.5, physical_cost=100, proposal_mask={}),
               dict(epoch=2, accuracy=.76, ce_loss=.6, physical_cost=60, proposal_mask={"g": [0]}),
               dict(epoch=3, accuracy=.70, ce_loss=.4, physical_cost=30, proposal_mask={"g": [0, 1]})]
    before = deepcopy(records)
    selected, report = select_checkpoint_records(records, .80, .05)
    assert selected["epoch"] == 2
    assert report["quality_threshold"] == pytest.approx(.75)
    assert records == before
    selected, report = select_checkpoint_records(records, .95, .05)
    assert selected["epoch"] == 1 and report["no_feasible_search"]
    ties = [dict(epoch=e, accuracy=.8, ce_loss=.4, physical_cost=30) for e in (3, 2)]
    assert select_checkpoint_records(ties, .8, .05)[0]["epoch"] == 2
    records[0]["accuracy"] = float("nan")
    with pytest.raises(FloatingPointError):
        select_checkpoint_records(records, .8, .05)


@pytest.mark.parametrize("learned_closed", [False, True])
def test_real_iterative_noop_or_learned_recovery_carry_rebase_and_ledger(tmp_path, learned_closed):
    cfg = make_synthetic_config(tmp_path / "inputs", learned_closed=learned_closed)
    initial_bytes = Path(cfg.accuracy_guided.initializer.path).read_bytes()
    reference_bytes = Path(cfg.accuracy_guided.reference.history_path).read_bytes()
    state = run_accuracy_guided_pruning(cfg, tmp_path / "run")
    assert state["status"] == "completed" and not state["test_evaluated"]
    assert state["ledger"] == dict(global_training_epoch=8, search_epochs_consumed=6,
                                   optimizer_updates=16, consumed_training_examples=64)
    assert [s["kind"] for s in state["stages"]] == ["search", "recovery", "search", "recovery"]
    assert state["learned_candidates_materialized"] == (2 if learned_closed else 0)
    assert state["decisions"][1]["new_pruning"] is False
    assert all(d["quality_feasible"] for d in state["decisions"])
    assert len(state["transitions"]) == 1
    transition = state["transitions"][0]
    assert transition["first_forward_after_apply_initial_state"]
    assert transition["alpha_preserved"] == state["stages"][0]["alpha_next"]
    assert transition["predictors"]["all_open_vs_physical"]["logits_max_abs"] < 1e-4
    events = state["epoch_events"]
    assert [row["ledger"]["global_training_epoch"] for row in events] == list(range(1, 9))
    assert [row["ledger"]["search_epochs_consumed"] for row in events] == [1, 2, 3, 3, 4, 5, 6, 6]
    recovery = [r for r in events if r["stage"].startswith("recovery")]
    assert all(r["actual_L_gate"] == 0 for r in recovery)
    deployment = torch.load(tmp_path / "run/deployment.pt", weights_only=True)
    assert deployment["model_state_hash"] == state_hash(deployment["model_state_dict"])
    assert deployment["ledger"] == state["ledger"]
    assert deployment["artifact_type"] == "physical_ungated"
    assert not any("gumbel" in key for key in deployment["model_state_dict"])
    assert Path(cfg.accuracy_guided.initializer.path).read_bytes() == initial_bytes
    assert Path(cfg.accuracy_guided.reference.history_path).read_bytes() == reference_bytes
    with pytest.raises(FileExistsError):
        run_accuracy_guided_pruning(cfg, tmp_path / "run")


@pytest.mark.parametrize("reject_epoch", [4, 8])
def test_recovery_quality_reject_rollback_and_fallback_charge_all_epochs(tmp_path, reject_epoch):
    cfg = make_synthetic_config(tmp_path / "inputs", learned_closed=True, reject_after=reject_epoch)
    state = run_accuracy_guided_pruning(cfg, tmp_path / "run")
    assert state["status"] == "infeasible"
    assert state["global_epochs_completed"] == 8
    assert state["ledger"]["optimizer_updates"] == 16
    assert state["ledger"]["consumed_training_examples"] == 64
    rejection = state["decisions"][-1]
    assert rejection["status"] == "rejected_after_recovery"
    assert rejection["rejected_recovery_epochs_charged"] == 1
    assert rejection["quality_threshold"] == pytest.approx(.95)
    assert rejection["commits_stopped"]
    if reject_epoch == 4:
        assert state["accepted_mask"] == {}  # rejected provisional mask is not accepted regrowth
        assert state["stages"][-1]["kind"] == "fallback"
        assert state["stages"][-1]["epochs"] == 4
        assert state["ledger"]["search_epochs_consumed"] == 3
    else:
        assert state["accepted_mask"] == state["decisions"][0]["proposal_mask"]
        assert state["ledger"]["search_epochs_consumed"] == 6
        assert state["deployment_origin"]["stage"] == "recovery_0"
        deployed = torch.load(tmp_path / "run/deployment.pt", weights_only=True)
        assert deployed["provenance"]["selected_origin"] == state["deployment_origin"]


def test_no_feasible_search_keeps_mask_and_runs_allocated_recovery(tmp_path):
    cfg = make_synthetic_config(tmp_path / "inputs", learned_closed=True, reject_after=1)
    state = run_accuracy_guided_pruning(cfg, tmp_path / "run")
    first = state["decisions"][0]
    assert first["selection"]["no_feasible_search"]
    assert first["proposal_mask"] == {}
    assert not first["new_pruning"]
    assert first["selector"]["no_op_reason"] == "no_feasible_search"
    assert state["stages"][1]["kind"] == "recovery"
    assert state["global_epochs_completed"] == 8


def test_real_calibration_counts_overhead_outside_training_budget(tmp_path):
    cfg = make_synthetic_config(tmp_path / "inputs", learned_closed=True)
    cfg.accuracy_guided.guard.train_bn_calibration_batches = 1
    state = run_accuracy_guided_pruning(cfg, tmp_path / "run")
    assert state["status"] == "completed"
    assert state["ledger"]["consumed_training_examples"] == 64
    assert state["orchestration_overhead"]["calibration_forward_examples"] == 8
    assert all(d["calibration_overhead"]["batches"] == 1 for d in state["decisions"])
    assert all(d["calibration_overhead"]["forward_examples"] == 4 for d in state["decisions"])
    assert all(d["transfer_equivalence"]["status"].startswith("passed") for d in state["decisions"])


def test_partial_interruption_persists_real_steps_and_examples(tmp_path, monkeypatch):
    from net_complexity.training import interruption
    from net_complexity.training.interruption import TrainingInterrupted
    cfg = make_synthetic_config(tmp_path / "inputs")
    def stop_after_first_batch(**context):
        if context.get("batches") == 1:
            raise TrainingInterrupted(epoch=context.get("epoch", 0), batches=1)
    monkeypatch.setattr(interruption, "check_stop", stop_after_first_batch)
    with pytest.raises(TrainingInterrupted):
        run_accuracy_guided_pruning(cfg, tmp_path / "run")
    state = json.loads((tmp_path / "run/protocol_state.json").read_text())
    assert state["status"] == "interrupted"
    assert state["ledger"] == dict(global_training_epoch=0, search_epochs_consumed=0,
                                   optimizer_updates=1, consumed_training_examples=4)
    assert state["exact_whole_plan_resume_available"] is False


def test_technical_transfer_error_is_not_hidden_by_calibration(tmp_path, monkeypatch):
    from net_complexity.training import accuracy_guided_pruning as runtime
    cfg = make_synthetic_config(tmp_path / "inputs", learned_closed=True)
    cfg.accuracy_guided.guard.train_bn_calibration_batches = 1
    original = runtime.committed_equivalence
    def broken_transfer(carrier, physical, *args, **kwargs):
        with torch.no_grad():
            physical.backbone.fc.bias.add_(10)
        return original(carrier, physical, *args, **kwargs)
    monkeypatch.setattr(runtime, "committed_equivalence", broken_transfer)
    with pytest.raises(AssertionError):
        runtime.run_accuracy_guided_pruning(cfg, tmp_path / "run")
    state = json.loads((tmp_path / "run/protocol_state.json").read_text())
    assert state["status"] == "failed"
    assert state["ledger"]["global_training_epoch"] == 3
    assert state["ledger"]["optimizer_updates"] == 6
    assert all(s["kind"] == "search" for s in state["stages"])
    assert state["decisions"] == []


def test_exact_whole_plan_resume_refuses_ambiguous_restart(tmp_path):
    cfg = make_synthetic_config(tmp_path / "inputs")
    with pytest.raises(ValueError, match="Exact whole-plan resume"):
        run_accuracy_guided_pruning(cfg, tmp_path / "run", resume_from=tmp_path / "unknown")
    assert not (tmp_path / "run").exists()


def test_diagnostic_preserves_rng_batchnorm_and_source_modes(tmp_path):
    from hydra.utils import instantiate
    from net_complexity.training.pruning_audit import build_structural
    cfg = make_synthetic_config(tmp_path / "inputs")
    model = instantiate(cfg.model).train()
    physical = build_structural(cfg, model, {}).train()
    data = instantiate(cfg.dataloaders)
    hashes = [state_hash(m.state_dict()) for m in (model, physical)]
    before_torch, before_python, before_numpy = torch.get_rng_state(), random.getstate(), np.random.get_state()
    loader_state = data.valid_dataloader.generator.get_state()
    result = compare_predictors(physical, model, data.valid_dataloader)
    assert hashes == [state_hash(m.state_dict()) for m in (model, physical)]
    assert model.training and physical.training
    assert torch.equal(before_torch, torch.get_rng_state())
    assert random.getstate() == before_python
    assert np.array_equal(before_numpy[1], np.random.get_state()[1])
    assert torch.equal(loader_state, data.valid_dataloader.generator.get_state())
    assert result["overhead"]["forward_examples"] == 15
