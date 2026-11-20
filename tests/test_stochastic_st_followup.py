"""Single inherited handoff, exact five-model queue and immutable source preflight."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stochastic_st_followup as followup
import stochastic_pruning_test
from net_complexity.models.pruning_budget import gates
from net_complexity.training.one_shot_pruning_config import compose_config, validate_config, resolved_branch_plan
from test_pruning_ablations import fixture


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def arguments(output, stage="run"):
    return SimpleNamespace(stage=stage, output=output, dense_source=None, device="cpu",
                           dry_run=False, skip_wait=True, keep_epoch_checkpoints=False)


def test_manifest_budget_and_exact_recipe_diff(tmp_path):
    study = followup.load_study(followup.DEFAULT_CONFIG)
    plan = followup.main(["--dry-run", "--output", str(tmp_path / "absent")])
    assert plan["models"] == 5 and plan["existing_search_recoveries"] == 3 and plan["new_searches"] == 2
    assert plan["new_training_epochs"] == 570 and plan["per_model_total_epochs"] == 150
    assert [r["id"] for r in plan["models_in_order"]] == ["dense_like", "pruned", "third_search", "gap_003_006pp", "gap_005_010pp"]
    assert [(r["soft_drop"], r["hard_drop"]) for r in study["searches"]] == [(.0003, .0006), (.0005, .001)]
    for row in study["recoveries"] + study["searches"]:
        cfg = followup.configured(study, row, arguments(None))
        control = compose_config("experiment/pruning_v3/ablations_internal_stochastic_st", [
            f"training_arguments.adaptive_lambda.soft_drop={row['soft_drop']}",
            f"training_arguments.adaptive_lambda.hard_drop={row['hard_drop']}",
        ], dense_source=ROOT / study["reference_output"])
        control.device = "cpu"
        expected = OmegaConf.to_container(control, resolve=True)
        expected["one_shot"]["branches"] = ["inherited"]
        assert OmegaConf.to_container(cfg, resolve=True) == expected
        assert validate_config(cfg) == 150 and [b["id"] for b in resolved_branch_plan(cfg)] == ["inherited"]


def make_source(path, config, study, row):
    """Minimal snapshot for reader identity checks; not a trainable ResNet fixture."""
    from test_pruning_ablations import fake_finished_search
    from net_complexity.training.pruning_measurement import state_hash, mask_hash
    saved = deepcopy(config); saved.one_shot.branches = ["inherited", "scratch", "random"]
    fake_finished_search(path, saved)
    weights = {"weight": torch.tensor([1.])}
    model_hash = state_hash(weights)
    torch.save({"epoch": 50, "model_state_hash": model_hash, "model_state_dict": weights}, path / "selected_checkpoint.pt")
    selection = followup.read_json(path / "selection.json")
    selection.update(pruning_mask={}, mask_hash=mask_hash({}), selected_model_state_hash=model_hash,
                     selected_checkpoint_hash=followup.digest(path / "selected_checkpoint.pt"))
    export = followup.read_json(path / "export_only/diagnostics.json")
    export.update(mask_hash=selection["mask_hash"], selected_model_state_hash=model_hash,
                  selected_checkpoint_hash=selection["selected_checkpoint_hash"])
    state = followup.read_json(path / "one_shot_state.json")
    state.update(selection=selection, provenance={"inputs": {"sha256": study["reference_sha256"],
                  "initializer_model_state_hash": study["initializer_model_state_hash"]},
                  "split_indices_hash": study["split_indices_hash"]})
    for name, value in (("selection.json", selection), ("one_shot_state.json", state), ("export_only/diagnostics.json", export)):
        followup.write_json(path / name, value)
    if "expected_parameters" in row:
        row.update(expected_parameters=123, selected_epoch=50,
                   selected_checkpoint_sha256_prefix=selection["selected_checkpoint_hash"][:16])
    return selection


def test_source_reader_pins_mask_and_actual_checkpoint_tensors(tmp_path):
    study = followup.load_study(followup.DEFAULT_CONFIG)
    row = study["recoveries"][0]
    config = followup.configured(study, row, arguments(tmp_path / "output"))
    source = tmp_path / "source"
    selection = make_source(source, config, study, row)
    review = followup.review_source(source, config, study, row, from_old_queue=True)
    assert review["selected_epoch"] == 50 and review["physical_parameters"] == 123
    wrong_pin = {**row, "expected_parameters": 124}
    with pytest.raises(ValueError, match="Measured search identity"):
        followup.review_source(source, config, study, wrong_pin, from_old_queue=True)
    # Even consistently updated file checksums cannot hide different raw tensors.
    torch.save({"epoch": 50, "model_state_hash": selection["selected_model_state_hash"],
                "model_state_dict": {"weight": torch.tensor([2.])}}, source / "selected_checkpoint.pt")
    selection["selected_checkpoint_hash"] = followup.digest(source / "selected_checkpoint.pt")
    row["selected_checkpoint_sha256_prefix"] = selection["selected_checkpoint_hash"][:16]
    state = followup.read_json(source / "one_shot_state.json"); state["selection"] = selection
    export = followup.read_json(source / "export_only/diagnostics.json")
    export["selected_checkpoint_hash"] = selection["selected_checkpoint_hash"]
    for name, value in (("selection.json", selection), ("one_shot_state.json", state), ("export_only/diagnostics.json", export)):
        followup.write_json(source / name, value)
    with pytest.raises(ValueError, match="tensors/epoch differ"):
        followup.review_source(source, config, study, row, from_old_queue=True)


def test_bad_third_source_blocks_all_new_training(tmp_path, monkeypatch):
    import net_complexity.training.one_shot_pruning as runtime
    study = followup.load_study(followup.DEFAULT_CONFIG)
    monkeypatch.setattr(followup, "ROOT", tmp_path)
    args = arguments(tmp_path / "new_outputs")
    for row in study["recoveries"]:
        cfg = followup.configured(study, row, args)
        source = tmp_path / row["source"]
        make_source(source, cfg, study, row)
    third = tmp_path / study["recoveries"][2]["source"]
    # Selection is complete, but its source split differs from the pinned study.
    value = followup.read_json(third / "one_shot_state.json")
    value["provenance"]["split_indices_hash"] = "f" * 64
    followup.write_json(third / "one_shot_state.json", value)
    monkeypatch.setattr(followup, "validate_inputs", lambda cfg: {"sha256": study["reference_sha256"],
                         "initializer_model_state_hash": study["initializer_model_state_hash"]})
    monkeypatch.setattr(runtime, "run_one_shot_pruning", lambda *a, **kw: pytest.fail("training started before all sources checked"))
    with pytest.raises(ValueError, match="reference/initializer/split changed"):
        followup.execute(study, args)
    assert not args.output.exists()
    assert not (tmp_path / "absent").exists()


def test_all_missing_original_sources_block_before_inputs_or_training(tmp_path, monkeypatch):
    monkeypatch.setattr(followup, "ROOT", tmp_path)
    monkeypatch.setattr(followup, "validate_inputs", lambda *a: pytest.fail("inputs read before missing sources reported"))
    study = followup.load_study(followup.DEFAULT_CONFIG)
    with pytest.raises(FileNotFoundError) as exc:
        followup.preflight(study, arguments(tmp_path / "new"))
    assert all(row["id"] in str(exc.value) for row in study["recoveries"])
    assert "selected_checkpoint.pt" in str(exc.value) and not (tmp_path / "new").exists()


def test_actual_three_branch_search_can_handoff_only_inherited(tmp_path, monkeypatch):
    import net_complexity.training.one_shot_pruning as runtime
    import net_complexity.training.pruning_ablations as ablations
    cfg = fixture(tmp_path / "inputs")
    cfg.model.backbone.resnet_block.train_gate_mode = "gumbel_hard"
    source = tmp_path / "original_search"
    initial = runtime.run_one_shot_pruning(cfg, source, search_only=True)
    before = {p.relative_to(source): followup.digest(p) for p in source.rglob("*") if p.is_file()}
    cfg = deepcopy(cfg); cfg.one_shot.branches = ["inherited"]
    monkeypatch.setattr(ablations, "make_matched_random_mask", lambda *a, **kw: pytest.fail("unrequested random ablation created"))
    result = runtime.run_one_shot_pruning(cfg, tmp_path / "main", reuse_search_from=source,
                                         keep_epoch_checkpoints=False)
    assert set(result["branches"]) == {"inherited"}
    assert result["compute_ledger"]["actual_training_epochs_executed"] == 2
    branch = result["branches"]["inherited"]
    assert branch["ledger"]["search_epochs_consumed"] == 3 and branch["ledger"]["global_training_epoch"] == 5
    assert branch["source_learned_mask_hash"] == initial["selection"]["mask_hash"]
    assert before == {p.relative_to(source): followup.digest(p) for p in source.rglob("*") if p.is_file()}


def fake_cells(study, output, completed=False):
    result = []
    for row in study["recoveries"] + study["searches"]:
        existing = "source" in row
        result.append({"row": row, "existing": existing,
                       "source": output / "sources" / row["id"],
                       "output": output / row["id"] / "internal/recovery",
                       "config": followup.configured(study, row, arguments(output)),
                       "review": {"physical_parameters": row.get("expected_parameters", 123), "selected_epoch": 50,
                                  "validation_accuracy": .9, "fallback_used": False} if existing or completed else None,
                       "completed": {"status": "completed"} if completed else None})
    return result


@pytest.mark.parametrize("completed", [False, True])
def test_exact_queue_no_source_search_restarts_and_test_after_each(tmp_path, monkeypatch, completed):
    import net_complexity.training.one_shot_pruning as runtime
    study = followup.load_study(followup.DEFAULT_CONFIG)
    cells = fake_cells(study, tmp_path / "queue", completed)
    args = arguments(tmp_path / "queue")
    monkeypatch.setattr(followup, "preflight", lambda *a: (args.output, cells))
    monkeypatch.setattr(followup, "review_source", lambda source, cfg, study, row, **kw:
                        {"physical_parameters": row.get("expected_parameters", 123), "selected_epoch": 50,
                         "validation_accuracy": .9, "fallback_used": False})
    monkeypatch.setattr(followup, "check_completed", lambda *a: {"status": "completed"})
    events = []
    class Reporter:
        def __init__(self, output, ids, device, branches):
            assert len(ids) == 5 and branches == ("inherited",)
        def callback(self, config):
            return lambda model, branch, root: events.append(("test", root.parent.parent.name))
        def existing(self, root):
            events.append(("reuse_test", root.parent.parent.name))
        def finish(self):
            return {"status": "completed"}
    monkeypatch.setattr(stochastic_pruning_test, "DeploymentTests", Reporter)
    def run(config, output, **kwargs):
        assert list(config.one_shot.branches) == ["inherited"]
        if kwargs.get("search_only"):
            index = next(cell["row"]["id"] for cell in cells if cell["source"] == output)
            events.append(("search60", index))
        else:
            events.append(("recover90", output.parent.parent.name))
            assert kwargs["keep_epoch_checkpoints"] is False
            kwargs["frozen_deployment_callback"](None, {}, output)
    monkeypatch.setattr(runtime, "run_one_shot_pruning", run)
    result = followup.execute(study, args)
    assert result["status"] == "training_completed" and len(result["completed_models"]) == 5
    if completed:
        assert [e[0] for e in events] == ["reuse_test"] * 5
    else:
        assert [e[0] for e in events] == ["recover90", "test"] * 3 + ["search60", "recover90", "test"] * 2
        assert sum(60 if e[0] == "search60" else 90 if e[0] == "recover90" else 0 for e in events) == 570
    assert not (args.output / "followup.lock").exists()


def test_standalone_test_does_not_require_original_sources(tmp_path, monkeypatch):
    study = followup.load_study(followup.DEFAULT_CONFIG)
    args = arguments(tmp_path, "test")
    for row in study["recoveries"] + study["searches"]:
        path = tmp_path / row["id"] / "internal/recovery"
        path.mkdir(parents=True)
        (path / "one_shot_state.json").write_text('{"status":"completed"}')
    monkeypatch.setattr(followup, "preflight", lambda *a: pytest.fail("inference requested original searches"))
    calls = []
    class Reporter:
        def __init__(self, *a, **kw): pass
        def existing(self, root): calls.append(root)
        def finish(self): return {"status": "completed"}
    monkeypatch.setattr(stochastic_pruning_test, "DeploymentTests", Reporter)
    assert followup.execute(study, args)["status"] == "completed" and len(calls) == 5


def test_waits_for_old_search_queue_without_starting_training(monkeypatch):
    answers = iter([[12277], []]); sleeps = []
    monkeypatch.setattr(followup, "running_original_searches", lambda: next(answers))
    monkeypatch.setattr(followup.time, "sleep", lambda seconds: sleeps.append(seconds))
    followup.wait_for_original_searches()
    assert sleeps == [15]


def test_original_search_matcher_handles_paths_equals_flags_and_shell_wrappers():
    assert followup.matches_original_search(["python", "-u", "scripts/launch_pruning_ablations.py", "--config",
                                            "configs/experiment/pruning_v3/ablations_stochastic_st4.yaml", "--stage", "search"])
    assert followup.matches_original_search(["python", "/repo/scripts/launch_pruning_ablations.py",
                                            "--stage=search", "--config=/repo/configs/experiment/pruning_v3/ablations_stochastic_st4.yaml"])
    assert not followup.matches_original_search(["bash", "-c", "python scripts/launch_pruning_ablations.py --config ablations_stochastic_st4.yaml --stage search"])
    assert not followup.matches_original_search(["python", "scripts/run_stochastic_st_followup.py"])
