"""Eight measured timing-grid sources: recovery only, pinned identity and720 epochs."""
from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stochastic_st_followup as queue
import run_stochastic_st_timing12 as timing
import stochastic_pruning_test
from test_stochastic_st_followup import arguments, fake_cells, make_source

CONFIG = ROOT / "configs/experiment/pruning_v3/stochastic_st_timing_recovery8.yaml"


def test_manifest_selects_eight_original_recipes_and_only720_recovery_epochs(tmp_path):
    study = queue.load_study(CONFIG)
    plan = queue.main(["--config", str(CONFIG), "--dry-run", "--output", str(tmp_path / "absent")])
    assert (plan["models"], plan["existing_search_recoveries"], plan["new_searches"], plan["new_training_epochs"]) == (8, 8, 0, 720)
    assert plan["test_after_each_model"] is True and plan["wait_for_original_search_process"] is False
    assert plan["bounded_epoch_checkpoints"] is True and plan["recovery_branches_per_model"] == ["inherited"]
    assert all((r["new_search_epochs"], r["recovery_epochs"]) == (0, 90) for r in plan["models_in_order"])
    assert [r["expected_parameters"] for r in study["recoveries"]] == [5110943, 5174199, 6415681, 7045869, 7398628, 7983520, 9680738, 11883222]
    assert sum(r["expected_fallback"] for r in study["recoveries"]) == 2
    original = queue.load_study(timing.DEFAULT_CONFIG)
    for row in study["recoveries"]:
        source_row = next(r for r in original["searches"] if r["id"] == row["id"])
        cfg = queue.configured(study, row, arguments(None))
        source_cfg = queue.configured(original, source_row, arguments(None))
        assert OmegaConf.to_container(cfg, resolve=True) == OmegaConf.to_container(source_cfg, resolve=True)
        assert row["source"] == original["output"] + "/" + row["id"] + "/internal/search"
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("completed", [False, True])
def test_eight_recoveries_reuse_sources_never_search_or_wait_and_test_after_each(tmp_path, monkeypatch, completed):
    import net_complexity.training.one_shot_pruning as runtime
    study = queue.load_study(CONFIG); args = arguments(tmp_path / "out"); args.skip_wait = False
    cells = fake_cells(study, args.output, completed)
    for cell in cells:
        cell["review"].update(selected_epoch=cell["row"]["selected_epoch"],
                             validation_accuracy=cell["row"]["expected_validation_accuracy"],
                             fallback_used=cell["row"]["expected_fallback"])
    monkeypatch.setattr(queue, "preflight", lambda *a: (args.output, cells))
    monkeypatch.setattr(queue, "wait_for_original_searches", lambda: pytest.fail("recovery waited for an old queue"))
    def review(source, config, study, row, *, from_old_queue):
        assert from_old_queue is False
        return next(cell["review"] for cell in cells if cell["source"] == source)
    monkeypatch.setattr(queue, "review_source", review)
    monkeypatch.setattr(queue, "check_completed", lambda *a: {"status": "completed"})
    events = []
    class Reporter:
        def __init__(self, output, ids, device, branches): assert len(ids) == 8 and branches == ("inherited",)
        def existing(self, path): events.append("existing_test")
        def callback(self, config): return lambda *a: events.append("test")
        def finish(self): return {"status": "completed"}
    monkeypatch.setattr(stochastic_pruning_test, "DeploymentTests", Reporter)
    def run(config, output, **kwargs):
        assert "search_only" not in kwargs
        cell = next(c for c in cells if c["output"] == output)
        assert kwargs["reuse_search_from"] == cell["source"]
        assert kwargs["bounded_checkpoints"] is True and kwargs["keep_epoch_checkpoints"] is False
        assert config.model.lambda_coef == cell["row"]["lambda_init"]
        assert config.training_arguments.adaptive_lambda.initial_search_warmup == cell["row"]["warmup"]
        events.append("recovery90"); kwargs["frozen_deployment_callback"](None, {}, output)
    monkeypatch.setattr(runtime, "run_one_shot_pruning", run)
    state = queue.execute(study, args)
    assert events == (["existing_test"] * 8 if completed else ["recovery90", "test"] * 8)
    assert state["planned_models"] == 8 and len(state["completed_models"]) == 8
    assert state["new_training_epochs_allocated"] == 720 and state["status"] == "training_completed"


def test_missing_eighth_checkpoint_blocks_all_training_before_input_loading(tmp_path, monkeypatch):
    import net_complexity.training.one_shot_pruning as runtime
    study = queue.load_study(CONFIG); monkeypatch.setattr(queue, "ROOT", tmp_path)
    for row in study["recoveries"][:7]:
        for name in queue.SOURCE_FILES:
            path = tmp_path / row["source"] / name
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"present")
    monkeypatch.setattr(queue, "validate_inputs", lambda *a: pytest.fail("inputs loaded before missing source reported"))
    monkeypatch.setattr(runtime, "run_one_shot_pruning", lambda *a, **k: pytest.fail("training started with missing source"))
    with pytest.raises(FileNotFoundError, match=study["recoveries"][7]["id"]):
        queue.execute(study, arguments(tmp_path / "new"))
    assert not (tmp_path / "new").exists()


def test_source_reader_accepts_explicit_fallback_and_pins_epoch_validation_channels(tmp_path):
    study = queue.load_study(CONFIG)
    row = {**study["recoveries"][4], "expected_parameters": 123, "expected_remaining_channels": 7552, "selected_epoch": 50}
    cfg = queue.configured(study, row, arguments(None)); source = tmp_path / "source"
    make_source(source, cfg, study, {"id": row["id"]})
    OmegaConf.save(cfg, source / "resolved_config.yaml")
    selection = queue.read_json(source / "selection.json")
    selection["policy"] = "best_validation_last_epochs_fallback"
    selection["trace"]["fallback_used"] = True
    state = queue.read_json(source / "one_shot_state.json"); state["selection"] = selection
    queue.write_json(source / "selection.json", selection); queue.write_json(source / "one_shot_state.json", state)
    diag = queue.read_json(source / "export_only/diagnostics.json")
    row["expected_validation_accuracy"] = diag["physical_validation_before_bn_calibration"]["accuracy"]
    result = queue.review_source(source, cfg, study, row, from_old_queue=False)
    assert result["fallback_used"] is True and result["selected_epoch"] == 50
    for key, value, message in [("expected_fallback", False, "channels/fallback"),
                                ("expected_validation_accuracy", .123, "validation"),
                                ("selected_epoch", 51, "identity"),
                                ("expected_remaining_channels", 7551, "channels/fallback")]:
        with pytest.raises(ValueError, match=message):
            queue.review_source(source, cfg, study, {**row, key: value}, from_old_queue=False)
