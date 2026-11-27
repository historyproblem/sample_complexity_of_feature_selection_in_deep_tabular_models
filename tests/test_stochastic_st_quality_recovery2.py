"""Two pinned quality-search sources recover without starting any new search."""
from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stochastic_st_followup as queue
import stochastic_pruning_test
from test_stochastic_st_followup import arguments, fake_cells, make_source

CONFIG = ROOT / "configs/experiment/pruning_v3/stochastic_st_quality17m_recovery2.yaml"
ORIGINAL = ROOT / "configs/experiment/pruning_v3/stochastic_st_quality17m3.yaml"


def test_two_pinned_recoveries_preserve_first_two_search_recipes_and_budget(tmp_path):
    study = queue.load_study(CONFIG)
    original = queue.load_study(ORIGINAL)
    plan = queue.main(["--config", str(CONFIG), "--stage", "run", "--dry-run", "--output", str(tmp_path / "absent")])
    assert (plan["models"], plan["existing_search_recoveries"], plan["new_searches"]) == (2, 2, 0)
    assert plan["new_training_epochs"] == 180 and plan["per_model_total_epochs"] == 150
    assert plan["test_after_each_model"] and plan["bounded_epoch_checkpoints"]
    assert plan["wait_for_original_search_process"] is False
    assert [(r["new_search_epochs"], r["recovery_epochs"]) for r in plan["models_in_order"]] == [(0, 90)] * 2
    assert [row["id"] for row in study["recoveries"]] == [row["id"] for row in original["searches"][:2]]
    assert [(r["expected_parameters"], r["expected_remaining_channels"], r["selected_epoch"], r["expected_validation_accuracy"])
            for r in study["recoveries"]] == [(18344554, 6566, 33, .9158), (17462511, 6462, 34, .9146)]
    for row, source_row in zip(study["recoveries"], original["searches"][:2]):
        assert row["source"] == original["output"] + "/" + row["id"] + "/internal/search"
        recovery_cfg = queue.configured(study, row, arguments(None))
        source_cfg = queue.configured(original, source_row, arguments(None))
        assert OmegaConf.to_container(recovery_cfg, resolve=True) == OmegaConf.to_container(source_cfg, resolve=True)
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("count", [0, 1, 3, 4])
def test_recovery_manifest_rejects_other_counts(tmp_path, count):
    study = queue.load_study(CONFIG)
    study["recoveries"] = [{**study["recoveries"][0], "id": f"run_{i}"} for i in range(count)]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(study))
    with pytest.raises(ValueError, match="Unexpected number of models"):
        queue.load_study(path)


@pytest.mark.parametrize("completed", [False, True])
def test_recoveries_never_search_or_wait_and_test_each_model(tmp_path, monkeypatch, completed):
    import net_complexity.training.one_shot_pruning as runtime
    study = queue.load_study(CONFIG)
    args = arguments(tmp_path / "output")
    args.skip_wait = False
    cells = fake_cells(study, args.output, completed)
    monkeypatch.setattr(queue, "preflight", lambda *a: (args.output, cells))
    monkeypatch.setattr(queue, "wait_for_original_searches", lambda: pytest.fail("Recovery waited for an old queue"))
    monkeypatch.setattr(queue, "review_source", lambda source, *a, **k: next(c["review"] for c in cells if c["source"] == source))
    monkeypatch.setattr(queue, "check_completed", lambda *a: {"status": "completed"})
    events = []
    class Reporter:
        def __init__(self, output, ids, device, branches):
            assert list(ids) == ["tighter_hard", "earlier_pressure"] and branches == ("inherited",)
        def existing(self, path): events.append("existing_test")
        def callback(self, config): return lambda *a: events.append("test")
        def finish(self): return {"status": "completed"}
    monkeypatch.setattr(stochastic_pruning_test, "DeploymentTests", Reporter)
    def run(config, output, **kwargs):
        assert "search_only" not in kwargs and "reuse_search_from" in kwargs
        assert kwargs["bounded_checkpoints"] is True and kwargs["keep_epoch_checkpoints"] is False
        events.append("recovery90")
        kwargs["frozen_deployment_callback"](None, {}, output)
    monkeypatch.setattr(runtime, "run_one_shot_pruning", run)
    state = queue.execute(study, args)
    assert events == (["existing_test"] * 2 if completed else ["recovery90", "test"] * 2)
    assert state["new_training_epochs_allocated"] == 180 and state["planned_models"] == 2
    assert state["completed_models"] == ["tighter_hard", "earlier_pressure"]


def test_missing_second_checkpoint_blocks_before_inputs_or_training(tmp_path, monkeypatch):
    import net_complexity.training.one_shot_pruning as runtime
    study = queue.load_study(CONFIG)
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    for filename in queue.SOURCE_FILES:
        p = tmp_path / study["recoveries"][0]["source"] / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"present")
    monkeypatch.setattr(queue, "validate_inputs", lambda *a: pytest.fail("Inputs read before missing-source check"))
    monkeypatch.setattr(runtime, "run_one_shot_pruning", lambda *a, **k: pytest.fail("Training started with missing source"))
    with pytest.raises(FileNotFoundError, match="earlier_pressure"):
        queue.execute(study, arguments(tmp_path / "output"))
    assert not (tmp_path / "output").exists()


def test_search_stage_is_forbidden(tmp_path, monkeypatch):
    study = queue.load_study(CONFIG)
    monkeypatch.setattr(queue, "preflight", lambda *a: pytest.fail("Search stage reached preflight"))
    with pytest.raises(ValueError, match="Search-only requires"):
        queue.execute(study, arguments(tmp_path / "output", stage="search"))


def test_source_reader_checks_all_logged_identity_pins(tmp_path):
    study = queue.load_study(CONFIG)
    row = {**study["recoveries"][0], "expected_parameters": 123,
           "expected_remaining_channels": 7552, "expected_validation_accuracy": .9}
    cfg = queue.configured(study, row, arguments(None))
    source = tmp_path / "source"
    make_source(source, cfg, study, row)
    OmegaConf.save(cfg, source / "resolved_config.yaml")
    good = queue.review_source(source, cfg, study, row, from_old_queue=False)
    assert good["physical_parameters"] == 123 and good["selected_epoch"] == 50
    for key, value in [("expected_parameters", 124), ("expected_remaining_channels", 7551),
                       ("expected_validation_accuracy", .5), ("selected_epoch", 51),
                       ("selected_checkpoint_sha256_prefix", "0" * 16), ("expected_fallback", True)]:
        with pytest.raises(ValueError, match="Measured"):
            queue.review_source(source, cfg, study, {**row, key: value}, from_old_queue=False)
