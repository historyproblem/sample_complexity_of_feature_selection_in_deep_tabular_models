"""Authorized12-factorial search budget, derived gains and preserved legacy guards."""
from copy import deepcopy
from itertools import product
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stochastic_st_followup as queue
import run_stochastic_st_full4 as full4
import run_stochastic_st_timing12 as timing
import stochastic_pruning_test
from net_complexity.training.one_shot_pruning_config import validate_config


def test_cartesian_grid_defaults_to720search_epochs_and_changes_only_authorized_fields(tmp_path):
    plan = timing.main(["--dry-run", "--output", str(tmp_path / "absent"),
                        "--reference-archive", str(tmp_path / "missing.tar.gz")])
    assert (plan["stage"], plan["models"], plan["new_searches"], plan["new_training_epochs"]) == ("search", 12, 12, 720)
    assert plan["recovery_branches_per_model"] == [] and plan["test_after_each_model"] is False
    assert plan["wait_for_original_search_process"] is False and plan["bounded_epoch_checkpoints"] is True
    assert all((r["new_search_epochs"], r["recovery_epochs"]) == (60, 0) for r in plan["models_in_order"])
    study = queue.load_study(timing.DEFAULT_CONFIG)
    combinations = {(r["warmup"], r["lambda_init"], r["soft_drop"]) for r in study["searches"]}
    assert combinations == set(product((10, 15), (.001, .01), (.005, .01, .015)))
    previous = queue.load_study(full4.DEFAULT_CONFIG)
    assert study["reference_sha256"] == previous["reference_sha256"]
    assert study["initializer_model_state_hash"] == previous["initializer_model_state_hash"]
    assert study["split_indices_hash"] == previous["split_indices_hash"]
    args = SimpleNamespace(dense_source=tmp_path / "reference", device=None)
    baseline = OmegaConf.to_container(queue.configured(previous, previous["searches"][1], args), resolve=True)
    factors = {(10, .001): 2.51188643150958, (10, .01): 1.99526231496888,
               (15, .001): 6.30957344480193, (15, .01): 3.98107170553497}
    for row, cell in zip(study["searches"], plan["models_in_order"]):
        cfg = queue.configured(study, row, args)
        assert validate_config(cfg) == 150
        expected = deepcopy(baseline)
        expected["one_shot"]["lambda_timing_ablation"] = "warmup_init_grid_v1"
        expected["model"]["lambda_coef"] = row["lambda_init"]
        expected["training_arguments"]["adaptive_lambda"].update(alpha_init=row["lambda_init"],
            initial_search_warmup=row["warmup"], soft_drop=row["soft_drop"], hard_drop=.03)
        assert OmegaConf.to_container(cfg, resolve=True) == expected
        assert math.isclose(cell["lambda_update_factor"], factors[row["warmup"], row["lambda_init"]])
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("stage", ["run", "test"])
def test_night_queue_rejects_recovery_and_test_before_reference_import(tmp_path, monkeypatch, stage):
    monkeypatch.setattr(full4, "prepare_reference", lambda *a: pytest.fail("reference import before stage rejection"))
    with pytest.raises(SystemExit) as error:
        timing.main(["--stage", stage, "--output", str(tmp_path / "absent")])
    assert error.value.code == 2 and not (tmp_path / "absent").exists()


@pytest.mark.parametrize("warmup,initial", [(10, .01), (15, .001), (15, .01)])
def test_legacy_recipe_still_rejects_unmarked_warmup_or_init_changes(warmup, initial):
    study = queue.load_study(full4.DEFAULT_CONFIG)
    cfg = queue.configured(study, study["searches"][0], SimpleNamespace(dense_source=None, device=None))
    cfg.model.lambda_coef = cfg.training_arguments.adaptive_lambda.alpha_init = initial
    cfg.training_arguments.adaptive_lambda.initial_search_warmup = warmup
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize("change", ["wrong_initial", "wrong_warmup", "duplicate_combination", "wrong_hard"])
def test_manifest_rejects_an_unapproved_or_incomplete_grid(tmp_path, change):
    study = queue.load_study(timing.DEFAULT_CONFIG)
    if change == "wrong_initial": study["searches"][0]["lambda_init"] = .1
    elif change == "wrong_warmup": study["searches"][0]["warmup"] = 20
    elif change == "wrong_hard": study["searches"][0]["hard_drop"] = .04
    else:
        study["searches"][0].update({k: v for k, v in study["searches"][1].items() if k != "id"})
    path = tmp_path / "bad.yaml"; path.write_text(yaml.safe_dump(study))
    with pytest.raises(ValueError): queue.load_study(path)


@pytest.mark.parametrize("change", ["lambda_mismatch", "cap", "floor", "entropy", "marker"])
def test_timing_variant_keeps_other_invariants(change):
    study = queue.load_study(timing.DEFAULT_CONFIG)
    cfg = queue.configured(study, study["searches"][0], SimpleNamespace(dense_source=None, device=None))
    if change == "lambda_mismatch": cfg.model.lambda_coef = .01
    elif change == "cap": cfg.training_arguments.adaptive_lambda.alpha_max = 81
    elif change == "floor": cfg.accuracy_guided.eligibility.min_keep_ratio = .001
    elif change == "entropy": cfg.model.entropy_regularization_coef = .1
    else: cfg.one_shot.lambda_timing_ablation = "unrecognized"
    with pytest.raises(ValueError): validate_config(cfg)


def test_twelve_searches_stop_without_recovery_or_test_and_log_the_factors(tmp_path, monkeypatch, capsys):
    import net_complexity.training.one_shot_pruning as runtime
    from test_stochastic_st_followup import fake_cells
    study = queue.load_study(timing.DEFAULT_CONFIG)
    output = tmp_path / "queue"; cells = fake_cells(study, output)
    monkeypatch.setattr(full4, "prepare_reference", lambda *a: None)
    monkeypatch.setattr(queue, "preflight", lambda *a: (output, cells))
    monkeypatch.setattr(queue, "wait_for_original_searches", lambda: pytest.fail("night queue waited for an old search"))
    monkeypatch.setattr(stochastic_pruning_test, "DeploymentTests", lambda *a, **k: pytest.fail("search-only constructed a tester"))
    def review(source, config, study, row, **kwargs):
        return {"physical_parameters": 123, "selected_epoch": 31, "validation_accuracy": .9, "fallback_used": False}
    monkeypatch.setattr(queue, "review_source", review)
    calls = []
    def run(config, path, **kwargs):
        assert kwargs == {"search_only": True, "keep_epoch_checkpoints": False, "bounded_checkpoints": True}
        row = next(c["row"] for c in cells if c["source"] == path)
        assert config.model.lambda_coef == config.training_arguments.adaptive_lambda.alpha_init == row["lambda_init"]
        assert config.training_arguments.adaptive_lambda.initial_search_warmup == row["warmup"]
        calls.append(row["id"])
        path.mkdir(parents=True); queue.write_json(path / "selection.json", {"pruning_mask": {}})
    monkeypatch.setattr(runtime, "run_one_shot_pruning", run)
    state = timing.main(["--output", str(output)])
    assert calls == [r["id"] for r in study["searches"]]
    assert state["status"] == "search_completed" and state["new_training_epochs_allocated"] == 720
    assert state["recovery_epochs_executed"] == 0 and state["test_evaluated"] is False
    assert len(state["completed_searches"]) == 12
    summaries = queue.read_json(output / "search_summary.json")["searches"]
    assert len(summaries) == 12 and all(r["remaining_internal_channels"] == 7552 for r in summaries)
    assert [r["lambda_init"] for r in summaries] == [r["lambda_init"] for r in study["searches"]]
    assert [r["warmup"] for r in summaries] == [r["warmup"] for r in study["searches"]]
    assert capsys.readouterr().out.count("[timing]") == 12
    assert not (output / "test_evaluation").exists() and all(not c["output"].exists() for c in cells)
