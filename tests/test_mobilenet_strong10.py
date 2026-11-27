"""Ten full45-epoch MobileNet probes with a strict method-parameter whitelist."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from net_complexity.models.channel_pruning import apply_channel_mask, build_structurally_pruned_model_from_config
from net_complexity.models.feature_selection import get_gumbel_modules
from net_complexity.training import cyclic_channel_pruning as cyclic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_mobilenet_strong10 as grid
from test_mobilenet_open98_ablation import completed_arm


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def configuration(row):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name=grid.BASE_CONFIG, overrides=grid.run_overrides(row))


def test_ten_actual_configs_preserve_model_recipe_and_resolve_each_auto_horizon():
    protocol = grid.validate_configs(ROOT)
    assert protocol["models"] == 10 and protocol["total_training_epochs"] == 450
    assert protocol["removal_threshold"] == .9 and protocol["search_forward_threshold"] == .5
    rows = protocol["runs"]
    assert [(r["lambda_init"], r["mask_init"]) for r in rows[:6]] == [
        (value, mode) for value in (.01, .1, 1.) for mode in ("paper", "paper_resnet50")]
    assert [(r["search_epochs"], r["recovery_epochs"]) for r in rows] == [(30, 15)] * 8 + [(35, 10), (25, 20)]
    for row in rows:
        assert row["search_epochs"] + row["recovery_epochs"] == 45
        assert row["auto_raw_log_step"] == pytest.approx(math.log(10 / row["lambda_init"]) / (row["search_epochs"] - row["warmup_epochs"]))
        assert row["lambda_update_factor"] == pytest.approx(math.exp(row["auto_raw_log_step"]))
        cfg = configuration(row)
        assert cfg.model.backbone.block.train_gate_mode == "gumbel_hard"
        assert cfg.model.backbone.block.eval_gate_mode == "deterministic_hard"
        assert cfg.optimizer.lr == .001 and cfg.optimizer.pretrained_lr_scale == .1
        assert cfg.optimizer.weight_decay == .0005 and cfg.model.backbone.pretrained_weights == "IMAGENET1K_V2"


@pytest.mark.parametrize("corruption", ["count", "duplicate", "optimizer", "fixed_step", "budget"])
def test_manifest_rejects_unapproved_or_invalid_changes(tmp_path, corruption):
    (tmp_path / "configs").mkdir()
    spec = grid.load_manifest(ROOT)
    if corruption == "count": spec["runs"].pop()
    elif corruption == "duplicate": spec["runs"][-1] = deepcopy(spec["runs"][0])
    elif corruption == "optimizer": spec["runs"][0]["optimizer_lr"] = .01
    elif corruption == "fixed_step": spec["runs"][0]["log_step"] = .5
    else: spec["runs"][0]["recovery_epochs"] += 1
    (tmp_path / grid.MANIFEST).write_text(json.dumps(spec))
    with pytest.raises(ValueError): grid.load_manifest(tmp_path)


def test_config_diff_guard_detects_a_global_optimizer_change(monkeypatch):
    import hydra
    actual_compose = hydra.compose
    def altered(*args, **kwargs):
        cfg = actual_compose(*args, **kwargs)
        if kwargs.get("config_name") == grid.BASE_CONFIG: cfg.optimizer.lr *= 2
        return cfg
    monkeypatch.setattr(hydra, "compose", altered)
    with pytest.raises(ValueError, match="beyond the authorized"):
        grid.validate_configs(ROOT)


def test_dry_run_uses_only_standard_library_and_contains_ten_complete_commands(tmp_path):
    result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/run_mobilenet_strong10.py"), "--dry-run",
                             "--baseline-archive", str(tmp_path / "absent.tar.gz")], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "450 training epochs" in result.stdout
    assert result.stdout.count("src/net_complexity/cli/cyclic_train.py") == 10
    assert "cyclic_channel_pruning.gumbel_epochs=35" in result.stdout
    assert "cyclic_channel_pruning.final_epochs=20" in result.stdout
    assert "log_step_init=" not in result.stdout  # Inherited literal auto, never numeric overrides.


@pytest.mark.parametrize("index", [0, 8, 9])
def test_public_cycle_uses_requested_phases_and_inherits_selected_weights(tmp_path, monkeypatch, index):
    row = grid.load_manifest(ROOT)["runs"][index]
    cfg = configuration(row)
    cfg.model.backbone.pretrained_weights = None
    cfg.model.backbone.width_mult = .25
    cfg.model.backbone.num_classes = 3
    source = instantiate(cfg.model)
    with torch.no_grad():
        for gate in get_gumbel_modules(source).values(): gate.logits[:] = torch.tensor([0., 5.])
        name, gate = next((n, g) for n, g in get_gumbel_modules(source).items() if "mid_gumbel_layer" in n)
        gate.logits[0] = torch.tensor([5., 0.])
        source.backbone.classifier[1].weight.add_(.25)
    metrics = {"valid_accuracy": .65}
    for layer, gate in get_gumbel_modules(source).items():
        for i, probability in enumerate(gate.get_selection_probs().tolist()):
            metrics[f"valid_{layer}.channel_{i:03d}_zero_prob"] = 1 - probability
    held = {"version": 1, "runtime": {"last_epoch": 2, "log_lambda": -3.}}
    calls = []
    def train(stage_cfg, *, progress_context=None, model_initializer=None):
        calls.append(stage_cfg)
        path = tmp_path / ("search" if len(calls) == 1 else "recovery")
        (path / "checkpoints").mkdir(parents=True)
        if len(calls) == 1:
            assert stage_cfg.training_arguments.num_epochs == stage_cfg.scheduler.T_max == row["search_epochs"]
            assert stage_cfg.training_arguments.adaptive_lambda.log_step_init == "auto"
            assert model_initializer is None
            torch.save({"epoch": 2, "model_state_dict": source.state_dict(), "metrics": metrics,
                        "extra_state": {"adaptive_lambda_state": held}}, path / "checkpoints/best.pt")
            return {"run_dir": str(path), "best_epoch": 2, "num_epochs_executed": row["search_epochs"], "last_valid_metrics": {}}
        assert stage_cfg.training_arguments.num_epochs == stage_cfg.scheduler.T_max == row["recovery_epochs"]
        assert stage_cfg.model.lambda_coef == 0 and not stage_cfg.training_arguments.adaptive_lambda.enabled
        assert stage_cfg.model.criterion.label_smoothing == 0 and model_initializer is not None
        target = build_structurally_pruned_model_from_config(stage_cfg, stage_cfg.channel_pruning)
        model_initializer(target)
        torch.testing.assert_close(target.backbone.classifier[1].weight, source.backbone.classifier[1].weight, rtol=0, atol=0)
        mask = {name: [0]}
        apply_channel_mask(source, mask)
        source.eval(); target.eval()
        images, labels = torch.randn(2, 3, 32, 32), torch.tensor([1, 2])
        with torch.inference_mode():
            torch.testing.assert_close(source(images, labels).logits, target(images, labels).logits, rtol=1e-4, atol=1e-5)
        assert not get_gumbel_modules(target)
        torch.save({"epoch": 1, "model_state_dict": target.state_dict()}, path / "checkpoints/best.pt")
        return {"run_dir": str(path), "best_epoch": 1, "num_epochs_executed": row["recovery_epochs"], "test_metrics": {"test_accuracy": .7}}
    monkeypatch.setattr(cyclic, "run_training", train)
    result = cyclic.run_cyclic_channel_pruning_training(cfg, tmp_path / "cycle")
    assert len(calls) == 2 and result["training_cost"]["num_epochs_executed"] == 45
    assert result["final_disabled_channels"] == {name: [0]}
    assert result["search_selections"][0]["held_adaptive_lambda_state"] == held


def test_variable_phase_completion_and_final_physical_test_record(completed_arm):
    row = grid.load_manifest(ROOT)["runs"][8]
    stage = completed_arm
    path = stage.directory / "cyclic_summary.json"
    summary = json.loads(path.read_text())
    for result, count in [(summary["cycle_results"][0], 35), (summary["recovery_results"][0], 10)]:
        result["num_epochs_executed"] = count
        root = Path(result["run_dir"])
        (root / "history.csv").write_text("epoch,valid_accuracy\n" + "".join(f"{i},0.65\n" for i in range(1, count + 1)))
        (root / "checkpoints/last.pt").write_bytes(b"last-model")
    summary["final_result"] = summary["recovery_results"][0]
    path.write_text(json.dumps(summary))
    assert grid.validate_run(stage, row)
    record = grid.comparison_record(stage, row)
    assert record["test_accuracy"] == .7 and record["compact_parameters"] == 2_000_000
    assert record["total_epochs"] == 45 and record["search_epochs"] == 35
    with pytest.raises(RuntimeError): grid.previous.validate_stage(stage)


@pytest.mark.parametrize("scenario", ["success", "reuse_first", "reuse_all", "fail_first", "partial_last", "fail_reference", "inspect"])
def test_launcher_finishes_each_full_model_before_next_and_preserves_completed(tmp_path, monkeypatch, scenario):
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "scripts").mkdir()
    shutil.copyfile(ROOT / grid.MANIFEST, repo / grid.MANIFEST)
    spec = grid.load_manifest(repo)
    output = repo / spec["output_root"]
    stages = [grid.single.Stage("gumbel_hard", output / row["id"], [row["id"]], row["id"]) for row in spec["runs"]]
    for i, stage in enumerate(stages):
        if scenario == "reuse_all" or (scenario == "reuse_first" and i == 0):
            stage.directory.mkdir(parents=True); (stage.directory / "done").touch()
        if scenario == "partial_last" and i == 9:
            stage.directory.mkdir(parents=True); (stage.directory / "partial").touch()
    calls, preflights = [], []
    def validate(stage, row):
        if (stage.directory / "partial").exists(): raise RuntimeError("Partial fit; refusing repeat")
        return (stage.directory / "done").exists()
    def run(command, **kwargs):
        if command[1:2] == ["scripts/prepare_mobilenet_reference.py"]:
            calls.append("reference_import")
            if scenario == "fail_reference": raise subprocess.CalledProcessError(1, command)
            return
        calls.append(command[0])
        if scenario == "fail_first": raise subprocess.CalledProcessError(1, command)
        index = next(i for i, stage in enumerate(stages) if stage.command == command)
        if index:
            records = json.loads((output / "comparison.json").read_text())["results"]
            assert len(records) == index
        (stages[index].directory / "done").touch()
    monkeypatch.chdir(repo)
    monkeypatch.setattr(grid, "__file__", str(repo / "scripts/run_mobilenet_strong10.py"))
    monkeypatch.setattr(grid, "build_plan", lambda *a: stages)
    monkeypatch.setattr(grid, "validate_configs", lambda *a: {"models": 10, "total_training_epochs": 450})
    monkeypatch.setattr(grid, "validate_reference", lambda *a: {"identity": "same"})
    monkeypatch.setattr(grid, "validate_run", validate)
    monkeypatch.setattr(grid, "comparison_record", lambda stage, row: {"id": row["id"], "test_accuracy": .7})
    monkeypatch.setattr(grid.single, "preflight", lambda *a, **k: preflights.append(True))
    monkeypatch.setattr(grid.subprocess, "run", run)
    assert grid.main(["--inspect"] if scenario == "inspect" else []) == (1 if scenario in {"fail_first", "fail_reference", "partial_last"} else 0)
    expected = ["reference_import"]
    if scenario == "success": expected += [r["id"] for r in spec["runs"]]
    elif scenario == "reuse_first": expected += [r["id"] for r in spec["runs"][1:]]
    elif scenario == "fail_first": expected += [spec["runs"][0]["id"]]
    assert calls == expected
    if scenario in {"success", "reuse_first", "reuse_all"}:
        assert len(json.loads((output / "comparison.json").read_text())["results"]) == 10
        assert len((output / "comparison.csv").read_text().splitlines()) == 11
    if scenario in {"reuse_all", "partial_last", "fail_reference"}: assert not preflights
