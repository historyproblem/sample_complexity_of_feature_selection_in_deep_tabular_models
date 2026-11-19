import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from net_complexity.models.channel_pruning import apply_channel_mask, build_structurally_pruned_model_from_config
from net_complexity.models.feature_selection import GumbelLayer, get_gumbel_modules
from net_complexity.training import cyclic_channel_pruning as cyclic


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("open98_workflow", ROOT / "scripts/run_mobilenet_open98_ablation.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def configuration(mode="gumbel"):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name=f"mobilenet_open98_{mode}")


def test_actual_configs_match_except_noise_and_resolve_auto_step():
    protocol = workflow.validate_configs(ROOT)
    assert (protocol["search_epochs"], protocol["recovery_epochs"]) == (30, 15)
    assert protocol["lambda_log_step"] == pytest.approx(math.log(10000) / 20)
    assert protocol["initial_p_open"] == pytest.approx(.9801596942659225)
    assert [stage.kind for stage in workflow.build_plan(ROOT)] == ["gumbel_hard", "ste_hard"]


def test_both_native_models_have_open98_internal_gates_and_identical_initial_weights():
    states = []
    for arm, mode in [("gumbel", "gumbel_hard"), ("ste", "ste_hard")]:
        cfg = configuration(arm)
        cfg.model.backbone.pretrained_weights = None
        torch.manual_seed(42)
        model = instantiate(cfg.model)
        gates = get_gumbel_modules(model)
        assert len(gates) == 26
        assert sum(gate.logits.shape[0] for gate in gates.values()) == 7896
        assert sum("mid_gumbel_layer" in name for name in gates) == 16
        assert sum(p.numel() for p in model.parameters()) == 2495864
        for gate in gates.values():
            assert gate.train_gate_mode == mode
            assert gate.eval_gate_mode == "deterministic_hard"
            assert .97 < float(gate.get_selection_probs().mean()) < .99
        states.append(model.state_dict())
    assert states[0].keys() == states[1].keys()
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)


def test_noise_changes_hard_samples_but_both_modes_keep_ste_gradients():
    samples = {}
    for mode in ["gumbel_hard", "ste_hard"]:
        torch.manual_seed(42)
        gate = GumbelLayer(96, train_gate_mode=mode, eval_gate_mode="deterministic_hard")
        gate.reset_parameters("paper_resnet50")
        gate.train()
        sample = gate(torch.ones(32, 96, 1, 1))
        assert bool(((sample == 0) | (sample == 1)).all())
        sample.sum().backward()
        assert gate.logits.grad is not None and bool((gate.logits.grad != 0).any())
        samples[mode] = sample.detach()
    assert bool((samples["ste_hard"] == 1).all())
    assert bool((samples["gumbel_hard"] == 0).any())
    assert len(torch.unique(samples["gumbel_hard"].flatten(1), dim=0)) > 1


def channel_metrics(model):
    result = {"valid_accuracy": .65}
    for name, gate in get_gumbel_modules(model).items():
        for index, value in enumerate(gate.get_selection_probs().tolist()):
            result[f"valid_{name}.channel_{index:03d}_zero_prob"] = 1 - value
    return result


@pytest.mark.parametrize("corruption", [None, "tie", "probabilities", "controller"])
def test_public_cycle_uses_best_mask_and_trained_survivors_in_15epoch_recovery(tmp_path, monkeypatch, corruption):
    cfg = configuration("ste")
    cfg.model.backbone.pretrained_weights = None
    cfg.model.backbone.width_mult = .25
    cfg.model.backbone.num_classes = 3
    source = instantiate(cfg.model)
    name, gate = next((name, gate) for name, gate in get_gumbel_modules(source).items() if "mid_gumbel_layer" in name)
    with torch.no_grad():
        gate.logits[0] = torch.tensor([0., 0.] if corruption == "tie" else [4., .1])
        source.backbone.classifier[1].weight.add_(.25)
    best_metrics = channel_metrics(source)
    last_metrics = dict(best_metrics)
    last_metrics[f"valid_{name}.channel_000_zero_prob"] = .01
    last_metrics[f"valid_{name}.channel_001_zero_prob"] = .99
    calls = []
    held = {"version": 1, "runtime": {"last_epoch": 12, "log_lambda": -3.}}

    def train(stage_cfg, *, progress_context=None, model_initializer=None):
        calls.append(stage_cfg)
        root = tmp_path / ("search" if len(calls) == 1 else "recovery")
        (root / "checkpoints").mkdir(parents=True)
        if len(calls) == 1:
            assert stage_cfg.training_arguments.num_epochs == stage_cfg.scheduler.T_max == 30
            assert stage_cfg.training_arguments.adaptive_lambda.enabled
            assert model_initializer is None
            metrics = {} if corruption == "probabilities" else best_metrics
            extra = {} if corruption == "controller" else {"adaptive_lambda_state": held, "model_lambda_coef": .02}
            torch.save({"epoch": 12, "model_state_dict": source.state_dict(), "metrics": metrics, "extra_state": extra}, root / "checkpoints/best.pt")
            return {"run_dir": str(root), "best_epoch": 12, "num_epochs_executed": 30,
                    "last_valid_metrics": last_metrics, "adaptive_lambda_state": {"end_epoch": 30}}
        assert stage_cfg.training_arguments.num_epochs == stage_cfg.scheduler.T_max == 15
        assert stage_cfg.model.criterion.label_smoothing == 0
        assert stage_cfg.model.lambda_coef == 0
        assert not stage_cfg.training_arguments.adaptive_lambda.enabled
        mask = OmegaConf.to_container(stage_cfg.channel_pruning.mask)
        assert mask == {name: [0]}
        target = build_structurally_pruned_model_from_config(stage_cfg, stage_cfg.channel_pruning)
        model_initializer(target)
        torch.testing.assert_close(target.backbone.classifier[1].weight, source.backbone.classifier[1].weight, rtol=0, atol=0)
        apply_channel_mask(source, mask)
        source.eval()
        target.eval()
        torch.manual_seed(42)
        images, labels = torch.randn(2, 3, 32, 32), torch.tensor([1, 2])
        with torch.inference_mode():
            torch.testing.assert_close(source(images, labels).logits, target(images, labels).logits, rtol=1e-4, atol=1e-5)
        assert not get_gumbel_modules(target)
        torch.save({"epoch": 3, "model_state_dict": target.state_dict()}, root / "checkpoints/best.pt")
        return {"run_dir": str(root), "best_epoch": 3, "num_epochs_executed": 15,
                "test_metrics": {"test_accuracy": .7}}

    monkeypatch.setattr(cyclic, "run_training", train)
    if corruption in {"probabilities", "controller"}:
        with pytest.raises(ValueError):
            cyclic.run_cyclic_channel_pruning_training(cfg, tmp_path)
        assert len(calls) == 1
    else:
        result = cyclic.run_cyclic_channel_pruning_training(cfg, tmp_path)
        assert len(calls) == 2
        assert result["training_cost"]["num_epochs_executed"] == 45
        assert result["final_disabled_channels"] == {name: [0]}
        assert result["search_selections"][0]["epoch"] == 12
        assert result["search_selections"][0]["held_adaptive_lambda_state"] == held
        assert result["cycle_results"][0]["adaptive_lambda_state"] == {"end_epoch": 30}


def test_broken_reference_fails_before_launch(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    shutil.copyfile(ROOT / workflow.MANIFEST, tmp_path / workflow.MANIFEST)
    manifest = json.loads((tmp_path / workflow.MANIFEST).read_text())
    root = tmp_path / manifest["reference_dir"]
    root.mkdir(parents=True)
    archive = tmp_path / "synthetic-reference.tar.gz"
    archive.write_bytes(b"synthetic reference input")
    digest = workflow.hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(workflow.single, "EXPECTED_ARCHIVE_SHA256", digest)
    path = root / "history.csv"
    path.write_text("epoch,valid_accuracy\n" + "".join(f"{epoch},0.65\n" for epoch in range(1, 87)))
    workflow.write_json(root / "reference_provenance.json", {
        "archive_sha256": digest, "actual_training_horizon_epochs": 150,
        "actual_cosine_T_max": 150, "available_history_epochs": 86,
        "selected_reference_epochs": [1, 30], "training_performed": False,
        "history_sha256": workflow.hashlib.sha256(path.read_bytes()).hexdigest(),
    })
    assert workflow.validate_reference(tmp_path, archive)["available_history_epochs"] == 86
    path.write_text("epoch,valid_accuracy\n1,0.1\n")
    with pytest.raises(RuntimeError):
        workflow.validate_reference(tmp_path, archive)


@pytest.fixture
def completed_arm(tmp_path):
    stage = workflow.single.Stage("gumbel_hard", tmp_path / "arm", [], "same-source")
    stage.directory.mkdir()
    workflow.single.write_receipt(stage, "running")
    search, recovery = stage.directory / "search", stage.directory / "recovery"
    for root, epochs in [(search, 30), (recovery, 15)]:
        (root / "checkpoints").mkdir(parents=True)
        (root / "checkpoints/best.pt").write_bytes(b"selected-model")
        (root / "history.csv").write_text("epoch,valid_accuracy\n" + "".join(f"{epoch},0.65\n" for epoch in range(1, epochs + 1)))
    sr = {"run_dir": str(search), "num_epochs_executed": 30, "best_epoch": 12}
    rr = {"run_dir": str(recovery), "num_epochs_executed": 15, "best_epoch": 5,
          "test_metrics": {"test_accuracy": .7}, "model_parameters": {"total": 2000000}}
    selected = search / "checkpoints/best.pt"
    summary = {"search_only": False, "num_cycles_completed": 1,
               "channel_selection_source": "best_checkpoint", "training_cost": {"num_epochs_executed": 45},
               "cycle_results": [sr], "recovery_results": [rr], "final_result": rr,
               "weight_handoff": {"enabled": True, "surviving_weights_transferred": True},
               "search_selections": [{"epoch": 12, "checkpoint": str(selected),
                                      "checkpoint_sha256": workflow.hashlib.sha256(selected.read_bytes()).hexdigest(),
                                      "held_adaptive_lambda_state": {"epoch": 12}}],
               "final_disabled_channels": {"backbone.features.2.mid_gumbel_layer": [0, 1]}}
    workflow.write_json(stage.directory / "cyclic_summary.json", summary)
    workflow.write_json(recovery / "summary.json", {"runtime": {"test_evaluation": {"checkpoint_epoch": 5},
                                                               "model_parameters": {"total": 2000000}},
                                                   "best_valid": {"epoch": 5, "metrics": {"valid_accuracy": .66}}})
    return stage


def test_completed_matching_arm_reuses_and_reports_official_test(completed_arm):
    assert workflow.validate_stage(completed_arm)
    row = workflow.comparison_record(completed_arm)
    assert row["test_accuracy"] == .7
    assert row["validation_accuracy"] == .66
    assert row["total_epochs"] == 45
    assert row["removed_internal_channels"] == 2


@pytest.mark.parametrize("corruption", ["partial", "fingerprint", "source_weights", "test_epoch", "epoch_budget"])
def test_existing_incomplete_or_mismatched_arm_never_repeats(completed_arm, corruption):
    stage = completed_arm
    if corruption == "partial":
        (stage.directory / "cyclic_summary.json").unlink()
    elif corruption == "fingerprint":
        workflow.write_json(stage.directory / "launcher_receipt.json", {"fingerprint": "different"})
    elif corruption == "source_weights":
        (stage.directory / "search/checkpoints/best.pt").write_bytes(b"replaced")
    elif corruption == "test_epoch":
        path = stage.directory / "recovery/summary.json"
        summary = json.loads(path.read_text())
        summary["runtime"]["test_evaluation"]["checkpoint_epoch"] = 15
        workflow.write_json(path, summary)
    else:
        path = stage.directory / "cyclic_summary.json"
        summary = json.loads(path.read_text())
        summary["training_cost"]["num_epochs_executed"] = 75
        workflow.write_json(path, summary)
    with pytest.raises(RuntimeError, match="refusing repeat"):
        workflow.validate_stage(stage)


@pytest.mark.parametrize("scenario", ["success", "fail_reference", "fail_first", "reuse_first", "reuse_both"])
def test_public_launcher_is_sequential_fail_fast_and_reuses_completed(tmp_path, monkeypatch, scenario):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "configs").mkdir()
    manifest = json.loads((ROOT / workflow.MANIFEST).read_text())
    workflow.write_json(repo / workflow.MANIFEST, manifest)
    output = repo / manifest["output_root"]
    plan = [workflow.single.Stage(mode, output / mode, [mode], mode)
            for mode in ["gumbel_hard", "ste_hard"]]
    for stage in plan:
        if scenario == "reuse_both" or (scenario == "reuse_first" and stage.kind == "gumbel_hard"):
            stage.directory.mkdir(parents=True)
            (stage.directory / "done").touch()
    calls, preflights = [], []

    def run(command, **kwargs):
        if len(command) > 1 and command[1] == "scripts/prepare_mobilenet_reference.py":
            calls.append("reference_import")
            if scenario == "fail_reference":
                raise workflow.subprocess.CalledProcessError(1, command)
            return
        calls.append(command[0])
        if scenario == "fail_first":
            raise workflow.subprocess.CalledProcessError(1, command)
        (next(stage for stage in plan if stage.kind == command[0]).directory / "done").touch()

    monkeypatch.chdir(repo)
    monkeypatch.setattr(workflow, "__file__", str(repo / "scripts/run_mobilenet_open98_ablation.py"))
    monkeypatch.setattr(workflow, "build_plan", lambda root, reference=None: plan)
    monkeypatch.setattr(workflow, "validate_reference", lambda root, archive: {"saved": True})
    monkeypatch.setattr(workflow, "validate_configs", lambda root: {"search_epochs": 30, "recovery_epochs": 15})
    monkeypatch.setattr(workflow, "validate_stage", lambda stage: (stage.directory / "done").exists())
    monkeypatch.setattr(workflow, "comparison_record", lambda stage: {"mode": stage.kind, "test_accuracy": .7})
    monkeypatch.setattr(workflow.single, "preflight", lambda *args, **kwargs: preflights.append(True))
    monkeypatch.setattr(workflow.subprocess, "run", run)
    assert workflow.main([]) == (1 if scenario in {"fail_first", "fail_reference"} else 0)
    expected = {"success": ["reference_import", "gumbel_hard", "ste_hard"],
                "fail_reference": ["reference_import"], "fail_first": ["reference_import", "gumbel_hard"],
                "reuse_first": ["reference_import", "ste_hard"], "reuse_both": ["reference_import"]}
    assert calls == expected[scenario]
    assert bool(preflights) == (scenario not in {"reuse_both", "fail_reference"})
    if scenario not in {"fail_first", "fail_reference"}:
        assert (output / "comparison.csv").is_file()
        records = json.loads((output / "comparison.json").read_text())["results"]
        assert [row["mode"] for row in records] == ["gumbel_hard", "ste_hard"]
