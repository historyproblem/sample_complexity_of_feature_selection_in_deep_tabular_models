"""Read-only previews, explicit budgets/reference, and durable per-job test order."""
from argparse import Namespace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_one_shot_pruning as launcher
from net_complexity.training.pruning_measurement import write_json


def arguments(**changes):
    base = dict(config_name="pruning_one_shot", cfg=None, data=None, device=None,
                hours=None, output=None, reference_source=None, job=None, job_config=None)
    return Namespace(**{**base, **changes})


def test_default_and_queue_explicit_300_plus_shared_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    single = launcher.resolve_plan(arguments())
    assert single["jobs"] == [launcher.JOBS[0]]
    assert single["total_new_training_epochs_allocated"] == 450
    assert single["new_dense_reference_epochs"] == 150
    assert single["new_training_epochs_per_job"] == 300
    assert single["hours"] is None and single["evaluate_test"] is True
    assert single["reference_source"] is None
    assert Path(single["output"]).parent == tmp_path / "outputs/runs"
    queue = launcher.resolve_plan(arguments(config_name="pruning_one_shot_queue"))
    assert queue["jobs"] == list(launcher.JOBS)
    assert queue["total_new_training_epochs_allocated"] == 1050
    reused = launcher.resolve_plan(arguments(reference_source=tmp_path / "dense"))
    assert reused["total_new_training_epochs_allocated"] == 300
    assert reused["new_dense_reference_epochs"] == 0
    assert reused["dense_reference_epochs"] == 150
    assert not (tmp_path / "outputs").exists()
    assert not GlobalHydra.instance().is_initialized()


def test_all_variants_are_mask_based_adaptive_continuous_150_plus_150():
    for job, soft, hard in zip(launcher.JOBS, (0.005, 0.01, 0.02), (0.01, 0.02, 0.05)):
        cfg = launcher.config_for(job)
        launcher.validate_model(cfg)
        assert cfg.training_arguments.adaptive_lambda.soft_drop == soft
        assert cfg.training_arguments.adaptive_lambda.hard_drop == hard
        assert cfg.scheduler.T_max == cfg.training_arguments.num_epochs == 150
        assert cfg.one_shot_pruning.search_epochs + cfg.one_shot_pruning.retrain_epochs == 300
        assert cfg.one_shot_pruning.probability_source == "raw_logits"
        assert cfg.one_shot_pruning.mask_threshold == 0.5
        assert cfg.one_shot_pruning.initial_checkpoint is None
        assert cfg.model.backbone.resnet_block.regularization_normalization == "initial_channels"
        assert cfg.run_history.monitor == "valid_accuracy"
        assert OmegaConf.select(cfg, "cyclic_channel_pruning") is None
        assert "param_budget" not in OmegaConf.to_yaml(cfg)
        assert "max_param_fraction" not in OmegaConf.to_yaml(cfg)
        assert cfg.training_arguments.adaptive_lambda.baseline_history_dir is None


@pytest.mark.parametrize("key,value", [
    ("training_arguments.adaptive_lambda.enabled", False),
    ("training_arguments.adaptive_lambda.baseline_history_dir", "hidden_baseline"),
    ("training_arguments.evaluate_test", True), ("dataloaders.include_test", True),
    ("one_shot_pruning.search_epochs", 149), ("one_shot_pruning.retrain_epochs", 151),
    ("one_shot_pruning.mask_threshold", 0.7), ("one_shot_pruning.probability_source", "effective"),
    ("model.backbone.resnet_block.regularization_normalization", "enabled_channels"),
    ("scheduler.T_max", 200), ("seed", 43),
    ("training_arguments.adaptive_lambda.hard_drop", 0.001),
    ("cyclic_channel_pruning.enabled", True),
])
def test_invalid_training_config_refused(key, value):
    cfg = launcher.config_for(launcher.JOBS[0])
    OmegaConf.update(cfg, key, value, force_add=True)
    with pytest.raises(ValueError):
        launcher.validate_model(cfg)


@pytest.mark.parametrize("changes", [dict(hours=-1), dict(hours=float("inf")), dict(hours=True),
                                          dict(device="cpu"), dict(device="cuda"), dict(config_name="../bad")])
def test_invalid_plan_refused(changes):
    with pytest.raises(ValueError):
        launcher.resolve_plan(arguments(**changes))


def test_preview_contains_full_config_without_gpu_or_source_reads(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("GPU checked in preview"))
    monkeypatch.setattr(launcher, "validate_reference", lambda *a: pytest.fail("Reference read in preview"))
    launcher.main(["--cfg", "job", "--reference-source", "does_not_exist", "--hours", "12"])
    cfg = OmegaConf.create(capsys.readouterr().out)
    assert cfg.total_new_training_epochs_allocated == 300
    assert cfg.hours == 12
    assert cfg.resolved_jobs[launcher.JOBS[0]].training_arguments.adaptive_lambda.soft_drop == 0.005
    assert cfg.resolved_jobs[launcher.DENSE].scheduler.T_max == 150
    assert not (tmp_path / "outputs").exists()


def test_cpu_refused_before_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        launcher.main([])
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("ending", ["complete", "train_failure", "test_failure"])
def test_queue_tests_dense_and_each_job_before_next_preserves_previous(tmp_path, monkeypatch, reuse, ending):
    output = tmp_path / "queue"
    dense = tmp_path / "prior" / launcher.DENSE if reuse else output / launcher.DENSE
    plan = launcher.resolve_plan(arguments(config_name="pruning_one_shot_queue", output=output,
                                          reference_source=dense if reuse else None))
    events = []
    reference_state = {"protocol": launcher.PROTOCOL, "status": "completed", "kind": "dense_reference"}

    def dense_artifacts():
        dense.mkdir(parents=True, exist_ok=True)
        write_json(dense / "one_shot_state.json", reference_state)
        (dense / "initializer.pt").write_bytes(b"zero-epoch initializer fixture")
        (dense / "global_history.csv").write_text("fixture\n")
        OmegaConf.save(launcher.config_for(launcher.DENSE), dense / "resolved_config.yaml")

    if reuse:
        dense_artifacts()

    def reference(source, cfg):
        assert Path(source) == dense
        assert (dense / "one_shot_state.json").exists()
        launcher.validate_model(cfg)
        return reference_state

    def child(command, log, deadline):
        if "--job" in command:
            job = command[command.index("--job") + 1]
            events.append(("train", job))
            if job == launcher.DENSE:
                assert not reuse
                dense_artifacts()
                return
            cfg = OmegaConf.load(command[command.index("--job-config") + 1])
            launcher.validate_model(cfg)
            assert Path(cfg.one_shot_pruning.initial_checkpoint).is_file()
            assert Path(cfg.one_shot_pruning.reference_history).is_file()
            if ending == "train_failure" and job == launcher.JOBS[1]:
                raise RuntimeError("training failed")
            write_json(output / job / "one_shot_state.json", {
                "protocol": launcher.PROTOCOL, "status": "completed", "kind": "one_shot_reinit",
                "search_epochs_completed": 150, "retrain_epochs_completed": 150,
                "global_epochs_completed": 300, "total_epochs_allocated": 300})
        elif any(str(arg).endswith("evaluate_one_shot_pruning.py") for arg in command):
            job = command[command.index("--jobs") + 1]
            events.append(("test", job))
            if ending == "test_failure" and job == launcher.JOBS[1]:
                raise RuntimeError("test failed")
            destination = Path(command[command.index("--output") + 1])
            assert not destination.exists()
            assert command[command.index("--reference-source") + 1] == str(dense)
            write_json(destination / "test_summary.json", {
                "status": "completed", "test_evaluated": True,
                "runs": [{"job": job, "physical_parameters": 13000000,
                          "kind": "dense_reference" if job == launcher.DENSE else "one_shot_reinit",
                          "validation": {"accuracy": 0.94},
                          "test": {"accuracy": 0.93, "correct_count": 9300,
                                   "example_count": 10000, "ce_loss": 0.3},
                          "search_epochs": 0 if job == launcher.DENSE else 150,
                          "retrain_epochs": 0 if job == launcher.DENSE else 150,
                          "dense_epochs": 150 if job == launcher.DENSE else 0,
                          "epochs_consumed": 150 if job == launcher.DENSE else 300,
                          "conv_linear_macs_per_image": 1000000000,
                          "checkpoint_sha256": job, "predictions": f"{job}.npz"}]})

    monkeypatch.setattr(launcher, "validate_reference", reference)
    monkeypatch.setattr(launcher, "run_child", child)
    monkeypatch.setattr(launcher, "provenance", lambda _: {})
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024 ** 3))
    if ending == "complete":
        status = launcher.run_queue(plan)
        assert status["status"] == "completed"
        assert status["all_planned_models_test_evaluated"] is True
    else:
        with pytest.raises(RuntimeError):
            launcher.run_queue(plan)
        status = json.loads((output / "one_shot_queue_status.json").read_text())
        assert status["status"] == "failed_or_interrupted"
        assert status["incomplete_jobs"] == ([launcher.JOBS[1]] if ending == "train_failure" else [])
        assert status["all_completed_models_test_evaluated"] == (ending == "train_failure")
    expected = ([] if reuse else [("train", launcher.DENSE)]) + [
        ("test", launcher.DENSE), ("train", launcher.JOBS[0]), ("test", launcher.JOBS[0]),
        ("train", launcher.JOBS[1])]
    if ending != "train_failure":
        expected.append(("test", launcher.JOBS[1]))
    if ending == "complete":
        expected += [("train", launcher.JOBS[2]), ("test", launcher.JOBS[2])]
    assert events == expected
    summary = json.loads((output / "test_evaluation/test_summary.json").read_text())
    assert summary["status"] == status["status"]
    assert summary["evaluated_jobs"][:2] == [launcher.DENSE, launcher.JOBS[0]]
    first_report = output / "test_evaluation" / launcher.JOBS[0] / "test_summary.json"
    assert json.loads(first_report.read_text())["status"] == "completed"
    assert "93.00%" in (output / "test_evaluation/test_comparison.md").read_text()
    with pytest.raises(ValueError, match="overwrite"):
        launcher.run_queue(plan)


def test_invalid_reference_fails_before_reserving_output(tmp_path, monkeypatch):
    output = tmp_path / "new"
    plan = launcher.resolve_plan(arguments(output=output, reference_source=tmp_path / "old_J1"))
    monkeypatch.setattr(launcher, "validate_reference", lambda *args: (_ for _ in ()).throw(
        ValueError("not matching continuous reference")))
    with pytest.raises(ValueError, match="continuous"):
        launcher.run_queue(plan)
    assert not output.exists()
