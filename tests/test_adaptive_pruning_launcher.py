import csv
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_adaptive_pruning as launcher
from pruning_pilot_common import config_for
from net_complexity.training.pruning_measurement import state_hash, write_json


def arguments(**overrides):
    return SimpleNamespace(**{"config_name": "pruning_adaptive_nightly", "hours": None,
                              "data": None, "output": None, "dense_source": None, "job": None,
                              **overrides})


def adaptive_config(job="A1_internal_p18"):
    config = config_for(job)
    config.cyclic_channel_pruning.adaptive_reference_history = "verified_reference.csv"
    return config


def test_four_yaml_jobs_are_adaptive_150_epoch_learned_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUDIT_INIT_CHECKPOINT", "unused.pt")
    plan = launcher.resolve_plan(arguments())
    assert plan["jobs"] == list(launcher.ADAPTIVE_JOBS)
    assert plan["hours"] == 12.5 and plan["evaluate_test"] is True
    assert Path(plan["output"]).parent == tmp_path / "outputs/runs"
    assert Path(plan["output"]).name.endswith("_pruning_adaptive_nightly")
    assert not Path(plan["output"]).exists()
    assert not GlobalHydra.instance().is_initialized()
    assert plan["dense_source"].endswith("20260907_134446_507828_pruning_dense_control")
    for job, fraction in zip(plan["jobs"], (0.18, 0.12, 0.23, 0.05)):
        config = adaptive_config(job)
        launcher.validate_nightly_config(config)
        c, adaptive = config.cyclic_channel_pruning, config.training_arguments.adaptive_lambda
        assert c.max_param_fraction == fraction
        assert 3 * c.gumbel_epochs + 2 * c.recovery_epochs + c.final_epochs == 150
        assert adaptive.enabled is True and adaptive.recovery.enabled is True
        assert adaptive.warmup_epochs == 10 and adaptive.update_every_epochs == 3
        assert adaptive.baseline_history_dir is None and config.model.lambda_coef == 0.001
        assert config.optimizer.lr == 0.001
        assert OmegaConf.select(config, "model.criterion.label_smoothing", default=0.0) == 0.0
        assert not config.training_arguments.evaluate_test and not config.dataloaders.include_test
        assert config.metrics.valid_metrics[-1]._target_.endswith("GumbelProbMetric")
        assert config.metrics.valid_metrics[-1].log_channel_zero_probs is False
        assert config.run_history.monitor == "valid_accuracy" and list(config.metrics.test_metrics) == []


@pytest.mark.parametrize("key,value", [
    ("training_arguments.adaptive_lambda.enabled", False),
    ("cyclic_channel_pruning.final_epochs", 59),
    ("cyclic_channel_pruning.final_epochs", 61),
    ("training_arguments.adaptive_lambda.baseline_history_dir", "auto_train"),
    ("cyclic_channel_pruning.ranking", "random"), ("seed", 43),
    ("training_arguments.evaluate_test", True),
])
def test_invalid_model_cannot_launch(tmp_path, monkeypatch, key, value):
    monkeypatch.setenv("AUDIT_INIT_CHECKPOINT", "unused.pt")
    config = adaptive_config()
    OmegaConf.update(config, key, value)
    with pytest.raises(ValueError):
        launcher.validate_nightly_config(config)


@pytest.mark.parametrize("key,value", [
    ("jobs", []), ("jobs", ["J1_dense_control"]), ("jobs", ["A1_internal_p18", "A1_internal_p18"]),
    ("evaluate_test", False), ("protocol", "fixed"), ("hours", 0), ("hours", 25), ("hours", True),
    ("name", "../bad"), ("data", ""), ("dense_source", ""), ("run_history.root_dir", ""),
    ("unknown", True),
])
def test_bad_plan_refused_without_writes(tmp_path, monkeypatch, key, value):
    config = OmegaConf.load(launcher.ROOT / "configs/pruning_adaptive_nightly.yaml")
    OmegaConf.update(config, key, value, force_add=True)
    (tmp_path / "configs").mkdir()
    OmegaConf.save(config, tmp_path / "configs/bad.yaml")
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        launcher.resolve_plan(arguments(config_name="bad", output=tmp_path / "run"))
    assert not (tmp_path / "run").exists()


def test_cfg_preview_does_not_check_gpu_or_write(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("GPU queried during preview"))
    launcher.main(["--cfg", "job", "--hours", "11", "--data", "mydata", "--dense-source", "oldrun"])
    output = OmegaConf.create(capsys.readouterr().out)
    assert output.hours == 11 and output.data == str(tmp_path / "mydata")
    assert output.dense_source == str(tmp_path / "oldrun")
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("remaining,previous,expected", [
    (0, None, False), (-1, 100, False), (1, None, True), (114, 100, False), (116, 100, True),
])
def test_deadline_does_not_start_another_likely_incomplete_model(remaining, previous, expected):
    assert launcher.enough_time_for_job(remaining, previous) is expected


@pytest.fixture
def dense(tmp_path, monkeypatch):
    source = tmp_path / "dense"
    job = source / launcher.DENSE
    job.mkdir(parents=True)
    initializer = source / "shared_random_seed42.pt"
    monkeypatch.setenv("AUDIT_INIT_CHECKPOINT", str(initializer))
    config = config_for(launcher.DENSE)
    OmegaConf.save(config, job / "resolved_config.yaml", resolve=True)
    weights = {"weight": torch.tensor([1.0])}
    common_hash = state_hash(weights)
    torch.save({"model_state_dict": weights, "model_state_hash": common_hash,
                "seed": 42, "trained_epochs": 0}, initializer)
    state = {"status": "completed", "global_epochs_completed": 150, "total_epochs_allocated": 150,
             "accepted_mask": {}, "test_evaluated": False, "seed": 42,
             "common_init_hash": common_hash, "split_indices_hash": "split", "optimizer_steps_total": 52650,
             "validation": {"accuracy": 0.9442, "example_count": 5000}}
    write_json(job / "pilot_state.json", state)
    with (job / "global_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["global_epoch", "optimizer_steps_total", "valid_accuracy"])
        writer.writeheader()
        writer.writerows({"global_epoch": e, "optimizer_steps_total": e * 351, "valid_accuracy": 0.9}
                        for e in range(1, 151))
    monkeypatch.setattr(launcher, "prepare_job", lambda *a: (None, {"checked": True}))
    return source


def test_dense_reference_relocated_and_never_trained_or_changed(dense):
    before = {str(p): launcher.file_hash(p) for p in dense.rglob("*") if p.is_file()}
    result = launcher.validate_dense_source(dense / launcher.DENSE, adaptive_config())
    assert result["state"]["reused_from"] == str(dense / launcher.DENSE)
    assert result["initializer"] == dense / "shared_random_seed42.pt"
    assert result["provenance"]["deployment"] == {"checked": True}
    assert before == {str(p): launcher.file_hash(p) for p in dense.rglob("*") if p.is_file()}


@pytest.mark.parametrize("mutation", ["incomplete", "trained_init", "weights", "seed", "history", "unweighted"])
def test_unsafe_dense_reference_rejected(dense, mutation):
    state_path = dense / launcher.DENSE / "pilot_state.json"
    state = json.loads(state_path.read_text())
    initial_path = dense / "shared_random_seed42.pt"
    initial = torch.load(initial_path, weights_only=True)
    if mutation == "incomplete":
        state["global_epochs_completed"] = 149
    elif mutation == "trained_init":
        initial["trained_epochs"] = 150
    elif mutation == "weights":
        initial["model_state_dict"]["weight"][0] = 2
    elif mutation == "seed":
        state["seed"] = 43
    elif mutation == "history":
        (dense / launcher.DENSE / "global_history.csv").write_text("global_epoch,valid_accuracy\n1,0.9\n")
    elif mutation == "unweighted":
        path = dense / launcher.DENSE / "resolved_config.yaml"
        config = OmegaConf.load(path)
        config.metrics.valid_metrics[1].return_counts = False
        OmegaConf.save(config, path)
    write_json(state_path, state)
    torch.save(initial, initial_path)
    with pytest.raises(ValueError):
        launcher.validate_dense_source(dense, adaptive_config())


@pytest.mark.parametrize("ending", ["complete", "deadline", "interrupt"])
def test_queue_freezes_all_jobs_before_test_and_never_retrains_dense(tmp_path, dense, monkeypatch, ending):
    monkeypatch.setattr(launcher, "provenance", lambda: {"protocol": launcher.PROTOCOL})
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024 ** 3))
    monkeypatch.setattr(launcher, "write_comparison", lambda *a: None)
    plan = launcher.resolve_plan(arguments(output=tmp_path / "night", dense_source=dense))
    output = Path(plan["output"])
    commands, completed = [], []
    clock = {"now": 1000.0}
    monkeypatch.setattr(launcher.time, "monotonic", lambda: clock["now"])
    source_state = json.loads((dense / launcher.DENSE / "pilot_state.json").read_text())

    def child(command, log, deadline):
        commands.append(command)
        if "--job" in command:
            job = command[command.index("--job") + 1]
            assert job in launcher.ADAPTIVE_JOBS
            config = OmegaConf.load(command[command.index("--job-config") + 1])
            launcher.validate_nightly_config(config)
            assert Path(config.cyclic_channel_pruning.adaptive_reference_history).is_file()
            assert Path(config.cyclic_channel_pruning.weight_handoff.initial_checkpoint).is_file()
            if job == plan["jobs"][1] and ending != "complete":
                if ending == "deadline":
                    clock["now"] = deadline + 0.1
                raise TimeoutError("simulated deadline or manual interruption")
            completed.append(job)
            write_json(output / job / "pilot_state.json", {
                **source_state, "pilot_version": 2, "protocol": launcher.PROTOCOL,
                "final_cost": {"physical_total_parameters": 13000000}})
        if any(str(part).endswith("evaluate_pruning_test.py") for part in command):
            assert ending != "interrupt", "Manual interruption must not start test inference"
            assert completed == (plan["jobs"] if ending == "complete" else plan["jobs"][:1])
            assert command[command.index("--jobs") + 1:] == [launcher.DENSE, *completed]
            write_json(output / "test_evaluation/test_summary.json", {"status": "completed", "test_evaluated": True})

    monkeypatch.setattr(launcher, "run_child", child)
    if ending == "interrupt":
        with pytest.raises(TimeoutError):
            launcher.run_queue(plan)
        assert not (output / "test_evaluation").exists()
        assert json.loads((output / "nightly_status.json").read_text())["status"] == "failed_or_interrupted"
        return
    status = launcher.run_queue(plan)
    assert status["test_evaluated"] is True
    if ending == "deadline":
        assert status["status"] == "completed_partial_wall_budget"
        assert status["completed_jobs"] == plan["jobs"][:1]
        assert status["incomplete_jobs"] == plan["jobs"][1:2]
        assert status["skipped_jobs"] == plan["jobs"][2:]
        assert len(commands) == 5  # tests, smoke, one success, one timeout, final test
    else:
        assert status["status"] == "completed"
        assert status["completed_jobs"] == plan["jobs"]
        assert len(commands) == 7  # tests, real smoke, 4 new jobs, final test
    assert json.loads((output / launcher.DENSE / "pilot_state.json").read_text())["reused_from"]
    assert (output / "adaptive_reference_history.csv").read_bytes() == (dense / launcher.DENSE / "global_history.csv").read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        launcher.run_queue(plan)
