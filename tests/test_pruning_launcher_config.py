from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_pruning_pilot as launcher
from pruning_pilot_common import config_for
from net_complexity.training.pruning_audit import validate_config


def _args(**overrides):
    return SimpleNamespace(**{
        "config_name": "pruning_nightly", "profile": None, "hours": None,
        "with_random_control": False, "data": None, "output": None,
        "preflight_only": False, "resume_from": None, "job": None,
        **overrides,
    })


def test_nightly_yaml_keeps_four_jobs_and_standard_auto_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = _args()
    jobs, plan = launcher.resolve_launcher_plan(args)
    assert jobs == ["J1_dense_control", "J2_output_fixed", "J3_internal_fixed", "J4_internal_random"]
    assert plan["hours"] == 12.5
    assert args.output.parent == Path("outputs/runs")
    assert args.output.name.endswith("_pruning_nightly")
    assert Path(plan["output"]).parent == tmp_path / "outputs/runs"
    assert plan["data"] == str(tmp_path / "data")
    assert not args.output.exists()
    assert not GlobalHydra.instance().is_initialized()
    monkeypatch.setenv("AUDIT_INIT_CHECKPOINT", "unused")
    for job in jobs:
        # Composition of the queue must not leave Hydra active for model configs.
        cfg = config_for(job)
        assert validate_config(cfg) == 150
        assert cfg.seed == 42
        assert not cfg.training_arguments.evaluate_test


def test_explicit_cli_values_override_yaml_and_random_is_not_duplicated(tmp_path):
    args = _args(hours=10.5, output=tmp_path / "explicit", data=tmp_path / "dataset", with_random_control=True)
    jobs, plan = launcher.resolve_launcher_plan(args)
    assert plan["hours"] == 10.5
    assert plan["output"] == str(tmp_path / "explicit")
    assert plan["data"] == str(tmp_path / "dataset")
    assert jobs.count("J4_internal_random") == 1


def test_old_cli_preserves_profiles_and_now_also_has_auto_output(tmp_path):
    args = _args(config_name=None, output=tmp_path / "explicit")
    jobs, plan = launcher.resolve_launcher_plan(args)
    assert jobs == launcher.JOBS
    assert plan["hours"] == 11.75
    args = _args(config_name=None, profile="daytime")
    jobs, plan = launcher.resolve_launcher_plan(args)
    assert jobs == launcher.DAYTIME_JOBS
    assert plan["hours"] == 2.0
    assert args.output.parent == Path("outputs/runs")


@pytest.mark.parametrize("key,value", [
    ("jobs", []), ("jobs", ["J1_dense_control", "J1_dense_control"]),
    ("jobs", ["J3_internal_fixed", "J1_dense_control"]),
    ("jobs", ["J1_dense_control", "D2_internal_fixed"]),
    ("jobs", "J1_dense_control"), ("jobs", [42]),
    ("profile", "unknown"), ("profile", ["nightly"]),
    ("hours", "12.5"), ("hours", True), ("hours", 0), ("hours", 25),
    ("name", "../escape"), ("data", ""), ("run_history.root_dir", ""),
    ("run_history.extra", True), ("unknown", True),
])
def test_invalid_yaml_fails_before_any_output(tmp_path, monkeypatch, key, value):
    cfg = OmegaConf.load(launcher.ROOT / "configs/pruning_nightly.yaml")
    OmegaConf.update(cfg, key, value, force_add=True)
    (tmp_path / "configs").mkdir()
    OmegaConf.save(cfg, tmp_path / "configs/invalid.yaml")
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        launcher.resolve_launcher_plan(_args(config_name="invalid", output=tmp_path / "result"))
    assert not (tmp_path / "result").exists()
    assert not GlobalHydra.instance().is_initialized()


def test_profile_override_cannot_silently_change_yaml_experiment():
    with pytest.raises(ValueError, match="independently"):
        launcher.resolve_launcher_plan(_args(profile="daytime"))


def test_cfg_job_prints_plan_without_gpu_env_or_files(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUDIT_INIT_CHECKPOINT", "unchanged")
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: pytest.fail("GPU checked during preview"))
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("process started during preview"))
    launcher.main(["--config-name", "pruning_nightly", "--cfg", "job"])
    plan = OmegaConf.create(capsys.readouterr().out)
    assert plan.hours == 12.5 and len(plan.jobs) == 4
    assert not (tmp_path / "outputs").exists()
    assert launcher.os.environ["AUDIT_INIT_CHECKPOINT"] == "unchanged"


def _mock_runtime(monkeypatch):
    monkeypatch.setattr(launcher, "sys", SimpleNamespace(
        version_info=(3, 12), executable=sys.executable, version=sys.version))
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(launcher, "provenance", lambda: {"mock": True})
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))


def test_yaml_launch_keeps_preflight_shared_init_deadline_and_comparison(tmp_path, monkeypatch):
    from test_pruning_pilot_launcher import _completed_result
    monkeypatch.chdir(tmp_path)
    _mock_runtime(monkeypatch)
    events, deadlines = [], []
    def initializer(cfg, path):
        events.append("initializer")
        assert path == Path(launcher.os.environ["AUDIT_INIT_CHECKPOINT"])
        path.touch()
    def child(command, log, deadline):
        deadlines.append(deadline)
        if "--job" not in command:
            events.append("tests" if "pytest" in command else "smoke")
            assert "initializer" not in events
            if events[-1] == "smoke":
                assert command[command.index("--profile") + 1] == "nightly"
            return
        job = command[command.index("--job") + 1]
        events.append(job)
        assert command[command.index("--profile") + 1] == "nightly"
        assert command[command.index("--data") + 1] == str(tmp_path / "dataset")
        assert Path(launcher.os.environ["AUDIT_INIT_CHECKPOINT"]).is_file()
        directory = Path(command[command.index("--output") + 1])
        launcher.write_json(directory / "pilot_state.json", _completed_result(epochs=150, accuracy=0.94))
    monkeypatch.setattr(launcher, "make_initializer", initializer)
    monkeypatch.setattr(launcher, "run_child", child)
    launcher.main(["--config-name", "pruning_nightly", "--data", "dataset"])
    assert events == ["tests", "smoke", "initializer", *launcher.JOBS, "J4_internal_random"]
    assert len(set(deadlines)) == 1
    output, = (tmp_path / "outputs/runs").iterdir()
    saved = OmegaConf.load(output / "launcher_config.yaml")
    assert saved.jobs == launcher.JOBS + ["J4_internal_random"]
    assert saved.hours == 12.5 and saved.output == str(output)
    assert saved.data == str(tmp_path / "dataset")
    status = json.loads((output / "nightly_status.json").read_text())
    assert status["status"] == "completed" and status["planned_jobs"] == list(saved.jobs)
    assert len(json.loads((output / "comparison.json").read_text())["runs"]) == 4


def test_existing_output_is_unchanged_and_no_process_starts(tmp_path, monkeypatch):
    _mock_runtime(monkeypatch)
    output = tmp_path / "previous"
    output.mkdir()
    original = b"preserve previous results"
    (output / "provenance.json").write_bytes(original)
    monkeypatch.setattr(launcher, "run_child", lambda *a: pytest.fail("process started"))
    with pytest.raises(FileExistsError):
        launcher.main(["--config-name", "pruning_nightly", "--output", str(output)])
    assert {p.name: p.read_bytes() for p in output.iterdir()} == {"provenance.json": original}


@pytest.mark.parametrize("resume", [False, True])
def test_internal_child_keeps_job_config_and_never_repeats_preflight(tmp_path, monkeypatch, resume):
    _mock_runtime(monkeypatch)
    job = "D2_internal_fixed" if resume else "J4_internal_random"
    profile = "daytime" if resume else "nightly"
    output, source = tmp_path / job, tmp_path / "source"
    initial = source / "shared_random_seed42.pt"
    monkeypatch.setenv("AUDIT_INIT_CHECKPOINT", str(initial))
    monkeypatch.setattr(launcher, "run_child", lambda *a: pytest.fail("child repeated preflight"))
    monkeypatch.setattr(launcher, "make_initializer", lambda *a: pytest.fail("child recreated initializer"))
    received = []
    def train(cfg, destination, *, resume_search_from):
        assert validate_config(cfg) == (25 if resume else 150)
        assert cfg.cyclic_channel_pruning.weight_handoff.initial_checkpoint == str(initial)
        assert cfg.dataloaders.path_to_data == str(tmp_path / "data")
        assert destination == output
        assert resume_search_from == (source / job if resume else None)
        received.append(cfg.cyclic_channel_pruning.ranking)
    monkeypatch.setattr(launcher, "run_fixed_pruning_pilot", train)
    args = ["--job", job, "--profile", profile, "--output", str(output), "--data", str(tmp_path / "data")]
    if resume:
        args.extend(["--resume-from", str(source)])
    launcher.main(args)
    assert received == ["learned" if resume else "random"]
    assert not output.exists()
