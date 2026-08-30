import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_resnet50_aig_adaptive_lambda_300ep.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_resnet50_aig_adaptive_lambda_300ep",
    SCRIPT_PATH,
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
runner = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(runner)


def _write_history(path, epochs=300):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["epoch,valid_accuracy"]
    rows.extend(f"{epoch},0.9" for epoch in range(1, epochs + 1))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_runner_trains_complete_baseline_before_starting_adaptive_tuning(
    tmp_path,
    monkeypatch,
):
    baseline_dir = tmp_path / "outputs/baselines/resnet50_adamw_cifar10_300ep"
    history_path = baseline_dir / "history.csv"
    checkpoint_path = baseline_dir / "checkpoints/best.pt"
    commands = []

    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "BASELINE_DIR", baseline_dir)
    monkeypatch.setattr(runner, "BASELINE_HISTORY", history_path)
    monkeypatch.setattr(runner, "BASELINE_CHECKPOINT", checkpoint_path)
    monkeypatch.setattr(
        runner,
        "sys",
        SimpleNamespace(
            version_info=(3, 10),
            version="3.10.0",
            executable="/test/python",
        ),
    )

    def _fake_run(command):
        commands.append(command)
        if "src/net_complexity/train.py" in command:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.touch()
            _write_history(history_path)

    monkeypatch.setattr(runner, "_run", _fake_run)

    runner.main()

    assert commands == [
        [
            "/test/python",
            "src/net_complexity/train.py",
            "experiment=best_practice_resnet50_adamw_300ep_on_cifar10",
            "hydra.run.dir=outputs/baselines/resnet50_adamw_cifar10_300ep",
        ],
        [
            "/test/python",
            "src/net_complexity/tune.py",
            "--config-name=tune_resnet50_aig_adaptive_lambda_checkpoint_history_gap_1_2_2_4_300ep",
        ],
    ]


def test_runner_rejects_history_with_fewer_than_300_epochs(tmp_path, monkeypatch):
    baseline_dir = tmp_path / "baseline"
    history_path = baseline_dir / "history.csv"
    checkpoint_path = baseline_dir / "checkpoints/best.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.touch()
    _write_history(history_path, epochs=299)

    monkeypatch.setattr(runner, "BASELINE_HISTORY", history_path)
    monkeypatch.setattr(runner, "BASELINE_CHECKPOINT", checkpoint_path)

    assert runner._baseline_is_complete() is False
