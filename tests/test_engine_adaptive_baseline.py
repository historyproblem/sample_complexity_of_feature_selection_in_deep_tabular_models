import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from net_complexity.training import engine


class _LambdaModel(nn.Module):
    def __init__(self, lambda_coef: float = 1e-6):
        super().__init__()
        self.lambda_coef = float(lambda_coef)


def _make_config(baseline_history_dir: str) -> object:
    return OmegaConf.create({
        "model": {
            "lambda_coef": 5.0,
            "gumbel_init_mode": "paper_resnet50",
            "bypass_on_zero_lambda": False,
        },
        "training_arguments": {
            "num_epochs": 2,
            "adaptive_lambda": {
                "enabled": True,
                "baseline_history_dir": baseline_history_dir,
            },
            "lambda_warmup": {
                "enabled": False,
            },
        },
        "run_history": {
            "root_dir": "outputs/runs",
            "use_hydra_output_dir": True,
        },
        "mlflow": {
            "enabled": False,
            "run_name": "adaptive_demo",
        },
    })


def test_build_baseline_training_config_disables_pruning_and_redirects_run_history(tmp_path):
    config = _make_config(str(tmp_path / "baseline_root"))
    OmegaConf.set_struct(config, True)

    baseline_config = engine._build_baseline_training_config(config, tmp_path / "baseline_root")

    assert baseline_config.training_arguments.adaptive_lambda.enabled is False
    assert baseline_config.training_arguments.lambda_warmup.enabled is False
    assert baseline_config.model.lambda_coef == 0.0
    assert baseline_config.model.gumbel_init_mode == "fully_open"
    assert baseline_config.model.bypass_on_zero_lambda is True
    assert baseline_config.run_history.use_hydra_output_dir is False
    assert baseline_config.run_history.root_dir == str(tmp_path / "baseline_root")
    assert baseline_config.run_history.run_name == "adaptive_demo_baseline_no_pruning"


def test_adaptive_lambda_auto_log_step_matches_missing_config_value():
    expected = math.log(10.0 / 1e-6) / 33.0

    def _build(log_step_init):
        adaptive = {
            "enabled": True,
            "warmup_epochs": 0,
            "update_every_epochs": 2,
            "lambda_max": 10.0,
        }
        if log_step_init != "missing":
            adaptive["log_step_init"] = log_step_init
        training_arguments = OmegaConf.create(
            {
                "num_epochs": 200,
                "adaptive_lambda": adaptive,
            }
        )
        return engine._build_adaptive_lambda(training_arguments, _LambdaModel())

    missing_controller = _build("missing")
    auto_controller = _build("auto")
    explicit_controller = _build(0.25)

    assert missing_controller.step == pytest.approx(expected)
    assert auto_controller.step == pytest.approx(expected)
    assert explicit_controller.step == pytest.approx(0.25)


def test_adaptive_lambda_auto_log_step_uses_actual_initial_lambda():
    training_arguments = OmegaConf.create(
        {
            "num_epochs": 150,
            "adaptive_lambda": {
                "enabled": True,
                "warmup_epochs": 0,
                "update_every_epochs": 1,
                "lambda_max": None,
                "log_step_init": "auto",
            },
        }
    )

    controller = engine._build_adaptive_lambda(
        training_arguments,
        _LambdaModel(lambda_coef=1e-4),
    )

    expected = math.log(10.0 / 1e-4) / 50.0
    assert controller is not None
    assert controller.step == pytest.approx(expected)
    assert 1e-4 * math.exp(controller.step * 50) == pytest.approx(10.0)


def test_ensure_adaptive_baseline_reference_runs_baseline_when_folder_is_empty(tmp_path, monkeypatch):
    baseline_root = tmp_path / "baseline_root"
    config = _make_config(str(baseline_root))
    calls: list[Path] = []
    observed_progress_contexts: list[dict] = []

    def _fake_run_training(baseline_config, epoch_end_callback=None, progress_context=None):
        del epoch_end_callback
        observed_progress_contexts.append(dict(progress_context or {}))
        run_root = Path(str(baseline_config.run_history.root_dir))
        run_dir = run_root / "generated_baseline_run"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "history.csv").write_text(
            "epoch,valid_accuracy,epoch_time_sec\n1,0.81,2.5\n2,0.84,3.5\n",
            encoding="utf-8",
        )
        calls.append(run_dir)
        return {"run_dir": str(run_dir)}

    monkeypatch.setattr(engine, "run_training", _fake_run_training)

    baseline_reference = engine._ensure_adaptive_baseline_reference(
        config,
        progress_context={
            "display_trial_idx": 1,
            "grid_params": {"model.lambda_coef": 1e-8},
            "optuna_trial_params": {"model.lambda_coef": 1e-8},
        },
    )

    assert calls == [baseline_root / "generated_baseline_run"]
    assert observed_progress_contexts == [
        {
            "display_trial_idx": 1,
            "baseline_reference_run": True,
        }
    ]
    assert baseline_reference is not None
    assert baseline_reference.metric_name == "valid_accuracy"
    assert baseline_reference.history_path == baseline_root / "generated_baseline_run" / "history.csv"
    assert baseline_reference.accuracy_by_epoch == {1: 0.81, 2: 0.84}
    assert baseline_reference.full_train_time_sec == 6.0
    assert baseline_reference.num_epochs_executed == 2
    assert baseline_reference.generated_for_current_run is True


def test_ensure_adaptive_baseline_reference_reuses_existing_history_without_rerunning(tmp_path, monkeypatch):
    baseline_root = tmp_path / "baseline_root"
    existing_run_dir = baseline_root / "existing_run"
    existing_run_dir.mkdir(parents=True, exist_ok=False)
    (existing_run_dir / "history.csv").write_text(
        "epoch,valid_accuracy\n1,0.79\n2,0.83\n",
        encoding="utf-8",
    )
    config = _make_config(str(baseline_root))

    def _unexpected_run_training(*args, **kwargs):
        raise AssertionError("baseline training should not run when history already exists")

    monkeypatch.setattr(engine, "run_training", _unexpected_run_training)

    baseline_reference = engine._ensure_adaptive_baseline_reference(config)

    assert baseline_reference is not None
    assert baseline_reference.history_path == existing_run_dir / "history.csv"
    assert baseline_reference.accuracy_by_epoch == {1: 0.79, 2: 0.83}
    assert baseline_reference.generated_for_current_run is False


def test_ensure_adaptive_baseline_reference_uses_checkpoint_run_epoch_history(
    tmp_path,
    monkeypatch,
):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "best.pt"
    torch.save(
        {
            "epoch": 137,
            "metrics": {
                "valid_accuracy": 0.9432,
                "valid_loss": 0.21,
            },
        },
        checkpoint_path,
    )
    (tmp_path / "history.csv").write_text(
        "epoch,valid_accuracy,epoch_time_sec\n"
        "1,0.71,2.5\n"
        "2,0.83,3.5\n"
        "137,0.9432,4.0\n",
        encoding="utf-8",
    )
    config = _make_config("")
    config.training_arguments.adaptive_lambda.baseline_checkpoint_path = str(
        checkpoint_dir
    )

    def _unexpected_run_training(*args, **kwargs):
        raise AssertionError("checkpoint baseline must not start a baseline run")

    monkeypatch.setattr(engine, "run_training", _unexpected_run_training)

    baseline_reference = engine._ensure_adaptive_baseline_reference(config)

    assert baseline_reference is not None
    assert baseline_reference.checkpoint_path == checkpoint_path
    assert baseline_reference.root_dir == tmp_path
    assert baseline_reference.history_path == tmp_path / "history.csv"
    assert baseline_reference.metric_name == "valid_accuracy"
    assert baseline_reference.accuracy_by_epoch == {
        1: 0.71,
        2: 0.83,
        137: 0.9432,
    }
    assert baseline_reference.full_train_time_sec == 10.0
    assert baseline_reference.num_epochs_executed == 3
    assert baseline_reference.generated_for_current_run is False

    controller = engine._build_adaptive_lambda(
        config.training_arguments,
        _LambdaModel(),
        baseline_accuracy_by_epoch=baseline_reference.accuracy_by_epoch,
        baseline_reference_source="baseline_checkpoint_history",
    )
    assert controller is not None
    assert controller.summary_state()["reference_source"] == "baseline_checkpoint_history"
    assert controller._resolve_reference_accuracy(2) == (2, 0.83)


def test_checkpoint_baseline_rejects_constant_best_accuracy_without_history(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "best.pt"
    torch.save(
        {"epoch": 7, "metrics": {"valid_accuracy": 0.95}},
        checkpoint_path,
    )
    config = _make_config("")
    config.training_arguments.adaptive_lambda.baseline_checkpoint_path = str(
        checkpoint_path
    )

    with pytest.raises(FileNotFoundError, match="history.csv for epoch-aligned accuracy"):
        engine._ensure_adaptive_baseline_reference(config)


def test_checkpoint_baseline_requires_accuracy_for_every_training_epoch(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "best.pt"
    torch.save({"epoch": 2}, checkpoint_path)
    (tmp_path / "history.csv").write_text(
        "epoch,valid_accuracy\n1,0.71\n",
        encoding="utf-8",
    )
    config = _make_config("")
    config.training_arguments.adaptive_lambda.baseline_checkpoint_path = str(
        checkpoint_path
    )

    with pytest.raises(ValueError, match="missing epochs: 2"):
        engine._ensure_adaptive_baseline_reference(config)


def test_adaptive_baseline_rejects_history_and_checkpoint_at_the_same_time(tmp_path):
    checkpoint_path = tmp_path / "best.pt"
    torch.save({"metrics": {"valid_accuracy": 0.9}}, checkpoint_path)
    config = _make_config(str(tmp_path / "history"))
    config.training_arguments.adaptive_lambda.baseline_checkpoint_path = str(
        checkpoint_path
    )

    with pytest.raises(ValueError, match="Configure only one"):
        engine._ensure_adaptive_baseline_reference(config)
