from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from net_complexity.cli.infer_stochastic_depth import (
    _set_stochastic_depth_inference_mode,
    discover_checkpoint_targets,
)
from net_complexity.models.stochastic_depth import HuangStochasticDepthBottleneck


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_stochastic_depth_inference_config_targets_existing_checkpoints():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        config = compose(config_name="infer_stochastic_depth")

    assert config.inference.eval_mode == "stochastic"
    assert config.inference.checkpoint_name == "best.pt"
    assert config.inference.num_repeats == 1
    assert list(config.inference.sources) == [
        "outputs/studies/"
        "20260819_221331_stochastic_depth_resnet50_"
        "pL_flops_grid_dense_10_09_08_07_05_03_ordered"
    ]


def test_discover_checkpoint_targets_accepts_study_runs_and_run_dirs(tmp_path):
    study_dir = tmp_path / "study"
    run_dir = study_dir / "runs" / "run-1"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    (run_dir / "config_resolved.yaml").write_text("model: {}\n", encoding="utf-8")
    (checkpoints_dir / "best.pt").touch()

    from_study = discover_checkpoint_targets(
        [study_dir],
        checkpoint_name="best.pt",
        root=tmp_path,
    )
    from_runs = discover_checkpoint_targets(
        [study_dir / "runs"],
        checkpoint_name="best.pt",
        root=tmp_path,
    )
    from_run = discover_checkpoint_targets(
        [run_dir],
        checkpoint_name="best.pt",
        root=tmp_path,
    )

    assert from_study == from_runs == from_run
    assert from_study[0].run_dir == run_dir.resolve()


def test_discover_checkpoint_targets_rejects_missing_checkpoint(tmp_path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "config_resolved.yaml").write_text("model: {}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Checkpoint 'best.pt' is missing"):
        discover_checkpoint_targets(
            [run_dir],
            checkpoint_name="best.pt",
            root=tmp_path,
        )


def test_stochastic_inference_uses_train_depth_with_eval_batchnorm(monkeypatch):
    block = HuangStochasticDepthBottleneck(
        1,
        1,
        survival_probability=0.0,
    )
    model = torch.nn.Sequential(block)
    monkeypatch.setattr(
        block.conv1,
        "forward",
        lambda _input: (_ for _ in ()).throw(
            AssertionError("residual branch should not be computed")
        ),
    )

    blocks = _set_stochastic_depth_inference_mode(model, "stochastic")
    inputs = torch.ones(2, 1, 4, 4)
    with torch.no_grad():
        outputs = model(inputs)

    torch.testing.assert_close(outputs, inputs)
    assert list(blocks) == ["0"]
    assert block.training is True
    assert block.batch_norm1.training is False
    assert block.last_residual_branch_active is False
