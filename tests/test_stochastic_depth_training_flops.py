from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from net_complexity.metrics.stochastic_depth import StochasticDepthFLOPsMetric
from net_complexity.models.stochastic_depth import (
    HuangStochasticDepthBottleneck,
    get_stochastic_depth_blocks,
)


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


class _TinyStochasticDepthNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 4, kernel_size=1)
        self.block1 = HuangStochasticDepthBottleneck(
            4,
            1,
            survival_probability=0.5,
        )
        self.block2 = HuangStochasticDepthBottleneck(
            4,
            1,
            survival_probability=0.25,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def _force_all_train_masks(model: nn.Module, value: float) -> None:
    for block in get_stochastic_depth_blocks(model).values():
        block._sample_survival_mask = lambda x, value=value: x.new_tensor(value)


def test_training_flops_use_dense_reference_and_sample_weighted_masks():
    model = _TinyStochasticDepthNet()
    metric = StochasticDepthFLOPsMetric(
        sample_size=1,
        backward_flops_multiplier=2.0,
    )

    model.train()
    _force_all_train_masks(model, 0.0)
    first_batch = torch.randn(2, 3, 8, 8)
    metric.update(first_batch, model(first_batch), torch.zeros(2), model)

    _force_all_train_masks(model, 1.0)
    second_batch = torch.randn(1, 3, 8, 8)
    metric.update(second_batch, model(second_batch), torch.zeros(1), model)

    computed = metric.compute()
    dense_forward = computed[
        "stochastic_depth_dense_reference_forward_flops_per_sample"
    ]
    always_forward = computed[
        "stochastic_depth_always_computed_forward_flops_per_sample"
    ]
    expected_actual_forward = (2.0 * always_forward + dense_forward) / 3.0

    assert computed["stochastic_depth_train_samples"] == pytest.approx(3.0)
    assert computed[
        "stochastic_depth_inference_forward_flops_per_sample"
    ] == pytest.approx(dense_forward)
    assert computed[
        "stochastic_depth_actual_train_forward_flops_per_sample"
    ] == pytest.approx(expected_actual_forward)
    assert computed[
        "stochastic_depth_actual_train_forward_backward_flops_per_sample"
    ] == pytest.approx(3.0 * expected_actual_forward)
    assert computed[
        "stochastic_depth_actual_train_forward_flops_epoch"
    ] == pytest.approx(3.0 * expected_actual_forward)
    assert computed[
        "stochastic_depth_actual_train_forward_backward_flops_epoch"
    ] == pytest.approx(9.0 * expected_actual_forward)
    assert computed["stochastic_depth_actual_flops_skip_ratio"] > 0.0

    # Dense profiling is temporary and must not change the configured schedule.
    assert [
        block.survival_probability
        for block in get_stochastic_depth_blocks(model).values()
    ] == pytest.approx([0.5, 0.25])


def test_dense_pL1_training_matches_inference_reference():
    model = _TinyStochasticDepthNet()
    for block in get_stochastic_depth_blocks(model).values():
        block.survival_probability = 1.0

    model.train()
    batch = torch.randn(2, 3, 8, 8)
    metric = StochasticDepthFLOPsMetric(sample_size=1)
    metric.update(batch, model(batch), torch.zeros(2), model)
    computed = metric.compute()

    dense_forward = computed[
        "stochastic_depth_dense_reference_forward_flops_per_sample"
    ]
    assert computed[
        "stochastic_depth_expected_train_forward_flops_per_sample"
    ] == pytest.approx(dense_forward)
    assert computed[
        "stochastic_depth_actual_train_forward_flops_per_sample"
    ] == pytest.approx(dense_forward)
    assert computed["stochastic_depth_expected_flops_skip_ratio"] == pytest.approx(0.0)
    assert computed["stochastic_depth_actual_flops_skip_ratio"] == pytest.approx(0.0)


def test_inference_flops_count_all_samples_on_the_dense_graph():
    model = _TinyStochasticDepthNet().eval()
    batch = torch.randn(3, 3, 8, 8)
    metric = StochasticDepthFLOPsMetric(sample_size=1)
    metric.update(batch, model(batch), torch.zeros(3), model)
    computed = metric.compute()

    dense_forward = computed[
        "stochastic_depth_dense_reference_forward_flops_per_sample"
    ]
    assert computed["stochastic_depth_inference_samples"] == pytest.approx(3.0)
    assert computed[
        "stochastic_depth_inference_forward_flops_total"
    ] == pytest.approx(3.0 * dense_forward)
    assert "stochastic_depth_actual_train_forward_flops_per_sample" not in computed


def test_flops_sweep_has_dense_reference_then_five_stochastic_depth_runs():
    tuning_name = (
        "stochastic_depth_resnet50_pL_flops_grid_dense_10_09_08_07_05_03_ordered"
    )
    tuning_cfg = OmegaConf.load(CONFIGS_DIR / "tuning" / f"{tuning_name}.yaml")
    tune_cfg = OmegaConf.load(
        CONFIGS_DIR / "tune_stochastic_depth_resnet50_pL_flops_grid.yaml"
    )

    assert tune_cfg.defaults == [
        {"experiment": "stochastic_depth_resnet50_cifar10"},
        {"tuning": tuning_name},
        "_self_",
    ]
    assert tuning_cfg.tuning.n_trials == 6
    assert tuning_cfg.tuning.points_in_order is True

    points = OmegaConf.to_container(tuning_cfg.tuning.points, resolve=False)
    assert [point["model.backbone.final_survival_probability"] for point in points] == [
        1.0,
        0.9,
        0.8,
        0.7,
        0.5,
        0.3,
    ]
    assert points[0]["mlflow.tags.variant"] == "dense_reference"
    assert all(
        point["mlflow.tags.variant"] == "stochastic_depth"
        for point in points[1:]
    )

    metrics_cfg = OmegaConf.load(CONFIGS_DIR / "metrics" / "stochastic_depth.yaml")
    for stage in ("train_metrics", "valid_metrics", "test_metrics"):
        flops_metric = metrics_cfg.metrics[stage][-1]
        assert flops_metric.backward_flops_multiplier == pytest.approx(2.0)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        composed = compose(
            config_name="tune_stochastic_depth_resnet50_pL_flops_grid",
        )

    assert composed.tuning.study_name == tuning_name
    assert composed.tuning.n_trials == 6
    assert composed.model.backbone._target_ == (
        "net_complexity.wrappers.StochasticDepthResNet50"
    )
