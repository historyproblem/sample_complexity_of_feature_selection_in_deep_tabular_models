from __future__ import annotations

import importlib.util
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
SEARCH_MODULE_PATH = REPO_ROOT / "src" / "net_complexity" / "tuning" / "search.py"

SPEC = importlib.util.spec_from_file_location("net_complexity.tuning.search", SEARCH_MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

build_grid_search_space = MODULE.build_grid_search_space
count_grid_trials = MODULE.count_grid_trials


def test_stg_grid_tuning_config_uses_exact_45_point_search_space():
    cfg = OmegaConf.load(CONFIGS_DIR / "tuning" / "stg_lambda_initmu_grid_sigma05.yaml")

    assert cfg.tuning.mode == "grid"
    assert cfg.tuning.study_name == "stg_lambda_initmu_grid_sigma05"
    assert cfg.tuning.n_trials == 45
    assert cfg.tuning.pruner._target_ == "optuna.pruners.NopPruner"
    assert set(cfg.tuning.search_space.keys()) == {
        "model.lambda_coef",
        "model.backbone.resnet_block.init_mu",
    }

    grid_space = build_grid_search_space(
        OmegaConf.to_container(cfg.tuning.search_space, resolve=True),
    )

    assert grid_space == {
        "model.lambda_coef": [0.03, 0.06, 0.1, 0.15, 0.22, 0.33, 0.5, 0.75, 1.0],
        "model.backbone.resnet_block.init_mu": [0.5, 0.7, 0.9, 1.1, 1.3],
    }
    assert count_grid_trials(grid_space) == 45


def test_stg_experiment_recipe_is_composed_from_layered_config_groups():
    cfg = OmegaConf.load(CONFIGS_DIR / "experiment" / "stg_cifar10_120.yaml")

    assert cfg.defaults == [
        {"/data": "cifar10"},
        {"/model": "cifar_resnet20"},
        {"/method": "stg"},
        {"/train": "long"},
        {"/optimizer": "adamw"},
        {"/scheduler": "multistep_91"},
        {"/metrics": "stg"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert OmegaConf.select(cfg, "model.backbone.resnet_block.sigma") == 0.5
    assert OmegaConf.select(cfg, "model.backbone.resnet_block.init_mu") == 1.0


def test_stg_tune_entrypoint_pins_sigma_and_uses_grid_config():
    cfg = OmegaConf.load(CONFIGS_DIR / "tune_stg_cifar10_120.yaml")

    assert cfg.defaults[0]["experiment"] == "stg_cifar10_120"
    assert cfg.defaults[1]["tuning"] == "stg_lambda_initmu_grid_sigma05"
    assert OmegaConf.select(cfg, "model.backbone.resnet_block.sigma") == 0.5
