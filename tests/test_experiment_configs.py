from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


def test_gumbel_cifar10_uses_article_training_stack_with_original_lambda():
    cfg = OmegaConf.load(CONFIGS_DIR / "experiment" / "gumbel_cifar10.yaml")

    assert cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "cifar_resnet20"},
        {"/method": "gumbel"},
        {"/train": "best_practice"},
        {"/optimizer": "sgd_resnet20"},
        {"/scheduler": "multistep_91_136"},
        {"/metrics": "gumbel"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 1.479470
    assert cfg.mlflow.tags.recipe == "gumbel_cifar10"


def test_stg_cifar10_uses_article_training_stack_with_original_stg_parameters():
    cfg = OmegaConf.load(
        CONFIGS_DIR / "experiment" / "stg_cifar10_best_practice_valid_accuracy.yaml"
    )

    assert cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "cifar_resnet20"},
        {"/method": "stg"},
        {"/train": "best_practice"},
        {"/optimizer": "sgd_resnet20"},
        {"/scheduler": "multistep_91_136"},
        {"/metrics": "stg"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 1.479470
    assert cfg.model.backbone.resnet_block.sigma == 0.5
    assert cfg.model.backbone.resnet_block.init_mu == 1.0
    assert cfg.mlflow.tags.recipe == "stg_cifar10_best_practice_valid_accuracy"


def test_before_refactor_gumbel_cifar10_preserves_old_recipe():
    cfg = OmegaConf.load(CONFIGS_DIR / "experiment" / "before_refactor" / "gumbel_cifar10.yaml")

    assert cfg.defaults == [
        {"/data": "cifar10"},
        {"/model": "cifar_resnet20"},
        {"/method": "gumbel"},
        {"/train": "default"},
        {"/optimizer": "adamw"},
        {"/scheduler": "none"},
        {"/metrics": "gumbel"},
        {"/run_history": "valid_loss_min"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 1.479470
    assert cfg.mlflow.tags.recipe == "before_refactor_gumbel_cifar10"


def test_before_refactor_stg_cifar10_preserves_old_recipe():
    cfg = OmegaConf.load(CONFIGS_DIR / "experiment" / "before_refactor" / "stg_cifar10.yaml")

    assert cfg.defaults == [
        {"/data": "cifar10"},
        {"/model": "cifar_resnet20"},
        {"/method": "stg"},
        {"/train": "default"},
        {"/optimizer": "adamw"},
        {"/scheduler": "multistep_91_136"},
        {"/metrics": "stg"},
        {"/run_history": "valid_loss_min"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 1.479470
    assert cfg.model.backbone.resnet_block.sigma == 0.5
    assert cfg.model.backbone.resnet_block.init_mu == 1.0
    assert cfg.mlflow.tags.recipe == "before_refactor_stg_cifar10"


def test_default_optuna_profile_matches_sgd_based_gumbel_recipe():
    cfg = OmegaConf.load(CONFIGS_DIR / "tuning" / "optuna.yaml")

    assert cfg.tuning.study_name == "gumbel_cifar10_optuna"
    assert cfg.tuning.search_space["optimizer.lr"].low == 0.01
    assert cfg.tuning.search_space["optimizer.lr"].high == 0.3
    assert cfg.tuning.search_space["dataloaders.batch_size"].choices == [128, 256, 512]


def test_before_refactor_stg_cifar10_120_preserves_old_recipe():
    cfg = OmegaConf.load(CONFIGS_DIR / "experiment" / "before_refactor" / "stg_cifar10_120.yaml")

    assert cfg.defaults == [
        {"/data": "cifar10"},
        {"/model": "cifar_resnet20"},
        {"/method": "stg"},
        {"/train": "default"},
        {"/optimizer": "adamw"},
        {"/scheduler": "multistep_91"},
        {"/metrics": "stg"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 1.479470
    assert cfg.model.backbone.resnet_block.sigma == 0.5
    assert cfg.model.backbone.resnet_block.init_mu == 1.0
    assert cfg.mlflow.tags.recipe == "before_refactor_stg_cifar10_120"
