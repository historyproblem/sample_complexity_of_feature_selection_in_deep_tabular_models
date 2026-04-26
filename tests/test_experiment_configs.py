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
        {"/data/before_refactor": "cifar10"},
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
        {"/data/before_refactor": "cifar10"},
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


def test_best_practice_resnet50_aig_baseline_uses_full_cifar_recipe_with_zero_lambda():
    cfg = OmegaConf.load(CONFIGS_DIR / "experiment" / "best_practice_resnet50_aig_on_cifar10.yaml")

    assert cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "resnet50"},
        {"/method": "aig"},
        {"/train": "resnet50_best_practice"},
        {"/optimizer": "sgd_resnet50"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "aig"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 0.0
    assert cfg.model.backbone.resnet_block.temperature == 1.0
    assert cfg.mlflow.tags.recipe == "best_practice_resnet50_aig_on_cifar10"


def test_best_practice_resnet50_plain_baseline_matches_requested_cifar_style_recipe():
    cfg = OmegaConf.load(CONFIGS_DIR / "experiment" / "best_practice_resnet50_on_cifar10.yaml")
    model_cfg = OmegaConf.load(CONFIGS_DIR / "model" / "resnet50.yaml")
    optimizer_cfg = OmegaConf.load(CONFIGS_DIR / "optimizer" / "sgd_resnet50.yaml")
    scheduler_cfg = OmegaConf.load(CONFIGS_DIR / "scheduler" / "cosine_200.yaml")
    train_cfg = OmegaConf.load(CONFIGS_DIR / "train" / "resnet50_best_practice.yaml")
    data_cfg = OmegaConf.load(CONFIGS_DIR / "data" / "cifar10_best_practice.yaml")

    assert cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "resnet50"},
        {"/method": "plain"},
        {"/train": "resnet50_best_practice"},
        {"/optimizer": "sgd_resnet50"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "classification"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 0.0
    assert cfg.mlflow.tags.recipe == "best_practice_resnet50_on_cifar10"

    assert model_cfg.model.backbone.stem_kernel_size == 3
    assert model_cfg.model.backbone.stem_stride == 1
    assert model_cfg.model.backbone.stem_padding == 1
    assert model_cfg.model.backbone.use_maxpool is False
    assert optimizer_cfg.optimizer.lr == 0.1
    assert optimizer_cfg.optimizer.momentum == 0.9
    assert optimizer_cfg.optimizer.weight_decay == 0.0005
    assert optimizer_cfg.optimizer.nesterov is False
    assert scheduler_cfg.scheduler._target_ == "torch.optim.lr_scheduler.CosineAnnealingLR"
    assert scheduler_cfg.scheduler.T_max == 200
    assert scheduler_cfg.scheduler.eta_min == 0.0
    assert train_cfg.training_arguments.num_epochs == 200
    assert data_cfg.dataloaders.batch_size == 128


def test_aig_optuna_profile_tunes_temperature_around_zero_lambda_baseline():
    cfg = OmegaConf.load(CONFIGS_DIR / "tuning" / "aig_cifar10_optuna120.yaml")

    assert cfg.training_arguments.num_epochs == 120
    assert cfg.tuning.study_name == "aig_cifar10_120_optuna"
    assert cfg.tuning.search_space["model.lambda_coef"].low == 0.0001
    assert cfg.tuning.search_space["model.lambda_coef"].high == 1.0
    assert cfg.tuning.search_space["model.backbone.resnet_block.temperature"].low == 0.5
    assert cfg.tuning.search_space["model.backbone.resnet_block.temperature"].high == 2.0
    assert cfg.tuning.search_space["dataloaders.batch_size"].choices == [64, 128, 256]


def test_aig_lambda_grid_150ep_includes_baseline_and_requested_lambda_values_without_pruning():
    cfg = OmegaConf.load(CONFIGS_DIR / "tuning" / "aig_lambda_grid_150ep.yaml")

    assert cfg.training_arguments.num_epochs == 150
    assert cfg.tuning.mode == "grid"
    assert cfg.tuning.study_name == "aig_lambda_grid_150ep"
    assert cfg.tuning.n_trials == 4
    assert cfg.tuning.sampler is None
    assert cfg.tuning.pruner._target_ == "optuna.pruners.NopPruner"
    assert cfg.tuning.search_space["model.lambda_coef"].choices == [0.0, 0.01, 0.05, 0.5]


def test_before_refactor_stg_cifar10_120_preserves_old_recipe():
    cfg = OmegaConf.load(CONFIGS_DIR / "experiment" / "before_refactor" / "stg_cifar10_120.yaml")

    assert cfg.defaults == [
        {"/data/before_refactor": "cifar10"},
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
