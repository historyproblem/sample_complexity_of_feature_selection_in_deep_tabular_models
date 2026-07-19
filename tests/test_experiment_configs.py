from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


def _assert_clean_aig_adaptive_lambda(training_arguments) -> None:
    adaptive = training_arguments.adaptive_lambda
    assert "lambda_warmup" not in training_arguments
    assert "adaptive_log_step_enabled" not in adaptive
    assert "prune_rate_low_per_epoch" not in adaptive
    assert "prune_rate_high_per_epoch" not in adaptive
    assert "log_step_boost_factor" not in adaptive
    assert "log_step_max_boost_level" not in adaptive
    assert "adaptive_log_step_max_epoch" not in adaptive
    assert "recovery" not in adaptive
    assert "collapse_guard" not in training_arguments


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
    assert cfg.model.gumbel_init_mode == "auto"
    assert cfg.model.backbone.resnet_block.beta == 1.0
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
    assert cfg.model.gumbel_init_mode == "auto"
    assert cfg.model.backbone.resnet_block.beta == 1.0
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


def test_gumbel_method_defaults_zero_lambda_bypass_to_true():
    cfg = OmegaConf.load(CONFIGS_DIR / "method" / "gumbel.yaml")

    assert cfg.optimizer.gate_weight_decay_scale is None
    assert cfg.model.gumbel_init_mode == "auto"
    assert cfg.model.bypass_on_zero_lambda is True
    assert cfg.model.backbone.resnet_block.beta == 1.0
    assert cfg.model.backbone.resnet_block.force_ones_mask is False
    assert cfg.model.backbone.resnet_block.deterministic_soft_mask is False
    assert cfg.model.backbone.resnet_block.deterministic_hard_mask is False
    assert cfg.model.backbone.resnet_block.train_gate_mode is None
    assert cfg.model.backbone.resnet_block.eval_gate_mode is None
    assert cfg.model.backbone.resnet_block.gate_threshold == 0.5


def test_masked_gumbel_method_matches_current_gumbel_defaults():
    cfg = OmegaConf.load(CONFIGS_DIR / "method" / "gumbel_masked.yaml")

    assert cfg.optimizer.gate_weight_decay_scale is None
    assert cfg.model.gumbel_init_mode == "auto"
    assert cfg.model.bypass_on_zero_lambda is True
    assert (
        cfg.model.backbone.resnet_block._target_
        == "net_complexity.wrappers.CIFARMaskedGumbelBasicBlock"
    )
    assert cfg.model.backbone.resnet_block.beta == 1.0
    assert cfg.model.backbone.resnet_block.force_ones_mask is False
    assert cfg.model.backbone.resnet_block.deterministic_soft_mask is False
    assert cfg.model.backbone.resnet_block.deterministic_hard_mask is False
    assert cfg.model.backbone.resnet_block.train_gate_mode is None
    assert cfg.model.backbone.resnet_block.eval_gate_mode is None
    assert cfg.model.backbone.resnet_block.gate_threshold == 0.5


def test_channel_pruning_and_layer_skipping_experiment_configs():
    soft_cfg = OmegaConf.load(CONFIGS_DIR / "experiment" / "gumbel_cifar10_pruned.yaml")
    structural_cfg = OmegaConf.load(
        CONFIGS_DIR / "experiment" / "gumbel_cifar10_structural_pruned.yaml"
    )
    layer_cfg = OmegaConf.load(
        CONFIGS_DIR / "experiment" / "resnet50_layer_skipping_cifar10.yaml"
    )

    assert soft_cfg.defaults[2] == {"/method": "gumbel_masked"}
    assert soft_cfg.channel_pruning.enabled is True
    assert soft_cfg.channel_pruning.mode == "explicit"
    assert structural_cfg.channel_pruning.structural is True
    assert structural_cfg.mlflow.tags.method == "structural_pruning"
    assert layer_cfg.layer_skipping.enabled is True
    assert layer_cfg.mlflow.tags.method == "layer_skipping"


def test_cyclic_aig_configs_use_current_adaptive_lambda_defaults():
    cfg = OmegaConf.load(CONFIGS_DIR / "experiment" / "cyclic_aig_resnet50_cifar10.yaml")

    assert cfg.defaults[3] == {"/train": "aig_adaptive_lambda"}
    assert cfg.model.lambda_coef == 1e-6
    assert cfg.model.backbone.resnet_block.gate_regularization == "l2_gate"
    assert cfg.training_arguments.adaptive_lambda.lambda_max == 10.0
    assert cfg.training_arguments.adaptive_lambda.log_step_init == "auto"
    assert "lambda_warmup" not in cfg.training_arguments
    assert "adaptive_log_step_enabled" not in cfg.training_arguments.adaptive_lambda
    assert cfg.mlflow.tags.gate_regularization == "l2_gate"


def test_train_profiles_enable_batchnorm_recalibration_by_default():
    default_cfg = OmegaConf.load(CONFIGS_DIR / "train" / "default.yaml")
    best_practice_cfg = OmegaConf.load(CONFIGS_DIR / "train" / "best_practice.yaml")
    resnet50_cfg = OmegaConf.load(CONFIGS_DIR / "train" / "resnet50_best_practice.yaml")
    removed_adaptive_fields = {
        "soft_step_shrink",
        "hard_step_shrink",
        "collapse_acc_threshold",
        "collapse_loss_threshold",
        "collapse_zero_prob_threshold",
        "collapse_acc_drop_threshold",
        "rollback_check_every_epochs",
        "rollback_acc_drop_threshold",
        "rollback_compare_epoch_lookback",
        "rollback_epoch_lookback",
        "lambda_increase_cooldown_epochs",
        "rollback_on_degradation",
        "rollback_on_collapse",
        "max_rollbacks",
        "freeze_on_rollback_limit",
    }

    for cfg in (default_cfg, best_practice_cfg, resnet50_cfg):
        assert cfg.training_arguments.adaptive_lambda.enabled is False
        assert cfg.training_arguments.adaptive_lambda.warmup_epochs == 10
        assert cfg.training_arguments.adaptive_lambda.update_every_epochs == 3
        assert cfg.training_arguments.adaptive_lambda.acc_window == 3
        assert cfg.training_arguments.adaptive_lambda.baseline_history_dir is None
        assert cfg.training_arguments.adaptive_lambda.lambda_min == 1e-8
        assert cfg.training_arguments.adaptive_lambda.lambda_max == 80.0
        assert removed_adaptive_fields.isdisjoint(cfg.training_arguments.adaptive_lambda)
        assert cfg.training_arguments.adaptive_lambda.adaptive_log_step_enabled is True
        assert cfg.training_arguments.adaptive_lambda.prune_rate_low_per_epoch == 0.02
        assert cfg.training_arguments.adaptive_lambda.prune_rate_high_per_epoch == 0.07
        assert cfg.training_arguments.adaptive_lambda.log_step_boost_factor == 2.0
        assert cfg.training_arguments.adaptive_lambda.log_step_max_boost_level == 2
        assert cfg.training_arguments.adaptive_lambda.adaptive_log_step_max_epoch is None
        assert cfg.training_arguments.adaptive_lambda.recovery.enabled is True
        assert cfg.training_arguments.adaptive_lambda.recovery.min_epoch == 80
        assert cfg.training_arguments.adaptive_lambda.recovery.recovery_epochs == 5
        assert cfg.training_arguments.adaptive_lambda.recovery.open_bias_start == 0.15
        assert cfg.training_arguments.adaptive_lambda.recovery.p_open_min == 0.02
        assert cfg.training_arguments.adaptive_lambda.recovery.p_open_max == 0.50
        assert cfg.training_arguments.batchnorm_recalibration.enabled is True
        assert cfg.training_arguments.batchnorm_recalibration.num_batches == 200
        assert cfg.training_arguments.batchnorm_recalibration.reset_running_stats is True
        assert cfg.training_arguments.batchnorm_recalibration.train_gate_mode == "deterministic_hard"
        assert cfg.training_arguments.batchnorm_recalibration.eval_gate_mode == "deterministic_hard"


def test_aig_adaptive_lambda_train_profile_keeps_only_bn_recalibration_extra():
    cfg = OmegaConf.load(CONFIGS_DIR / "train" / "aig_adaptive_lambda.yaml")

    assert cfg.training_arguments.adaptive_lambda.enabled is False
    assert cfg.training_arguments.adaptive_lambda.warmup_epochs == 0
    assert cfg.training_arguments.adaptive_lambda.update_every_epochs == 2
    assert cfg.training_arguments.adaptive_lambda.lambda_max == 10.0
    assert cfg.training_arguments.adaptive_lambda.log_step_init == "auto"
    _assert_clean_aig_adaptive_lambda(cfg.training_arguments)
    assert cfg.training_arguments.batchnorm_recalibration.enabled is True
    assert cfg.training_arguments.batchnorm_recalibration.num_batches == 200
    assert cfg.training_arguments.batchnorm_recalibration.reset_running_stats is True


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


def test_aig_method_preserves_dynamic_gates_for_existing_zero_lambda_recipes():
    cfg = OmegaConf.load(CONFIGS_DIR / "method" / "aig.yaml")

    assert cfg.model.bypass_on_zero_lambda is False


def test_best_practice_resnet20_gumbel_baseline_sets_beta_to_one():
    cfg = OmegaConf.load(CONFIGS_DIR / "experiment" / "best_practice_resnet20_on_cifar10.yaml")

    assert cfg.model.lambda_coef == 0.0
    assert cfg.model.backbone.resnet_block.beta == 1.0


def test_best_practice_resnet50_gumbel_baseline_uses_full_cifar_recipe_with_zero_lambda():
    cfg = OmegaConf.load(CONFIGS_DIR / "experiment" / "best_practice_resnet50_gumbel_on_cifar10.yaml")

    assert cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "resnet50"},
        {"/method": "gumbel"},
        {"/train": "resnet50_best_practice"},
        {"/optimizer": "adamw"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "gumbel_resnet50"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.optimizer.gate_weight_decay_scale == 20.0
    assert cfg.model.lambda_coef == 0.0
    assert cfg.model.gumbel_init_mode == "paper_resnet50"
    assert cfg.model.bypass_on_zero_lambda is True
    assert cfg.model.backbone.resnet_block._target_ == "net_complexity.wrappers.GumbelBottleneckLayer"
    assert cfg.model.backbone.resnet_block.temperature == 1.0
    assert cfg.model.backbone.resnet_block.beta == 1.0
    assert cfg.run_history.log_gate_history is True
    assert cfg.mlflow.enabled is False
    assert cfg.mlflow.tags.recipe == "best_practice_resnet50_gumbel_on_cifar10"


def test_resnet50_gumbel_adaptive_lambda_recovery_v1_config_enables_recovery():
    cfg = OmegaConf.load(
        CONFIGS_DIR / "experiment" / "resnet50_gumbel_adaptive_lambda_recovery_v1.yaml"
    )

    assert cfg.training_arguments.adaptive_lambda.enabled is True
    adaptive = cfg.training_arguments.adaptive_lambda
    assert adaptive.adaptive_log_step_enabled is True
    assert adaptive.prune_rate_low_per_epoch == 0.02
    assert adaptive.prune_rate_high_per_epoch == 0.07
    assert adaptive.log_step_boost_factor == 2.0
    assert adaptive.log_step_max_boost_level == 2
    assert adaptive.adaptive_log_step_max_epoch is None
    recovery = cfg.training_arguments.adaptive_lambda.recovery
    assert recovery.enabled is True
    assert recovery.decay_lambda is False
    assert recovery.open_bias_start == 0.15
    assert recovery.open_bias_decay == 0.90
    assert recovery.p_open_min == 0.02
    assert recovery.p_open_max == 0.50
    assert recovery.recovery_epochs == 5
    assert recovery.patience == 5
    assert recovery.max_reopen_delta == 0.02
    assert recovery.min_epoch == 80
    assert recovery.require_slow_recovery is True
    assert recovery.recovery_slope_window == 5
    assert recovery.min_acc_delta_over_window == 0.005
    assert recovery.use_zero_prob_filter is True
    assert recovery.zero_prob_window == 10
    assert recovery.zero_prob_delta_min == 0.02


def test_best_practice_resnet50_gumbel_paper_init_lambda0_disables_bypass_and_warmup():
    cfg = OmegaConf.load(
        CONFIGS_DIR / "experiment" / "best_practice_resnet50_gumbel_paper_init_lambda0_on_cifar10.yaml"
    )

    assert cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "resnet50"},
        {"/method": "gumbel"},
        {"/train": "resnet50_best_practice"},
        {"/optimizer": "sgd_resnet50"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "gumbel_resnet50"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 0.0
    assert cfg.model.gumbel_init_mode == "paper"
    assert cfg.model.bypass_on_zero_lambda is False
    assert cfg.model.backbone.resnet_block._target_ == "net_complexity.wrappers.GumbelBottleneckLayer"
    assert cfg.model.backbone.resnet_block.temperature == 1.0
    assert cfg.model.backbone.resnet_block.beta == 1.0
    assert cfg.training_arguments.lambda_warmup.enabled is False
    assert cfg.run_history.log_gate_history is True
    assert cfg.mlflow.enabled is False
    assert cfg.mlflow.tags.recipe == "best_practice_resnet50_gumbel_paper_init_lambda0_on_cifar10"


def test_best_practice_resnet50_gumbel_paper_resnet50_init_lambda0_uses_resnet50_specific_init():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "experiment"
        / "best_practice_resnet50_gumbel_paper_resnet50_init_lambda0_on_cifar10.yaml"
    )

    assert cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "resnet50"},
        {"/method": "gumbel"},
        {"/train": "resnet50_best_practice"},
        {"/optimizer": "sgd_resnet50"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "gumbel_resnet50"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 0.0
    assert cfg.model.gumbel_init_mode == "paper_resnet50"
    assert cfg.model.bypass_on_zero_lambda is False
    assert cfg.model.backbone.resnet_block._target_ == "net_complexity.wrappers.GumbelBottleneckLayer"
    assert cfg.model.backbone.resnet_block.temperature == 1.0
    assert cfg.model.backbone.resnet_block.beta == 1.0
    assert cfg.training_arguments.lambda_warmup.enabled is False
    assert cfg.run_history.log_gate_history is True
    assert cfg.mlflow.enabled is False
    assert (
        cfg.mlflow.tags.recipe
        == "best_practice_resnet50_gumbel_paper_resnet50_init_lambda0_on_cifar10"
    )


def test_best_practice_resnet50_gumbel_paper_resnet50_ramp30_lambda001_starts_ramp_from_epoch_one():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "experiment"
        / "best_practice_resnet50_gumbel_paper_resnet50_ramp30_lambda001_on_cifar10.yaml"
    )

    assert cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "resnet50"},
        {"/method": "gumbel"},
        {"/train": "resnet50_best_practice"},
        {"/optimizer": "sgd_resnet50"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "gumbel_resnet50"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 0.0
    assert cfg.model.gumbel_init_mode == "paper_resnet50"
    assert cfg.model.bypass_on_zero_lambda is False
    assert cfg.model.backbone.resnet_block._target_ == "net_complexity.wrappers.GumbelBottleneckLayer"
    assert cfg.model.backbone.resnet_block.temperature == 1.0
    assert cfg.model.backbone.resnet_block.beta == 1.0
    assert cfg.training_arguments.lambda_warmup.enabled is True
    assert cfg.training_arguments.lambda_warmup.start_epoch == 1
    assert cfg.training_arguments.lambda_warmup.initial_lambda_coef == 0.0
    assert cfg.training_arguments.lambda_warmup.target_lambda_coef == 0.01
    assert cfg.training_arguments.lambda_warmup.ramp_epochs == 30
    assert cfg.training_arguments.lambda_warmup.bypass_during_warmup is False
    assert cfg.training_arguments.batchnorm_recalibration.enabled is True
    assert cfg.run_history.log_gate_history is True
    assert cfg.mlflow.enabled is False
    assert (
        cfg.mlflow.tags.recipe
        == "best_practice_resnet50_gumbel_paper_resnet50_ramp30_lambda001_on_cifar10"
    )


def test_resnet50_adaptive_lambda_experiments_use_requested_initial_lambda_grid():
    expected = {
        "resnet50_adaptive_lambda_init5.yaml": 5.0,
        "resnet50_adaptive_lambda_init15.yaml": 15.0,
        "resnet50_adaptive_lambda_init25.yaml": 25.0,
        "resnet50_adaptive_lambda_init35.yaml": 35.0,
    }

    for config_name, lambda_init in expected.items():
        cfg = OmegaConf.load(CONFIGS_DIR / "experiment" / config_name)
        assert cfg.defaults == [
            {"/data": "cifar10_best_practice"},
            {"/model": "resnet50"},
            {"/method": "gumbel"},
            {"/train": "resnet50_best_practice"},
            {"/optimizer": "sgd_resnet50"},
            {"/scheduler": "cosine_200"},
            {"/metrics": "gumbel_resnet50"},
            {"/run_history": "valid_accuracy_max"},
            {"/tracking": "default"},
            "_self_",
        ]
        assert cfg.model.lambda_coef == lambda_init
        assert cfg.model.gumbel_init_mode == "paper_resnet50"
        assert cfg.model.bypass_on_zero_lambda is False
        assert cfg.training_arguments.lambda_warmup.enabled is False
        assert cfg.training_arguments.adaptive_lambda.enabled is True
        assert cfg.training_arguments.adaptive_lambda.warmup_epochs == 10
        assert cfg.training_arguments.adaptive_lambda.update_every_epochs == 3
        assert cfg.training_arguments.adaptive_lambda.acc_window == 3
        assert (
            cfg.training_arguments.adaptive_lambda.baseline_history_dir
            == "outputs/baselines/resnet50_gumbel_cifar10_no_pruning_temp_1.0"
        )
        assert cfg.training_arguments.adaptive_lambda.lambda_min == 1e-8
        assert cfg.training_arguments.adaptive_lambda.lambda_max == 80.0
        assert cfg.training_arguments.adaptive_lambda.log_step_init == 0.6931471805599453
        assert cfg.training_arguments.adaptive_lambda.log_step_min == 0.04879016416943205
        assert cfg.training_arguments.adaptive_lambda.soft_drop == 0.02
        assert cfg.training_arguments.adaptive_lambda.hard_drop == 0.05
        assert cfg.run_history.log_gate_history is True
        assert cfg.mlflow.enabled is False
        assert cfg.mlflow.tags.recipe == config_name.removesuffix(".yaml")


def test_resnet50_adaptive_lambda_init1em6_no_warmup_uses_requested_recipe():
    cfg = OmegaConf.load(
        CONFIGS_DIR / "experiment" / "resnet50_adaptive_lambda_init1em6_no_warmup.yaml"
    )

    assert cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "resnet50"},
        {"/method": "gumbel"},
        {"/train": "resnet50_best_practice"},
        {"/optimizer": "sgd_resnet50"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "gumbel_resnet50"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 1.0e-6
    assert cfg.model.gumbel_init_mode == "paper_resnet50"
    assert cfg.model.bypass_on_zero_lambda is False
    assert cfg.training_arguments.lambda_warmup.enabled is False
    assert cfg.training_arguments.adaptive_lambda.enabled is True
    assert cfg.training_arguments.adaptive_lambda.warmup_epochs == 0
    assert cfg.training_arguments.adaptive_lambda.update_every_epochs == 3
    assert cfg.training_arguments.adaptive_lambda.acc_window == 3
    assert (
        cfg.training_arguments.adaptive_lambda.baseline_history_dir
        == "outputs/baselines/resnet50_gumbel_cifar10_no_pruning_temp_1.0"
    )
    assert cfg.run_history.log_gate_history is True
    assert cfg.mlflow.enabled is False
    assert cfg.mlflow.tags.recipe == "resnet50_adaptive_lambda_init1em6_no_warmup"


def test_resnet50_adaptive_lambda_step125_no_warmup_base_uses_requested_recipe():
    cfg = OmegaConf.load(
        CONFIGS_DIR / "experiment" / "resnet50_adaptive_lambda_step125_no_warmup_base.yaml"
    )

    assert cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "resnet50"},
        {"/method": "gumbel"},
        {"/train": "resnet50_best_practice"},
        {"/optimizer": "sgd_resnet50"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "gumbel_resnet50"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 1.0e-3
    assert cfg.model.gumbel_init_mode == "paper_resnet50"
    assert cfg.model.bypass_on_zero_lambda is False
    assert cfg.training_arguments.lambda_warmup.enabled is False
    assert cfg.training_arguments.adaptive_lambda.enabled is True
    assert cfg.training_arguments.adaptive_lambda.warmup_epochs == 0
    assert cfg.training_arguments.adaptive_lambda.update_every_epochs == 1
    assert cfg.training_arguments.adaptive_lambda.lambda_max == 100.0
    assert cfg.training_arguments.adaptive_lambda.log_step_init == 0.22314355131420976
    assert cfg.training_arguments.adaptive_lambda.hard_drop == 0.04
    assert cfg.run_history.log_gate_history is True
    assert cfg.mlflow.enabled is False
    assert cfg.mlflow.tags.recipe == "resnet50_adaptive_lambda_step125_no_warmup_base"


def test_best_practice_resnet50_gumbel_gate_modes_lambda001_uses_requested_base_recipe():
    cfg = OmegaConf.load(
        CONFIGS_DIR / "experiment" / "best_practice_resnet50_gumbel_gate_modes_lambda001_on_cifar10.yaml"
    )

    assert cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "resnet50"},
        {"/method": "gumbel"},
        {"/train": "resnet50_best_practice"},
        {"/optimizer": "sgd_resnet50"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "gumbel_resnet50"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 0.01
    assert cfg.model.gumbel_init_mode == "paper"
    assert cfg.model.bypass_on_zero_lambda is False
    assert cfg.model.backbone.resnet_block._target_ == "net_complexity.wrappers.GumbelBottleneckLayer"
    assert cfg.model.backbone.resnet_block.temperature == 1.0
    assert cfg.model.backbone.resnet_block.beta == 1.0
    assert cfg.training_arguments.lambda_warmup.enabled is False
    assert cfg.training_arguments.gate_mode_schedule.enabled is False
    assert cfg.run_history.log_gate_history is True
    assert cfg.mlflow.enabled is False
    assert cfg.mlflow.tags.recipe == "best_practice_resnet50_gumbel_gate_modes_lambda001_on_cifar10"


def test_best_practice_resnet50_gumbel_bn_recalibration_lambda001_uses_requested_recipe():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "experiment"
        / "best_practice_resnet50_gumbel_bn_recalibration_lambda001_on_cifar10.yaml"
    )

    assert cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "resnet50"},
        {"/method": "gumbel"},
        {"/train": "resnet50_best_practice"},
        {"/optimizer": "sgd_resnet50"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "gumbel_resnet50"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 0.01
    assert cfg.model.gumbel_init_mode == "paper"
    assert cfg.model.bypass_on_zero_lambda is False
    assert cfg.model.backbone.resnet_block._target_ == "net_complexity.wrappers.GumbelBottleneckLayer"
    assert cfg.model.backbone.resnet_block.temperature == 1.0
    assert cfg.model.backbone.resnet_block.beta == 1.0
    assert cfg.training_arguments.lambda_warmup.enabled is False
    assert cfg.training_arguments.gate_mode_schedule.enabled is False
    assert cfg.training_arguments.batchnorm_recalibration.enabled is True
    assert cfg.training_arguments.batchnorm_recalibration.num_batches == 200
    assert cfg.training_arguments.batchnorm_recalibration.train_gate_mode == "deterministic_hard"
    assert cfg.training_arguments.batchnorm_recalibration.eval_gate_mode == "deterministic_hard"
    assert cfg.run_history.log_gate_history is True
    assert cfg.mlflow.enabled is False
    assert (
        cfg.mlflow.tags.recipe
        == "best_practice_resnet50_gumbel_bn_recalibration_lambda001_on_cifar10"
    )


def test_best_practice_resnet20_gumbel_warmup30_lambda1479470_uses_requested_recipe():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "experiment"
        / "best_practice_resnet20_gumbel_warmup30_lambda1479470_on_cifar10.yaml"
    )

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
    assert cfg.model.lambda_coef == 0.0
    assert cfg.model.gumbel_init_mode == "fully_open"
    assert cfg.training_arguments.lambda_warmup.enabled is True
    assert cfg.training_arguments.lambda_warmup.start_epoch == 30
    assert cfg.training_arguments.lambda_warmup.initial_lambda_coef == 0.0
    assert cfg.training_arguments.lambda_warmup.target_lambda_coef == 1.479470
    assert cfg.training_arguments.lambda_warmup.ramp_epochs == 30
    assert cfg.training_arguments.lambda_warmup.bypass_during_warmup is False
    assert (
        cfg.mlflow.tags.recipe
        == "best_practice_resnet20_gumbel_warmup30_lambda1479470_on_cifar10"
    )


def test_best_practice_resnet50_gumbel_warmup30_noise_scaling_only_lambda001_uses_requested_recipe():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "experiment"
        / "best_practice_resnet50_gumbel_warmup30_noise_scaling_only_lambda001_on_cifar10.yaml"
    )

    assert cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "resnet50"},
        {"/method": "gumbel"},
        {"/train": "resnet50_best_practice"},
        {"/optimizer": "sgd_resnet50"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "gumbel_resnet50"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 0.0
    assert cfg.model.gumbel_init_mode == "fully_open"
    assert cfg.model.backbone.resnet_block.temperature == 1.0
    assert cfg.model.backbone.resnet_block.beta == 0.15
    assert cfg.optimizer.gate_weight_decay_scale is None
    assert cfg.training_arguments.lambda_warmup.enabled is True
    assert cfg.training_arguments.lambda_warmup.start_epoch == 30
    assert cfg.training_arguments.lambda_warmup.initial_lambda_coef == 0.0
    assert cfg.training_arguments.lambda_warmup.target_lambda_coef == 0.01
    assert cfg.training_arguments.lambda_warmup.ramp_epochs == 30
    assert cfg.training_arguments.lambda_warmup.bypass_during_warmup is False
    assert cfg.training_arguments.batchnorm_recalibration.enabled is True
    assert cfg.training_arguments.batchnorm_recalibration.num_batches == 200
    assert cfg.training_arguments.batchnorm_recalibration.train_gate_mode == "deterministic_hard"
    assert cfg.training_arguments.batchnorm_recalibration.eval_gate_mode == "deterministic_hard"
    assert cfg.run_history.log_gate_history is True
    assert (
        cfg.mlflow.tags.recipe
        == "best_practice_resnet50_gumbel_warmup30_noise_scaling_only_lambda001_on_cifar10"
    )


def test_best_practice_resnet50_gumbel_warmup30_gate_weight_decay_only_lambda001_uses_requested_recipe():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "experiment"
        / "best_practice_resnet50_gumbel_warmup30_gate_weight_decay_only_lambda001_on_cifar10.yaml"
    )

    assert cfg.model.lambda_coef == 0.0
    assert cfg.model.gumbel_init_mode == "fully_open"
    assert cfg.model.backbone.resnet_block.temperature == 1.0
    assert cfg.model.backbone.resnet_block.beta == 1.0
    assert cfg.optimizer.gate_weight_decay_scale == 20.0
    assert cfg.training_arguments.lambda_warmup.enabled is True
    assert cfg.training_arguments.lambda_warmup.target_lambda_coef == 0.01
    assert cfg.training_arguments.lambda_warmup.ramp_epochs == 30
    assert cfg.training_arguments.batchnorm_recalibration.enabled is True
    assert cfg.run_history.log_gate_history is True
    assert (
        cfg.mlflow.tags.recipe
        == "best_practice_resnet50_gumbel_warmup30_gate_weight_decay_only_lambda001_on_cifar10"
    )


def test_best_practice_resnet50_gumbel_warmup30_noise_scaling_and_gate_weight_decay_lambda001_uses_requested_recipe():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "experiment"
        / "best_practice_resnet50_gumbel_warmup30_noise_scaling_and_gate_weight_decay_lambda001_on_cifar10.yaml"
    )

    assert cfg.model.lambda_coef == 0.0
    assert cfg.model.gumbel_init_mode == "fully_open"
    assert cfg.model.backbone.resnet_block.temperature == 1.0
    assert cfg.model.backbone.resnet_block.beta == 0.15
    assert cfg.optimizer.gate_weight_decay_scale == 20.0
    assert cfg.training_arguments.lambda_warmup.enabled is True
    assert cfg.training_arguments.lambda_warmup.target_lambda_coef == 0.01
    assert cfg.training_arguments.lambda_warmup.ramp_epochs == 30
    assert cfg.training_arguments.batchnorm_recalibration.enabled is True
    assert cfg.training_arguments.batchnorm_recalibration.num_batches == 200
    assert cfg.training_arguments.batchnorm_recalibration.train_gate_mode == "deterministic_hard"
    assert cfg.training_arguments.batchnorm_recalibration.eval_gate_mode == "deterministic_hard"
    assert cfg.run_history.log_gate_history is True
    assert (
        cfg.mlflow.tags.recipe
        == "best_practice_resnet50_gumbel_warmup30_noise_scaling_and_gate_weight_decay_lambda001_on_cifar10"
    )


def test_best_practice_resnet50_gumbel_warmup30_baseline_lambda001_uses_requested_recipe():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "experiment"
        / "best_practice_resnet50_gumbel_warmup30_baseline_lambda001_on_cifar10.yaml"
    )

    assert cfg.model.lambda_coef == 0.0
    assert cfg.model.gumbel_init_mode == "fully_open"
    assert cfg.model.backbone.resnet_block.temperature == 1.0
    assert cfg.model.backbone.resnet_block.beta == 1.0
    assert cfg.optimizer.gate_weight_decay_scale is None
    assert cfg.training_arguments.lambda_warmup.enabled is True
    assert cfg.training_arguments.lambda_warmup.target_lambda_coef == 0.01
    assert cfg.training_arguments.lambda_warmup.ramp_epochs == 30
    assert cfg.training_arguments.batchnorm_recalibration.enabled is True
    assert cfg.run_history.log_gate_history is True
    assert (
        cfg.mlflow.tags.recipe
        == "best_practice_resnet50_gumbel_warmup30_baseline_lambda001_on_cifar10"
    )


def test_gumbel_resnet50_metrics_disable_per_channel_zero_probability_logging():
    cfg = OmegaConf.load(CONFIGS_DIR / "metrics" / "gumbel_resnet50.yaml")

    assert cfg.metrics.train_metrics[2].log_channel_zero_probs is False
    assert cfg.metrics.valid_metrics[2].log_channel_zero_probs is False
    assert cfg.metrics.test_metrics[2].log_channel_zero_probs is False


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


def test_gumbel_resnet50_lambda_grid_150ep_includes_baseline_and_requested_lambda_values_without_pruning():
    cfg = OmegaConf.load(CONFIGS_DIR / "tuning" / "gumbel_resnet50_lambda_grid_150ep.yaml")

    assert cfg.training_arguments.num_epochs == 150
    assert cfg.scheduler.T_max == 150
    assert cfg.tuning.mode == "grid"
    assert cfg.tuning.study_name == "gumbel_resnet50_lambda_grid_150ep"
    assert cfg.tuning.n_trials == 4
    assert cfg.tuning.sampler is None
    assert cfg.tuning.pruner._target_ == "optuna.pruners.NopPruner"
    assert cfg.tuning.search_space["model.lambda_coef"].choices == [0.0, 0.01, 0.05, 0.5]


def test_gumbel_resnet50_lambda_grid_150ep_narrow_uses_requested_manual_lambda_points():
    cfg = OmegaConf.load(CONFIGS_DIR / "tuning" / "gumbel_resnet50_lambda_grid_150ep_narrow.yaml")

    assert cfg.training_arguments.num_epochs == 150
    assert cfg.scheduler.T_max == 150
    assert cfg.tuning.mode == "grid"
    assert cfg.tuning.study_name == "gumbel_resnet50_lambda_grid_150ep_narrow"
    assert cfg.tuning.n_trials == 5
    assert cfg.tuning.sampler is None
    assert cfg.tuning.pruner._target_ == "optuna.pruners.NopPruner"
    assert cfg.tuning.search_space["model.lambda_coef"].choices == [0.0, 0.01, 0.2, 0.25, 0.3]


def test_gumbel_resnet50_paper_resnet50_ramp30_lambda_grid_200ep_uses_requested_manual_lambda_points():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "gumbel_resnet50_paper_resnet50_ramp30_lambda_grid_200ep_5_15_25_35.yaml"
    )

    assert cfg.training_arguments.num_epochs == 200
    assert cfg.scheduler.T_max == 200
    assert cfg.tuning.mode == "grid"
    assert (
        cfg.tuning.study_name
        == "gumbel_resnet50_paper_resnet50_ramp30_lambda_grid_200ep_5_15_25_35"
    )
    assert cfg.tuning.n_trials == 4
    assert cfg.tuning.sampler is None
    assert cfg.tuning.pruner._target_ == "optuna.pruners.NopPruner"
    assert (
        cfg.tuning.search_space["training_arguments.lambda_warmup.target_lambda_coef"].choices
        == [5.0, 15.0, 25.0, 35.0]
    )


def test_gumbel_resnet20_warmup30_lambda_grid_160ep_uses_requested_manual_lambda_points():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "gumbel_resnet20_warmup30_lambda_grid_160ep_5_15_25_35.yaml"
    )

    assert cfg.training_arguments.num_epochs == 160
    assert cfg.tuning.mode == "grid"
    assert cfg.tuning.study_name == "gumbel_resnet20_warmup30_lambda_grid_160ep_5_15_25_35"
    assert cfg.tuning.n_trials == 4
    assert cfg.tuning.sampler is None
    assert cfg.tuning.pruner._target_ == "optuna.pruners.NopPruner"
    assert (
        cfg.tuning.search_space["training_arguments.lambda_warmup.target_lambda_coef"].choices
        == [5.0, 15.0, 25.0, 35.0]
    )


def test_resnet20_gumbel_adaptive_lambda_adamw_base_uses_requested_recipe():
    cfg = OmegaConf.load(CONFIGS_DIR / "experiment" / "resnet20_gumbel_adaptive_lambda_adamw_v1.yaml")

    assert cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "cifar_resnet20"},
        {"/method": "gumbel"},
        {"/train": "best_practice"},
        {"/optimizer": "adamw"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "gumbel"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 1.0e-6
    assert cfg.model.gumbel_init_mode == "auto"
    assert cfg.model.bypass_on_zero_lambda is False
    assert cfg.model.backbone.resnet_block.temperature == 1.0
    assert cfg.training_arguments.lambda_warmup.enabled is False
    adaptive = cfg.training_arguments.adaptive_lambda
    assert adaptive.enabled is True
    assert adaptive.warmup_epochs == 0
    assert adaptive.update_every_epochs == 1
    assert (
        adaptive.baseline_history_dir
        == "outputs/baselines/resnet20_gumbel_cifar10_no_pruning_temp_1.0"
    )
    assert adaptive.lambda_min == 1e-8
    assert adaptive.lambda_max == 100.0
    assert adaptive.recovery.enabled is False
    assert cfg.mlflow.tags.recipe == "resnet20_gumbel_adaptive_lambda_adamw_v1"


def test_resnet20_gumbel_adaptive_lambda_adamw_grid_runs_requested_points_in_order():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "resnet20_gumbel_adaptive_lambda_adamw_grid_200ep_1em6_20_50_ordered.yaml"
    )

    assert cfg.training_arguments.num_epochs == 200
    assert cfg.training_arguments.adaptive_lambda.enabled is True
    assert cfg.training_arguments.adaptive_lambda.update_every_epochs == 1
    assert cfg.training_arguments.adaptive_lambda.recovery.enabled is False
    assert cfg.scheduler.T_max == 200
    assert cfg.optimizer._target_ == "torch.optim.AdamW"
    assert cfg.optimizer.lr == 0.001
    assert cfg.optimizer.weight_decay == 0.0005
    assert cfg.optimizer.gate_weight_decay_scale is None
    assert cfg.tuning.mode == "grid"
    assert (
        cfg.tuning.study_name
        == "resnet20_gumbel_adaptive_lambda_adamw_grid_200ep_1em6_20_50_ordered"
    )
    assert cfg.tuning.n_trials == 3
    assert cfg.tuning.points_in_order is True
    assert [point["model.lambda_coef"] for point in cfg.tuning.points] == [
        0.000001,
        20.0,
        50.0,
    ]
    assert [
        point["training_arguments.adaptive_lambda.update_every_epochs"]
        for point in cfg.tuning.points
    ] == [1, 1, 1]


def test_tune_resnet20_gumbel_adaptive_lambda_adamw_grid_uses_requested_defaults():
    cfg = OmegaConf.load(CONFIGS_DIR / "tune_resnet20_gumbel_adaptive_lambda_adamw_grid.yaml")

    assert cfg.defaults == [
        {"experiment": "resnet20_gumbel_adaptive_lambda_adamw_v1"},
        {"tuning": "resnet20_gumbel_adaptive_lambda_adamw_grid_200ep_1em6_20_50_ordered"},
        "_self_",
    ]


def test_resnet50_adaptive_lambda_nightly_grid_runs_requested_points_in_order():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "resnet50_adaptive_lambda_init_grid_200ep_5_15_25_35_ordered.yaml"
    )

    assert cfg.training_arguments.num_epochs == 200
    assert cfg.scheduler.T_max == 200
    assert cfg.tuning.mode == "grid"
    assert cfg.tuning.study_name == "resnet50_adaptive_lambda_init_grid_200ep_5_15_25_35_ordered"
    assert cfg.tuning.n_trials == 4
    assert cfg.tuning.points_in_order is True
    assert [point["model.lambda_coef"] for point in cfg.tuning.points] == [5.0, 15.0, 25.0, 35.0]


def test_resnet50_adaptive_lambda_updatefreq_nightly_grid_runs_requested_points_in_order():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "resnet50_adaptive_lambda_updatefreq_grid_200ep_init1em3_1em1_ordered.yaml"
    )

    assert cfg.training_arguments.num_epochs == 200
    assert cfg.scheduler.T_max == 200
    assert cfg.tuning.mode == "grid"
    assert (
        cfg.tuning.study_name
        == "resnet50_adaptive_lambda_updatefreq_grid_200ep_init1em3_1em1_ordered"
    )
    assert cfg.tuning.n_trials == 4
    assert cfg.tuning.points_in_order is True
    assert [point["model.lambda_coef"] for point in cfg.tuning.points] == [0.001, 0.1, 0.001, 0.1]
    assert [
        point["training_arguments.adaptive_lambda.update_every_epochs"]
        for point in cfg.tuning.points
    ] == [2, 2, 1, 1]


def test_tune_resnet50_adaptive_lambda_nightly_uses_adaptive_base_and_ordered_grid():
    cfg = OmegaConf.load(CONFIGS_DIR / "tune_resnet50_adaptive_lambda_nightly.yaml")

    assert cfg.defaults == [
        {"experiment": "resnet50_adaptive_lambda_init5"},
        {"tuning": "resnet50_adaptive_lambda_init_grid_200ep_5_15_25_35_ordered"},
        "_self_",
    ]


def test_tune_resnet50_adaptive_lambda_updatefreq_nightly_uses_adaptive_base_and_ordered_grid():
    cfg = OmegaConf.load(CONFIGS_DIR / "tune_resnet50_adaptive_lambda_updatefreq_nightly.yaml")

    assert cfg.defaults == [
        {"experiment": "resnet50_adaptive_lambda_step125_no_warmup_base"},
        {"tuning": "resnet50_adaptive_lambda_updatefreq_grid_200ep_init1em3_1em1_ordered"},
        "_self_",
    ]


def test_resnet50_adaptive_lambda_base_disables_boost_and_recovery():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "experiment"
        / "resnet50_adaptive_lambda_init1em3_no_boost_no_recovery.yaml"
    )

    adaptive = cfg.training_arguments.adaptive_lambda
    assert cfg.model.lambda_coef == 0.001
    assert adaptive.enabled is True
    assert adaptive.adaptive_log_step_enabled is False
    assert adaptive.recovery.enabled is False


def test_resnet50_adaptive_lambda_step_grid_runs_requested_points_in_order():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / (
            "resnet50_adaptive_lambda_init1em3_no_boost_no_recovery_"
            "step_grid_125_150_175_200_ordered.yaml"
        )
    )

    assert cfg.tuning.mode == "grid"
    assert (
        cfg.tuning.study_name
        == (
            "resnet50_adaptive_lambda_init1em3_no_boost_no_recovery_"
            "step_grid_125_150_175_200_ordered"
        )
    )
    assert cfg.tuning.n_trials == 4
    assert cfg.tuning.points_in_order is True
    assert [point["model.lambda_coef"] for point in cfg.tuning.points] == [
        0.001,
        0.001,
        0.001,
        0.001,
    ]
    assert [
        point["training_arguments.adaptive_lambda.log_step_init"]
        for point in cfg.tuning.points
    ] == [
        0.22314355131420976,
        0.4054651081081644,
        0.5596157879354227,
        0.6931471805599453,
    ]
    assert [point["mlflow.tags.lambda_multiplier"] for point in cfg.tuning.points] == [
        "1.25",
        "1.5",
        "1.75",
        "2",
    ]


def test_tune_resnet50_adaptive_lambda_step_grid_uses_requested_defaults():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tune_resnet50_adaptive_lambda_init1em3_no_boost_no_recovery_step_grid.yaml"
    )

    assert cfg.defaults == [
        {"experiment": "resnet50_adaptive_lambda_init1em3_no_boost_no_recovery"},
        {
            "tuning": (
                "resnet50_adaptive_lambda_init1em3_no_boost_no_recovery_"
                "step_grid_125_150_175_200_ordered"
            )
        },
        "_self_",
    ]


def test_resnet50_gumbel_adamw_lambda_grid_runs_requested_points():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "resnet50_gumbel_adaptive_lambda_adamw_lambda_grid_ordered.yaml"
    )

    adaptive = cfg.training_arguments.adaptive_lambda
    assert cfg.training_arguments.num_epochs == 200
    assert cfg.scheduler.T_max == 200
    assert cfg.optimizer._target_ == "torch.optim.AdamW"
    assert adaptive.update_every_epochs == 1
    assert adaptive.recovery.enabled is False
    assert cfg.tuning.study_name == "resnet50_gumbel_adaptive_lambda_adamw_lambda_grid_ordered"
    assert cfg.tuning.n_trials == 4
    assert cfg.tuning.points_in_order is True
    assert [point["model.lambda_coef"] for point in cfg.tuning.points] == [
        0.00000001,
        0.000001,
        10.0,
        50.0,
    ]
    assert [point["mlflow.tags.recovery"] for point in cfg.tuning.points] == [
        "disabled",
        "disabled",
        "disabled",
        "disabled",
    ]


def test_tune_resnet50_gumbel_adamw_lambda_grid_uses_requested_defaults():
    cfg = OmegaConf.load(
        CONFIGS_DIR / "tune_resnet50_gumbel_adaptive_lambda_adamw_lambda_grid.yaml"
    )

    assert cfg.defaults == [
        {"experiment": "resnet50_gumbel_adaptive_lambda_v1"},
        {"tuning": "resnet50_gumbel_adaptive_lambda_adamw_lambda_grid_ordered"},
        {"override /optimizer": "adamw"},
        "_self_",
    ]


def test_resnet50_gumbel_adaptive_logstep_until50_grid_runs_requested_points_in_order():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / (
            "resnet50_gumbel_adaptive_lambda_recovery_v1_"
            "logstep_grid_1em6_1em4_boost5_until50_ordered.yaml"
        )
    )

    assert cfg.tuning.mode == "grid"
    assert (
        cfg.tuning.study_name
        == (
            "resnet50_gumbel_adaptive_lambda_recovery_v1_"
            "logstep_grid_1em6_1em4_boost5_until50_ordered"
        )
    )
    assert cfg.tuning.n_trials == 4
    assert cfg.tuning.points_in_order is True
    assert [point["model.lambda_coef"] for point in cfg.tuning.points] == [
        0.000001,
        0.0001,
        0.000001,
        0.0001,
    ]
    assert [
        point["training_arguments.adaptive_lambda.log_step_init"]
        for point in cfg.tuning.points
    ] == [
        0.22314355131420976,
        0.22314355131420976,
        0.09531017980432493,
        0.09531017980432493,
    ]
    assert [
        point["training_arguments.adaptive_lambda.log_step_max_boost_level"]
        for point in cfg.tuning.points
    ] == [5, 5, 5, 5]
    assert [
        point["training_arguments.adaptive_lambda.adaptive_log_step_max_epoch"]
        for point in cfg.tuning.points
    ] == [50, 50, 50, 50]


def test_tune_resnet50_gumbel_adaptive_logstep_until50_uses_recovery_base_and_ordered_grid():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tune_resnet50_gumbel_adaptive_lambda_recovery_v1_logstep_grid_until50.yaml"
    )

    assert cfg.defaults == [
        {"experiment": "resnet50_gumbel_adaptive_lambda_recovery_v1"},
        {
            "tuning": (
                "resnet50_gumbel_adaptive_lambda_recovery_v1_"
                "logstep_grid_1em6_1em4_boost5_until50_ordered"
            )
        },
        "_self_",
    ]


def test_resnet50_gumbel_constant_lambda_nightly_grid_runs_requested_points_in_order():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "resnet50_gumbel_constant_lambda_grid_200ep_27_28_29_31_ordered.yaml"
    )

    assert cfg.training_arguments.num_epochs == 200
    assert cfg.scheduler.T_max == 200
    assert cfg.tuning.mode == "grid"
    assert (
        cfg.tuning.study_name
        == "resnet50_gumbel_constant_lambda_grid_200ep_27_28_29_31_ordered"
    )
    assert cfg.tuning.n_trials == 4
    assert cfg.tuning.points_in_order is True
    assert [point["model.lambda_coef"] for point in cfg.tuning.points] == [27.0, 28.0, 29.0, 31.0]


def test_tune_resnet50_gumbel_constant_lambda_nightly_uses_stock_gumbel_base_and_ordered_grid():
    cfg = OmegaConf.load(CONFIGS_DIR / "tune_resnet50_gumbel_constant_lambda_nightly.yaml")

    assert cfg.defaults == [
        {"experiment": "best_practice_resnet50_gumbel_on_cifar10"},
        {"tuning": "resnet50_gumbel_constant_lambda_grid_200ep_27_28_29_31_ordered"},
        "_self_",
    ]


def test_resnet50_gumbel_constant_lambda_adamw_grid_runs_requested_points_in_order():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "resnet50_gumbel_constant_lambda_adamw_grid_200ep_1em8_1em6_10_50_ordered.yaml"
    )

    assert cfg.training_arguments.num_epochs == 200
    assert cfg.training_arguments.lambda_warmup.enabled is False
    assert cfg.training_arguments.adaptive_lambda.enabled is False
    assert cfg.scheduler.T_max == 200
    assert cfg.optimizer._target_ == "torch.optim.AdamW"
    assert cfg.optimizer.lr == 0.001
    assert cfg.optimizer.weight_decay == 0.0005
    assert cfg.optimizer.gate_weight_decay_scale is None
    assert cfg.tuning.mode == "grid"
    assert (
        cfg.tuning.study_name
        == "resnet50_gumbel_constant_lambda_adamw_grid_200ep_1em8_1em6_10_50_ordered"
    )
    assert cfg.tuning.n_trials == 4
    assert cfg.tuning.points_in_order is True
    assert [point["model.lambda_coef"] for point in cfg.tuning.points] == [
        0.00000001,
        0.000001,
        10.0,
        50.0,
    ]
    assert [point["mlflow.tags.adaptive_lambda"] for point in cfg.tuning.points] == [
        "disabled",
        "disabled",
        "disabled",
        "disabled",
    ]


def test_tune_resnet50_gumbel_constant_lambda_adamw_grid_uses_stock_gumbel_base():
    cfg = OmegaConf.load(CONFIGS_DIR / "tune_resnet50_gumbel_constant_lambda_adamw_grid.yaml")

    assert cfg.defaults == [
        {"experiment": "best_practice_resnet50_gumbel_on_cifar10"},
        {
            "tuning": (
                "resnet50_gumbel_constant_lambda_adamw_grid_200ep_"
                "1em8_1em6_10_50_ordered"
            )
        },
        {"override /optimizer": "adamw"},
        "_self_",
    ]


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


def test_tinyimagenet_data_and_model_configs_use_200_classes_and_64px_images():
    data_cfg = OmegaConf.load(CONFIGS_DIR / "data" / "tinyimagenet200_best_practice.yaml")
    resnet50_cfg = OmegaConf.load(CONFIGS_DIR / "model" / "resnet50_tinyimagenet200.yaml")
    resnet20_cfg = OmegaConf.load(CONFIGS_DIR / "model" / "cifar_resnet20_tinyimagenet200.yaml")
    cifar10_resnet50_cfg = OmegaConf.load(CONFIGS_DIR / "model" / "resnet50.yaml")
    data_raw = OmegaConf.to_container(data_cfg, resolve=False)

    assert data_cfg.dataloaders.taskname == "tinyimagenet200"
    assert data_raw["dataloaders"]["path_to_data"] == "${hydra:runtime.cwd}/data/tiny-imagenet-200"
    assert data_cfg.dataloaders.num_classes == 200
    assert data_cfg.dataloaders.image_size == 64
    assert data_cfg.dataloaders.valid_ratio == 0.1
    assert data_cfg.dataloaders.seed == 42
    assert data_cfg.dataloaders.batch_size == 128

    assert resnet50_cfg.model.backbone.num_classes == 200
    assert resnet50_cfg.model.backbone.stem_kernel_size == 3
    assert resnet50_cfg.model.backbone.stem_stride == 1
    assert resnet50_cfg.model.backbone.use_maxpool is False
    assert resnet20_cfg.model.backbone.num_classes == 200
    assert cifar10_resnet50_cfg.model.backbone.num_classes == 10


def test_tinyimagenet_baseline_experiment_configs_use_requested_recipes():
    resnet50_cfg = OmegaConf.load(
        CONFIGS_DIR / "experiment" / "best_practice_resnet50_on_tinyimagenet200.yaml"
    )
    resnet50_gumbel_cfg = OmegaConf.load(
        CONFIGS_DIR / "experiment" / "best_practice_resnet50_gumbel_on_tinyimagenet200.yaml"
    )
    resnet20_cfg = OmegaConf.load(
        CONFIGS_DIR / "experiment" / "best_practice_resnet20_on_tinyimagenet200.yaml"
    )

    assert resnet50_cfg.defaults == [
        {"/data": "tinyimagenet200_best_practice"},
        {"/model": "resnet50_tinyimagenet200"},
        {"/method": "plain"},
        {"/train": "resnet50_best_practice"},
        {"/optimizer": "sgd_resnet50"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "classification"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert resnet50_cfg.training_arguments.num_epochs == 240
    assert resnet50_cfg.scheduler.T_max == 240
    assert resnet50_cfg.model.lambda_coef == 0.0
    assert resnet50_cfg.mlflow.tags.recipe == "best_practice_resnet50_on_tinyimagenet200"

    assert resnet50_gumbel_cfg.defaults[0] == {"/data": "tinyimagenet200_best_practice"}
    assert resnet50_gumbel_cfg.defaults[1] == {"/model": "resnet50_tinyimagenet200"}
    assert resnet50_gumbel_cfg.defaults[2] == {"/method": "gumbel"}
    assert resnet50_gumbel_cfg.defaults[4] == {"/optimizer": "sgd_resnet50"}
    assert resnet50_gumbel_cfg.training_arguments.num_epochs == 240
    assert resnet50_gumbel_cfg.scheduler.T_max == 240
    assert resnet50_gumbel_cfg.model.lambda_coef == 0.0
    assert resnet50_gumbel_cfg.model.gumbel_init_mode == "paper_resnet50"
    assert (
        resnet50_gumbel_cfg.model.backbone.resnet_block._target_
        == "net_complexity.wrappers.GumbelBottleneckLayer"
    )
    assert resnet50_gumbel_cfg.mlflow.tags.recipe == (
        "best_practice_resnet50_gumbel_on_tinyimagenet200"
    )

    assert resnet20_cfg.defaults == [
        {"/data": "tinyimagenet200_best_practice"},
        {"/model": "cifar_resnet20_tinyimagenet200"},
        {"/method": "plain"},
        {"/train": "best_practice"},
        {"/optimizer": "sgd_resnet20"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "classification"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert resnet20_cfg.training_arguments.num_epochs == 200
    assert resnet20_cfg.scheduler.T_max == 200
    assert resnet20_cfg.model.lambda_coef == 0.0
    assert resnet20_cfg.mlflow.tags.recipe == "best_practice_resnet20_on_tinyimagenet200"


def test_tinyimagenet_adaptive_lambda_configs_use_tiny_baselines():
    resnet50_cfg = OmegaConf.load(
        CONFIGS_DIR
        / "experiment"
        / "resnet50_gumbel_adaptive_lambda_tinyimagenet200_v1.yaml"
    )
    resnet20_cfg = OmegaConf.load(
        CONFIGS_DIR
        / "experiment"
        / "resnet20_gumbel_adaptive_lambda_adamw_tinyimagenet200_v1.yaml"
    )

    for cfg in (resnet50_cfg, resnet20_cfg):
        adaptive = cfg.training_arguments.adaptive_lambda
        assert cfg.defaults[0] == {"/data": "tinyimagenet200_best_practice"}
        assert adaptive.enabled is True
        assert "tinyimagenet200" in adaptive.baseline_history_dir
        assert cfg.mlflow.tags.dataset == "tinyimagenet200"
        assert "tinyimagenet200" in cfg.mlflow.tags.recipe

    assert resnet50_cfg.defaults[1] == {"/model": "resnet50_tinyimagenet200"}
    assert resnet50_cfg.defaults[4] == {"/optimizer": "sgd_resnet50"}
    assert resnet50_cfg.training_arguments.num_epochs == 240
    assert resnet50_cfg.scheduler.T_max == 240
    assert resnet50_cfg.model.gumbel_init_mode == "paper_resnet50"

    assert resnet20_cfg.defaults[1] == {"/model": "cifar_resnet20_tinyimagenet200"}
    assert resnet20_cfg.defaults[4] == {"/optimizer": "adamw"}
    assert resnet20_cfg.training_arguments.num_epochs == 200
    assert resnet20_cfg.scheduler.T_max == 200
    assert resnet20_cfg.model.gumbel_init_mode == "auto"


def test_tinyimagenet_resnet20_resnet50_ordered_lambda10_adaptive_config_runs_requested_sequence():
    tuning_cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "tinyimagenet200_resnet20_resnet50_lambda10_adaptive_init001_ordered.yaml"
    )
    tune_cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tune_tinyimagenet200_resnet20_resnet50_lambda10_adaptive_init001.yaml"
    )

    assert tune_cfg.defaults == [
        {"experiment": "resnet20_gumbel_adaptive_lambda_adamw_tinyimagenet200_v1"},
        {"tuning": "tinyimagenet200_resnet20_resnet50_lambda10_adaptive_init001_ordered"},
        "_self_",
    ]
    assert tuning_cfg.tuning.enabled is True
    assert tuning_cfg.tuning.mode == "grid"
    assert tuning_cfg.tuning.points_in_order is True
    assert tuning_cfg.tuning.n_jobs == 1
    assert tuning_cfg.tuning.n_trials == 4
    assert (
        tuning_cfg.tuning.study_name
        == "tinyimagenet200_resnet20_resnet50_lambda10_adaptive_init001_ordered"
    )

    points = OmegaConf.to_container(tuning_cfg.tuning.points, resolve=False)
    assert [point["model.lambda_coef"] for point in points] == [10.0, 0.01, 10.0, 0.01]
    assert [point["training_arguments.adaptive_lambda.enabled"] for point in points] == [
        False,
        True,
        False,
        True,
    ]
    assert [point["model.backbone"]["_target_"] for point in points] == [
        "net_complexity.wrappers.CIFARResNet20",
        "net_complexity.wrappers.CIFARResNet20",
        "net_complexity.wrappers.ResNet50",
        "net_complexity.wrappers.ResNet50",
    ]
    assert [point["model.backbone"]["num_classes"] for point in points] == [200, 200, 200, 200]
    assert [point["optimizer"]["_target_"] for point in points] == [
        "torch.optim.AdamW",
        "torch.optim.AdamW",
        "torch.optim.AdamW",
        "torch.optim.AdamW",
    ]
    assert points[1]["training_arguments.adaptive_lambda.baseline_history_dir"].startswith(
        "outputs/baselines/resnet20_gumbel_tinyimagenet200"
    )
    assert points[3]["training_arguments.adaptive_lambda.baseline_history_dir"].startswith(
        "outputs/baselines/resnet50_gumbel_tinyimagenet200"
    )


def test_tinyimagenet_resnet50_only_ordered_lambda10_adaptive_config_uses_batch224_adamw():
    tuning_cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "tinyimagenet200_resnet50_lambda10_adaptive_init001_batch224_ordered.yaml"
    )
    tune_cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tune_tinyimagenet200_resnet50_lambda10_adaptive_init001_batch224.yaml"
    )

    assert tune_cfg.defaults == [
        {"experiment": "resnet50_gumbel_adaptive_lambda_tinyimagenet200_v1"},
        {"tuning": "tinyimagenet200_resnet50_lambda10_adaptive_init001_batch224_ordered"},
        {"override /optimizer": "adamw"},
        "_self_",
    ]
    assert tuning_cfg.dataloaders.batch_size == 224
    assert tuning_cfg.optimizer._target_ == "torch.optim.AdamW"
    assert tuning_cfg.optimizer.lr == 0.001
    assert tuning_cfg.optimizer.weight_decay == 0.0005
    assert tuning_cfg.optimizer.gate_weight_decay_scale is None
    assert tuning_cfg.tuning.enabled is True
    assert tuning_cfg.tuning.mode == "grid"
    assert tuning_cfg.tuning.points_in_order is True
    assert tuning_cfg.tuning.n_jobs == 1
    assert tuning_cfg.tuning.n_trials == 2
    assert (
        tuning_cfg.tuning.study_name
        == "tinyimagenet200_resnet50_lambda10_adaptive_init001_batch224_ordered"
    )

    points = OmegaConf.to_container(tuning_cfg.tuning.points, resolve=False)
    assert [point["model.lambda_coef"] for point in points] == [10.0, 0.01]
    assert [point["training_arguments.adaptive_lambda.enabled"] for point in points] == [
        False,
        True,
    ]
    assert points[0]["training_arguments.adaptive_lambda.baseline_history_dir"] is None
    assert points[1]["training_arguments.adaptive_lambda.baseline_history_dir"].startswith(
        "outputs/baselines/resnet50_gumbel_tinyimagenet200"
    )
    assert [point["mlflow.tags.optimizer"] for point in points] == ["AdamW", "AdamW"]
    assert [point["mlflow.tags.batch_size"] for point in points] == ["224", "224"]


def test_tinyimagenet_resnet50_adaptive_lambda_init001_config_runs_single_batch128_point():
    tuning_cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "tinyimagenet200_resnet50_adaptive_lambda_init001_ordered.yaml"
    )
    tune_cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tune_tinyimagenet200_resnet50_adaptive_lambda_init001.yaml"
    )

    assert tune_cfg.defaults == [
        {"experiment": "resnet50_gumbel_adaptive_lambda_tinyimagenet200_v1"},
        {"tuning": "tinyimagenet200_resnet50_adaptive_lambda_init001_ordered"},
        {"override /optimizer": "adamw"},
        "_self_",
    ]
    assert tuning_cfg.dataloaders.batch_size == 128
    assert tuning_cfg.optimizer._target_ == "torch.optim.AdamW"
    assert tuning_cfg.optimizer.lr == 0.001
    assert tuning_cfg.optimizer.weight_decay == 0.0005
    assert tuning_cfg.optimizer.gate_weight_decay_scale is None
    assert tuning_cfg.tuning.enabled is True
    assert tuning_cfg.tuning.mode == "grid"
    assert tuning_cfg.tuning.points_in_order is True
    assert tuning_cfg.tuning.n_jobs == 1
    assert tuning_cfg.tuning.n_trials == 1
    assert (
        tuning_cfg.tuning.study_name
        == "tinyimagenet200_resnet50_adaptive_lambda_init001_ordered"
    )

    points = OmegaConf.to_container(tuning_cfg.tuning.points, resolve=False)
    assert len(points) == 1
    point = points[0]
    assert point["model.lambda_coef"] == 0.01
    assert point["training_arguments.adaptive_lambda.enabled"] is True
    assert point["training_arguments.adaptive_lambda.update_every_epochs"] == 2
    assert point["training_arguments.adaptive_lambda.baseline_history_dir"].startswith(
        "outputs/baselines/resnet50_gumbel_tinyimagenet200"
    )
    assert point["training_arguments.adaptive_lambda.recovery.enabled"] is False
    assert point["mlflow.tags.optimizer"] == "AdamW"
    assert point["mlflow.tags.initial_lambda"] == "0.01"
    assert point["mlflow.tags.adaptive_lambda"] == "enabled"
    assert point["mlflow.tags.batch_size"] == "128"


def test_resnet50_aig_adaptive_lambda_config_uses_clean_adaptive_profile():
    experiment_cfg = OmegaConf.load(
        CONFIGS_DIR / "experiment" / "resnet50_aig_adaptive_lambda_v1.yaml"
    )
    tuning_cfg = OmegaConf.load(
        CONFIGS_DIR / "tuning" / "resnet50_aig_adaptive_lambda_init001_ordered.yaml"
    )
    tune_cfg = OmegaConf.load(
        CONFIGS_DIR / "tune_resnet50_aig_adaptive_lambda_init001.yaml"
    )

    assert experiment_cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "resnet50"},
        {"/method": "aig"},
        {"/train": "aig_adaptive_lambda"},
        {"/optimizer": "adamw"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "aig"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert experiment_cfg.model.lambda_coef == 1e-6
    assert experiment_cfg.model.bypass_on_zero_lambda is False
    assert experiment_cfg.model.backbone.resnet_block.gate_regularization == "l2_gate"
    adaptive_cfg = experiment_cfg.training_arguments.adaptive_lambda
    assert adaptive_cfg.enabled is True
    assert adaptive_cfg.baseline_history_dir.startswith(
        "outputs/baselines/resnet50_aig_cifar10_no_pruning"
    )
    assert adaptive_cfg.lambda_max == 10.0
    assert adaptive_cfg.log_step_init == "auto"
    _assert_clean_aig_adaptive_lambda(experiment_cfg.training_arguments)
    assert experiment_cfg.training_arguments.batchnorm_recalibration.enabled is False
    assert experiment_cfg.mlflow.tags.method == "aig"
    assert experiment_cfg.mlflow.tags.gate_regularization == "l2_gate"
    assert experiment_cfg.mlflow.tags.initial_lambda == "1e-6"
    assert experiment_cfg.mlflow.tags.target_lambda == "10"

    assert tune_cfg.defaults == [
        {"experiment": "resnet50_aig_adaptive_lambda_v1"},
        {"tuning": "resnet50_aig_adaptive_lambda_init001_ordered"},
        "_self_",
    ]
    assert tuning_cfg.training_arguments.num_epochs == 200
    assert tuning_cfg.training_arguments.adaptive_lambda.enabled is True
    assert "recovery" not in tuning_cfg.training_arguments.adaptive_lambda
    assert tuning_cfg.scheduler.T_max == 200
    assert tuning_cfg.tuning.n_trials == 1
    assert tuning_cfg.tuning.points_in_order is True

    points = OmegaConf.to_container(tuning_cfg.tuning.points, resolve=False)
    assert len(points) == 1
    point = points[0]
    assert point["model.lambda_coef"] == 0.01
    assert point["training_arguments.adaptive_lambda.enabled"] is True
    assert "training_arguments.adaptive_lambda.recovery.enabled" not in point
    assert point["mlflow.tags.optimizer"] == "AdamW"
    assert "mlflow.tags.recovery" not in point


def test_resnet50_aig_adaptive_lambda_no_extra_grid_runs_requested_initial_lambdas():
    tuning_cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "resnet50_aig_adaptive_lambda_no_extra_init_grid_1em8_1em5_1em2_1_ordered.yaml"
    )
    tune_cfg = OmegaConf.load(
        CONFIGS_DIR / "tune_resnet50_aig_adaptive_lambda_no_extra_init_grid.yaml"
    )

    assert tune_cfg.defaults == [
        {"experiment": "resnet50_aig_adaptive_lambda_v1"},
        {
            "tuning": (
                "resnet50_aig_adaptive_lambda_no_extra_"
                "init_grid_1em8_1em5_1em2_1_ordered"
            )
        },
        "_self_",
    ]
    assert tuning_cfg.training_arguments.adaptive_lambda.enabled is True
    assert "adaptive_log_step_enabled" not in tuning_cfg.training_arguments.adaptive_lambda
    assert "recovery" not in tuning_cfg.training_arguments.adaptive_lambda
    assert tuning_cfg.training_arguments.batchnorm_recalibration.enabled is False
    assert "collapse_guard" not in tuning_cfg.training_arguments
    assert "restart_guard" not in tuning_cfg.tuning
    assert tuning_cfg.tuning.mode == "grid"
    assert tuning_cfg.tuning.n_trials == 4
    assert tuning_cfg.tuning.points_in_order is True

    points = OmegaConf.to_container(tuning_cfg.tuning.points, resolve=False)
    assert [point["model.lambda_coef"] for point in points] == [
        1e-8,
        1e-5,
        1e-2,
        1.0,
    ]
    assert [point["mlflow.tags.initial_lambda"] for point in points] == [
        "1e-8",
        "1e-5",
        "1e-2",
        "1",
    ]


def test_resnet50_aig_adaptive_lambda_no_lambda_max_recipe_uses_step150():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "experiment"
        / "resnet50_aig_adaptive_lambda_init1em8_step150_no_lambda_max_200ep.yaml"
    )

    assert cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "resnet50"},
        {"/method": "aig"},
        {"/train": "aig_adaptive_lambda"},
        {"/optimizer": "adamw"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "aig"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 1e-8
    adaptive = cfg.training_arguments.adaptive_lambda
    assert adaptive.enabled is True
    assert adaptive.lambda_min == 1e-8
    assert adaptive.lambda_max is None
    assert adaptive.log_step_init == 0.4054651081081644
    _assert_clean_aig_adaptive_lambda(cfg.training_arguments)
    assert cfg.training_arguments.num_epochs == 200
    assert cfg.scheduler.T_max == 200
    assert cfg.mlflow.tags.initial_lambda == "1e-8"
    assert cfg.mlflow.tags.lambda_multiplier == "1.5"
    assert cfg.mlflow.tags.lambda_max == "none"


def test_resnet50_aig_adaptive_lambda_no_lambda_max_init_grid_runs_requested_initial_lambdas():
    tuning_cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "resnet50_aig_adaptive_lambda_no_lambda_max_init_grid_5_1_1em1_1em2_1em3_ordered.yaml"
    )
    tune_cfg = OmegaConf.load(
        CONFIGS_DIR / "tune_resnet50_aig_adaptive_lambda_no_lambda_max_init_grid.yaml"
    )

    assert tune_cfg.defaults == [
        {"experiment": "resnet50_aig_adaptive_lambda_init1em8_step150_no_lambda_max_200ep"},
        {
            "tuning": (
                "resnet50_aig_adaptive_lambda_no_lambda_max_"
                "init_grid_5_1_1em1_1em2_1em3_ordered"
            )
        },
        "_self_",
    ]
    adaptive = tuning_cfg.training_arguments.adaptive_lambda
    assert adaptive.enabled is True
    assert adaptive.lambda_max is None
    assert adaptive.log_step_init == 0.4054651081081644
    assert adaptive.update_every_epochs == 2
    assert "adaptive_log_step_enabled" not in adaptive
    assert "recovery" not in adaptive
    assert tuning_cfg.training_arguments.batchnorm_recalibration.enabled is False
    assert "collapse_guard" not in tuning_cfg.training_arguments
    assert tuning_cfg.tuning.mode == "grid"
    assert tuning_cfg.tuning.n_trials == 5
    assert tuning_cfg.tuning.points_in_order is True

    points = OmegaConf.to_container(tuning_cfg.tuning.points, resolve=False)
    assert [point["model.lambda_coef"] for point in points] == [
        5.0,
        1.0,
        0.1,
        0.01,
        0.001,
    ]
    assert [point["mlflow.tags.initial_lambda"] for point in points] == [
        "5",
        "1",
        "0.1",
        "0.01",
        "0.001",
    ]
    assert {point["mlflow.tags.lambda_max"] for point in points} == {"none"}


def test_resnet50_aig_adaptive_lambda_no_lambda_max_scaled_step_epochs_grid():
    tuning_cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "resnet50_aig_adaptive_lambda_no_lambda_max_init1em4_scaled_step_epochs_50_70_90_140_200_ordered.yaml"
    )
    tune_cfg = OmegaConf.load(
        CONFIGS_DIR / "tune_resnet50_aig_adaptive_lambda_no_lambda_max_scaled_step_epochs.yaml"
    )

    assert tune_cfg.defaults == [
        {"experiment": "resnet50_aig_adaptive_lambda_init1em8_step150_no_lambda_max_200ep"},
        {
            "tuning": (
                "resnet50_aig_adaptive_lambda_no_lambda_max_"
                "init1em4_scaled_step_epochs_50_70_90_140_200_ordered"
            )
        },
        "_self_",
    ]
    adaptive = tuning_cfg.training_arguments.adaptive_lambda
    assert adaptive.enabled is True
    assert adaptive.lambda_max is None
    assert adaptive.update_every_epochs == 2
    assert "adaptive_log_step_enabled" not in adaptive
    assert "recovery" not in adaptive
    assert tuning_cfg.training_arguments.batchnorm_recalibration.enabled is False
    assert "collapse_guard" not in tuning_cfg.training_arguments
    assert tuning_cfg.tuning.mode == "grid"
    assert tuning_cfg.tuning.n_trials == 5
    assert tuning_cfg.tuning.points_in_order is True

    points = OmegaConf.to_container(tuning_cfg.tuning.points, resolve=False)
    assert [point["model.lambda_coef"] for point in points] == [1e-4] * 5
    assert [point["training_arguments.num_epochs"] for point in points] == [
        50,
        70,
        90,
        140,
        200,
    ]
    assert [point["scheduler.T_max"] for point in points] == [50, 70, 90, 140, 200]
    assert [
        point["training_arguments.adaptive_lambda.log_step_init"]
        for point in points
    ] == [
        0.6907755278982136,
        0.49341109135586697,
        0.38376418216567426,
        0.24670554567793349,
        0.1726938819745534,
    ]
    assert [point["mlflow.tags.target_lambda"] for point in points] == ["10"] * 5
    assert {point["mlflow.tags.lambda_max"] for point in points} == {"none"}


def test_resnet50_aig_adaptive_lambda_no_lambda_max_scaled_step_150ep():
    tuning_cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "resnet50_aig_adaptive_lambda_no_lambda_max_init1em4_scaled_step_150ep_ordered.yaml"
    )
    tune_cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tune_resnet50_aig_adaptive_lambda_no_lambda_max_scaled_step_150ep.yaml"
    )

    assert tune_cfg.defaults == [
        {"experiment": "resnet50_aig_adaptive_lambda_v1"},
        {
            "tuning": (
                "resnet50_aig_adaptive_lambda_no_lambda_max_"
                "init1em4_scaled_step_150ep_ordered"
            )
        },
        "_self_",
    ]
    adaptive = tuning_cfg.training_arguments.adaptive_lambda
    assert tuning_cfg.training_arguments.num_epochs == 150
    assert adaptive.enabled is True
    assert adaptive.lambda_max is None
    assert adaptive.update_every_epochs == 2
    assert adaptive.log_step_init == 0.23025850929940458
    assert tuning_cfg.training_arguments.batchnorm_recalibration.enabled is False
    assert tuning_cfg.scheduler.T_max == 150
    assert tuning_cfg.tuning.mode == "grid"
    assert tuning_cfg.tuning.n_trials == 1
    assert tuning_cfg.tuning.points_in_order is True

    points = OmegaConf.to_container(tuning_cfg.tuning.points, resolve=False)
    assert len(points) == 1
    point = points[0]
    assert point["model.lambda_coef"] == 0.0001
    assert (
        point["training_arguments.adaptive_lambda.log_step_init"]
        == 0.23025850929940458
    )
    assert point["mlflow.tags.initial_lambda"] == "0.0001"
    assert point["mlflow.tags.target_lambda"] == "10"
    assert point["mlflow.tags.target_growth_steps"] == "50"
    assert point["mlflow.tags.lambda_multiplier"] == "1.25892541179"
    assert point["mlflow.tags.lambda_max"] == "none"
    assert point["mlflow.tags.num_epochs"] == "150"


def test_resnet50_aig_adaptive_lambda_l1_p_vs_g_120ep_repeats7():
    tuning_cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "resnet50_aig_adaptive_lambda_l1_p_vs_g_no_lambda_max_init1em4_scaled_step_120ep_repeats7_ordered.yaml"
    )
    tune_cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tune_resnet50_aig_adaptive_lambda_l1_p_vs_g_no_lambda_max_120ep_repeats7.yaml"
    )

    assert tune_cfg.defaults == [
        {"experiment": "resnet50_aig_adaptive_lambda_v1"},
        {
            "tuning": (
                "resnet50_aig_adaptive_lambda_l1_p_vs_g_no_lambda_max_"
                "init1em4_scaled_step_120ep_repeats7_ordered"
            )
        },
        "_self_",
    ]

    adaptive = tuning_cfg.training_arguments.adaptive_lambda
    assert tuning_cfg.training_arguments.num_epochs == 120
    assert adaptive.enabled is True
    assert adaptive.lambda_max is None
    assert adaptive.update_every_epochs == 2
    assert adaptive.log_step_init == 0.28782313662425574
    assert tuning_cfg.training_arguments.batchnorm_recalibration.enabled is False
    assert tuning_cfg.scheduler.T_max == 120
    assert tuning_cfg.tuning.mode == "grid"
    assert tuning_cfg.tuning.n_trials == 2
    assert tuning_cfg.tuning.repeats_per_trial == 7
    assert tuning_cfg.tuning.seed_base == 42
    assert tuning_cfg.tuning.seed_stride == 1
    assert tuning_cfg.tuning.points_in_order is True

    points = OmegaConf.to_container(tuning_cfg.tuning.points, resolve=False)
    assert [point["model.lambda_coef"] for point in points] == [0.0001, 0.0001]
    assert [
        point["model.backbone.resnet_block.gate_regularization"]
        for point in points
    ] == ["l1_probability", "l1_activation"]
    assert [
        point["training_arguments.adaptive_lambda.log_step_init"]
        for point in points
    ] == [0.28782313662425574, 0.28782313662425574]
    assert [point["mlflow.tags.regularization_target"] for point in points] == [
        "p",
        "g",
    ]
    assert [point["mlflow.tags.gate_regularization"] for point in points] == [
        "l1_probability",
        "l1_activation",
    ]
    assert [point["mlflow.tags.target_growth_steps"] for point in points] == [
        "40",
        "40",
    ]
    assert [point["mlflow.tags.lambda_multiplier"] for point in points] == [
        "1.33352143216",
        "1.33352143216",
    ]
    assert {point["mlflow.tags.lambda_max"] for point in points} == {"none"}
    assert {point["mlflow.tags.num_epochs"] for point in points} == {"120"}
    assert {point["mlflow.tags.repeats_per_trial"] for point in points} == {"7"}


def test_resnet50_aig_probability_divergences_90ep_repeats2():
    tuning_cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "resnet50_aig_probability_divergences_90ep_repeats2_ordered.yaml"
    )
    tune_cfg = OmegaConf.load(
        CONFIGS_DIR / "tune_resnet50_aig_probability_divergences_90ep.yaml"
    )

    assert tune_cfg.defaults == [
        {"experiment": "resnet50_aig_adaptive_lambda_v1"},
        {"tuning": "resnet50_aig_probability_divergences_90ep_repeats2_ordered"},
        "_self_",
    ]
    assert tuning_cfg.training_arguments.num_epochs == 90
    assert tuning_cfg.scheduler.T_max == 90
    assert tuning_cfg.tuning.n_trials == 3
    assert tuning_cfg.tuning.repeats_per_trial == 2

    points = OmegaConf.to_container(tuning_cfg.tuning.points, resolve=False)
    assert [
        point["model.backbone.resnet_block.gate_regularization"]
        for point in points
    ] == [
        "l1_probability",
        "jensen_shannon_probability",
        "hellinger_probability",
    ]
