from pathlib import Path

import torch
from omegaconf import OmegaConf

from net_complexity.metrics.aig import AIGFLOPsMetric
from net_complexity.models.efficientnet_v2_aig import AIGEfficientNetV2M, AIGEfficientNetV2S
from net_complexity.models.feature_selection import get_AIG_modules
from net_complexity.models.outputs import ClassifModelOutput


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


def test_aig_efficientnetv2_s_smoke_forward_32x32():
    model = AIGEfficientNetV2S(num_classes=10, lambda_coef=0.001)
    model.train()

    logits, aux = model(torch.randn(2, 3, 32, 32))

    assert logits.shape == (2, 10)
    assert aux["gate_probabilities"].shape == (2, 35)
    assert aux["gate_values"].shape == (2, 35)
    assert aux["mean_active_ratio"].ndim == 0
    assert aux["gate_loss"].ndim == 0
    assert 0.0 <= aux["mean_active_ratio"].item() <= 1.0
    assert 0.0 <= aux["gate_loss"].item() <= 1.0


def test_aig_efficientnetv2_s_gates_only_shape_safe_blocks():
    model = AIGEfficientNetV2S(num_classes=10)
    safe_blocks = [
        block
        for block in model.blocks
        if getattr(block, "use_skip_connection", False)
    ]
    changing_blocks = [
        block
        for block in model.blocks
        if not getattr(block, "use_skip_connection", False)
    ]

    assert len(safe_blocks) == 35
    assert len(get_AIG_modules(model)) == len(safe_blocks)
    assert all(block.gate is not None for block in safe_blocks)
    assert all(block.gate is None for block in changing_blocks)


def test_aig_efficientnetv2_m_smoke_forward_32x32_and_gates_shape_safe_blocks():
    model = AIGEfficientNetV2M(num_classes=10, lambda_coef=0.001)
    model.train()

    logits, aux = model(torch.randn(2, 3, 32, 32))
    safe_blocks = [
        block
        for block in model.blocks
        if getattr(block, "use_skip_connection", False)
    ]

    assert logits.shape == (2, 10)
    assert aux["gate_probabilities"].shape == (2, 51)
    assert aux["gate_values"].shape == (2, 51)
    assert len(safe_blocks) == 51
    assert len(get_AIG_modules(model)) == 51


def test_aig_efficientnetv2_s_training_engine_forward_contract():
    model = AIGEfficientNetV2S(num_classes=10, lambda_coef=0.25)
    inputs = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([0, 1])

    output = model(inputs, targets)

    assert isinstance(output, ClassifModelOutput)
    assert output.logits.shape == (2, 10)
    torch.testing.assert_close(
        output.loss,
        output.ce_loss + 0.25 * output.regularization_loss,
    )


def test_aig_efficientnetv2_s_bypass_on_zero_lambda():
    model = AIGEfficientNetV2S(
        num_classes=10,
        lambda_coef=0.0,
        bypass_on_zero_lambda=True,
    )
    model.train()

    _, aux = model(torch.randn(2, 3, 32, 32))

    assert torch.all(aux["gate_values"] == 1.0)
    assert aux["gate_loss"].item() == 0.0

    model.set_lambda_coef(0.001)
    assert all(not module.bypass for module in get_AIG_modules(model).values())


def test_efficientnetv2_aig_experiment_config_points_to_cifar10_adaptive_lambda():
    cfg = OmegaConf.load(
        CONFIGS_DIR / "experiment" / "efficientnetv2_s_aig_adaptive_lambda_cifar10.yaml"
    )

    assert cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "efficientnetv2_s_aig"},
        {"/train": "resnet50_best_practice"},
        {"/optimizer": "adamw"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "aig_train_valid"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 0.001
    assert cfg.training_arguments.adaptive_lambda.enabled is True
    assert cfg.training_arguments.adaptive_lambda.recovery.enabled is False
    assert cfg.dataloaders.taskname == "CIFAR10"


def test_aig_flops_metric_reports_static_and_gate_adjusted_flops():
    model = AIGEfficientNetV2S(num_classes=10)
    model.eval()
    for gate in get_AIG_modules(model).values():
        final_linear = gate.router[-1]
        final_linear.bias.data.fill_(-10.0)

    inputs = torch.randn(2, 3, 32, 32)
    logits, _ = model(inputs)
    output = ClassifModelOutput(logits=logits)
    metric = AIGFLOPsMetric(sample_size=1)

    metric.update(inputs, output, torch.tensor([0, 1]), model)
    computed = metric.compute()

    assert computed["aig_num_gated_blocks"] == 35
    assert computed["aig_static_flops_per_sample"] > 0
    assert computed["aig_active_flops_per_sample"] > 0
    assert computed["aig_skipped_flops_per_sample"] > 0
    assert computed["aig_active_flops_per_sample"] < computed["aig_static_flops_per_sample"]
    assert 0.0 < computed["aig_flops_skip_ratio"] < 1.0


def test_efficientnetv2_aig_scaled_epoch_tuning_config_uses_expected_lambda_steps():
    cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "efficientnetv2_s_aig_adaptive_lambda_init1em4_scaled_step_epochs_50_70_90_140_200_ordered.yaml"
    )

    expected = {
        50: 0.6907755278982136,
        70: 0.49341109135586697,
        90: 0.38376418216567426,
        140: 0.24670554567793349,
        200: 0.1726938819745534,
    }

    assert cfg.training_arguments.adaptive_lambda.lambda_max is None
    assert cfg.training_arguments.adaptive_lambda.update_every_epochs == 2
    assert cfg.training_arguments.adaptive_lambda.adaptive_log_step_enabled is False
    assert len(cfg.tuning.points) == len(expected)

    for point in cfg.tuning.points:
        num_epochs = int(point["training_arguments.num_epochs"])
        assert point["model.lambda_coef"] == 0.0001
        assert point["scheduler.T_max"] == num_epochs
        assert point["training_arguments.adaptive_lambda.log_step_init"] == expected[num_epochs]
        assert point["mlflow.tags.target_lambda"] == "10"
        assert point["mlflow.tags.flops_logging"] == "enabled"


def test_efficientnetv2_aig_init1em4_experiment_uses_flops_metrics():
    cfg = OmegaConf.load(
        CONFIGS_DIR / "experiment" / "efficientnetv2_s_aig_adaptive_lambda_init1em4_cifar10.yaml"
    )

    assert cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "efficientnetv2_s_aig"},
        {"/train": "resnet50_best_practice"},
        {"/optimizer": "adamw"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "aig_train_valid_flops"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert cfg.model.lambda_coef == 0.0001
    assert cfg.training_arguments.adaptive_lambda.lambda_max is None


def test_efficientnetv2_m_aig_init1em4_experiment_and_tuning_config():
    exp_cfg = OmegaConf.load(
        CONFIGS_DIR / "experiment" / "efficientnetv2_m_aig_adaptive_lambda_init1em4_cifar10.yaml"
    )
    tune_cfg = OmegaConf.load(
        CONFIGS_DIR
        / "tuning"
        / "efficientnetv2_m_aig_adaptive_lambda_init1em4_scaled_step_epochs_50_70_90_140_200_ordered.yaml"
    )

    assert exp_cfg.defaults == [
        {"/data": "cifar10_best_practice"},
        {"/model": "efficientnetv2_m_aig"},
        {"/train": "resnet50_best_practice"},
        {"/optimizer": "adamw"},
        {"/scheduler": "cosine_200"},
        {"/metrics": "aig_train_valid_flops"},
        {"/run_history": "valid_accuracy_max"},
        {"/tracking": "default"},
        "_self_",
    ]
    assert exp_cfg.model.lambda_coef == 0.0001
    assert exp_cfg.mlflow.tags.model_type == "EfficientNetV2-M_AIG"
    assert tune_cfg.training_arguments.adaptive_lambda.lambda_max is None
    assert len(tune_cfg.tuning.points) == 5
    assert tune_cfg.tuning.points[0]["training_arguments.num_epochs"] == 50
    assert tune_cfg.tuning.points[-1]["training_arguments.num_epochs"] == 200
