from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_iterative_channel_resnet50_cifar10_uses_point_seven_pruning_threshold():
    with initialize_config_dir(
        config_dir=str(CONFIGS_DIR),
        version_base=None,
    ):
        cfg = compose(
            config_name="cyclic_channel_train",
            overrides=[
                "experiment=ablation/ours_iterative_channels/resnet50_cifar10"
            ],
        )

    assert cfg.cyclic_channel_pruning.g_prob_threshold == 0.7
    assert float(cfg.mlflow.tags.channel_prune_threshold) == 0.7


def test_iterative_channel_gap_0_25_0_50_uses_weight_handoff_and_adaptive_lambda():
    with initialize_config_dir(
        config_dir=str(CONFIGS_DIR),
        version_base=None,
    ):
        cfg = compose(
            config_name="cyclic_channel_train",
            overrides=[
                "experiment=ablation/ours_iterative_channels/"
                "resnet50_cifar10_adaptive_lambda_gap_0_25_0_50"
            ],
        )

    adaptive = cfg.training_arguments.adaptive_lambda
    cyclic = cfg.cyclic_channel_pruning

    assert cfg.model.lambda_coef == 0.0001
    assert adaptive.enabled is True
    assert adaptive.soft_drop == 0.0025
    assert adaptive.hard_drop == 0.0050
    assert adaptive.update_every_epochs == 1
    assert adaptive.log_step_init == "auto"
    assert adaptive.lambda_max is None
    assert adaptive.recovery.enabled is False
    assert cyclic.drop_mode == "threshold"
    assert cyclic.g_prob_threshold == 0.7
    assert cyclic.weight_handoff.enabled is True
    assert cyclic.weight_handoff.checkpoint_name == "best.pt"
    assert cfg.mlflow.tags.gap_pair_pp == "0.25/0.50"
    assert cfg.mlflow.tags.post_prune_training == "fine_tune_surviving_weights"


def test_iterative_channel_gap_150ep_config_has_exact_four_cycle_budget():
    with initialize_config_dir(
        config_dir=str(CONFIGS_DIR),
        version_base=None,
    ):
        cfg = compose(
            config_name="cyclic_channel_train",
            overrides=[
                "experiment=ablation/ours_iterative_channels/"
                "resnet50_cifar10_adaptive_lambda_gap_0_25_0_50_150ep_4cycles"
            ],
        )

    cyclic = cfg.cyclic_channel_pruning
    assert cyclic.max_cycles == 4
    assert cyclic.stop_on_convergence is False
    assert cyclic.gumbel_epochs == 20
    assert cyclic.recovery_epochs == 10
    assert cyclic.final_epochs == 40
    assert (
        cyclic.max_cycles * cyclic.gumbel_epochs
        + (cyclic.max_cycles - 1) * cyclic.recovery_epochs
        + cyclic.final_epochs
    ) == 150
    assert cfg.training_arguments.adaptive_lambda.soft_drop == 0.0025
    assert cfg.training_arguments.adaptive_lambda.hard_drop == 0.005
    assert cfg.cyclic_channel_pruning.weight_handoff.enabled is True
    assert cfg.cyclic_channel_pruning.g_prob_threshold == 0.7


def test_iterative_channel_nightly_has_four_equal_150_epoch_budgets():
    schedules = {
        "four_cycles_150ep_search25_recovery5": (25, 5, 35),
        "four_cycles_150ep_search20_recovery10": (20, 10, 40),
        "four_cycles_150ep_search15_recovery15": (15, 15, 45),
        "four_cycles_150ep_search10_recovery20": (10, 20, 50),
    }

    nightly = OmegaConf.load(CONFIGS_DIR / "cyclic_channel_nightly.yaml")
    configured_schedules = {
        value.strip()
        for value in str(nightly.hydra.sweeper.params.cyclic_schedule).split(",")
    }
    assert configured_schedules == set(schedules)
    assert str(nightly.hydra.mode) == "MULTIRUN"

    with initialize_config_dir(
        config_dir=str(CONFIGS_DIR),
        version_base=None,
    ):
        for schedule, expected_epochs in schedules.items():
            cfg = compose(
                config_name="cyclic_channel_nightly",
                overrides=[f"cyclic_schedule={schedule}"],
            )
            cyclic = cfg.cyclic_channel_pruning
            epochs = (
                cyclic.gumbel_epochs,
                cyclic.recovery_epochs,
                cyclic.final_epochs,
            )
            assert epochs == expected_epochs
            assert cyclic.max_cycles == 4
            assert cyclic.stop_on_convergence is False
            assert cyclic.g_prob_threshold == 0.7
            assert (
                cyclic.max_cycles * cyclic.gumbel_epochs
                + (cyclic.max_cycles - 1) * cyclic.recovery_epochs
                + cyclic.final_epochs
            ) == 150
