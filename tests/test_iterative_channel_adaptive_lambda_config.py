from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


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
    assert cyclic.g_prob_threshold == 0.1
    assert cyclic.weight_handoff.enabled is True
    assert cyclic.weight_handoff.checkpoint_name == "best.pt"
    assert cfg.mlflow.tags.gap_pair_pp == "0.25/0.50"
    assert cfg.mlflow.tags.post_prune_training == "fine_tune_surviving_weights"
