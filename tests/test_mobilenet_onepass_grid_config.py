import json
from itertools import product
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
MANIFEST_PATH = CONFIGS_DIR / "mobilenet_onepass_grid_thr05_09_search_only_warmup10.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text())
GRID_POINTS = [
    (threshold, band["soft_drop"], band["hard_drop"])
    for threshold, band in product(MANIFEST["thresholds"], MANIFEST["lambda_bands"])
]


def test_manifest_defines_eight_unique_points_including_both_requested_eight_twelve_bands():
    assert set(MANIFEST) == {
        "base_config", "output_root", "seed", "search_only", "thresholds", "lambda_bands",
        "warmup_epochs", "auto_log_step_epoch_fraction",
    }
    assert MANIFEST["base_config"] == "mobilenet_onepass_thr05_gap8_12pp"
    assert MANIFEST["output_root"] == (
        "outputs/runs/mobilenet_onepass_grid_thr05_09_search_only_warmup10"
    )
    assert MANIFEST["search_only"] is True
    assert MANIFEST["seed"] == 42
    assert MANIFEST["warmup_epochs"] == 10
    assert MANIFEST["auto_log_step_epoch_fraction"] == 1.0
    assert MANIFEST["thresholds"] == [0.5, 0.9]
    assert MANIFEST["lambda_bands"] == [
        {"soft_drop": 0.04, "hard_drop": 0.08},
        {"soft_drop": 0.06, "hard_drop": 0.10},
        {"soft_drop": 0.08, "hard_drop": 0.12},
        {"soft_drop": 0.10, "hard_drop": 0.14},
    ]
    assert len(GRID_POINTS) == len(set(GRID_POINTS)) == 8
    assert (0.5, 0.08, 0.12) in GRID_POINTS
    assert (0.9, 0.08, 0.12) in GRID_POINTS
    for band in MANIFEST["lambda_bands"]:
        assert band["hard_drop"] - band["soft_drop"] == pytest.approx(0.04)


@pytest.mark.parametrize(
    ("threshold", "soft_drop", "hard_drop"),
    GRID_POINTS,
    ids=[f"thr{threshold}_soft{soft}_hard{hard}" for threshold, soft, hard in GRID_POINTS],
)
def test_grid_point_composes_threshold_and_band_while_preserving_onepass_method(
    threshold, soft_drop, hard_drop
):
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(
            config_name=MANIFEST["base_config"],
            overrides=[
                "training_arguments.adaptive_lambda.baseline_history_dir="
                "/test/imported_dense_reference_20261002",
                f"++seed={MANIFEST['seed']}",
                f"cyclic_channel_pruning.g_prob_threshold={threshold}",
                "cyclic_channel_pruning.threshold_max_param_fraction=null",
                "++cyclic_channel_pruning.search_only=true",
                "cyclic_channel_pruning.recovery_epochs=0",
                "cyclic_channel_pruning.final_epochs=0",
                f"training_arguments.adaptive_lambda.soft_drop={soft_drop}",
                f"training_arguments.adaptive_lambda.hard_drop={hard_drop}",
                f"training_arguments.adaptive_lambda.warmup_epochs={MANIFEST['warmup_epochs']}",
                "++training_arguments.adaptive_lambda.auto_log_step_epoch_fraction="
                f"{MANIFEST['auto_log_step_epoch_fraction']}",
            ],
        )

    adaptive = cfg.training_arguments.adaptive_lambda
    cyclic = cfg.cyclic_channel_pruning
    assert cfg.seed == 42
    assert cfg.dataloaders.seed == 42
    assert cyclic.enabled is True
    assert cyclic.drop_mode == "threshold"
    assert cyclic.g_prob_threshold == threshold
    assert cyclic.threshold_max_param_fraction is None
    assert cyclic.max_cycles == 1
    assert cyclic.search_only is True
    assert (cyclic.gumbel_epochs, cyclic.recovery_epochs, cyclic.final_epochs) == (30, 0, 0)
    assert cyclic.gumbel_epochs + cyclic.final_epochs == 30
    assert cfg.training_arguments.num_epochs == 30
    assert cfg.scheduler.T_max == 30
    assert adaptive.enabled is True
    assert (adaptive.soft_drop, adaptive.hard_drop) == (soft_drop, hard_drop)
    assert adaptive.log_step_init == "auto"
    assert adaptive.adaptive_log_step_enabled is False
    assert adaptive.warmup_epochs == 10
    assert adaptive.auto_log_step_epoch_fraction == 1.0
    assert adaptive.update_every_epochs == 1
    assert adaptive.recovery.enabled is False
    assert cfg.model.lambda_coef == pytest.approx(1e-4)
    # Structural selection at 0.9 leaves the search predictor's own gate
    # threshold at the inherited 0.5; these are separate decisions.
    assert cfg.model.backbone.block.gate_threshold == 0.5
    assert cfg.model.backbone.block.gate_internal_width is True
    assert cfg.model.backbone.pretrained_weights == "IMAGENET1K_V2"
    assert cfg.model.criterion.label_smoothing == 0
    assert cfg.model.entropy_regularization_coef == 0
