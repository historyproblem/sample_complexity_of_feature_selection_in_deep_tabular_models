"""Integration-style tests for the per-cycle recovery orchestration added to
cyclic_aig.py and the new cyclic_channel_pruning.py, using the same
monkeypatch(engine.run_training) pattern as test_engine_adaptive_baseline.py
so the full loop logic is exercised without any real training/data/GPU.
"""
from pathlib import Path

from omegaconf import OmegaConf

from net_complexity.training import cyclic_aig, cyclic_channel_pruning


def _make_layer_config():
    return OmegaConf.create(
        {
            "model": {"lambda_coef": 0.01, "backbone": {"resnet_block": {}}},
            "training_arguments": {"num_epochs": 5, "adaptive_lambda": {"enabled": False}},
            "layer_skipping": {"enabled": True, "disabled_layers": []},
            "cyclic_layer_dropping": {
                "enabled": True,
                "max_cycles": 5,
                "drop_mode": "threshold",
                "g_prob_threshold": 0.5,
                "aig_epochs": 5,
                "recovery_epochs": 2,
                "final_epochs": 9,
                "use_plain_model_for_final": True,
                "disable_mlflow_for_cycles": False,
            },
            "run_history": {"root_dir": "outputs/runs", "use_hydra_output_dir": False},
            "mlflow": {"enabled": True, "run_name": "cyclic_layer_demo"},
        }
    )


def test_cyclic_aig_runs_recovery_pass_every_cycle_and_uses_last_as_final(tmp_path, monkeypatch):
    config = _make_layer_config()
    calls: list[dict] = []

    # Cycle 1: g_prob for layer1.0 below threshold -> drop it.
    # Cycle 2: nothing left below threshold -> converged.
    cycle_metrics = [
        {"valid_g_prob_backbone.layer1.0": 0.1, "valid_g_prob_backbone.layer1.1": 0.9},
        {"valid_g_prob_backbone.layer1.1": 0.9},
    ]

    def _fake_run_training(cfg, epoch_end_callback=None, progress_context=None):
        del epoch_end_callback, progress_context
        is_recovery = float(cfg.model.lambda_coef) == 0.0
        call_index = len(calls)
        calls.append(
            {
                "lambda_coef": float(cfg.model.lambda_coef),
                "num_epochs": int(cfg.training_arguments.num_epochs),
                "disabled_layers": list(cfg.layer_skipping.disabled_layers),
                "run_name": str(cfg.mlflow.run_name),
                "mlflow_enabled": bool(cfg.mlflow.enabled),
            }
        )
        if is_recovery:
            return {"full_train_time_sec": 1.0, "num_epochs_executed": int(cfg.training_arguments.num_epochs)}
        # search cycle: index // 2 tracks which search step this is (0-based)
        search_step = sum(1 for c in calls if float(c["lambda_coef"]) != 0.0) - 1
        metrics = cycle_metrics[min(search_step, len(cycle_metrics) - 1)]
        return {
            "last_valid_metrics": metrics,
            "full_train_time_sec": 1.0,
            "num_epochs_executed": int(cfg.training_arguments.num_epochs),
        }

    monkeypatch.setattr(cyclic_aig, "run_training", _fake_run_training)

    result = cyclic_aig.run_cyclic_aig_training(config, output_root=tmp_path)

    search_calls = [c for c in calls if c["lambda_coef"] != 0.0]
    recovery_calls = [c for c in calls if c["lambda_coef"] == 0.0]

    # Two search cycles: cycle 1 finds a drop, cycle 2 converges.
    assert len(search_calls) == 2
    # A recovery pass runs after *every* cycle, including the converged one.
    assert len(recovery_calls) == 2

    assert search_calls[0]["disabled_layers"] == []
    assert search_calls[1]["disabled_layers"] == ["layer1.0"]

    # Recovery configs never carry regularization and always log to MLflow.
    for recovery in recovery_calls:
        assert recovery["mlflow_enabled"] is True
    assert recovery_calls[0]["disabled_layers"] == ["layer1.0"]
    assert recovery_calls[0]["num_epochs"] == 2  # recovery_epochs
    assert recovery_calls[1]["num_epochs"] == 9  # final_epochs (last recovery)

    assert result["num_cycles_completed"] == 2
    assert result["final_disabled_layers"] == ["layer1.0"]
    assert len(result["recovery_results"]) == 2
    assert result["final_result"] is result["recovery_results"][-1]
    assert (tmp_path / "cyclic_summary.json").exists()


def _make_channel_config():
    return OmegaConf.create(
        {
            "model": {"lambda_coef": 0.01},
            "training_arguments": {"num_epochs": 5, "adaptive_lambda": {"enabled": False}},
            "cyclic_channel_pruning": {
                "enabled": True,
                "max_cycles": 5,
                "drop_mode": "threshold",
                "g_prob_threshold": 0.5,
                "gumbel_epochs": 5,
                "recovery_epochs": 2,
                "final_epochs": 9,
                "disable_mlflow_for_cycles": False,
            },
            "run_history": {"root_dir": "outputs/runs", "use_hydra_output_dir": False},
            "mlflow": {"enabled": True, "run_name": "cyclic_channel_demo"},
        }
    )


def test_cyclic_channel_pruning_accumulates_disabled_channels_and_prunes_structurally(
    tmp_path, monkeypatch
):
    config = _make_channel_config()
    calls: list[dict] = []

    cycle_metrics = [
        {
            "valid_backbone.layer2.0.gumbel_layer.channel_000_zero_prob": 0.9,  # prob 0.1 -> drop
            "valid_backbone.layer2.0.gumbel_layer.channel_001_zero_prob": 0.1,  # prob 0.9 -> keep
        },
        {
            "valid_backbone.layer2.0.gumbel_layer.channel_001_zero_prob": 0.1,
        },
    ]

    def _fake_run_training(cfg, epoch_end_callback=None, progress_context=None):
        del epoch_end_callback, progress_context
        is_recovery = float(cfg.model.lambda_coef) == 0.0
        mask = (
            OmegaConf.to_container(cfg.channel_pruning.mask, resolve=True)
            if "channel_pruning" in cfg
            else {}
        )
        calls.append(
            {
                "lambda_coef": float(cfg.model.lambda_coef),
                "num_epochs": int(cfg.training_arguments.num_epochs),
                "structural": bool(cfg.channel_pruning.structural) if "channel_pruning" in cfg else None,
                "mask": mask,
                "mlflow_enabled": bool(cfg.mlflow.enabled),
            }
        )
        if is_recovery:
            return {"full_train_time_sec": 1.0, "num_epochs_executed": int(cfg.training_arguments.num_epochs)}
        search_step = sum(1 for c in calls if c["lambda_coef"] != 0.0) - 1
        metrics = cycle_metrics[min(search_step, len(cycle_metrics) - 1)]
        return {
            "last_valid_metrics": metrics,
            "full_train_time_sec": 1.0,
            "num_epochs_executed": int(cfg.training_arguments.num_epochs),
        }

    monkeypatch.setattr(cyclic_channel_pruning, "run_training", _fake_run_training)

    result = cyclic_channel_pruning.run_cyclic_channel_pruning_training(config, output_root=tmp_path)

    search_calls = [c for c in calls if c["lambda_coef"] != 0.0]
    recovery_calls = [c for c in calls if c["lambda_coef"] == 0.0]

    assert len(search_calls) == 2
    assert len(recovery_calls) == 2

    # Search cycles use soft (non-structural) masking.
    assert all(c["structural"] is False for c in search_calls)
    # Recovery cycles physically prune.
    assert all(c["structural"] is True for c in recovery_calls)

    assert search_calls[0]["mask"] == {}
    assert search_calls[1]["mask"] == {"backbone.layer2.0.gumbel_layer": [0]}
    assert recovery_calls[0]["mask"] == {"backbone.layer2.0.gumbel_layer": [0]}

    assert result["num_cycles_completed"] == 2
    assert result["final_disabled_channels"] == {"backbone.layer2.0.gumbel_layer": [0]}
    assert result["final_result"] is result["recovery_results"][-1]
    assert (tmp_path / "cyclic_summary.json").exists()


def test_cyclic_channel_pruning_hands_recovery_weights_to_next_search(
    tmp_path, monkeypatch
):
    config = _make_channel_config()
    config.cyclic_channel_pruning.weight_handoff = OmegaConf.create(
        {
            "enabled": True,
            "checkpoint_name": "best.pt",
            "initial_checkpoint": None,
            "initial_run_dir": None,
        }
    )
    calls: list[dict] = []
    initializer_events: list[tuple] = []
    cycle_metrics = [
        {
            "valid_backbone.layer2.0.gumbel_layer.channel_000_zero_prob": 0.9,
            "valid_backbone.layer2.0.gumbel_layer.channel_001_zero_prob": 0.1,
        },
        {
            "valid_backbone.layer2.0.gumbel_layer.channel_001_zero_prob": 0.1,
        },
    ]

    def _search_to_structural(_cfg, checkpoint_path):
        initializer_events.append(("search_to_structural", checkpoint_path))
        return lambda _model: None

    def _next_search(
        _cfg,
        disabled_channels,
        previous_search_checkpoint,
        previous_recovery_checkpoint,
    ):
        initializer_events.append(
            (
                "next_search",
                dict(disabled_channels),
                previous_search_checkpoint,
                previous_recovery_checkpoint,
            )
        )
        return lambda _model: None

    def _fake_run_training(
        cfg,
        epoch_end_callback=None,
        progress_context=None,
        model_initializer=None,
    ):
        del epoch_end_callback, progress_context
        is_recovery = float(cfg.model.lambda_coef) == 0.0
        run_dir = tmp_path / f"fake_run_{len(calls)}"
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        (checkpoints_dir / "best.pt").write_bytes(b"checkpoint-placeholder")
        calls.append(
            {
                "is_recovery": is_recovery,
                "has_initializer": model_initializer is not None,
                "run_dir": run_dir,
            }
        )
        if is_recovery:
            return {
                "run_dir": str(run_dir),
                "full_train_time_sec": 1.0,
                "num_epochs_executed": int(cfg.training_arguments.num_epochs),
            }
        search_step = sum(not call["is_recovery"] for call in calls) - 1
        return {
            "run_dir": str(run_dir),
            "last_valid_metrics": cycle_metrics[min(search_step, 1)],
            "full_train_time_sec": 1.0,
            "num_epochs_executed": int(cfg.training_arguments.num_epochs),
        }

    monkeypatch.setattr(cyclic_channel_pruning, "run_training", _fake_run_training)
    monkeypatch.setattr(
        cyclic_channel_pruning,
        "_search_to_structural_initializer",
        _search_to_structural,
    )
    monkeypatch.setattr(
        cyclic_channel_pruning,
        "_next_search_initializer",
        _next_search,
    )

    result = cyclic_channel_pruning.run_cyclic_channel_pruning_training(
        config,
        output_root=tmp_path / "cyclic",
    )

    assert [call["has_initializer"] for call in calls] == [False, True, True, True]
    assert [event[0] for event in initializer_events] == [
        "search_to_structural",
        "next_search",
        "search_to_structural",
    ]
    assert initializer_events[1][1] == {
        "backbone.layer2.0.gumbel_layer": [0]
    }
    assert result["weight_handoff"] == {
        "enabled": True,
        "checkpoint_name": "best.pt",
        "initial_checkpoint": None,
        "optimizer_state_transferred": False,
    }
