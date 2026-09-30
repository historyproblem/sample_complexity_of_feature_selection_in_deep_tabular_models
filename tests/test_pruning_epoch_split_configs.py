import pytest

from net_complexity.training import one_shot_pruning_config as schema


SPLITS = [
    (30, 120, 0.001, 0.002),
    (45, 105, 0.00067, 0.00133),
    (60, 90, 0.0005, 0.001),
    (75, 75, 0.0004, 0.0008),
    (90, 60, 0.00033, 0.00067),
]


@pytest.mark.parametrize("search_epochs,recovery_epochs,soft_drop,hard_drop", SPLITS)
def test_epoch_split_pair_is_a_strict_150_epoch_plain_ce_experiment(
    search_epochs, recovery_epochs, soft_drop, hard_drop
):
    search = schema.compose_config(
        f"experiment/pruning_v3/epoch_split_{search_epochs}_{recovery_epochs}_search"
    )
    recovery = schema.compose_config(
        f"experiment/pruning_v3/epoch_split_{search_epochs}_{recovery_epochs}_recovery"
    )

    assert schema.validate_config(search) == schema.validate_config(recovery) == 150
    assert search.one_shot.protocol == schema.EPOCH_SPLIT_SEARCH_PROTOCOL
    assert recovery.one_shot.protocol == schema.EPOCH_SPLIT_RECOVERY_PROTOCOL
    assert recovery.one_shot.reuse_search_required is True

    for config in (search, recovery):
        assert (config.one_shot.search_epochs, config.one_shot.final_epochs) == (
            search_epochs,
            recovery_epochs,
        )
        assert [stage.epochs for stage in config.accuracy_guided.stage_plan] == [
            search_epochs,
            0,
            recovery_epochs,
        ]
        adaptive = config.training_arguments.adaptive_lambda
        assert adaptive.enabled is True
        assert adaptive.soft_drop == pytest.approx(soft_drop)
        assert adaptive.hard_drop == pytest.approx(hard_drop)
        assert adaptive.log_step == pytest.approx(0.9210340371976183)
        assert config.model.criterion == {"_target_": "torch.nn.CrossEntropyLoss"}
        assert "label_smoothing" not in config.model.criterion
        assert config.optimizer.lr == pytest.approx(0.001)
        assert config.optimizer.weight_decay == pytest.approx(0.0005)
        assert config.scheduler.eta_min == pytest.approx(0.0)
        assert config.accuracy_guided.eligibility.min_keep_ratio == pytest.approx(0.08)

    assert search.accuracy_guided.guard.train_bn_calibration_batches == 0
    assert recovery.accuracy_guided.guard.train_bn_calibration_batches == 200
    assert search.one_shot.search_scheduler_eta_min == pytest.approx(0.0)
    assert recovery.one_shot.search_scheduler_horizon_epochs == search_epochs
    assert schema.resolved_branch_plan(recovery) == [{
        "id": "fresh_optimizer_fresh_scheduler__repeat_1",
        "method": "fresh_optimizer_fresh_scheduler",
        "repeat": "repeat_1",
        "training_seed": 42,
        "model_state": "selected_surviving_state",
        "optimizer_state": "fresh",
        "scheduler_state": schema.FRESH_SCHEDULER,
    }]
    report = schema.resolved_one_shot(recovery, check_inputs=False)
    assert report["budget"]["per_branch_budget_including_shared_search"] == 150
    assert report["budget"]["shared_search_epochs_executed_once"] == search_epochs
    assert report["budget"]["physical_training_epochs_per_branch"] == recovery_epochs
    assert report["execution_policy"]["comparison_axis"] == (
        "search/recovery epoch allocation; ordinary recovery recipe fixed"
    )


def test_epoch_split_protocol_rejects_an_unplanned_split():
    config = schema.compose_config("experiment/pruning_v3/epoch_split_60_90_search")
    config.one_shot.search_epochs = 50
    config.one_shot.final_epochs = 100
    config.accuracy_guided.stage_plan[0].epochs = 50
    config.accuracy_guided.stage_plan[2].epochs = 100
    with pytest.raises(ValueError, match="epoch-split study requires"):
        schema.validate_config(config)


def test_epoch_split_recovery_preflight_allows_only_lr_ablation(monkeypatch):
    config = schema.compose_config(
        "experiment/pruning_v3/epoch_split_45_105_recovery",
        overrides=["optimizer.lr=0.000857142857"],
    )
    observed = {}

    def fake_validate(shared, **kwargs):
        observed.update(kwargs)
        return {"status": "ready"}

    monkeypatch.setattr(schema, "_validate_v3_inputs", fake_validate)

    assert schema.validate_inputs(config)["status"] == "ready"
    assert observed == {"allow_optimizer_lr_difference": True}
