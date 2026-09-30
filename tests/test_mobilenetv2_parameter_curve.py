import json
import math
from pathlib import Path
import sys

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest
import torch

from net_complexity.models.pruning_budget import PhysicalBudget, gates
from net_complexity.training.one_shot_pruning import run_one_shot_pruning
from net_complexity.training.one_shot_pruning_config import (
    compose_config,
    resolved_branch_plan,
    validate_config,
)
from net_complexity.training.one_shot_reference import prepare_dense_reference


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_mobilenetv2_parameter_curve as curve


EXPECTED_DROPS = [
    (0.0050, 0.0100),
    (0.0075, 0.0100),
    (0.0100, 0.0150),
    (0.0125, 0.0175),
]


def test_nightly_plan_has_four_checked_drop_bands_and_feasible_mobilenet_hints():
    plan = curve.load_plan()

    assert [(point["soft_drop"], point["hard_drop"]) for point in plan["points"]] == EXPECTED_DROPS
    assert [point["target_hint_parameters"] for point in plan["points"]] == [
        1_900_000, 1_370_000, 950_000, 820_000,
    ]
    assert [point["resnet50_analogue_parameters"] for point in plan["points"]] == [
        18_000_000, 13_000_000, 9_000_000, 4_000_000,
    ]
    assert plan["minimum_physical_parameters"] <= min(
        point["target_hint_parameters"] for point in plan["points"]
    )
    assert max(point["target_hint_parameters"] for point in plan["points"]) < plan["dense_physical_parameters"]


def test_all_curve_profiles_are_adaptive_independent_60_plus_90_mobilenet_models():
    plan = curve.load_plan()
    profiles = curve._compose_profiles(plan)

    assert len(profiles) == 4
    for expected, (point, search, recovery) in zip(EXPECTED_DROPS, profiles):
        assert search.model.backbone._target_ == "net_complexity.wrappers.MobileNetV2TinyImageNet200"
        assert recovery.model.backbone._target_ == "net_complexity.wrappers.MobileNetV2TinyImageNet200"
        assert search.model.backbone.block.gate_output is False
        assert search.model.backbone.block.gate_internal_width is True
        assert search.training_arguments.adaptive_lambda.enabled is True
        assert (search.training_arguments.adaptive_lambda.soft_drop,
                search.training_arguments.adaptive_lambda.hard_drop) == expected
        assert (recovery.training_arguments.adaptive_lambda.soft_drop,
                recovery.training_arguments.adaptive_lambda.hard_drop) == expected
        assert [(stage.kind, stage.epochs) for stage in search.accuracy_guided.stage_plan] == [
            ("search", 60), ("commit", 0), ("recovery", 90),
        ]
        assert [(stage.kind, stage.epochs) for stage in recovery.accuracy_guided.stage_plan] == [
            ("search", 60), ("commit", 0), ("recovery", 90),
        ]
        branch_plan = resolved_branch_plan(recovery)
        assert len(branch_plan) == 1
        assert branch_plan[0]["model_state"] == "selected_surviving_state"
        assert branch_plan[0]["optimizer_state"] == "fresh"
        assert branch_plan[0]["scheduler_state"] == "fresh_final_stage_cosine"
        assert recovery.model.criterion.label_smoothing == pytest.approx(0.05)
        assert point["soft_drop"] == expected[0]


def test_checked_dense_size_and_minimum_structural_floor_are_exact():
    plan = curve.load_plan()
    _, search, _ = curve._compose_profiles(plan)[0]
    model = instantiate(search.model)

    assert sum(parameter.numel() for parameter in model.parameters()) == 2_494_280
    assert PhysicalBudget(model, {}).total() == plan["dense_physical_parameters"] == 2_480_072
    ratio = float(search.accuracy_guided.eligibility.min_keep_ratio)
    mask = {}
    for name, gate in gates(model).items():
        keep = max(1, math.ceil(gate.initial_channels * ratio))
        mask[name] = list(range(keep, gate.initial_channels))
    assert PhysicalBudget(model, mask).total() == plan["minimum_physical_parameters"] == 818_984
    assert curve._model_preflight(plan, search, require_cuda=False) == {
        "dense_physical_parameters": 2_480_072,
        "minimum_physical_parameters": 818_984,
    }


def test_child_commands_bind_each_search_to_its_own_recovery(tmp_path):
    plan = curve.load_plan()
    point = plan["points"][2]
    dense = tmp_path / "dense"
    search = tmp_path / "point/search60"
    recovery = tmp_path / "point/recovery90"

    search_command = curve._command(
        plan["search_config"], dense, search, point, search_only=True,
    )
    recovery_command = curve._command(
        plan["recovery_config"], dense, recovery, point,
        reuse_search=search, test_config=tmp_path / "test.yaml", skip_test=True,
    )

    assert search_command.count("--search-only") == 1
    assert "--reuse-search" not in search_command
    assert recovery_command[recovery_command.index("--reuse-search") + 1] == str(search)
    assert "--search-only" not in recovery_command
    assert "--skip-test" in recovery_command
    for command in (search_command, recovery_command):
        assert "training_arguments.adaptive_lambda.soft_drop=0.01" in command
        assert "training_arguments.adaptive_lambda.hard_drop=0.015" in command


def test_dry_run_from_scratch_is_side_effect_free(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(curve.torch.cuda, "is_available", lambda: pytest.fail("CUDA queried"))
    output = tmp_path / "curve"
    reference = tmp_path / "reference"

    report = curve.main([
        "--from-scratch", "--dry-run",
        "--output", str(output), "--reference-output", str(reference),
    ])

    assert json.loads(capsys.readouterr().out) == report
    assert report["training_performed"] is False
    assert report["budget"] == {
        "dense_reference_epochs": 150,
        "epochs_per_curve_model": 150,
        "curve_models": 4,
        "total_curve_training_epochs": 600,
        "total_training_epochs_this_launch": 750,
    }
    assert all(point["adaptive_lambda_enabled"] for point in report["points"])
    assert all((point["search_epochs"], point["recovery_epochs"]) == (60, 90)
               for point in report["points"])
    assert not list(tmp_path.iterdir())


def test_search_and_recovery_records_reject_incomplete_ledgers(tmp_path):
    search = tmp_path / "search"
    (search / "export_only").mkdir(parents=True)
    (search / "one_shot_state.json").write_text(json.dumps({
        "status": "search_only_completed",
        "shared_search_ledger": {"search_epochs_consumed": 59},
    }))
    (search / "export_only/diagnostics.json").write_text("{}")
    (search / "selection.json").write_text("{}")
    with pytest.raises(RuntimeError, match="exactly 60"):
        curve._search_record(search)

    recovery = tmp_path / "recovery"
    recovery.mkdir()
    (recovery / "one_shot_state.json").write_text(json.dumps({
        "status": "completed",
        "branches": {"only": {"ledger": {"global_training_epoch": 149}}},
    }))
    with pytest.raises(RuntimeError, match=r"60\+90"):
        curve._recovery_record(recovery)


def _smoke_config(config_name, reference):
    cfg = compose_config(config_name, dense_source=reference)
    cfg.device = "cpu"
    cfg.accuracy_guided.smoke = True
    cfg.accuracy_guided.total_epochs = cfg.training_arguments.num_epochs = 2
    cfg.one_shot.search_epochs = cfg.one_shot.final_epochs = 1
    if "search_scheduler_horizon_epochs" in cfg.one_shot:
        cfg.one_shot.search_scheduler_horizon_epochs = 1
    if "search_scheduler_eta_min" in cfg.one_shot:
        cfg.one_shot.search_scheduler_eta_min = 0.0
    cfg.accuracy_guided.stage_plan[0].epochs = 1
    cfg.accuracy_guided.stage_plan[2].epochs = 1
    adaptive = cfg.training_arguments.adaptive_lambda
    adaptive.initial_search_warmup = 0
    adaptive.gap_window = adaptive.reentry_samples = adaptive.update_every_search_epochs = 1
    adaptive.soft_drop, adaptive.hard_drop = 0.0, 1.0
    adaptive.log_step = math.log(2.0)
    cfg.model.backbone.num_classes = 3
    cfg.model.backbone.width_mult = 0.25
    cfg.dataloaders = OmegaConf.create({
        "_target_": "net_complexity.training.pruning_synthetic.SyntheticDataloaders",
        "seed": 42, "loader_seed": 42, "include_test": False, "batch_size": 4,
    })
    assert validate_config(cfg) == 2
    return cfg


def test_real_mobilenet_dense_search_export_and_physical_recovery_smoke(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        reference = tmp_path / "reference"
        search_cfg = _smoke_config(
            "experiment/pruning_v3/mobilenetv2_tinyimagenet200_parameter_curve_search60",
            reference,
        )
        recovery_cfg = _smoke_config(
            "experiment/pruning_v3/mobilenetv2_tinyimagenet200_parameter_curve_recovery90",
            reference,
        )
        reference_state = prepare_dense_reference(search_cfg, reference)
        search_state = run_one_shot_pruning(
            search_cfg, tmp_path / "search", search_only=True,
        )
        recovery_state = run_one_shot_pruning(
            recovery_cfg, tmp_path / "recovery", reuse_search_from=tmp_path / "search",
        )
    finally:
        torch.set_num_threads(previous_threads)

    assert reference_state["status"] == "completed"
    assert search_state["status"] == "search_only_completed"
    assert search_state["export_only"]["gated_equivalent_on_checked_batch"] is True
    assert recovery_state["status"] == "completed"
    assert len(recovery_state["branches"]) == 1
    branch = next(iter(recovery_state["branches"].values()))
    assert branch["ledger"]["global_training_epoch"] == 2
    assert branch["final_cost"]["physical_total_parameters"] == (
        search_state["export_only"]["physical_cost"]["physical_total_parameters"]
    )
