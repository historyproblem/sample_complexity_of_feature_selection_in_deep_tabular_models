import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launch_resnet50_cifar10_60_90_parameter_curve",
    ROOT / "scripts/launch_resnet50_cifar10_60_90_parameter_curve.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_curve_is_ten_strict_standard_60_90_profiles():
    rows = launcher.validate_curve_configs()

    assert [row["id"] for row in rows] == [
        "18m", "15m", "12m", "10m", "8m",
        "6m", "5m", "4m", "3m", "2m_floor_probe",
    ]
    assert [row["hard_drop"] for row in rows] == pytest.approx([
        0.0005, 0.00075, 0.001, 0.0025, 0.005,
        0.01, 0.015, 0.03, 0.06, 0.10,
    ])
    assert all(row["soft_drop"] * 2 == row["hard_drop"] for row in rows)
    assert all(row["split"] == "60/90" for row in rows)
    assert all(row["min_keep_ratio"] == pytest.approx(0.001) for row in rows)
    assert all(row["resolved_log_step"] == pytest.approx(0.9210340371976183)
               for row in rows)


def test_floor_probe_is_honestly_marked_unreachable():
    rows = launcher.validate_curve_configs()
    floors = {row["theoretical_parameter_floor"] for row in rows}

    assert floors == {2_876_538}
    assert next(row for row in rows if row["id"] == "3m")[
        "target_architecturally_reachable"
    ] is True
    assert next(row for row in rows if row["id"] == "2m_floor_probe")[
        "target_architecturally_reachable"
    ] is False


def test_plan_is_thirty_gpu_hours_when_all_points_are_fresh(tmp_path):
    plan = launcher._estimate_plan(tmp_path / "curve")

    assert plan["fresh_pairs"] == 10
    assert plan["estimated_single_gpu_hours"] == pytest.approx(30.0)
