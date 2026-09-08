"""Real two-cycle adaptive/structural optimizer smoke; synthetic data, never test."""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import tempfile

from pruning_pilot_common import config_for, make_initializer

import torch
from net_complexity.training.pruning_audit import run_adaptive_pruning_pilot
from net_complexity.training.pruning_measurement import write_json


def run_smoke(destination, device="cpu"):
    torch.set_num_threads(1)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    initializer = destination / "shared_random_seed42.pt"
    os.environ["AUDIT_INIT_CHECKPOINT"] = str(initializer)
    reference = destination / "reference.csv"
    reference.write_text("global_epoch,valid_accuracy\n" +
                         "".join(f"{epoch},0.0\n" for epoch in range(1, 11)))
    make_initializer(config_for("A1_internal_p18"), initializer)
    summaries = {}
    # Exercise both different physical pruning boundaries used tonight.
    for job in ("A1_internal_p18", "A4_output_p05"):
        config = config_for(job)
        config.device = device
        config.dataloaders = {"_target_": "smoke_pruning_pilot.SmokeDataloaders", "loader_seed": 42}
        c = config.cyclic_channel_pruning
        c.adaptive_reference_history = str(reference)
        c.max_cycles, c.gumbel_epochs, c.recovery_epochs, c.final_epochs = 2, 4, 1, 1
        c.max_param_fraction = 0.03
        c.commit_guard.train_bn_calibration_batches = 1
        c.commit_guard.max_immediate_accuracy_drop = 1.0
        c.commit_guard.max_recovered_accuracy_drop = 1.0
        adaptive = config.training_arguments.adaptive_lambda
        adaptive.warmup_epochs, adaptive.update_every_epochs, adaptive.acc_window = 0, 1, 1
        result = run_adaptive_pruning_pilot(config, destination / job)
        assert result["status"] == "completed" and result["pilot_version"] == 2
        assert result["protocol"] == "adaptive_lambda_v1" and result["adaptive_lambda_enabled"] is True
        assert result["global_epochs_completed"] == 10 and result["optimizer_steps_total"] == 20
        assert not result["test_evaluated"] and result["validation"]["example_count"] == 5
        assert result["accepted_mask"] and all(d["status"] == "accepted" for d in result["decisions"])
        with (destination / job / "global_history.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        search = [row for row in rows if row["stage"].endswith("search")]
        recovery = [row for row in rows if row["stage"].endswith("recovery")]
        assert [int(row["global_epoch"]) for row in rows] == list(range(1, 11))
        assert len(search) == 8 and len(recovery) == 2
        assert all(float(row["lambda_next"]) >= float(row["lambda_used"]) for row in search)
        assert all(float(row["lambda_next"]) <= float(adaptive.lambda_max) * (1 + 1e-12) for row in search)
        for cycle in range(2):
            assert any(float(row["lambda_next"]) > float(row["lambda_used"])
                       for row in search if row["stage"] == f"cycle_{cycle}_search")
        assert all(row["valid_average_zero_prob"] != "" for row in search)
        assert all(float(row["lambda_used"]) == 0.0 for row in recovery)
        handoffs = result["adaptive_controller_handoffs"]
        assert len(handoffs) == 2
        expected_lambda = math.exp(handoffs[0]["controller_state"]["runtime"]["log_lambda"])
        assert math.isclose(float(search[4]["lambda_used"]), expected_lambda, rel_tol=1e-10)
        assert expected_lambda > float(config.model.lambda_coef), "Controller reset to initial lambda"
        for stage, handoff in zip((s for s in result["stages"] if s["name"].endswith("search")), handoffs):
            checkpoint = torch.load(Path(stage["run_dir"]) / "checkpoints/best.pt", weights_only=True)
            assert checkpoint["extra_state"]["adaptive_lambda_state"] == handoff["controller_state"]
        summaries[job] = {"status": "passed", "epochs": 10, "adaptive_lambda_changed": True,
                          "selected_controller_carried": True, "test_access": False,
                          "physical_parameters": result["final_cost"]["physical_total_parameters"]}
        write_json(destination / "smoke_summary.json", summaries)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="adaptive-pruning-smoke-")) / "run"
    run_smoke(output, args.device)
    print(f"Adaptive pruning smoke passed: {output}")
