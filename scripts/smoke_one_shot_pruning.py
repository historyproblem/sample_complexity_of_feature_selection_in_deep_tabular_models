"""Short real ResNet50 one-shot training on synthetic data; never CIFAR/test.

The deliberately near-0.5 synthetic gate initialization exercises physical
removal in seconds. It is not the production initialization or a quality run.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import gc
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from launch_one_shot_pruning import config_for
from net_complexity.models.feature_selection import GumbelLayer
from net_complexity.training.one_shot_pruning import run_dense_reference, run_one_shot_pruning
from net_complexity.training.pruning_measurement import write_json


def run_smoke(output, device="cpu"):
    torch.set_num_threads(1)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke output: {output}")
    output.mkdir(parents=True)
    config = config_for("O1_internal_gap05_1")
    config.device = device
    config.dataloaders = {"_target_": "smoke_pruning_pilot.SmokeDataloaders",
                          "loader_seed": 42, "seed": 42, "include_test": False}
    config.one_shot_pruning.search_epochs = 2
    config.one_shot_pruning.retrain_epochs = 2
    adaptive = config.training_arguments.adaptive_lambda
    adaptive.warmup_epochs, adaptive.update_every_epochs, adaptive.acc_window = 0, 1, 1
    reference_path = output / "D0_dense_reference"
    reference = run_dense_reference(config, reference_path, allow_short_run=True)
    assert reference["global_epochs_completed"] == reference["dense_epochs_completed"] == 2
    config.one_shot_pruning.initial_checkpoint = str(reference_path / "initializer.pt")
    config.one_shot_pruning.reference_history = str(reference_path / "global_history.csv")
    records = {}
    # Only the synthetic smoke uses balanced near-threshold gates. Production
    # paper_resnet50 initialization remains p(open) ~ 0.98, without modification.
    with patch.object(GumbelLayer, "PAPER_RESNET50_INIT_ON_LOGIT", 0.1):
        for mode in ("internal", "output"):
            job = deepcopy(config)
            job.model.backbone.resnet_block.gate_internal_width = mode == "internal"
            job.model.backbone.resnet_block.gate_output = mode == "output"
            destination = output / f"smoke_{mode}"
            state = run_one_shot_pruning(job, destination, allow_short_run=True)
            assert state["status"] == "completed"
            assert state["search_epochs_completed"] == state["retrain_epochs_completed"] == 2
            assert state["global_epochs_completed"] == 4
            assert state["physical_prune_events"] == 1 and state["compression_occurred"]
            assert state["final_cost"]["physical_total_parameters"] < state["initial_cost"]["physical_total_parameters"]
            assert state["scratch_init_hash"] != state["mask_selection"]["disposable_trained_state_hash"]
            assert state["mask_selection"]["parameter_budget"] is None
            assert not state["mask_selection"]["collapsed_boundaries"]
            assert not state["search_weights_reused_for_scratch"] and not state["search_bn_reused_for_scratch"]
            assert not state["test_evaluated"]
            with (destination / "global_history.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            search = [row for row in rows if row["stage"] == "search"]
            scratch = [row for row in rows if row["stage"] == "scratch"]
            assert [int(row["global_epoch"]) for row in rows] == [1, 2, 3, 4]
            assert any(float(row["lambda_used"]) != float(row["lambda_next"]) for row in search)
            assert all(float(row["lambda_used"]) == float(row["lambda_next"]) == 0 for row in scratch)
            assert [int(row["local_epoch"]) for row in scratch] == [1, 2]
            assert state["stages"][1]["initial_state_hash"] == state["scratch_init_hash"]
            records[mode] = {"physical_parameters": state["final_cost"]["physical_total_parameters"],
                             "search_epochs": 2, "scratch_epochs": 2, "status": "passed"}
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    report = {"status": "passed", "device": str(device), "synthetic_only": True,
              "production_quality_claim": False, "test_accessed": False,
              "synthetic_gate_initialization": "p_open around 0.5 to exercise removal",
              "runs": records}
    write_json(output / "smoke_summary.json", report)
    print(f"One-shot real optimizer/physical prune/fresh scratch smoke passed: {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = args.output
    if output is None:
        output = Path(tempfile.mkdtemp(prefix="one-shot-smoke-")) / "run"
    run_smoke(output, args.device)


if __name__ == "__main__":
    main()
