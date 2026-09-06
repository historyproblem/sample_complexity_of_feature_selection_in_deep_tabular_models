"""Real optimizer/BN/checkpoint smoke on tiny synthetic data; no downloads or test."""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from pruning_pilot_common import ROOT, config_for, make_initializer

import torch
from torch.utils.data import DataLoader, TensorDataset

from net_complexity.training.pruning_audit import run_fixed_pruning_pilot


class SmokeDataloaders:
    def __init__(self, batch_size=4, include_test=False, loader_seed=42, **kwargs):
        if include_test:
            raise AssertionError("Smoke sentinel: test dataset must never be constructed.")
        rng = torch.Generator().manual_seed(1729)
        train = TensorDataset(torch.randn(8, 3, 16, 16, generator=rng), torch.arange(8) % 10)
        valid = TensorDataset(torch.randn(5, 3, 16, 16, generator=rng), torch.arange(5) % 10)
        self.train_dataloader = DataLoader(train, batch_size=4, shuffle=True,
                                           generator=torch.Generator().manual_seed(loader_seed))
        self.valid_dataloader = DataLoader(valid, batch_size=4)

    @property
    def test_dataloader(self):
        raise AssertionError("Smoke sentinel: test loader accessed.")


def run_smoke(destination, device="cpu", profile="nightly"):
    if profile not in ("nightly", "daytime"):
        raise ValueError("Unknown smoke profile.")
    torch.set_num_threads(1)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    os.environ["AUDIT_INIT_CHECKPOINT"] = str(destination / "shared_init.pt")
    jobs = (("D1_dense_control", "D2_internal_fixed") if profile == "daytime" else
            ("J1_dense_control", "J2_output_fixed", "J3_internal_fixed"))
    base = config_for(jobs[0])
    make_initializer(base, destination / "shared_init.pt")
    summaries = {}
    for job in jobs:
        config = config_for(job)
        config.device = device
        config.dataloaders = {"_target_": "smoke_pruning_pilot.SmokeDataloaders", "loader_seed": 42}
        config.run_history.log_channel_history = False
        config.cyclic_channel_pruning.max_cycles = 1 if profile == "daytime" else 2
        config.cyclic_channel_pruning.gumbel_epochs = 1
        config.cyclic_channel_pruning.recovery_epochs = 1
        config.cyclic_channel_pruning.final_epochs = 1
        config.cyclic_channel_pruning.commit_guard.train_bn_calibration_batches = 1
        # Synthetic labels do not measure quality; allow both transactions to exercise handoff.
        config.cyclic_channel_pruning.commit_guard.max_immediate_accuracy_drop = 1.0
        config.cyclic_channel_pruning.commit_guard.max_recovered_accuracy_drop = 1.0
        result = run_fixed_pruning_pilot(config, destination / job)
        assert result["status"] == "completed"
        expected_epochs = 2 if profile == "daytime" else 4
        assert result["global_epochs_completed"] == expected_epochs
        assert result["optimizer_steps_total"] == expected_epochs * 2
        assert all(d["status"] == "accepted" for d in result["decisions"])
        assert result["validation"]["example_count"] == 5
        if job == jobs[0]:
            assert result["accepted_mask"] == {}
            assert result["final_cost"]["physical_total_parameters"] == 23547338
        else:
            assert result["accepted_mask"]
            if job in ("J3_internal_fixed", "D2_internal_fixed"):
                assert all("mid1_" in key or "mid2_" in key for key in result["accepted_mask"])
        summaries[job] = result
    from launch_pruning_pilot import write_comparison
    write_comparison(destination, list(jobs), profile)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile", choices=["nightly", "daytime"], default="nightly")
    args = parser.parse_args()
    destination = args.output or Path(tempfile.mkdtemp(prefix="pruning-pilot-smoke-"))
    run_smoke(destination, device=args.device, profile=args.profile)
    print(f"Pruning pilot smoke passed: {destination}")
