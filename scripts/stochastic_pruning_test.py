"""Official frozen test immediately after each completed stochastic-ST recovery."""
from pathlib import Path
import json

import numpy as np
from omegaconf import OmegaConf
import torch

import evaluate_one_shot_pruning_test as evaluation
from net_complexity.models.pruning_budget import gates
from net_complexity.training.interruption import TrainingInterrupted
from net_complexity.training.pruning_measurement import isolated_diagnostic_rng, state_hash, write_json


PROTOCOL = "stochastic_st_per_deployment_official_test_v1"


class DeploymentTests:
    def __init__(self, output, profile_ids, device=None):
        self.output = Path(output) / "test_evaluation"
        self.expected = {profile + "__" + branch for profile in profile_ids
                         for branch in ("inherited", "scratch", "random")}
        self.device = device
        self.loader = None
        self.dataset = None
        path = self.output / "test_summary.json"
        self.report = json.loads(path.read_text()) if path.exists() else {
            "protocol": PROTOCOL, "status": "running", "comparison_scope": "exploratory",
            "training_performed": False, "bn_recalibration": False, "test_based_selection": False,
            "test_evaluated": False, "planned_jobs": sorted(self.expected), "runs": [], "errors": {}}
        evaluation.require(self.report["protocol"] == PROTOCOL
                           and set(self.report["planned_jobs"]) == self.expected, "Test report belongs to another study")

    def save(self):
        write_json(self.output / "test_summary.json", self.report)

    def callback(self, config):
        def completed(model, branch, run_dir):
            try:
                self.evaluate(model, branch, run_dir, config)
            except TrainingInterrupted:
                raise
            except Exception as exc:
                job = Path(run_dir).parent.parent.name + "__" + branch["branch"]
                self.report["errors"][job] = f"{type(exc).__name__}: {exc}"
                self.save()
                print(f"[official-test] {job}: failed ({exc}); recovery continues; retry with --stage test", flush=True)
            finally:
                model.cpu()
        return completed

    def evaluate(self, model, branch, run_dir, config):
        run_dir = Path(run_dir)
        profile, name = run_dir.parent.parent.name, branch["branch"]
        job = profile + "__" + name
        require = evaluation.require
        require(job in self.expected and not config.accuracy_guided.smoke,
                "Unexpected/synthetic deployment cannot access official test")
        block = config.model.backbone.resnet_block
        require(block.train_gate_mode == "gumbel_hard" and block.eval_gate_mode == "deterministic_hard"
                and block.beta == 1 and block.temperature == 1, "Expected the approved stochastic-ST recipe")
        require(branch["status"] in {"completed", "infeasible"}
                and branch["ledger"]["global_training_epoch"] == 150
                and branch["ledger"]["search_epochs_consumed"] == 60
                and branch["final_training_epochs_executed"] == 90,
                "Official test requires completed 60+90 training")
        require(not gates(model), "Official test requires a physical ungated deployment")
        model.cpu().eval()
        before = state_hash(model.state_dict())
        require(before == branch["model_state_hash"], "Test model differs from the selected deployment")
        checkpoint = run_dir / name / "deployment.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(state_hash(payload["model_state_dict"]) == before
                and payload["mask_hash"] == branch["mask_hash"]
                and payload["validation"] == branch["validation"], "Saved deployment identity differs")
        del payload
        record = {"job": job, "architecture": profile, "branch": name,
                  "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": evaluation.file_hash(checkpoint),
                  "config_sha256": evaluation.file_hash(run_dir / "resolved_config.yaml"),
                  "model_state_hash": before, "mask_hash": branch["mask_hash"],
                  "training_protocol": branch["protocol"], "gate_regularization_normalization": "initial_channels",
                  "search_gate_modes": {"train_gate_mode": block.train_gate_mode,
                                        "eval_gate_mode": block.eval_gate_mode,
                                        "beta": block.beta, "temperature": block.temperature},
                  "ledger": branch["ledger"], "selected_final_epoch": branch["selected_final_epoch"],
                  "validation": branch["validation"], "quality_feasible": branch["quality_feasible"],
                  "physical_parameters": branch["final_cost"]["physical_total_parameters"],
                  "conv_linear_macs_per_image": branch["final_cost"]["conv_linear_macs_per_image"]}
        freeze_path = self.output / "frozen_deployments" / (job + ".json")
        if freeze_path.exists():
            require(json.loads(freeze_path.read_text()) == record, "Frozen deployment changed")
        else:
            write_json(freeze_path, record)
        freeze_hash = evaluation.file_hash(freeze_path)
        previous = [r for r in self.report["runs"] if r["job"] == job]
        if previous:
            require(len(previous) == 1 and previous[0]["frozen_deployment_sha256"] == freeze_hash,
                    "Prior test used another frozen deployment")
            require(evaluation.file_hash(self.output / previous[0]["predictions"])
                    == previous[0]["predictions_sha256"], "Prior test predictions changed")
            self.report["errors"].pop(job, None)
            return previous[0]
        device = self.device or str(config.device)
        # Test loader construction/iteration must not advance later training RNGs.
        with isolated_diagnostic_rng():
            if self.loader is None:
                self.loader, self.dataset = evaluation.frozen.build_test_loader(
                    config.dataloaders.path_to_data, 128, int(config.dataloaders.num_workers), device, False)
            require(self.dataset["dataset"] == "CIFAR10" and self.dataset["split"] == "official_test"
                    and self.dataset["example_count"] == 10000, "Expected the full official CIFAR-10 test split")
            require("dataset" not in self.report or self.report["dataset"] == self.dataset, "Official test dataset changed")
            metrics, arrays = evaluation.frozen.evaluate_fixed(model, self.loader, device, job=job)
        require(state_hash(model.state_dict()) == before, "Test changed weights or BatchNorm buffers")
        require(evaluation.file_hash(checkpoint) == record["checkpoint_sha256"], "Test changed the saved checkpoint")
        require(np.array_equal(arrays["label"], np.asarray(self.loader.dataset.targets)), "Test sample order changed")
        require(metrics["example_count"] == 10000, "Incomplete official-test inference")
        prediction_path = self.output / (job + "_predictions.npz")
        np.savez_compressed(prediction_path, **arrays)
        result = {**record, "test": metrics, "predictions": prediction_path.name,
                  "predictions_sha256": evaluation.file_hash(prediction_path),
                  "frozen_deployment_sha256": freeze_hash, "model_state_unchanged": True}
        self.report["runs"].append(result)
        self.report["errors"].pop(job, None)
        self.report.update(dataset=self.dataset, test_evaluated=True)
        self.save()
        evaluation.frozen.write_comparison(self.output, self.report["runs"])
        comparison = self.output / "test_comparison.md"
        comparison.write_text(comparison.read_text().replace("\n\n", "\n\nExploratory comparison: prior test results informed experimentation.\n\n", 1))
        print(f"[official-test] {job}: test={metrics['accuracy']:.2%}; "
              f"validation={branch['validation']['accuracy']:.2%}; parameters={record['physical_parameters']:,}", flush=True)
        return result

    def existing(self, run_dir):
        """Resume only inference for already completed, independently verified fits."""
        run_dir = Path(run_dir)
        config = OmegaConf.load(run_dir / "resolved_config.yaml")
        state = json.loads((run_dir / "one_shot_state.json").read_text())
        for model, record in evaluation.prepare_branches(run_dir):
            self.evaluate(model, state["branches"][record["branch"]], run_dir, config)

    def finish(self):
        done = {r["job"] for r in self.report["runs"]}
        self.report["status"] = "completed" if done == self.expected and not self.report["errors"] else "incomplete"
        self.save()
        evaluation.require(self.report["status"] == "completed",
                           "Recovery artifacts retained; official test incomplete. Retry --stage test; no training rerun needed")
        print(f"Official test complete: {len(done)}/{len(self.expected)} deployments; "
              f"results: {self.output / 'test_comparison.csv'}", flush=True)
        return self.report
