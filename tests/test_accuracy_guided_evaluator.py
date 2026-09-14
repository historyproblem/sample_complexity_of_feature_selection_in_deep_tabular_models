from pathlib import Path
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_pruning_test as evaluation
from net_complexity.training.accuracy_guided_config import compose_config
from net_complexity.training.pruning_measurement import mask_hash, state_hash, write_json


class FrozenProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(10)

    def forward(self, x, y):
        assert not torch.is_grad_enabled() and not self.training
        return SimpleNamespace(logits=self.bn(x))


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    cfg = compose_config()
    OmegaConf.save(cfg, tmp_path / "resolved_config.yaml")
    model = FrozenProbe()
    weights = model.state_dict()
    widths = {f"backbone.layer{stage}.{index}.mid{mid}_gumbel_layer": 64 * 2 ** (stage - 1)
              for stage, blocks in enumerate((3, 4, 6, 3), 1) for index in range(blocks) for mid in (1, 2)}
    metadata = {"version": 1, "M0": len(widths), "n_b0": widths,
                "normalization": {name: "initial_channels" for name in widths},
                "scaling_contract": "survivor_equivalent_v1"}
    validation = {"accuracy": 0.9, "ce_loss": 0.2, "correct_count": 4500, "example_count": 5000}
    cost = {"physical_total_parameters": 20, "conv_linear_macs_per_image": 123}
    state = {"protocol": "accuracy_guided_gates_v3", "artifact_type": "physical_ungated", "status": "completed",
             "global_epochs_completed": 150, "total_epochs_allocated": 150, "accepted_mask": {},
             "accepted_mask_hash": mask_hash({}), "validation": validation, "common_init_hash": "zero_epoch",
             "seed": 42, "split_indices_hash": "split", "optimizer_steps_total": 100,
             "normalization": "initial_channels", "scaling_contract": "survivor_equivalent_v1",
             "provenance": {"normalization": metadata}, "final_cost": cost, "quality_feasible": True}
    checkpoint = {"protocol": "accuracy_guided_gates_v3", "artifact_type": "physical_ungated",
                  "model_state_dict": weights, "model_state_hash": state_hash(weights), "pruning_mask": {},
                  "mask_hash": mask_hash({}), "validation": validation, "common_init_hash": "zero_epoch",
                  "global_epochs_consumed": 148, "normalization_metadata": metadata,
                  "provenance": {"selected_origin": "validation_best"}}
    write_json(tmp_path / "protocol_state.json", state)
    torch.save(checkpoint, tmp_path / "deployment.pt")
    monkeypatch.setattr(evaluation, "build_structurally_pruned_model_from_config", lambda *a: FrozenProbe())
    monkeypatch.setattr(evaluation, "deployment_cost", lambda *a: cost)
    return tmp_path


def test_v3_load_and_frozen_pass_preserve_weights_bn_and_consumed_ledger(deployment):
    model, record = evaluation.prepare_v3_deployment(deployment)
    before = state_hash(model.state_dict())
    loader = DataLoader(TensorDataset(torch.zeros(5, 10), torch.zeros(5, dtype=torch.long)), batch_size=2)
    metrics, arrays = evaluation.evaluate_fixed(model, loader, "cpu", job="v3", expected_examples=5)
    assert metrics["example_count"] == 5 and len(arrays["label"]) == 5
    assert state_hash(model.state_dict()) == before
    assert model.bn.num_batches_tracked == 0
    assert record["epochs_consumed"] == 150 and record["deployment_epochs_consumed"] == 148
    assert record["artifact_type"] == "physical_ungated"


@pytest.mark.parametrize("mutation", ["gated_type", "gated_tensor", "weights", "normalization", "m0", "incomplete", "smoke"])
def test_v3_refuses_ambiguous_or_mismatched_artifacts(deployment, mutation):
    state_path, ckpt_path = deployment / "protocol_state.json", deployment / "deployment.pt"
    state, ckpt = evaluation.read_json(state_path), torch.load(ckpt_path, weights_only=True)
    if mutation == "gated_type":
        state["artifact_type"] = ckpt["artifact_type"] = "gated_only"
    elif mutation == "gated_tensor":
        ckpt["model_state_dict"]["gumbel_layer.logits"] = torch.zeros(2)
        ckpt["model_state_hash"] = state_hash(ckpt["model_state_dict"])
    elif mutation == "weights":
        ckpt["model_state_dict"]["bn.weight"][0] += 1
    elif mutation == "normalization":
        ckpt["normalization_metadata"]["scaling_contract"] = "double_decay"
    elif mutation == "m0":
        ckpt["normalization_metadata"]["M0"] = state["provenance"]["normalization"]["M0"] = 2
    elif mutation == "incomplete":
        state["global_epochs_completed"] = 149
    elif mutation == "smoke":
        OmegaConf.save(compose_config("accuracy_guided_gates_v3_smoke"), deployment / "resolved_config.yaml")
    write_json(state_path, state)
    torch.save(ckpt, ckpt_path)
    with pytest.raises(ValueError):
        evaluation.prepare_v3_deployment(deployment)


def test_check_only_cannot_construct_test_loader(deployment, monkeypatch):
    monkeypatch.setattr(evaluation, "build_test_loader", lambda *a, **k: pytest.fail("official test constructed"))
    args = SimpleNamespace(run_dir=deployment, output=deployment / "check", check_only=True,
                           protocol_v3=True, device="cpu", dense_source=None, jobs=[])
    assert evaluation.run(args) is None
    assert not args.output.exists()
