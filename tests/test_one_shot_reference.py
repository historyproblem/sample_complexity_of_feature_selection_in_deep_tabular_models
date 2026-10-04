"""A clean clone can prepare a measured common reference without historical files."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from net_complexity.models.pruning_budget import gates
from net_complexity.training.one_shot_pruning_config import (
    compose_config, dense_source_paths, resolved_one_shot, validate_config, validate_inputs,
)
from net_complexity.training.pruning_measurement import state_hash


@pytest.fixture(autouse=True)
def tiny_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def clean_config(directory):
    """Use real preparation; do not fabricate any reference/initializer file."""
    cfg = compose_config()
    cfg.device = "cpu"
    cfg.accuracy_guided.smoke = True
    cfg.accuracy_guided.total_epochs = cfg.training_arguments.num_epochs = 5
    cfg.scheduler.T_max = 5
    cfg.one_shot.search_epochs = 3
    cfg.one_shot.final_epochs = 2
    cfg.accuracy_guided.stage_plan[0].epochs = 3
    cfg.accuracy_guided.stage_plan[2].epochs = 2
    cfg.training_arguments.adaptive_lambda.initial_search_warmup = 0
    cfg.training_arguments.adaptive_lambda.gap_window = 1
    cfg.training_arguments.adaptive_lambda.update_every_search_epochs = 1
    cfg.training_arguments.adaptive_lambda.reentry_samples = 1
    # Quality on five synthetic examples is not a scientific claim. A permissive
    # technical threshold lets the complete handoff execute regardless of it.
    cfg.training_arguments.adaptive_lambda.soft_drop = 0.0
    cfg.training_arguments.adaptive_lambda.hard_drop = 1.0
    OmegaConf.update(cfg, "model.backbone.base_width", 2, force_add=True)
    cfg.model.backbone.num_classes = 3
    cfg.dataloaders = {
        "_target_": "net_complexity.training.pruning_synthetic.SyntheticDataloaders",
        "seed": 42, "loader_seed": 42, "include_test": False, "batch_size": 4,
    }
    cfg.accuracy_guided.reference = {
        "history_path": str(directory / "absent_history.csv"),
        "state_path": str(directory / "absent_state.json"),
        "config_path": str(directory / "absent_config.yaml"),
    }
    cfg.accuracy_guided.initializer.path = str(directory / "absent_initializer.pt")
    validate_config(cfg)
    return cfg


def use_reference(cfg, directory):
    paths = dense_source_paths(directory)
    cfg.accuracy_guided.initializer.path = paths.pop("initializer_path")
    cfg.accuracy_guided.reference = paths
    return cfg


def capture_runtime(module, monkeypatch, captures, epoch_captures=None):
    """Inspect the real post-initialization engine model, before its first batch."""
    original = module.run_training

    def run(config, *args, **kwargs):
        callback = kwargs.get("runtime_initialized_callback")
        epoch_callback = kwargs.get("epoch_end_callback")

        def initialized(model, controller):
            if callback is not None:
                callback(model, controller)
            captures.append({
                "config": deepcopy(config),
                "state": {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()},
                "has_gates": bool(gates(model)),
                "lambda_coef": model.lambda_coef,
                "has_controller": controller is not None,
            })

        kwargs["runtime_initialized_callback"] = initialized
        if epoch_captures is not None:
            def epoch_end(epoch, train, valid, model, optimizer, history):
                if epoch_callback is not None:
                    epoch_callback(epoch, train, valid, model, optimizer, history)
                epoch_captures.append({"epoch": epoch, "train": deepcopy(train), "valid": deepcopy(valid)})
            kwargs["epoch_end_callback"] = epoch_end
        return original(config, *args, **kwargs)

    monkeypatch.setattr(module, "run_training", run)


def test_clean_clone_prepares_measured_reference_and_search_uses_only_common_zero_epoch_state(tmp_path, monkeypatch):
    from net_complexity.training import one_shot_pruning as search_runtime
    from net_complexity.training import one_shot_reference as reference_runtime

    cfg = clean_config(tmp_path)
    before = OmegaConf.to_container(cfg, resolve=True)
    assert not any(Path(path).exists() for path in cfg.accuracy_guided.reference.values())
    assert not Path(cfg.accuracy_guided.initializer.path).exists()
    dense_runtime, dense_epochs = [], []
    capture_runtime(reference_runtime, monkeypatch, dense_runtime, dense_epochs)
    source = tmp_path / "new_reference"
    reference_runtime.prepare_dense_reference(cfg, source)
    assert OmegaConf.to_container(cfg, resolve=True) == before

    job = source / "J1_dense_control"
    state = json.loads((job / "pilot_state.json").read_text())
    initial_path = source / "shared_random_seed42.pt"
    initial_bytes = initial_path.read_bytes()
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    assert initial["trained_epochs"] == 0 and initial["seed"] == 42
    assert initial["model_state_hash"] == state_hash(initial["model_state_dict"])
    assert state["common_init_hash"] == initial["model_state_hash"]
    assert state["status"] == "completed" and not state["accepted_mask"]
    assert state["test_evaluated"] is False
    assert state["global_epochs_completed"] == state["total_epochs_allocated"] == 5
    assert state["reference_training_epochs_actually_executed"] == 5
    assert state["synthetic_reference"] is True
    assert state["reference_kind"] == "measured_synthetic_dense_reference"
    assert state["validation"]["example_count"] == 5
    assert state["ledger"] == {
        "global_training_epoch": 5, "search_epochs_consumed": 0,
        "optimizer_updates": 10, "consumed_training_examples": 40,
    }
    assert state["training_initializer_verified"]
    assert state["initialization_state_hash"] == initial["model_state_hash"]
    assert not any("gumbel_layer" in key for key in initial["model_state_dict"])
    assert len(dense_runtime) == 1
    runtime = dense_runtime[0]
    assert not runtime["has_gates"] and not runtime["has_controller"]
    assert runtime["lambda_coef"] == 0
    assert runtime["config"].scheduler.T_max == runtime["config"].training_arguments.num_epochs == 5
    assert runtime["config"].training_arguments.adaptive_lambda.enabled is False
    assert state_hash(runtime["state"]) == initial["model_state_hash"]

    assert len(dense_epochs) == 5
    with (job / "global_history.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["global_epoch"]) for row in rows] == list(range(1, 6))
    for row, actual in zip(rows, dense_epochs):
        metrics = actual["valid"]
        assert actual["epoch"] == int(row["global_epoch"])
        accuracy = float(row["valid_accuracy"])
        assert accuracy == pytest.approx(metrics["valid_accuracy"])
        assert accuracy * 5 == pytest.approx(round(accuracy * 5))
        assert int(row["valid_correct_count"]) == metrics["valid_correct_count"]
        assert int(row["valid_example_count"]) == metrics["valid_example_count"] == 5
        assert float(row["valid_ce_loss"]) == pytest.approx(metrics["valid_ce_loss"])
        assert actual["train"]["train_regularization_loss"] == 0
    final_paths = list(job.rglob("last.pt"))
    assert len(final_paths) == 1
    final_epoch = torch.load(final_paths[0], map_location="cpu", weights_only=True)
    assert final_epoch["scheduler_state_dict"]["T_max"] == 5
    assert final_epoch["scheduler_step_count"] == 5
    assert not any("gumbel_layer" in key for key in final_epoch["model_state_dict"])
    optimizer_states = final_epoch["optimizer_state_dict"]["state"]
    assert optimizer_states and all(int(item["step"]) == 10 for item in optimizer_states.values())
    assert state_hash(final_epoch["model_state_dict"]) != initial["model_state_hash"]
    selected_row = min(rows, key=lambda row: (
        -float(row["valid_accuracy"]), float(row["valid_ce_loss"]), int(row["global_epoch"])))
    assert state["selected_epoch"] == int(selected_row["global_epoch"])
    deployed = torch.load(job / "deployment.pt", map_location="cpu", weights_only=True)
    selected = torch.load(job / "selected_checkpoint.pt", map_location="cpu", weights_only=True)
    assert state_hash(deployed["model_state_dict"]) == state["model_state_hash"] == selected["model_state_hash"]
    assert deployed["artifact_type"] == "dense_reference_only"
    assert state["validation"]["accuracy"] == float(selected_row["valid_accuracy"])
    assert state["validation"]["ce_loss"] == pytest.approx(float(selected_row["valid_ce_loss"]))

    run_cfg = use_reference(deepcopy(cfg), source)
    inputs = validate_inputs(run_cfg)
    assert inputs["status"] == "ready"
    assert inputs["initializer_model_state_hash"] == initial["model_state_hash"]
    search_captures = []
    capture_runtime(search_runtime, monkeypatch, search_captures)
    result = search_runtime.run_one_shot_pruning(run_cfg, tmp_path / "one_shot")
    assert result["status"] == "completed"
    assert result["provenance"]["dense_reference_weights_loaded"] is False
    assert len(search_captures) == 3
    search = search_captures[0]
    assert search["has_gates"] and search["has_controller"] and search["lambda_coef"] > 0
    assert search["config"].scheduler.T_max == 3
    assert search["config"].training_arguments.adaptive_lambda.enabled is True
    for name, tensor in initial["model_state_dict"].items():
        assert torch.equal(search["state"][name], tensor), name
    common_keys = initial["model_state_dict"].keys()
    assert any(not torch.equal(search["state"][key], final_epoch["model_state_dict"][key])
               for key in common_keys if key.endswith("conv1.weight"))
    for branch in result["branches"].values():
        assert branch["ledger"]["global_training_epoch"] == 5
        assert branch["ledger"]["search_epochs_consumed"] == 3
    assert result["compute_ledger"]["actual_training_epochs_executed"] == 7
    assert result["test_evaluated"] is False
    assert initial_path.read_bytes() == initial_bytes
    assert hashlib.sha256(initial_path.read_bytes()).hexdigest() == inputs["sha256"]["initializer_path"]


def test_reference_output_is_never_overwritten_or_implicitly_resumed(tmp_path, monkeypatch):
    from net_complexity.training import one_shot_reference as runtime
    cfg = clean_config(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "user_file.txt"
    marker.write_text("preserve me")
    monkeypatch.setattr(runtime, "run_training", lambda *args, **kwargs: pytest.fail("must not train"))
    with pytest.raises(FileExistsError):
        runtime.prepare_dense_reference(cfg, output)
    assert marker.read_text() == "preserve me"
    assert list(output.iterdir()) == [marker]


def test_new_reference_rejects_job_path_that_would_escape_output_and_overwrite_parent_initializer(tmp_path, monkeypatch):
    from net_complexity.training import one_shot_reference as runtime
    cfg = clean_config(tmp_path)
    parent_initializer = tmp_path / "shared_random_seed42.pt"
    parent_initializer.write_bytes(b"preserve original user initializer")
    output = tmp_path / "J1_dense_control"
    monkeypatch.setattr(runtime, "run_training", lambda *args, **kwargs: pytest.fail("must not train"))
    with pytest.raises(ValueError, match="(?i)(J1_dense_control|root|inside|directory)"):
        runtime.prepare_dense_reference(cfg, output)
    assert parent_initializer.read_bytes() == b"preserve original user initializer"
    assert not output.exists()


def test_incomplete_dense_training_cannot_publish_a_valid_reference(tmp_path, monkeypatch):
    from net_complexity.training import one_shot_reference as runtime
    cfg = clean_config(tmp_path)
    source = tmp_path / "incomplete"

    def incomplete(*args, **kwargs):
        return {"num_epochs_executed": 0, "test_evaluation_disabled": True,
                "run_dir": str(source / "J1_dense_control" / "training"), "test_metrics": {}}

    monkeypatch.setattr(runtime, "run_training", incomplete)
    with pytest.raises((RuntimeError, ValueError, AssertionError), match="(?i)(incomplete|epoch|initializ|training)"):
        runtime.prepare_dense_reference(cfg, source)
    state = json.loads((source / "J1_dense_control" / "pilot_state.json").read_text())
    assert state["status"] != "completed"
    assert not (source / "J1_dense_control" / "deployment.pt").exists()
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_inputs(use_reference(cfg, source))


def test_full_reference_rejects_synthetic_runtime_before_training(tmp_path, monkeypatch):
    from net_complexity.training import one_shot_reference as runtime
    cfg = compose_config()
    cfg.dataloaders = clean_config(tmp_path).dataloaders
    monkeypatch.setattr(runtime, "run_training", lambda *args, **kwargs: pytest.fail("must not train"))
    with pytest.raises(ValueError, match="(?i)(synthetic|cifar|full|smoke)"):
        runtime.prepare_dense_reference(cfg, tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_default_full_reference_passes_domain_validation_before_cuda_availability_guard(tmp_path, monkeypatch):
    from net_complexity.training import one_shot_reference as runtime
    cfg = compose_config()
    assert cfg.device.startswith("cuda")
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runtime, "instantiate", lambda *args, **kwargs: pytest.fail("must not construct data or model"))
    monkeypatch.setattr(runtime, "run_training", lambda *args, **kwargs: pytest.fail("must not train"))
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        runtime.prepare_dense_reference(cfg, tmp_path / "cuda_unavailable")
    assert not (tmp_path / "cuda_unavailable").exists()


def test_smoke_flag_cannot_label_a_full_150_epoch_reference(tmp_path, monkeypatch):
    from net_complexity.training import one_shot_reference as runtime
    cfg = compose_config()
    cfg.accuracy_guided.smoke = True
    monkeypatch.setattr(runtime, "run_training", lambda *args, **kwargs: pytest.fail("must not train"))
    with pytest.raises(ValueError, match="(?i)(smoke|short)"):
        runtime.prepare_dense_reference(cfg, tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_unconfigured_smoke_reference_is_reported_as_missing_instead_of_type_error(tmp_path):
    cfg = clean_config(tmp_path)
    cfg.accuracy_guided.reference.state_path = None
    report = resolved_one_shot(cfg, output_root=tmp_path / "preview")
    assert report["inputs"]["status"] == "blocked_missing_inputs"
    assert "state_path" in report["inputs"]["missing_inputs"]
    assert report["inputs"]["paths"]["state_path"] is None
    with pytest.raises(FileNotFoundError, match="<not configured>"):
        validate_inputs(cfg)
    assert not (tmp_path / "preview").exists()
