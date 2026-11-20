"""Actual noisy ST, strict paired recipes and staged search/recovery handoff."""
from copy import deepcopy
from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import launch_pruning_ablations as launcher
from net_complexity.models.feature_selection import GumbelLayer
from net_complexity.models.pruning_budget import gates
from net_complexity.training.one_shot_pruning_config import (
    compose_config, validate_config, validate_search_handoff,
)
from test_pruning_ablations import fixture


CONFIG = ROOT / "configs/experiment/pruning_v3/ablations_stochastic_st4.yaml"
RECIPE = "experiment/pruning_v3/ablations_internal_stochastic_st"


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_paired_configs_change_only_train_mode_and_run_labels():
    study = launcher.load_study(CONFIG)
    old = {}
    for name in ("ablations_internal_midrange_pair.yaml", "ablations_internal_less_sparse_pair.yaml"):
        source = launcher.load_study(CONFIG.with_name(name))
        for row in source["profiles"]:
            old[row["id"]] = (source, row)
    reference = ROOT / study["reference_output"]
    assert len(study["profiles"]) == 4 and study["configs"] == {"internal": RECIPE}
    for row in study["profiles"]:
        source, original = old[row["id"]]
        assert row == original
        new = launcher.compose_cell(study, row, "internal", reference)
        previous = launcher.compose_cell(source, original, "internal", reference)
        assert validate_config(new) == 150
        assert new.training_arguments.adaptive_lambda.enabled
        assert new.training_arguments.adaptive_lambda.alpha_max == 80
        assert new.model.criterion.get("label_smoothing", 0) == 0
        assert new.one_shot.search_selection_last_epochs == 30
        assert new.one_shot.search_selection_bn_calibration_batches == 0
        before, after = [OmegaConf.to_container(cfg, resolve=True) for cfg in (previous, new)]
        assert after["model"]["backbone"]["resnet_block"]["train_gate_mode"] == "gumbel_hard"
        after["model"]["backbone"]["resnet_block"]["train_gate_mode"] = "ste_hard"
        for section in ("mlflow", "run_history"):
            after[section]["run_name"] = before[section]["run_name"]
        assert before == after
    assert study["output"] == "outputs/runs/pruning_internal_stochastic_st4_20261120"


@pytest.mark.parametrize("field,value", [("beta", 0), ("beta", .5), ("beta", float("nan")),
                                        ("temperature", 0), ("temperature", .5),
                                        ("eval_gate_mode", "gumbel_hard")])
def test_stochastic_recipe_rejects_noise_or_evaluation_drift(field, value):
    cfg = compose_config(RECIPE)
    cfg.model.backbone.resnet_block[field] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_stochastic_study_cannot_silently_fall_back_to_deterministic(monkeypatch):
    import net_complexity.training.one_shot_pruning_config as schema
    study = launcher.load_study(CONFIG)
    deterministic = compose_config("experiment/pruning_v3/ablations_internal")
    monkeypatch.setattr(schema, "compose_config", lambda *a, **kw: deterministic)
    with pytest.raises(ValueError, match="Stochastic study requires"):
        launcher.compose_cell(study, study["profiles"][0], "internal", ROOT / study["reference_output"])


def test_hard_st_sampling_changes_masks_and_keeps_gate_gradients():
    cfg = compose_config(RECIPE)
    block = cfg.model.backbone.resnet_block
    torch.manual_seed(42)
    gate = GumbelLayer(32, beta=block.beta, temperature=block.temperature,
                       train_gate_mode=block.train_gate_mode, eval_gate_mode=block.eval_gate_mode)
    with torch.no_grad():
        gate.logits[:, 0] = 0
        gate.logits[:, 1] = torch.log(torch.tensor(.7 / .3))
    gate.train()
    x = torch.ones(16, 32)
    first, second = gate.compute_gates(x), gate.compute_gates(x)
    assert set(first.detach().flatten().tolist()) == {0., 1.}
    assert not torch.equal(first, second)
    first.sum().backward()
    assert torch.isfinite(gate.logits.grad).all() and gate.logits.grad.abs().sum() > 0
    gate.eval()
    assert torch.equal(gate.compute_gates(x), torch.ones_like(x))
    assert torch.equal(gate.compute_gates(x), gate.compute_gates(x))


@pytest.mark.parametrize("stage,epochs", [("search", 240), ("inspect", 0), ("recover", 1080)])
def test_stochastic_dry_runs_are_explicit_and_training_free(tmp_path, stage, epochs):
    output = tmp_path / "absent"
    plan = launcher.main(["--config", str(CONFIG), "--stage", stage, "--subset", "full",
                          "--output", str(output), "--dry-run"])
    assert (plan["searches"], plan["recoveries"], plan["new_training_epochs"]) == (4, 12, epochs)
    assert plan["gate_modes"]["internal"] == {"train_gate_mode": "gumbel_hard",
        "eval_gate_mode": "deterministic_hard", "beta": 1., "temperature": 1.}
    assert plan["official_test"] is (stage == "recover") and plan["per_model_budget"] == 150
    assert not output.exists()


def test_launcher_dispatches_four_new_searches_only(tmp_path, monkeypatch):
    import net_complexity.training.one_shot_pruning_config as schema
    import net_complexity.training.one_shot_pruning as runtime
    monkeypatch.setattr(schema, "validate_inputs", lambda cfg: {"status": "ready"})
    calls = []
    monkeypatch.setattr(runtime, "run_one_shot_pruning", lambda *a, **kw: calls.append((a, kw)))
    launcher.main(["--config", str(CONFIG), "--stage", "search", "--subset", "full",
                   "--output", str(tmp_path / "searches")])
    assert len(calls) == 4
    assert all(args[0].model.backbone.resnet_block.train_gate_mode == "gumbel_hard" for args, _ in calls)
    assert all(kwargs == {"search_only": True, "reuse_search_from": None} for _, kwargs in calls)


def test_recovery_cannot_reuse_a_deterministic_search():
    stochastic = compose_config(RECIPE)
    deterministic = compose_config("experiment/pruning_v3/ablations_internal")
    with pytest.raises(ValueError, match="different gate setting: train_gate_mode"):
        validate_search_handoff(stochastic, deterministic, 40)


def test_standalone_test_stage_requires_no_search_reference_or_training(tmp_path, monkeypatch):
    import net_complexity.training.one_shot_pruning_config as schema
    import net_complexity.training.one_shot_pruning as runtime
    import stochastic_pruning_test
    study = launcher.load_study(CONFIG)
    for row in study["profiles"]:
        root = tmp_path / row["id"] / "internal/recovery"
        root.mkdir(parents=True)
        (root / "one_shot_state.json").write_text('{"status":"completed"}')
    monkeypatch.setattr(schema, "validate_inputs", lambda *a: pytest.fail("test requested original reference"))
    monkeypatch.setattr(runtime, "run_one_shot_pruning", lambda *a, **kw: pytest.fail("test restarted training"))
    calls = []
    class Reporter:
        def __init__(self, output, profiles, device):
            assert len(profiles) == 4
        def existing(self, root):
            calls.append(root)
        def finish(self):
            return {"status": "completed"}
    monkeypatch.setattr(stochastic_pruning_test, "DeploymentTests", Reporter)
    result = launcher.main(["--config", str(CONFIG), "--stage", "test", "--subset", "full",
                            "--output", str(tmp_path), "--dense-source", str(tmp_path / "missing_reference")])
    assert result["status"] == "completed" and len(calls) == 4


@pytest.mark.parametrize("keep", [False, True])
def test_stochastic_review_and_recovery_use_only_new_searches(tmp_path, monkeypatch, keep):
    import net_complexity.training.one_shot_pruning_config as schema
    import net_complexity.training.one_shot_pruning as runtime
    from test_pruning_ablations import fake_finished_search
    study = launcher.load_study(CONFIG)
    output, reference = tmp_path / "new_series", tmp_path / "reference"
    for row in study["profiles"]:
        cfg = launcher.compose_cell(study, row, "internal", reference)
        fake_finished_search(output / row["id"] / "internal/search", cfg)
    monkeypatch.setattr(schema, "validate_inputs", lambda cfg: {"status": "ready"})
    args = ["--config", str(CONFIG), "--subset", "full", "--output", str(output), "--skip-test",
            "--dense-source", str(reference)]
    receipt = launcher.main(["--stage", "inspect", *args])
    assert receipt["protocol"] == launcher.STOCHASTIC_PROTOCOL and len(receipt["searches"]) == 4
    calls = []
    monkeypatch.setattr(runtime, "run_one_shot_pruning", lambda *a, **kw: calls.append((a, kw)))
    plan = launcher.main(["--stage", "recover", *args, *(["--keep-epoch-checkpoints"] if keep else [])])
    assert len(calls) == 4 and plan["new_training_epochs"] == 1080
    assert all(kw["search_only"] is False and kw["keep_epoch_checkpoints"] is keep for _, kw in calls)
    assert all(kw["reuse_search_from"].is_relative_to(output) for _, kw in calls)


def test_actual_tiny_stochastic_search_then_ungated_recoveries(tmp_path, monkeypatch):
    import net_complexity.training.one_shot_pruning as runtime
    cfg = fixture(tmp_path / "inputs")
    cfg.model.backbone.resnet_block.train_gate_mode = "gumbel_hard"
    validate_config(cfg)
    original = runtime.run_training
    seen = []
    order = []
    def checked_training(config, **kwargs):
        initialized = kwargs["runtime_initialized_callback"]
        def check(model, controller):
            initialized(model, controller)
            active = gates(model)
            if config.training_arguments.accuracy_guided_stage.kind == "search":
                assert active and all(g.train_gate_mode == "gumbel_hard" and g.beta == 1 for g in active.values())
            else:
                assert not active and controller is None
                assert config.training_arguments.adaptive_lambda.enabled is False
                assert config.training_arguments.accuracy_guided_stage.held_controller_state is not None
            seen.append(config.run_history.run_name)
            order.append(config.run_history.run_name)
        kwargs["runtime_initialized_callback"] = check
        return original(config, **kwargs)
    monkeypatch.setattr(runtime, "run_training", checked_training)
    search = tmp_path / "search"
    first = runtime.run_one_shot_pruning(cfg, search, search_only=True)
    assert first["status"] == "search_only_completed"
    source_hash = launcher.digest(search / "selected_checkpoint.pt")
    def frozen_test(model, branch, root):
        assert branch["ledger"]["global_training_epoch"] == 5
        assert not gates(model) and (root / branch["branch"] / "deployment.pt").is_file()
        assert (root / branch["branch"] / "branch_state.json").is_file()
        order.append("test_" + branch["branch"])
    result = runtime.run_one_shot_pruning(cfg, tmp_path / "recovery", reuse_search_from=search,
                                         frozen_deployment_callback=frozen_test)
    assert seen == ["shared_search", "inherited", "scratch", "random"]
    assert order == ["shared_search", "inherited", "test_inherited", "scratch", "test_scratch", "random", "test_random"]
    assert result["status"] == "completed" and result["test_evaluated"] is False
    assert launcher.digest(search / "selected_checkpoint.pt") == source_hash
    for branch in result["branches"].values():
        assert branch["ledger"]["global_training_epoch"] == 5
        assert branch["source_learned_mask_hash"] == first["selection"]["mask_hash"]
