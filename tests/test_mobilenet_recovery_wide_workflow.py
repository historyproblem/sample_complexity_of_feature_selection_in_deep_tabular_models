import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from net_complexity.models.channel_pruning import (
    apply_channel_mask, build_structurally_pruned_model_from_config,
    transfer_gated_weights_to_structural,
)
from net_complexity.training import engine
from net_complexity.training import mobilenet_selected_recovery as recovery


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("wide_workflow", ROOT / "scripts/run_mobilenet_recovery_wide_grid.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def test_five_real_configs_have_requested_order_and_independent_searches():
    plan = workflow.build_plan(ROOT)
    assert [s.kind for s in plan] == ["recovery45"] + ["search_only"] * 4
    observed = []
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        for stage in plan:
            pos = stage.command.index("--config-name")
            cfg = compose(config_name=stage.command[pos + 1], overrides=stage.command[pos + 2:])
            assert cfg.model.criterion.label_smoothing == 0
            assert cfg.dataloaders.num_workers == 0
            assert cfg.model.backbone.block.gate_internal_width
            if stage.kind == "recovery45":
                assert cfg.training_arguments.num_epochs == cfg.scheduler.T_max == 45
                assert not cfg.training_arguments.adaptive_lambda.enabled
                assert cfg.model.backbone.pretrained_weights is None
            else:
                assert cfg.model.backbone.pretrained_weights == "IMAGENET1K_V2"
                assert cfg.model.lambda_coef == 1e-4
                assert cfg.training_arguments.num_epochs == 30
                assert cfg.training_arguments.adaptive_lambda.enabled
                assert cfg.training_arguments.adaptive_lambda.warmup_epochs == 10
                assert cfg.training_arguments.adaptive_lambda.log_step_init == "auto"
                assert not cfg.training_arguments.adaptive_lambda.adaptive_log_step_enabled
                assert cfg.training_arguments.adaptive_lambda.auto_log_step_epoch_fraction == 1
                assert cfg.cyclic_channel_pruning.search_only
                assert cfg.cyclic_channel_pruning.final_epochs == cfg.cyclic_channel_pruning.recovery_epochs == 0
                assert not cfg.cyclic_channel_pruning.weight_handoff.enabled
                assert cfg.cyclic_channel_pruning.threshold_max_param_fraction is None
                observed.append((cfg.cyclic_channel_pruning.g_prob_threshold,
                                 cfg.training_arguments.adaptive_lambda.soft_drop,
                                 cfg.training_arguments.adaptive_lambda.hard_drop))
    assert observed == [(0.5, .15, .20), (.9, .15, .20), (.5, .20, .25), (.9, .20, .25)]
    assert len({s.directory for s in plan}) == 5


def test_mobilenet_transfer_copies_trained_survivors_bn_and_classifier():
    cfg = OmegaConf.load(ROOT / "configs/mobilenet_onepass_thr05_gap8_12pp.yaml")
    cfg.model.backbone.pretrained_weights = None
    cfg.model.lambda_coef = 1e-4
    cfg.model.backbone.block.force_ones_mask = True
    source = instantiate(cfg.model)
    mask = json.loads((ROOT / "configs/masks/mobilenet_thr09_gap6_10_epoch30.json").read_text())
    pruning = OmegaConf.create({"mode": "explicit", "structural": True, "mask": mask})
    target = build_structurally_pruned_model_from_config(cfg, pruning)
    with torch.no_grad():
        for i, p in enumerate(source.parameters()):
            p.add_(i * 1e-5)
        for module in source.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.add_(.1)
                module.running_var.mul_(1.1)
                module.num_batches_tracked.add_(7)
    apply_channel_mask(source, mask)
    transfer_gated_weights_to_structural(source, target)
    assert sum(p.numel() for p in target.parameters()) == 2251882
    torch.testing.assert_close(source.backbone.classifier[1].weight, target.backbone.classifier[1].weight, rtol=0, atol=0)
    for src, dst in zip(source.backbone.features, target.backbone.features):
        if hasattr(dst, "narrow_full_width_state"):
            expected = dst.narrow_full_width_state({"branch." + k: v for k, v in src.branch.state_dict().items()})
            for key, tensor in dst.branch.state_dict().items():
                torch.testing.assert_close(tensor, expected["branch." + key], rtol=0, atol=0)
        else:
            for key, tensor in dst.state_dict().items():
                torch.testing.assert_close(tensor, src.state_dict()[key], rtol=0, atol=0)
    source.eval()
    target.eval()
    x, y = torch.randn(2, 3, 32, 32), torch.tensor([1, 2])
    with torch.no_grad():
        torch.testing.assert_close(source(x, y).logits, target(x, y).logits, rtol=1e-4, atol=1e-5)
    old = target.backbone.classifier[1].weight.detach().clone()
    target.train()
    optimizer = torch.optim.SGD(target.parameters(), lr=.01)
    target(x, y).loss.backward()
    optimizer.step()
    assert not torch.equal(old, target.backbone.classifier[1].weight)


@pytest.fixture
def source_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "configs/masks").mkdir(parents=True)
    manifest = workflow.load_workflow(ROOT)
    shutil.copyfile(ROOT / manifest["recovery_source"]["mask_file"], repo / manifest["recovery_source"]["mask_file"])
    profile = repo / "source"
    run = profile / "cycle_0/run30"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints/best.pt").write_bytes(b"checkpoint")
    (run / "history.csv").write_text("epoch,valid_accuracy\n" + "".join(f"{e},0.6\n" for e in range(1, 31)))
    cfg = OmegaConf.load(ROOT / "configs/mobilenet_onepass_thr05_gap8_12pp.yaml")
    cfg.model.backbone.pretrained_weights = None
    OmegaConf.save(cfg, run / "config_resolved.yaml")
    for filename, field in (("history.csv", "history_sha256"), ("config_resolved.yaml", "config_sha256")):
        manifest["recovery_source"][field] = recovery.sha256(run / filename)
    (repo / recovery.MANIFEST).write_text(json.dumps(manifest))
    mask = json.loads((repo / manifest["recovery_source"]["mask_file"]).read_text())
    (profile / "channel_mask.json").write_text(json.dumps(mask))
    (profile / "architecture.json").write_text(json.dumps({"total_params": 2251882, "channels_removed": 962}))
    (profile / "cyclic_summary.json").write_text(json.dumps({
        "cycle_results": [{"run_dir": "/old/location/run30", "best_epoch": 30,
                           "num_epochs_executed": 30, "adaptive_lambda_state": {"state": "held", "observed": {"30": .6}}}],
        "recovery_results": [], "final_result": None,
    }))
    return repo, profile, run, cfg


def test_source_identity_and_checkpoint_digest_are_bound(source_repo):
    repo, profile, run, _ = source_repo
    identity = recovery.inspect_source(repo, profile)
    assert identity["source_epoch"] == 30
    assert identity["held_adaptive_lambda_state"] == {"state": "held", "observed": {"30": .6}}
    (run / "checkpoints/best.pt").write_bytes(b"different")
    assert identity["checkpoint_sha256"] != recovery.inspect_source(repo, profile)["checkpoint_sha256"]


@pytest.mark.parametrize("corruption", ["mask", "history", "checkpoint", "epoch"])
def test_wrong_or_missing_source_fails_without_fallback(source_repo, corruption):
    repo, profile, run, _ = source_repo
    if corruption == "mask":
        (profile / "channel_mask.json").write_text("{}")
    elif corruption == "history":
        (run / "history.csv").write_text("epoch\n1\n")
    elif corruption == "checkpoint":
        (run / "checkpoints/best.pt").unlink()
    else:
        path = profile / "cyclic_summary.json"
        summary = json.loads(path.read_text())
        summary["cycle_results"][0]["best_epoch"] = 29
        path.write_text(json.dumps(summary))
    with pytest.raises((ValueError, FileNotFoundError)):
        recovery.inspect_source(repo, profile)


def test_recovery_public_driver_transfers_once_and_preserves_controller(source_repo, monkeypatch, tmp_path):
    repo, profile, run, cfg = source_repo
    source = instantiate(cfg.model)
    torch.save({"epoch": 30, "model_state_dict": source.state_dict(),
                "extra_state": {"adaptive_lambda_state": {"state": "held", "observed": {30: .6}}}}, run / "checkpoints/best.pt")
    cfg.model.lambda_coef = 0
    cfg.training_arguments.num_epochs = cfg.scheduler.T_max = 45
    cfg.training_arguments.adaptive_lambda.enabled = False
    cfg.selected_recovery = {"source_profile": str(profile)}
    output = tmp_path / "recovery_output"
    monkeypatch.setattr(recovery, "__file__", str(repo / "src/net_complexity/training/mobilenet_selected_recovery.py"))
    from hydra.core.hydra_config import HydraConfig
    monkeypatch.setattr(HydraConfig, "get", staticmethod(lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(output)))))
    calls = []

    def train(config, *, model_initializer):
        assert config.channel_pruning.structural
        model = build_structurally_pruned_model_from_config(config, config.channel_pruning)
        model_initializer(model)
        assert sum(p.numel() for p in model.parameters()) == 2251882
        assert model.backbone.pretrained_parameter_names
        torch.testing.assert_close(model.backbone.classifier[1].weight, source.backbone.classifier[1].weight, rtol=0, atol=0)
        calls.append(config.training_arguments.num_epochs)
        return {"num_epochs_executed": 45, "test_metrics": {"test_accuracy": .7}}

    monkeypatch.setattr(engine, "run_training", train)
    result = recovery.recover_selected_model(cfg)
    assert calls == [45]
    assert result["lifetime_training_epochs"] == 75
    assert json.loads((output / "source_provenance.json").read_text())["held_adaptive_lambda_state"] == {"state": "held", "observed": {"30": .6}}


@pytest.mark.parametrize("failure_at", [None, 3])
def test_workflow_runs_recovery_before_four_searches_and_stops_on_failure(monkeypatch, failure_at):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(workflow, "inspect_source", lambda *a: {"checkpoint_sha256": "fixed"})
    monkeypatch.setattr(workflow.single, "validate_imported_reference", lambda *a: {"history": "fixed"})
    monkeypatch.setattr(workflow.single, "preflight", lambda *a, **k: None)
    from contextlib import nullcontext
    monkeypatch.setattr(workflow.single, "exclusive_launcher", lambda *a, **k: nullcontext())
    plan = workflow.build_plan(ROOT)
    done = set()
    calls = []
    monkeypatch.setattr(workflow, "validate_stage", lambda s: s.directory in done)
    monkeypatch.setattr(workflow.single, "write_receipt", lambda *a: None)
    monkeypatch.setattr(workflow.previous_grid, "_write_json", lambda *a: None)
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: None)

    def run(command, **kwargs):
        if "prepare_mobilenet_reference.py" in " ".join(command):
            return
        stage = next(s for s in plan if s.command == command)
        calls.append(stage.kind)
        if len(calls) == failure_at:
            raise workflow.subprocess.CalledProcessError(1, command)
        done.add(stage.directory)

    monkeypatch.setattr(workflow.subprocess, "run", run)
    # build_plan obtains Git HEAD through check_output, which itself uses subprocess.run.
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda *a, **k: "fixed-head\n")
    assert workflow.main([]) == (1 if failure_at else 0)
    assert calls == ["recovery45"] + ["search_only"] * (2 if failure_at else 4)
    if failure_at is None:
        before = list(calls)
        assert workflow.main([]) == 0
        assert calls == before
