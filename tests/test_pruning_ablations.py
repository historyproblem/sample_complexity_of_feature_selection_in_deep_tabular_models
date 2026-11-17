"""Staged size ablations, real tiny CPU optimization, no CIFAR/test access."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest
import torch

from test_one_shot_pruning import one_shot_fixture, selected_fixture
from net_complexity.models.pruning_budget import gates, make_matched_random_mask
from net_complexity.training.one_shot_pruning import run_one_shot_pruning
from net_complexity.training.one_shot_pruning_config import ABLATION_PROTOCOL, compose_config, validate_config
from net_complexity.training.pruning_ablations import build_ablation_branches
from net_complexity.training.pruning_measurement import deployment_cost, mask_hash, state_hash

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location("ablation_launcher", ROOT / "scripts/launch_pruning_ablations.py")
launcher = importlib.util.module_from_spec(loader)
loader.loader.exec_module(launcher)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(directory):
    cfg = one_shot_fixture(directory)
    cfg.one_shot.protocol = ABLATION_PROTOCOL
    cfg.one_shot.branches = ["inherited", "scratch", "random"]
    cfg.one_shot.inherited_optimizer_state = "fresh"
    for name, value in {"random_mask_seed": 42, "search_scheduler_eta_min": 0.,
                        "search_selection_last_epochs": 2, "search_selection_bn_calibration_batches": 0}.items():
        OmegaConf.update(cfg, "one_shot." + name, value, force_add=True)
    cfg.accuracy_guided.eligibility.min_keep_ratio = .08
    cfg.accuracy_guided.guard.train_bn_calibration_batches = 1
    validate_config(cfg)
    return cfg


@pytest.mark.parametrize("placement,count", [("internal", 3), ("output", 1)])
def test_full_recipes(placement, count):
    cfg = compose_config("experiment/pruning_v3/ablations_" + placement)
    assert validate_config(cfg) == 150
    assert len(cfg.one_shot.branches) == count
    assert cfg.one_shot.inherited_optimizer_state == "fresh"
    assert cfg.accuracy_guided.eligibility.min_keep_ratio == .08
    assert cfg.training_arguments.adaptive_lambda.enabled
    assert cfg.training_arguments.adaptive_lambda.log_step == "auto"
    assert cfg.one_shot.search_selection_last_epochs == 30
    assert cfg.one_shot.search_selection_bn_calibration_batches == 0
    assert cfg.model.criterion.get("label_smoothing", 0) == 0


@pytest.mark.parametrize("key,value", [
    ("one_shot.inherited_optimizer_state", "mapped_adamw_moments_and_step"),
    ("model.criterion.label_smoothing", .05),
    ("accuracy_guided.eligibility.min_keep_ratio", .001),
    ("one_shot.search_selection_last_epochs", 60),
    ("training_arguments.adaptive_lambda.enabled", False),
])
def test_recipe_rejects_method_drift(key, value):
    cfg = compose_config("experiment/pruning_v3/ablations_internal")
    OmegaConf.update(cfg, key, value, force_add=True)
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_matched_masks_preserve_width_cost_rng_and_source(tmp_path):
    cfg = fixture(tmp_path / "inputs")
    carrier, learned, identity = selected_fixture(cfg)
    before, rng = state_hash(carrier.state_dict()), random.getstate()
    masks = [make_matched_random_mask(carrier, learned, seed=seed) for seed in range(8)]
    assert masks[0] == make_matched_random_mask(carrier, learned, seed=0)
    assert len({json.dumps(m, sort_keys=True) for m in masks}) > 1
    assert random.getstate() == rng and state_hash(carrier.state_dict()) == before
    for mask in masks:
        assert {k: len(v) for k, v in mask.items()} == {k: len(v) for k, v in learned.items()}
    models, report = build_ablation_branches(cfg, carrier, learned, identity)
    costs = {name: deployment_cost(m, image_shape=(3, 8, 8)) for name, m in models.items()}
    assert len({c["physical_total_parameters"] for c in costs.values()}) == 1
    assert len({c["conv_linear_macs_per_image"] for c in costs.values()}) == 1
    assert report["inherited"]["architecture_hash"] == report["scratch"]["architecture_hash"]
    assert report["inherited"]["initialization_state_hash"] != report["scratch"]["initialization_state_hash"]
    for module in models["scratch"].modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            assert int(module.num_batches_tracked) == 0
            assert torch.equal(module.running_mean, torch.zeros_like(module.running_mean))
    assert not any(gates(m) for m in models.values())
    assert state_hash(carrier.state_dict()) == before


def test_real_search_then_recovery_never_retrains_search(tmp_path, monkeypatch):
    cfg = fixture(tmp_path / "inputs")
    search = tmp_path / "search"
    initial = run_one_shot_pruning(cfg, search, search_only=True)
    assert initial["status"] == "search_only_completed" and initial["branches"] == {}
    assert initial["export_only"]["bn_calibration_batches"] == 0
    assert initial["compute_ledger"]["actual_training_epochs_executed"] == 3
    pins = {p.relative_to(search).as_posix(): launcher.digest(p) for p in search.rglob("*") if p.is_file()}
    import net_complexity.training.one_shot_pruning as runner
    original = runner.run_training
    calls = []
    def only_recovery(config, **kwargs):
        assert config.training_arguments.accuracy_guided_stage.kind == "physical_final"
        assert config.training_arguments.adaptive_lambda.enabled is False
        assert config.training_arguments.accuracy_guided_stage.held_controller_state is not None
        calls.append(config.run_history.run_name)
        return original(config, **kwargs)
    monkeypatch.setattr(runner, "run_training", only_recovery)
    result = runner.run_one_shot_pruning(cfg, tmp_path / "recovery", reuse_search_from=search)
    assert calls == ["inherited", "scratch", "random"]
    assert result["status"] == "completed" and result["test_evaluated"] is False
    assert result["compute_ledger"]["actual_training_epochs_executed"] == 6
    assert result["compute_ledger"]["reused_search_epochs"] == 3
    for name, branch in result["branches"].items():
        assert branch["ledger"]["global_training_epoch"] == 5
        assert branch["bn_calibration"]["batches"] == 1
        assert branch["optimizer_state_initialization"] == "fresh"
        assert branch["scheduler_state_initialization"] == "fresh_final_stage_cosine"
        assert branch["mask_hash"] == mask_hash(branch["pruning_mask"])
        assert branch["source_learned_mask_hash"] == initial["selection"]["mask_hash"]
        stage = result["stages"][name]
        assert stage["optimizer_state_initialization"] == "fresh"
        assert branch["training_initializer_verified"]
    assert result["branches"]["scratch"]["architecture_hash"] == result["branches"]["inherited"]["architecture_hash"]
    assert len({b["final_cost"]["physical_total_parameters"] for b in result["branches"].values()}) == 1
    assert len({b["final_cost"]["conv_linear_macs_per_image"] for b in result["branches"].values()}) == 1
    assert pins == {p.relative_to(search).as_posix(): launcher.digest(p) for p in search.rglob("*") if p.is_file()}


def test_all_stage_dry_runs_are_training_free(tmp_path):
    for subset, searches, recoveries in [("pilot", 6, 12), ("full", 10, 20)]:
        for stage, epochs in [("search", searches * 60), ("inspect", 0), ("recover", recoveries * 90)]:
            out = tmp_path / (stage + subset)
            code = f"""
import runpy,sys,torch
sys.path.insert(0,'src')
import net_complexity.data.dataloaders as data
def blocked(*a, **kw): raise AssertionError('dry-run touched data/CUDA')
torch.cuda.is_available=blocked
torch.cuda.init=blocked
data.ClassicCVDataloaders.__init__=blocked
sys.argv=['launch_pruning_ablations.py','--stage','{stage}','--subset','{subset}',
          '--output',{str(out)!r},'--dry-run']
def forbidden(name):
    if name == 'net_complexity.training.one_shot_pruning':
        raise AssertionError('dry-run imported training dependency: '+name)
class Trap:
    def find_spec(self, fullname, path=None, target=None): forbidden(fullname)
sys.meta_path.insert(0,Trap())
runpy.run_path('scripts/launch_pruning_ablations.py',run_name='__main__')
"""
            result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, check=True)
            plan = json.loads(result.stdout)
            assert (plan["searches"], plan["recoveries"], plan["new_training_epochs"]) == (searches, recoveries, epochs)
            assert plan["official_test"] is False and not out.exists()


def test_output_search_then_one_physical_recovery(tmp_path):
    cfg = fixture(tmp_path / "inputs")
    cfg.model.backbone.resnet_block.gate_internal_width = False
    cfg.model.backbone.resnet_block.gate_output = True
    cfg.one_shot.branches = ["inherited"]
    source = instantiate(cfg.model)
    with torch.no_grad():
        for gate in gates(source).values():
            gate.logits[:, 0].zero_()
            gate.logits[:, 1].fill_(2.)
            gate.logits[0, 1] = -2.
    h = state_hash(source.state_dict())
    torch.save({"trained_epochs": 0, "seed": 42, "model_state_dict": source.state_dict(),
                "model_state_hash": h}, cfg.accuracy_guided.initializer.path)
    p = Path(cfg.accuracy_guided.reference.state_path)
    state = json.loads(p.read_text()); state["common_init_hash"] = h
    p.write_text(json.dumps(state))
    search = tmp_path / "search"
    result = run_one_shot_pruning(cfg, search, search_only=True)
    assert result["export_only"]["bn_calibration_batches"] == 0
    result = run_one_shot_pruning(cfg, tmp_path / "recovery", reuse_search_from=search)
    assert list(result["branches"]) == ["inherited"]
    assert result["compute_ledger"]["actual_training_epochs_executed"] == 2
    assert result["branches"]["inherited"]["ledger"]["global_training_epoch"] == 5


def fake_finished_search(path, cfg):
    path.mkdir(parents=True)
    (path / "export_only").mkdir()
    OmegaConf.save(cfg, path / "resolved_config.yaml", resolve=True)
    for name in ["selected_checkpoint.pt", "export_only/deployment.pt", "export_only/transfer_equivalence.json"]:
        (path / name).write_text("fixture")
    selection = {"selected_epoch": 50, "reference_epoch": 60, "mask_hash": "mask",
                 "selected_model_state_hash": "model", "selected_checkpoint_hash": launcher.digest(path / "selected_checkpoint.pt"),
                 "policy": "best_feasible_compact",
                 "trace": {"candidate_epoch_start": 31, "candidate_epoch_end": 60, "fallback_used": False}}
    export = {**selection, "gated_equivalent_on_checked_batch": True, "bn_calibration_batches": 0,
              "physical_cost": {"physical_total_parameters": 123, "conv_linear_macs_per_image": 456},
              "physical_validation_before_bn_calibration": {"accuracy": .9}, "retained_learned_closed_channels": []}
    state = {"protocol": ABLATION_PROTOCOL, "status": "search_only_completed", "test_evaluated": False,
             "branches": {}, "selection": selection,
             "shared_search_ledger": {"global_training_epoch": 60, "search_epochs_consumed": 60}}
    for name, payload in [("one_shot_state.json", state), ("selection.json", selection), ("export_only/diagnostics.json", export)]:
        launcher.write_json(path / name, payload)


def test_inspection_receipt_and_all_cell_recovery_preflight(tmp_path, monkeypatch):
    import net_complexity.training.one_shot_pruning_config as schema
    import net_complexity.training.one_shot_pruning as runtime
    study = launcher.load_study(launcher.DEFAULT_CONFIG)
    output, reference = tmp_path / "runs", tmp_path / "reference"
    for profile in study["profiles"]:
        if profile["id"] in study["pilot_profiles"]:
            for placement in ("internal", "output"):
                cfg = launcher.compose_cell(study, profile, placement, reference)
                fake_finished_search(output / profile["id"] / placement / "search", cfg)
    monkeypatch.setattr(schema, "validate_inputs", lambda cfg: {"status": "ready"})
    common = ["--output", str(output), "--dense-source", str(reference), "--subset", "pilot"]
    receipt = launcher.main(["--stage", "inspect", *common])
    assert len(receipt["searches"]) == 6 and receipt["test_evaluated"] is False
    assert (output / "search_summary_pilot.csv").is_file()
    calls = []
    monkeypatch.setattr(runtime, "run_one_shot_pruning", lambda *a, **kw: calls.append((a, kw)))
    changed = output / "gap_300/output/search/selected_checkpoint.pt"
    changed.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        launcher.main(["--stage", "recover", *common])
    assert calls == []  # The first five valid cells must not start before the sixth is checked.
    changed.write_text("fixture")
    launcher.main(["--stage", "recover", *common])
    assert len(calls) == 6
    assert all(kw["search_only"] is False and kw["reuse_search_from"].name == "search" for _, kw in calls)


def test_reference_archive_restores_only_inputs_without_path_traversal(tmp_path):
    import io
    import tarfile
    archive = tmp_path / "reference.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ["bundle/shared_random_seed42.pt", "bundle/J1_dense_control/global_history.csv",
                     "bundle/J1_dense_control/pilot_state.json", "bundle/J1_dense_control/resolved_config.yaml",
                     "../../unrelated"]:
            info = tarfile.TarInfo(name); info.size = 4
            tar.addfile(info, io.BytesIO(b"data"))
    destination = tmp_path / "restored"
    launcher.restore_reference(archive, destination)
    assert len([p for p in destination.rglob("*") if p.is_file()]) == 4
    assert not (tmp_path / "unrelated").exists()
    with pytest.raises(ValueError, match="already exists"):
        launcher.restore_reference(archive, destination)


@pytest.mark.parametrize("use_archive", [False, True])
def test_missing_reference_cli_explains_actual_flags_without_training(tmp_path, use_archive):
    output = tmp_path / "not_started"
    args = [sys.executable, "scripts/launch_pruning_ablations.py", "--stage", "search",
            "--dense-source", str(tmp_path / "missing_reference"), "--output", str(output)]
    if use_archive:
        args += ["--reference-archive", str(tmp_path / "missing.tar.gz")]
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 2
    assert "--reference-archive" in result.stderr
    assert "epoch_split_dense_reference_seed42_fresh_upload.tar.gz" in result.stderr
    assert "has started" in result.stderr
    assert "--from-scratch" not in result.stderr and "--override" not in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists() and not (tmp_path / "missing_reference").exists()
    if not use_archive:
        assert "--dense-source" in result.stderr


PAIR_CONFIG = ROOT / 'configs/experiment/pruning_v3/ablations_internal_less_sparse_pair.yaml'


def test_less_sparse_pair_changes_only_adaptive_ceiling():
    study = launcher.load_study(PAIR_CONFIG)
    assert list(study['configs']) == ['internal'] and len(study['profiles']) == 2
    reference = ROOT / 'outputs/baseline/one_shot_dense_reference_seed42'
    baseline_study = launcher.load_study(launcher.DEFAULT_CONFIG)
    control = launcher.compose_cell(baseline_study, baseline_study['profiles'][0], 'internal', reference)
    control = OmegaConf.to_container(control, resolve=True)
    control['training_arguments']['adaptive_lambda'].pop('alpha_max')
    for row, ceiling in zip(study['profiles'], [.1, 1.]):
        cfg = launcher.compose_cell(study, row, 'internal', reference)
        assert cfg.training_arguments.adaptive_lambda.enabled is True
        assert cfg.model.backbone.resnet_block.gate_internal_width is True
        assert cfg.model.backbone.resnet_block.gate_output is False
        plain = OmegaConf.to_container(cfg, resolve=True)
        assert plain['training_arguments']['adaptive_lambda'].pop('alpha_max') == ceiling
        assert plain == control
    assert study['output'] != baseline_study['output']


def test_less_sparse_pair_dispatches_exactly_two_searches(tmp_path, monkeypatch):
    import net_complexity.training.one_shot_pruning_config as schema
    import net_complexity.training.one_shot_pruning as runtime
    monkeypatch.setattr(schema, 'validate_inputs', lambda cfg: {'status': 'ready'})
    calls = []
    monkeypatch.setattr(runtime, 'run_one_shot_pruning', lambda *a, **kw: calls.append((a, kw)))
    plan = launcher.main(['--config', str(PAIR_CONFIG), '--stage', 'search', '--subset', 'full',
                          '--output', str(tmp_path / 'searches')])
    assert len(calls) == plan['searches'] == 2
    assert plan['new_training_epochs'] == 120 and plan['placements'] == ['internal']
    assert plan['recoveries'] == 6 and plan['official_test'] is False
    assert [a[0].training_arguments.adaptive_lambda.alpha_max for a, _ in calls] == [.1, 1.]
    assert all(kw['search_only'] is True and kw['reuse_search_from'] is None for _, kw in calls)
    assert all(a[1].parts[-2:] == ('internal', 'search') for a, _ in calls)


def test_less_sparse_pair_review_protocol_and_recovery(tmp_path, monkeypatch):
    import net_complexity.training.one_shot_pruning_config as schema
    import net_complexity.training.one_shot_pruning as runtime
    study = launcher.load_study(PAIR_CONFIG)
    output, reference = tmp_path / 'pair', tmp_path / 'reference'
    for row in study['profiles']:
        cfg = launcher.compose_cell(study, row, 'internal', reference)
        fake_finished_search(output / row['id'] / 'internal/search', cfg)
    monkeypatch.setattr(schema, 'validate_inputs', lambda cfg: {'status': 'ready'})
    args = ['--config', str(PAIR_CONFIG), '--subset', 'full', '--output', str(output),
            '--dense-source', str(reference)]
    receipt = launcher.main(['--stage', 'inspect', *args])
    assert receipt['protocol'] == launcher.INTERNAL_PAIR_PROTOCOL and len(receipt['searches']) == 2
    calls = []
    monkeypatch.setattr(runtime, 'run_one_shot_pruning', lambda *a, **kw: calls.append((a, kw)))
    path = output / 'search_review_full.json'
    altered = deepcopy(receipt); altered['protocol'] = launcher.PROTOCOL
    launcher.write_json(path, altered)
    with pytest.raises(ValueError, match='Invalid search review'):
        launcher.main(['--stage', 'recover', *args])
    assert calls == []
    launcher.write_json(path, receipt)
    plan = launcher.main(['--stage', 'recover', *args])
    assert len(calls) == 2 and plan['new_training_epochs'] == 540
    assert all(kw['search_only'] is False and kw['reuse_search_from'].name == 'search' for _, kw in calls)


def test_less_sparse_pair_rejects_output_arm(tmp_path):
    study = launcher.load_study(PAIR_CONFIG)
    study['configs']['output'] = 'experiment/pruning_v3/ablations_output'
    path = tmp_path / 'bad.yaml'
    import yaml
    path.write_text(yaml.safe_dump(study))
    with pytest.raises(ValueError, match='checked size-ablation recipes'):
        launcher.load_study(path)


def test_pair_missing_reference_hint_preserves_internal_config(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        launcher.main(['--config', str(PAIR_CONFIG), '--stage', 'search', '--subset', 'full',
                       '--dense-source', str(tmp_path / 'missing'), '--output', str(tmp_path / 'out')])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert PAIR_CONFIG.name in error and '--config' in error
    assert '--reference-archive' in error
    assert not (tmp_path / 'out').exists()
