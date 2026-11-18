"""Selected-source orchestration, frozen evaluation and actual tiny recovery cleanup."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import launch_pruning_ablations as launcher
import pruning_recovery_matrix as matrix
import evaluate_one_shot_pruning_test as evaluation
from test_pruning_ablations import fixture
from test_one_shot_pruning_evaluator import FrozenProbe, pair
from net_complexity.training.one_shot_pruning import run_one_shot_pruning
from net_complexity.training.one_shot_pruning_config import compose_config, resolved_branch_plan, ABLATION_PROTOCOL
from net_complexity.training.pruning_measurement import state_hash, mask_hash

CONFIG = ROOT / 'configs/experiment/pruning_v3/ablations_recovery_selected5.yaml'


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def cli_args(output):
    return SimpleNamespace(stage='recover', output=output, dense_source=None, device='cpu',
                           dry_run=False, reference_archive=None, skip_test=True, keep_epoch_checkpoints=False)


def test_manifest_and_public_dry_run(tmp_path, capsys):
    manifest = launcher.load_study(CONFIG)
    assert len(manifest['architectures']) == 5
    assert {r['parameters'] for r in manifest['architectures']} == matrix.PARAMETERS
    for row in manifest['architectures']:
        cfg = compose_config(manifest['config'], [f"training_arguments.adaptive_lambda.soft_drop={row['soft_drop']}",
                                                  f"training_arguments.adaptive_lambda.hard_drop={row['hard_drop']}"])
        assert cfg.training_arguments.adaptive_lambda.alpha_max == 80
        assert cfg.model.backbone.resnet_block.gate_output is False
        assert [r['optimizer_state'] for r in resolved_branch_plan(cfg)] == ['fresh'] * 3
    output = tmp_path / 'absent'
    plan = launcher.main(['--config', str(CONFIG), '--stage', 'recover', '--dry-run', '--output', str(output)])
    assert (plan['models'], plan['new_training_epochs'], plan['new_searches']) == (15, 1350, 0)
    assert plan['automatic_test_after_all_models'] is True and not output.exists()


def test_all_missing_sources_are_reported_before_any_training(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix, 'ROOT', tmp_path)
    monkeypatch.setattr('net_complexity.training.one_shot_pruning.run_one_shot_pruning',
                        lambda *a, **kw: pytest.fail('training began with missing source files'))
    study = launcher.load_study(CONFIG)
    with pytest.raises(FileNotFoundError) as exc:
        matrix.preflight(study, cli_args(tmp_path / 'out'))
    for row in study['architectures']:
        assert row['id'] in str(exc.value)
    assert 'selected_checkpoint.pt' in str(exc.value) and not (tmp_path / 'out').exists()


def fake_prepared(study, tmp_path):
    cfg = compose_config(study['config'])
    return [{'row': row, 'source': tmp_path / 'sources' / row['id'],
             'output': tmp_path / 'out' / row['id'], 'config': cfg,
             'review': {'frozen': row['id']}, 'completed': []} for row in study['architectures']]


def test_dispatch_exactly_fifteen_recoveries_without_search(tmp_path, monkeypatch):
    study = launcher.load_study(CONFIG); args = cli_args(tmp_path / 'out')
    prepared = fake_prepared(study, tmp_path)
    monkeypatch.setattr(matrix, 'preflight', lambda s, a: (args.output, prepared))
    monkeypatch.setattr(launcher, 'inspect_search', lambda source, cfg, **kw: {'frozen': source.name})
    monkeypatch.setattr(matrix, 'check_completed', lambda path, cfg, row: [
        {'architecture': row['id'], 'branch': name, 'new_training_epochs': 90} for name in matrix.BRANCHES])
    monkeypatch.setattr(OmegaConf, 'load', lambda path: prepared[0]['config'])
    calls = []
    def recover(cfg, output, **kwargs):
        assert kwargs['reuse_search_from'].parent == tmp_path / 'sources'
        assert kwargs['keep_epoch_checkpoints'] is False and 'search_only' not in kwargs
        assert [r['id'] for r in resolved_branch_plan(cfg)] == list(matrix.BRANCHES)
        calls.append(output)
        return {}
    monkeypatch.setattr('net_complexity.training.one_shot_pruning.run_one_shot_pruning', recover)
    result = matrix.execute_matrix(study, args)
    assert len(calls) == 5 and result['completed_models'] == 15
    assert sum(r['new_training_epochs'] for r in result['validation_results']) == 1350
    assert (args.output / 'validation_results.csv').is_file()


def test_actual_cpu_cleanup_preserves_source_and_frozen_results(tmp_path):
    cfg = fixture(tmp_path / 'inputs')
    search = tmp_path / 'source'
    run_one_shot_pruning(cfg, search, search_only=True)
    before = {p.relative_to(search): matrix.digest(p) for p in search.rglob('*') if p.is_file()}
    kept = run_one_shot_pruning(cfg, tmp_path / 'kept', reuse_search_from=search)
    cleaned = run_one_shot_pruning(cfg, tmp_path / 'cleaned', reuse_search_from=search, keep_epoch_checkpoints=False)
    for name in matrix.BRANCHES:
        assert kept['branches'][name]['model_state_hash'] == cleaned['branches'][name]['model_state_hash']
        assert cleaned['branches'][name]['ledger']['global_training_epoch'] == 5
        assert cleaned['branches'][name]['completed_epoch_checkpoints_removed']['files'] == 2
        assert (tmp_path / 'cleaned' / name / 'deployment.pt').is_file()
        assert (tmp_path / 'cleaned' / name / 'initial_state.pt').is_file()
        assert not list((tmp_path / 'cleaned' / name).rglob('checkpoints/epoch_*.pt'))
    assert before == {p.relative_to(search): matrix.digest(p) for p in search.rglob('*') if p.is_file()}


@pytest.fixture
def ablation_pair(pair, monkeypatch):
    cfg = compose_config('experiment/pruning_v3/ablations_internal')
    OmegaConf.save(cfg, pair / 'resolved_config.yaml')
    selection = matrix.read_json(pair / 'selection.json')
    expected_models, expected_report = {}, {}
    for index, name in enumerate(matrix.BRANCHES):
        folder = pair / name; folder.mkdir(exist_ok=True)
        origin = pair / ('scratch' if name == 'scratch' else 'inherited')
        state = deepcopy(matrix.read_json(origin / 'branch_state.json'))
        checkpoint = deepcopy(torch.load(origin / 'deployment.pt', weights_only=True))
        mask = deepcopy(selection['pruning_mask'])
        if name == 'random':
            mask = {key: [1] for key in mask}
        expected = FrozenProbe()
        expected.bn.weight.data.fill_(index + 1)
        pre_hash = state_hash(expected.state_dict())
        expected_models[name] = expected
        expected_report[name] = {'pruning_mask': mask, 'initialization_state_hash': pre_hash}
        initial_weights = deepcopy(expected.state_dict())
        initial_weights['bn.running_mean'].fill_(.3)
        initial_weights['bn.num_batches_tracked'].fill_(200)
        initial_hash = state_hash(initial_weights)
        for artifact in (state, checkpoint):
            artifact.update(protocol=ABLATION_PROTOCOL, branch=name, method=name, repeat='repeat_1', training_seed=42,
                pruning_mask=mask, mask_hash=mask_hash(mask), source_learned_mask_hash=selection['mask_hash'],
                optimizer_state_initialization='fresh', scheduler_state_initialization='fresh_final_stage_cosine',
                initialization='pytorch_default_all_trainable_and_bn' if name == 'scratch' else 'selected_surviving_state',
                initialization_state_hash=initial_hash, pre_bn_calibration_state_hash=pre_hash,
                optimizer_handoff=None, scheduler_handoff=None,
                bn_calibration={'requested_batches': 200, 'batches': 200, 'trainable_weights_unchanged': True})
            artifact['provenance']['resolved_config_sha256'] = __import__('hashlib').sha256(OmegaConf.to_yaml(cfg, resolve=True).encode()).hexdigest()
        matrix.write_json(folder / 'branch_state.json', state)
        torch.save(checkpoint, folder / 'deployment.pt')
        torch.save({'model_state_dict': initial_weights, 'model_state_hash': initial_hash}, folder / 'initial_state.pt')
    monkeypatch.setattr(evaluation, 'ablation_expectations', lambda *a: (expected_models, expected_report))
    return pair


def test_frozen_evaluator_accepts_prescribed_random_and_scratch(ablation_pair):
    prepared = evaluation.prepare_branches(ablation_pair)
    assert [r['branch'] for _, r in prepared] == list(matrix.BRANCHES)
    assert prepared[0][1]['mask_hash'] != prepared[2][1]['mask_hash']
    assert len({r['physical_parameters'] for _, r in prepared}) == 1


def test_frozen_evaluator_refuses_unprescribed_random_mask(ablation_pair):
    folder = ablation_pair / 'random'
    state = matrix.read_json(folder / 'branch_state.json')
    checkpoint = torch.load(folder / 'deployment.pt', weights_only=True)
    altered = {key: [2] for key in state['pruning_mask']}
    for value in (state, checkpoint):
        value['pruning_mask'] = altered; value['mask_hash'] = mask_hash(altered)
    matrix.write_json(folder / 'branch_state.json', state)
    torch.save(checkpoint, folder / 'deployment.pt')
    with pytest.raises(ValueError, match='prescribed learned/random mask'):
        evaluation.prepare_branches(ablation_pair)


def test_all_models_frozen_before_test_and_completed_test_is_reused(tmp_path, monkeypatch):
    study = launcher.load_study(CONFIG); args = cli_args(tmp_path / 'out'); args.output.mkdir()
    prepared = fake_prepared(study, tmp_path)
    events = []
    monkeypatch.setattr(matrix, 'check_completed', lambda *a: [])
    def prepare(path):
        events.append(path.name)
        row = next(r for r in study['architectures'] if r['id'] == path.name)
        return [(FrozenProbe().eval(), {'branch': name, 'job': name, 'physical_parameters': row['parameters'],
                    'checkpoint_sha256': row['id'] + name, 'validation': {'accuracy': .9},
                    'conv_linear_macs_per_image': 123}) for name in matrix.BRANCHES]
    monkeypatch.setattr(evaluation, 'prepare_branches', prepare)
    def loader(*a):
        assert len(events) == 5
        assert len(matrix.read_json(args.output / 'frozen_deployments.json')['deployments']) == 15
        return SimpleNamespace(dataset=SimpleNamespace(targets=np.zeros(10000, dtype=np.int64))), {
            'dataset': 'CIFAR10', 'split': 'official_test', 'example_count': 10000}
    monkeypatch.setattr(evaluation.frozen, 'build_test_loader', loader)
    calls = []
    def evaluate(model, loader, device, *, job):
        calls.append(job)
        return {'accuracy': .9, 'ce_loss': .2, 'correct_count': 9000, 'example_count': 10000}, {
            'label': np.zeros(10000, dtype=np.int64), 'prediction': np.zeros(10000, dtype=np.int64)}
    monkeypatch.setattr(evaluation.frozen, 'evaluate_fixed', evaluate)
    result = matrix.evaluate_matrix(study, args.output, args, prepared)
    assert result['status'] == 'completed' and len(calls) == 15
    events.clear()
    monkeypatch.setattr(evaluation.frozen, 'build_test_loader', lambda *a: pytest.fail('completed test reran'))
    matrix.evaluate_matrix(study, args.output, args, prepared)
    assert len(calls) == 15


def test_incomplete_fifteenth_model_prevents_any_test_access(tmp_path, monkeypatch):
    study = launcher.load_study(CONFIG); args = cli_args(tmp_path / 'out'); args.output.mkdir()
    prepared = fake_prepared(study, tmp_path)
    def check(path, *a):
        if path.name == study['architectures'][-1]['id']:
            raise ValueError('incomplete budget')
    monkeypatch.setattr(matrix, 'check_completed', check)
    def prepare(path):
        row = next(r for r in study['architectures'] if r['id'] == path.name)
        return [(FrozenProbe(), {'branch': name, 'physical_parameters': row['parameters']})
                for name in matrix.BRANCHES]
    monkeypatch.setattr(evaluation, 'prepare_branches', prepare)
    monkeypatch.setattr(evaluation.frozen, 'build_test_loader', lambda *a: pytest.fail('test accessed before completion'))
    with pytest.raises(ValueError, match='incomplete budget'):
        matrix.evaluate_matrix(study, args.output, args, prepared)
    assert not (args.output / 'test_evaluation').exists()


def test_public_test_stage_does_not_require_original_search_folders(tmp_path, monkeypatch):
    prepared = [{'fixture': True}]
    monkeypatch.setattr(matrix, 'preflight', lambda *a: pytest.fail('test stage required original searches'))
    monkeypatch.setattr(matrix, 'completed_cells', lambda study, args: (tmp_path, prepared))
    monkeypatch.setattr(matrix, 'evaluate_matrix', lambda study, out, args, cells: cells)
    assert launcher.main(['--config', str(CONFIG), '--stage', 'test']) == prepared
