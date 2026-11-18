"""Run the frozen selected-architecture recovery matrix; no new mask searches."""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

PROTOCOL = 'pruning_v3_selected_internal_recovery5_v1'
ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ('resolved_config.yaml', 'one_shot_state.json', 'selection.json', 'selected_checkpoint.pt',
                'export_only/diagnostics.json', 'export_only/deployment.pt', 'export_only/transfer_equivalence.json')
BRANCHES = ('inherited', 'scratch', 'random')
PARAMETERS = {3923049, 5871099, 12373042, 16260925, 23507429}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_manifest(study):
    fields = {'protocol', 'config', 'reference_output', 'output', 'seed', 'architectures',
              'reference_sha256', 'initializer_model_state_hash', 'split_indices_hash'}
    require(set(study) == fields and study['protocol'] == PROTOCOL, 'Invalid selected recovery manifest')
    require(study['config'] == 'experiment/pruning_v3/ablations_internal' and study['seed'] == 42,
            'Selected recovery must preserve internal-only recipe and seed42')
    rows = study['architectures']
    sizes = {r['parameters'] for r in rows}
    require(len(rows) == len(sizes) and sizes in (PARAMETERS, PARAMETERS - {3923049}),
            'Expected the five selected sizes or the four-size matrix without 3.923M')
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate recovery architecture IDs')
    require(set(study['reference_sha256']) == {'history_path', 'state_path', 'config_path', 'initializer_path'},
            'Expected all four pinned reference artifacts')
    for value in [*study['reference_sha256'].values(), study['initializer_model_state_hash'], study['split_indices_hash']]:
        require(isinstance(value, str) and re.fullmatch(r'[a-f0-9]{64}', value), 'Invalid reference checksum')
    for row in rows:
        require(set(row) == {'id', 'search', 'soft_drop', 'hard_drop', 'parameters', 'selected_epoch',
                             'selected_checkpoint_sha256', 'selected_model_state_hash', 'mask_hash'},
                'Only frozen source identities and soft/hard thresholds are allowed')
        require(re.fullmatch(r'[a-zA-Z0-9_-]+', row['id']) is not None, 'Invalid architecture ID')
        path = Path(row['search'])
        require(not path.is_absolute() and '..' not in path.parts and path.parts[:2] == ('outputs', 'runs'),
                'Source searches must be explicit repository-relative run paths')
        require(path.parts[-2:] == ('internal', 'search'), 'Only internal search sources are allowed')
        require(0 <= row['soft_drop'] < row['hard_drop'] <= 1 and 31 <= row['selected_epoch'] <= 60,
                'Invalid source thresholds or selected epoch')
        for key in ('selected_checkpoint_sha256', 'selected_model_state_hash', 'mask_hash'):
            require(re.fullmatch(r'[a-f0-9]{64}', row[key]) is not None, 'Invalid source checksum')
    return study


def canonical(config):
    """Ignore machine locations only; retain every model/training setting."""
    from omegaconf import OmegaConf
    value = OmegaConf.to_container(config, resolve=True)
    value.pop('device')
    for key in value['accuracy_guided']['reference']:
        value['accuracy_guided']['reference'][key] = '<location>'
    value['accuracy_guided']['initializer']['path'] = '<location>'
    return value


def check_completed(path, config, row):
    from omegaconf import OmegaConf
    state = read_json(path / 'one_shot_state.json')
    require(state['status'] == 'completed' and state['test_evaluated'] is False,
            f"{row['id']}: partial recovery retained; refusing to restart or overwrite it")
    require(canonical(OmegaConf.load(path / 'resolved_config.yaml')) == canonical(config),
            f"{row['id']}: saved recovery configuration differs")
    require(digest(path / 'selected_checkpoint.pt') == row['selected_checkpoint_sha256'],
            f"{row['id']}: recovery used a different source checkpoint")
    require(set(state['branches']) == set(BRANCHES), f"{row['id']}: incomplete branch set")
    selection = read_json(path / 'selection.json')
    require(selection['mask_hash'] == row['mask_hash'] and selection['selected_epoch'] == row['selected_epoch'],
            f"{row['id']}: recovery reselected the source mask/epoch")
    records = []
    for name in BRANCHES:
        branch = state['branches'][name]
        require(branch['status'] in {'completed', 'infeasible'}
                and branch['ledger']['global_training_epoch'] == 150
                and branch['ledger']['search_epochs_consumed'] == 60
                and branch['final_training_epochs_executed'] == 90,
                f"{row['id']}/{name}: incomplete 150-epoch budget")
        require(branch['optimizer_state_initialization'] == 'fresh'
                and branch['scheduler_state_initialization'] == 'fresh_final_stage_cosine'
                and branch['training_seed'] == 42 and branch['training_initializer_verified'] is True,
                f"{row['id']}/{name}: recovery initialization policy differs")
        require(branch['bn_calibration']['requested_batches'] == branch['bn_calibration']['batches'] == 200,
                f"{row['id']}/{name}: expected train-only BN200")
        require(branch['final_cost']['physical_total_parameters'] == row['parameters'],
                f"{row['id']}/{name}: architecture size changed")
        require(branch['source_learned_mask_hash'] == row['mask_hash'], 'Source learned-mask identity changed')
        stage = state['stages'][name]
        require(stage['stage_metadata']['kind'] == 'physical_final'
                and stage['stage_metadata']['held_controller_state'] is not None,
                'Physical recovery lost the held adaptive-controller state')
        require(read_json(path / name / 'branch_state.json') == branch, 'Branch state files differ')
        for file in ('initial_state.pt', 'deployment.pt'):
            require((path / name / file).is_file(), f"Missing {path/name/file}")
        records.append({'architecture': row['id'], 'branch': name, 'parameters': row['parameters'],
                        'status': branch['status'], 'validation_accuracy': branch['validation']['accuracy'],
                        'selected_final_epoch': branch['selected_final_epoch'],
                        'new_training_epochs': branch['final_training_epochs_executed'],
                        'model_state_hash': branch['model_state_hash'],
                        'deployment_sha256': digest(path / name / 'deployment.pt')})
    require(len({state['branches'][name]['final_cost']['conv_linear_macs_per_image'] for name in BRANCHES}) == 1,
            'Matched variants changed MAC count')
    return records


def preflight(study, args):
    from launch_pruning_ablations import inspect_search
    from omegaconf import OmegaConf
    from net_complexity.training.one_shot_pruning_config import compose_config, validate_config, validate_inputs
    output = Path(args.output or ROOT / study['output']).resolve()
    reference = Path(args.dense_source or ROOT / study['reference_output']).resolve()
    missing = []
    for row in study['architectures']:
        source = (ROOT / row['search']).resolve()
        require(output != source and output not in source.parents and source not in output.parents,
                'Recovery output must be separate from every immutable source search')
        absent = [name for name in SOURCE_FILES if not (source / name).is_file()]
        if absent:
            missing.append(row['id'] + ': ' + str(source) + '\n  missing: ' + ', '.join(absent))
    if missing:
        raise FileNotFoundError('All configured source searches must be available before training:\n' + '\n'.join(missing)
            + '\nRestore original search folders including selected_checkpoint.pt. Metadata-only archives are insufficient.'
            + '\nNo search is restarted and no recovery has started.')
    prepared = []
    for row in study['architectures']:
        source = (ROOT / row['search']).resolve()
        saved = OmegaConf.load(source / 'resolved_config.yaml')
        validate_config(saved)
        cfg = compose_config(study['config'], [
            f"training_arguments.adaptive_lambda.soft_drop={row['soft_drop']}",
            f"training_arguments.adaptive_lambda.hard_drop={row['hard_drop']}",
        ], dense_source=reference)
        if args.device:
            cfg.device = args.device
        require(canonical(saved) == canonical(cfg), f"{row['id']}: source model/training settings differ")
        reviewed = inspect_search(source, saved, include_provenance=True)
        require(reviewed['physical_parameters'] == row['parameters']
                and reviewed['selected_epoch'] == row['selected_epoch']
                and reviewed['mask_hash'] == row['mask_hash']
                and reviewed['selected_model_state_hash'] == row['selected_model_state_hash']
                and reviewed['files']['selected_checkpoint.pt'] == row['selected_checkpoint_sha256'],
                f"{row['id']}: frozen checkpoint/mask/size differs from the selected experiment")
        require(reviewed['reference_sha256'] == study['reference_sha256']
                and reviewed['initializer_model_state_hash'] == study['initializer_model_state_hash']
                and reviewed['split_indices_hash'] == study['split_indices_hash'],
                f"{row['id']}: source reference/initializer/split differs")
        selection = read_json(source / 'selection.json')
        require(object_hash(selection['pruning_mask']) == row['mask_hash'], 'Source mask contents changed')
        path = output / row['id']
        completed = check_completed(path, cfg, row) if path.exists() else []
        prepared.append({'row': row, 'source': source, 'output': path, 'config': cfg,
                         'review': reviewed, 'completed': completed})
    inputs = validate_inputs(prepared[0]['config'])
    require(inputs['sha256'] == study['reference_sha256']
            and inputs['initializer_model_state_hash'] == study['initializer_model_state_hash'],
            'Current reference does not match the selected searches')
    return output, prepared


def write_summary(output, records):
    if not records:
        return
    with (output / 'validation_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def completed_cells(study, args):
    """Final evaluation needs recovery artifacts, not the original input folders."""
    from omegaconf import OmegaConf
    from net_complexity.training.one_shot_pruning_config import compose_config, validate_config
    output = Path(args.output or ROOT / study['output']).resolve()
    reference = Path(args.dense_source or ROOT / study['reference_output']).resolve()
    prepared = []
    for row in study['architectures']:
        path = output / row['id']
        saved = OmegaConf.load(path / 'resolved_config.yaml')
        validate_config(saved)
        expected = compose_config(study['config'], [
            f"training_arguments.adaptive_lambda.soft_drop={row['soft_drop']}",
            f"training_arguments.adaptive_lambda.hard_drop={row['hard_drop']}",
        ], dense_source=reference)
        require(canonical(saved) == canonical(expected), 'Frozen recovery recipe changed')
        prepared.append({'row': row, 'output': path, 'config': saved,
                         'completed': check_completed(path, saved, row)})
    return output, prepared


def evaluate_matrix(study, output, args, prepared):
    """Validate/freeze every deployment before constructing the official test loader."""
    import evaluate_one_shot_pruning_test as evaluation
    import numpy as np
    from net_complexity.training.pruning_measurement import state_hash
    models = []
    for cell in prepared:
        check_completed(cell['output'], cell['config'], cell['row'])
        branches = evaluation.prepare_branches(cell['output'])
        require(len(branches) == 3, 'Expected all three frozen variants')
        for model, record in branches:
            record.update(architecture=cell['row']['id'], job=cell['row']['id'] + '__' + record['branch'])
            require(record['physical_parameters'] == cell['row']['parameters'], 'Frozen model size differs')
            models.append((model, record))
    require(len(models) == len(study['architectures']) * len(BRANCHES),
            'Official test requires all configured frozen deployments')
    freeze = {'protocol': PROTOCOL, 'comparison_scope': 'exploratory',
              'deployments': [record for _, record in models]}
    frozen_path = output / 'frozen_deployments.json'
    if frozen_path.exists():
        require(read_json(frozen_path) == freeze, 'Frozen deployment list changed after it was written')
    else:
        write_json(frozen_path, freeze)
    freeze_hash = digest(frozen_path)
    test_output = output / 'test_evaluation'
    summary_path = test_output / 'test_summary.json'
    report = read_json(summary_path) if summary_path.exists() else {
        'status': 'running', 'protocol': PROTOCOL, 'comparison_scope': 'exploratory',
        'frozen_deployments_sha256': freeze_hash, 'training_performed': False,
        'bn_recalibration': False, 'test_based_selection': False, 'runs': []}
    require(report['frozen_deployments_sha256'] == freeze_hash, 'Test report uses different deployments')
    frozen_by_id = {record['job']: record for _, record in models}
    done = set()
    for record in report['runs']:
        require(record['job'] in frozen_by_id and record['job'] not in done, 'Unexpected prior test record')
        require(record['checkpoint_sha256'] == frozen_by_id[record['job']]['checkpoint_sha256'], 'Test checkpoint changed')
        require(digest(test_output / record['predictions']) == record['predictions_sha256'], 'Saved predictions changed')
        done.add(record['job'])
    if len(done) == len(models) and report['status'] == 'completed':
        print('All configured frozen official-test evaluations are already complete', flush=True)
        return report
    cfg = prepared[0]['config']
    device = args.device or str(cfg.device)
    loader, dataset = evaluation.frozen.build_test_loader(
        cfg.dataloaders.path_to_data, 128, int(cfg.dataloaders.num_workers), device, False)
    require(dataset['dataset'] == 'CIFAR10' and dataset['split'] == 'official_test'
            and dataset['example_count'] == 10000, 'Expected all official CIFAR10 test samples')
    if 'dataset' in report:
        require(report['dataset'] == dataset, 'Test dataset/preprocessing changed')
    report['dataset'] = dataset
    test_output.mkdir(parents=True, exist_ok=True)
    try:
        for model, record in models:
            if record['job'] in done:
                continue
            before = state_hash(model.state_dict())
            metrics, arrays = evaluation.frozen.evaluate_fixed(model, loader, device, job=record['job'])
            require(state_hash(model.state_dict()) == before, 'Test inference changed weights or BN buffers')
            require(np.array_equal(arrays['label'], np.asarray(loader.dataset.targets)), 'Test sample order changed')
            prediction_file = test_output / (record['job'] + '_predictions.npz')
            np.savez_compressed(prediction_file, **arrays)
            report['runs'].append({**record, 'test': metrics, 'predictions': prediction_file.name,
                                   'predictions_sha256': digest(prediction_file), 'model_state_unchanged': True})
            write_json(summary_path, report)
            evaluation.frozen.write_comparison(test_output, report['runs'])
            comparison = test_output / 'test_comparison.md'
            text = comparison.read_text()
            comparison.write_text(text.replace(
                '\n\n', '\n\nExploratory comparison: prior test results informed experimentation.\n\n', 1))
            print(f"[official-test] {record['job']}: {metrics['accuracy']:.2%}", flush=True)
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(summary_path, report)
    comparisons = []
    for cell in prepared:
        arch = cell['row']['id']
        scores = {r['branch']: r['test']['accuracy'] for r in report['runs'] if r['architecture'] == arch}
        comparisons.append({'architecture': arch, 'parameters': cell['row']['parameters'],
                            'inherited_minus_scratch_pp': 100 * (scores['inherited'] - scores['scratch']),
                            'learned_minus_random_pp': 100 * (scores['inherited'] - scores['random'])})
    write_json(output / 'test_comparisons.json', {'scope': 'exploratory', 'rows': comparisons})
    return report


def execute_matrix(study, args):
    require(args.stage in {'inspect', 'recover', 'test'}, 'Selected recovery matrix never starts a new search')
    require(args.reference_archive is None, 'Reuse the existing reference; this matrix does not create/restore one')
    model_count = len(study['architectures']) * len(BRANCHES)
    plan = {'protocol': PROTOCOL, 'stage': args.stage, 'architectures': study['architectures'],
            'models': model_count, 'search_epochs_reused_per_model': 60, 'recovery_epochs_per_model': 90,
            'new_training_epochs': model_count * 90 if args.stage == 'recover' else 0, 'per_model_budget': 150,
            'new_searches': 0, 'automatic_test_after_all_models': not args.skip_test,
            'source_preflight_before_any_training': True, 'output': str(Path(args.output or ROOT / study['output']).resolve())}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    if args.stage == 'test':
        output, prepared = completed_cells(study, args)
        return evaluate_matrix(study, output, args, prepared)
    try:
        output, prepared = preflight(study, args)
    except (ValueError, FileNotFoundError) as exc:
        print('Recovery matrix preflight blocked before training:\n' + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(2) from None
    output.mkdir(parents=True, exist_ok=True)
    source_review = {'protocol': PROTOCOL, 'manifest_hash': object_hash(study),
                     'sources': {c['row']['id']: c['review'] for c in prepared}}
    review_path = output / 'source_review.json'
    if review_path.exists():
        require(read_json(review_path) == source_review, 'Source files/manifest changed after preflight')
    else:
        write_json(review_path, source_review)
    if args.stage == 'inspect':
        print('All configured pinned searches and reference inputs verified; no training performed', flush=True)
        return source_review
    # Import the evaluator before long training, catching missing evaluation dependencies now.
    import evaluate_one_shot_pruning_test
    from launch_pruning_ablations import inspect_search
    from net_complexity.training.one_shot_pruning import run_one_shot_pruning
    records = [r for cell in prepared for r in cell['completed']]
    matrix_state = {'protocol': PROTOCOL, 'status': 'running', 'planned_models': model_count,
                    'completed_models': len(records), 'validation_results': records}
    write_summary(output, records)
    try:
        for cell in prepared:
            if cell['completed']:
                print('Reusing completed recovery: ' + cell['row']['id'], flush=True)
                continue
            from omegaconf import OmegaConf
            current = inspect_search(cell['source'], OmegaConf.load(cell['source'] / 'resolved_config.yaml'),
                                     include_provenance=True)
            require(current == cell['review'], 'Source search changed after all-source preflight')
            print(f"[matrix] {cell['row']['id']}: three90-epoch recoveries, no search", flush=True)
            result = run_one_shot_pruning(cell['config'], cell['output'], reuse_search_from=cell['source'],
                                         keep_epoch_checkpoints=args.keep_epoch_checkpoints)
            del result
            cell['completed'] = check_completed(cell['output'], cell['config'], cell['row'])
            records.extend(cell['completed'])
            matrix_state.update(completed_models=len(records), validation_results=records)
            write_json(output / 'matrix_state.json', matrix_state)
            write_summary(output, records)
        matrix_state['status'] = 'training_completed'
        write_json(output / 'matrix_state.json', matrix_state)
    except BaseException as exc:
        matrix_state.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'matrix_state.json', matrix_state)
        raise
    if not args.skip_test:
        evaluate_matrix(study, output, args, prepared)
    return matrix_state
