"""Dependency bootstrap ordering and version preservation; no real pip mutations."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = load_module('pruning_setup_test', ROOT / 'scripts/setup_pruning_dependencies.py')
launcher = load_module('pruning_launcher_setup_test', ROOT / 'scripts/launch_pruning_ablations.py')


@pytest.mark.parametrize('versions', [
    {'torch': '2.6.0+cu126', 'torchvision': '0.21.0+cu126'},
    {'torch': '2.8.0', 'torchvision': '0.23.0'},
])
def test_complete_tensor_pair_is_preserved(versions):
    assert setup.tensor_install_command(versions, 'linux') is None


@pytest.mark.parametrize('versions,expected', [
    ({}, ['torch==2.6.0+cu126', 'torchvision==0.21.0+cu126', '--index-url', 'https://download.pytorch.org/whl/cu126']),
    ({'torch': '2.6.0+cu126'}, ['torch==2.6.0+cu126', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu126']),
    ({'torchvision': '0.21.0+cu126'}, ['torch', 'torchvision==0.21.0+cu126', '--index-url', 'https://download.pytorch.org/whl/cu126']),
    ({'torch': '2.6.0+cpu'}, ['torch==2.6.0+cpu', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cpu']),
])
def test_missing_tensor_partner_uses_matching_build(versions, expected):
    assert setup.tensor_install_command(versions, 'linux') == expected


def test_system_python_is_never_modified(monkeypatch):
    monkeypatch.setattr(setup.sys, 'prefix', setup.sys.base_prefix)
    monkeypatch.setattr(setup.subprocess, 'run', lambda *a, **kw: pytest.fail('system pip was invoked'))
    with pytest.raises(RuntimeError, match='system interpreter'):
        setup.install_dependencies()


def prepare_fake_environment(tmp_path, monkeypatch, versions):
    environment = dict(versions)
    monkeypatch.setattr(setup, 'ROOT', tmp_path)
    requirements = tmp_path / 'requirements-pruning-search.txt'
    requirements.write_text('hydra-core\n')
    monkeypatch.setattr(setup, 'REQUIREMENTS', requirements)
    monkeypatch.setattr(setup, 'installed_versions', lambda: dict(environment))
    return environment


def test_install_pins_existing_versions_and_verifies_imports(tmp_path, monkeypatch):
    before = {'torch': '2.6.0+cu126', 'torchvision': '0.21.0+cu126', 'numpy': '1.26.4'}
    environment = prepare_fake_environment(tmp_path, monkeypatch, before)
    calls = []
    def run(args, check):
        calls.append(args)
        assert args[0] == sys.executable and check is True
        if 'pip' in args:
            assert '--upgrade' not in args and '--force-reinstall' not in args
            pins = Path(args[args.index('--constraint') + 1]).read_text()
            assert 'torch==2.6.0+cu126\n' in pins and 'numpy==1.26.4\n' in pins
            assert 'torch==2.6.0+cu126' in args and 'torchvision==0.21.0+cu126' in args
            environment['hydra-core'] = '1.3.2'
        else:
            assert '-c' in args and len(calls) == 2
            assert 'cuda.is_available' not in args[2]
            assert 'net_complexity.training.one_shot_pruning' in args[2]
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(setup.subprocess, 'run', run)
    result = setup.install_dependencies()
    assert result['before'] == before and result['after'] == environment
    assert result['existing_versions_preserved'] and result['training_started'] is False
    assert len(calls) == 2
    assert json.loads((tmp_path / 'outputs/setup/pruning-dependencies.json').read_text()) == result


def test_clean_environment_installs_tensor_pair_before_runtime(tmp_path, monkeypatch):
    environment = prepare_fake_environment(tmp_path, monkeypatch, {})
    command = setup.tensor_install_command
    monkeypatch.setattr(setup, 'tensor_install_command', lambda versions: command(versions, 'linux'))
    calls = []
    def run(args, check):
        calls.append(args)
        if 'pip' in args and '--index-url' in args:
            assert 'torch==2.6.0+cu126' in args and 'torchvision==0.21.0+cu126' in args
            environment.update(torch='2.6.0+cu126', torchvision='0.21.0+cu126')
        elif 'pip' in args:
            assert len(calls) == 2
            pins = Path(args[args.index('--constraint') + 1]).read_text()
            assert 'torchvision==0.21.0+cu126\n' in pins
            environment['hydra-core'] = '1.3.2'
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(setup.subprocess, 'run', run)
    setup.install_dependencies()
    assert len(calls) == 3 and '-c' in calls[-1]


def test_unexpected_existing_version_change_stops_before_runtime(tmp_path, monkeypatch):
    environment = prepare_fake_environment(tmp_path, monkeypatch,
        {'torch': '2.6.0+cu126', 'torchvision': '0.21.0+cu126', 'numpy': '1.26.4'})
    def run(args, check):
        assert 'pip' in args
        environment['numpy'] = '2.0.0'
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(setup.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='Installed versions changed'):
        setup.install_dependencies()
    assert not (tmp_path / 'outputs/setup/pruning-dependencies.json').exists()


def test_launcher_setup_precedes_config_loading(monkeypatch):
    calls = []
    def install(args, check):
        assert Path(args[-1]).name == 'setup_pruning_dependencies.py'
        assert args[0] == sys.executable and check
        calls.append('install')
    def load(path):
        assert calls == ['install']
        calls.append('config')
        return {'fixture': True}
    def execute(study, args):
        assert calls == ['install', 'config'] and args.install_deps
        return 'ready'
    monkeypatch.setattr(launcher.subprocess, 'run', install)
    monkeypatch.setattr(launcher, 'load_study', load)
    monkeypatch.setattr(launcher, 'execute', execute)
    assert launcher.main(['--install-deps', '--stage', 'search']) == 'ready'


def test_setup_failure_never_loads_or_runs_training(monkeypatch):
    def fail(args, check):
        raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(launcher.subprocess, 'run', fail)
    monkeypatch.setattr(launcher, 'load_study', lambda p: pytest.fail('training/config loaded after setup failure'))
    with pytest.raises(SystemExit) as exc:
        launcher.main(['--install-deps', '--stage', 'search'])
    assert exc.value.code == 2


def test_dry_run_skips_installation(monkeypatch):
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **kw: pytest.fail('dry-run invoked pip'))
    monkeypatch.setattr(launcher, 'load_study', lambda p: {})
    monkeypatch.setattr(launcher, 'execute', lambda study, args: args.dry_run)
    assert launcher.main(['--install-deps', '--stage', 'search', '--dry-run']) is True


def test_help_and_setup_plan_need_only_standard_library(tmp_path):
    for script, arguments in [('scripts/launch_pruning_ablations.py', ['--install-deps', '--help']),
                              ('scripts/setup_pruning_dependencies.py', ['--dry-run'])]:
        code = f"""
import runpy,sys
class Trap:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {{'yaml','hydra','torch','torchvision'}}:
            raise AssertionError('bootstrap imported a third-party module: '+name)
sys.meta_path.insert(0,Trap())
sys.argv=[{script!r}, *{arguments!r}]
runpy.run_path({script!r},run_name='__main__')
"""
        completed = subprocess.run([sys.executable, '-c', code], cwd=ROOT, text=True, capture_output=True)
        assert completed.returncode == 0, completed.stderr
