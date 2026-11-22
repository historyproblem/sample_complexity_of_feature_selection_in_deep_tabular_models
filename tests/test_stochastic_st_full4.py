"""Four fresh complete models and safe immutable-reference archive restoration."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stochastic_st_full4 as full4
import run_stochastic_st_followup as queue
import stochastic_pruning_test


STRONGER_CONFIG = ROOT / "configs/experiment/pruning_v3/stochastic_st_stronger2.yaml"


def test_stronger_pair_is_two_fresh_complete_models_with_only_threshold_changes(tmp_path):
    result = full4.main(["--config", str(STRONGER_CONFIG), "--dry-run",
                         "--output", str(tmp_path / "absent"),
                         "--reference-archive", str(tmp_path / "missing.tar.gz")])
    assert (result["models"], result["existing_search_recoveries"], result["new_searches"]) == (2, 0, 2)
    assert result["new_training_epochs"] == 300 and result["bounded_epoch_checkpoints"] is True
    assert result["wait_for_original_search_process"] is False and result["test_after_each_model"] is True
    assert [(r["new_search_epochs"], r["recovery_epochs"]) for r in result["models_in_order"]] == [(60, 90)] * 2
    assert [(r["soft_drop"], r["hard_drop"]) for r in result["models_in_order"]] == [(.00125, .0025), (.0025, .005)]
    previous = queue.load_study(full4.DEFAULT_CONFIG)
    study = queue.load_study(STRONGER_CONFIG)
    assert study["reference_sha256"] == previous["reference_sha256"]
    assert study["initializer_model_state_hash"] == previous["initializer_model_state_hash"]
    assert study["split_indices_hash"] == previous["split_indices_hash"]
    args = SimpleNamespace(dense_source=tmp_path / "same_reference", device=None)
    baseline = queue.configured(previous, previous["searches"][1], args)
    for row in study["searches"]:
        actual = queue.configured(study, row, args)
        expected = deepcopy(baseline)
        expected.training_arguments.adaptive_lambda.soft_drop = row["soft_drop"]
        expected.training_arguments.adaptive_lambda.hard_drop = row["hard_drop"]
        assert OmegaConf.to_container(actual, resolve=True) == OmegaConf.to_container(expected, resolve=True)
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("count", [0, 1, 3, 4])
def test_two_model_manifest_rejects_other_counts(tmp_path, count):
    import yaml
    study = queue.load_study(STRONGER_CONFIG)
    study["searches"] = [{**study["searches"][0], "id": f"run_{i}"} for i in range(count)]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(study))
    with pytest.raises(ValueError, match="Unexpected number of models"):
        queue.load_study(path)


def test_four_complete_models_dry_run_never_needs_old_searches(tmp_path):
    result = full4.main(["--dry-run", "--output", str(tmp_path / "absent"),
                         "--reference-archive", str(tmp_path / "missing.tar.gz")])
    assert (result["models"], result["existing_search_recoveries"], result["new_searches"]) == (4, 0, 4)
    assert result["new_training_epochs"] == 600 and result["bounded_epoch_checkpoints"] is True
    assert result["wait_for_original_search_process"] is False
    assert all(row["new_search_epochs"] == 60 and row["recovery_epochs"] == 90 for row in result["models_in_order"])
    assert [(r["soft_drop"], r["hard_drop"]) for r in result["models_in_order"]] == [
        (.000225, .00045), (.00025, .0005), (.0003, .0006), (.0005, .001)]
    assert not (tmp_path / "absent").exists()


def archive_fixture(tmp_path):
    study = queue.load_study(full4.DEFAULT_CONFIG)
    payloads = {"shared_random_seed42.pt": b"initializer fixture", "J1_dense_control/global_history.csv": b"history fixture",
                "J1_dense_control/pilot_state.json": b"state fixture", "J1_dense_control/resolved_config.yaml": b"config fixture"}
    archive = tmp_path / "reference.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, data in payloads.items():
            member = tarfile.TarInfo("old/run/" + name); member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    names = {"initializer_path": "shared_random_seed42.pt", "history_path": "J1_dense_control/global_history.csv",
             "state_path": "J1_dense_control/pilot_state.json", "config_path": "J1_dense_control/resolved_config.yaml"}
    study["reference_sha256"] = {key: hashlib.sha256(payloads[name]).hexdigest() for key, name in names.items()}
    return study, archive


def test_archive_restores_into_owned_cache_without_overwriting_partial_old_input(tmp_path):
    study, archive = archive_fixture(tmp_path)
    old = tmp_path / "partial_old_reference"; old.mkdir()
    (old / "shared_random_seed42.pt").write_bytes(b"old incomplete input")
    args = SimpleNamespace(output=tmp_path / "new", dense_source=old, reference_archive=archive)
    result = full4.prepare_reference(study, args)
    assert result == args.output / "reference_inputs" and args.dense_source == result
    assert (old / "shared_random_seed42.pt").read_bytes() == b"old incomplete input"
    assert full4.matching_reference(result, study)
    assert not list(args.output.glob(".reference_import_*"))
    args.reference_archive = tmp_path / "missing.tar.gz"
    assert full4.prepare_reference(study, args) == result


def test_wrong_archive_is_rejected_and_only_own_staging_files_removed(tmp_path):
    study, archive = archive_fixture(tmp_path)
    study["reference_sha256"]["history_path"] = "0" * 64
    args = SimpleNamespace(output=tmp_path / "out", dense_source=tmp_path / "missing", reference_archive=archive)
    with pytest.raises(ValueError, match="does not match"):
        full4.prepare_reference(study, args)
    assert not (args.output / "reference_inputs").exists()
    assert not list(args.output.glob(".reference_import_*"))


def test_missing_reference_blocks_before_queue_or_training(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(queue, "execute", lambda *a: pytest.fail("training queue entered without inputs"))
    with pytest.raises(SystemExit) as exc:
        full4.main(["--stage", "inspect", "--dense-source", str(tmp_path / "missing"),
                    "--reference-archive", str(tmp_path / "missing.tar.gz"), "--output", str(tmp_path / "new")])
    assert exc.value.code == 2 and "Upload epoch_split_dense_reference_seed42_fresh_upload.tar.gz" in capsys.readouterr().err
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("config_path, count", [(full4.DEFAULT_CONFIG, 4), (STRONGER_CONFIG, 2)])
def test_each_model_finishes_recovery_and_test_before_next_search(tmp_path, monkeypatch, config_path, count):
    import net_complexity.training.one_shot_pruning as runtime
    from test_stochastic_st_followup import fake_cells
    study = queue.load_study(config_path)
    cells = fake_cells(study, tmp_path / "queue")
    monkeypatch.setattr(full4, "prepare_reference", lambda *a: None)
    monkeypatch.setattr(queue, "preflight", lambda *a: (tmp_path / "queue", cells))
    monkeypatch.setattr(queue, "review_source", lambda *a, **kw: {"physical_parameters": 123})
    monkeypatch.setattr(queue, "check_completed", lambda *a: {"status": "completed"})
    monkeypatch.setattr(queue, "wait_for_original_searches", lambda: pytest.fail("fresh runs waited for old server"))
    events = []
    class Reporter:
        def __init__(self, output, ids, device, branches):
            assert len(ids) == count and branches == ("inherited",)
        def callback(self, config):
            return lambda model, branch, root: events.append("test")
        def finish(self): return {"status": "completed"}
    monkeypatch.setattr(stochastic_pruning_test, "DeploymentTests", Reporter)
    def run(config, output, **kwargs):
        assert kwargs["bounded_checkpoints"] is True and kwargs["keep_epoch_checkpoints"] is False
        events.append("search60" if kwargs.get("search_only") else "recovery90")
        if not kwargs.get("search_only"):
            kwargs["frozen_deployment_callback"](None, {}, output)
    monkeypatch.setattr(runtime, "run_one_shot_pruning", run)
    result = full4.main(["--config", str(config_path), "--output", str(tmp_path / "queue")])
    assert events == ["search60", "recovery90", "test"] * count
    assert result["planned_models"] == count and len(result["completed_models"]) == count
    assert result["new_training_epochs_allocated"] == 150 * count


def test_test_only_never_restores_reference(monkeypatch):
    monkeypatch.setattr(full4, "prepare_reference", lambda *a: pytest.fail("test-only required reference"))
    monkeypatch.setattr(queue, "execute", lambda study, args: {"stage": args.stage, "models": len(study["searches"])})
    assert full4.main(["--stage", "test"]) == {"stage": "test", "models": 4}
