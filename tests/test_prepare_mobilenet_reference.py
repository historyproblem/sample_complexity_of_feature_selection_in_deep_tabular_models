"""Reference import preserves accepted validation metadata without using weights."""

import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_mobilenet_reference as prepare


def make_archive(path, *, epochs=range(1, 87), invalid_accuracy=False, missing_config=False):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["epoch", "valid_accuracy"])
    for epoch in epochs:
        writer.writerow([epoch, "nan" if invalid_accuracy else 0.5 + epoch / 1000])
    history = output.getvalue().encode()
    config = b"training_arguments:\n  num_epochs: 150\nscheduler:\n  T_max: 150\n"
    contents = {"history.csv": history}
    if not missing_config:
        contents["config_resolved.yaml"] = config
    with tarfile.open(path, "w:gz") as archive:
        for filename, content in contents.items():
            member = tarfile.TarInfo(f"{prepare.SOURCE_RUN}/{filename}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        # Deliberately invalid checkpoint bytes: successful import must not deserialize them.
        weight = tarfile.TarInfo(f"{prepare.SOURCE_RUN}/checkpoints/best.pt")
        weight.size = 20
        archive.addfile(weight, io.BytesIO(b"not a torch file!!!!"))
    return hashlib.sha256(path.read_bytes()).hexdigest(), history, config


def test_import_only_preserves_full_history_config_and_provenance(tmp_path):
    archive = tmp_path / "baseline.tar.gz"
    digest, history, config = make_archive(archive)
    target = tmp_path / "reference"
    assert prepare.import_reference(archive, target, digest) == target
    assert {path.name for path in target.iterdir()} == {
        "history.csv", "config_resolved.yaml", "reference_provenance.json"
    }
    assert (target / "history.csv").read_bytes() == history
    assert (target / "config_resolved.yaml").read_bytes() == config
    provenance = json.loads((target / prepare.PROVENANCE_FILE).read_text())
    assert provenance["actual_training_horizon_epochs"] == 150
    assert provenance["actual_cosine_T_max"] == 150
    assert provenance["available_history_epochs"] == 86
    assert provenance["selected_reference_epochs"] == [1, 30]
    assert provenance["training_performed"] is False
    assert provenance["checkpoint_extraction_or_deserialization"] is False


def test_verified_import_reused_even_when_source_archive_removed(tmp_path):
    archive = tmp_path / "baseline.tar.gz"
    digest, _, _ = make_archive(archive)
    target = prepare.import_reference(archive, tmp_path / "reference", digest)
    original = (target / prepare.PROVENANCE_FILE).read_bytes()
    archive.unlink()
    assert prepare.import_reference(archive, target, digest) == target
    assert (target / prepare.PROVENANCE_FILE).read_bytes() == original


def test_missing_archive_requests_upload_without_creating_output(tmp_path):
    target = tmp_path / "reference"
    with pytest.raises(FileNotFoundError, match="Upload mobilenet_baseline_partial"):
        prepare.import_reference(tmp_path / "missing.tar.gz", target)
    assert not target.exists()


def test_corrupt_or_different_archive_rejected_before_publish(tmp_path):
    archive = tmp_path / "baseline.tar.gz"
    digest, _, _ = make_archive(archive)
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare.import_reference(archive, tmp_path / "reference", digest)
    assert not (tmp_path / "reference").exists()


@pytest.mark.parametrize("kwargs", [
    {"epochs": range(1, 31)},
    {"epochs": [1, 1, *range(3, 87)]},
    {"invalid_accuracy": True},
    {"missing_config": True},
])
def test_invalid_history_or_missing_metadata_is_not_published(tmp_path, kwargs):
    archive = tmp_path / "baseline.tar.gz"
    digest, _, _ = make_archive(archive, **kwargs)
    with pytest.raises(ValueError):
        prepare.import_reference(archive, tmp_path / "reference", digest)
    assert not (tmp_path / "reference").exists()


def test_partial_or_foreign_existing_directory_is_never_replaced(tmp_path):
    target = tmp_path / "reference"
    target.mkdir()
    marker = target / "user.txt"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="refusing to replace"):
        prepare.import_reference(tmp_path / "missing.tar.gz", target)
    assert marker.read_text() == "keep"


def test_tampered_existing_history_is_not_reused(tmp_path):
    archive = tmp_path / "baseline.tar.gz"
    digest, _, _ = make_archive(archive)
    target = prepare.import_reference(archive, tmp_path / "reference", digest)
    (target / "history.csv").write_text("changed")
    with pytest.raises(ValueError, match="refusing to replace"):
        prepare.import_reference(archive, target, digest)


def test_broken_output_symlink_is_not_replaced(tmp_path):
    target = tmp_path / "reference"
    target.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    with pytest.raises(ValueError, match="refusing to replace"):
        prepare.import_reference(tmp_path / "missing.tar.gz", target)
    assert target.is_symlink()
