"""Dataset upload preparation contracts; only small synthetic archives are used."""

import hashlib
import io
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_tinyimagenet as prepare


def dataset_zip(extra=None):
    files = {
        "tiny-imagenet-200/wnids.txt": "class_a\nclass_b\n",
        "tiny-imagenet-200/train/class_a/images/a.JPEG": "image a",
        "tiny-imagenet-200/train/class_b/images/b.JPEG": "image b",
        "tiny-imagenet-200/val/images/v.JPEG": "image v",
        "tiny-imagenet-200/val/val_annotations.txt": "v.JPEG\tclass_a\t0\t0\t1\t1\n",
    }
    files.update(extra or {})
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return stream.getvalue()


def write_parts(data_dir, content):
    # Write in reversed creation order to ensure directory order cannot affect assembly.
    for index in reversed(range(8)):
        start = len(content) * index // 8
        end = len(content) * (index + 1) // 8
        (data_dir / prepare.PART_NAMES[index]).write_bytes(content[start:end])
    return hashlib.sha256(content).hexdigest()


def test_named_parts_assemble_in_order_then_reuse_without_parts(tmp_path):
    content = dataset_zip()
    digest = write_parts(tmp_path, content)
    archive_path = prepare.assemble_archive(tmp_path, digest)
    assert archive_path.read_bytes() == content
    for name in prepare.PART_NAMES:
        (tmp_path / name).unlink()
    assert prepare.assemble_archive(tmp_path, digest) == archive_path


def test_missing_part_does_not_create_archive(tmp_path):
    digest = write_parts(tmp_path, dataset_zip())
    (tmp_path / prepare.PART_NAMES[3]).unlink()
    with pytest.raises(FileNotFoundError, match=prepare.PART_NAMES[3]):
        prepare.assemble_archive(tmp_path, digest)
    assert not (tmp_path / prepare.ARCHIVE_NAME).exists()


def test_corrupted_upload_does_not_publish_archive(tmp_path):
    digest = write_parts(tmp_path, dataset_zip())
    (tmp_path / prepare.PART_NAMES[0]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare.assemble_archive(tmp_path, digest)
    assert not (tmp_path / prepare.ARCHIVE_NAME).exists()
    assert not list(tmp_path.glob(".tiny-imagenet-200.zip.*"))


def test_differing_existing_zip_is_never_overwritten(tmp_path):
    digest = write_parts(tmp_path, dataset_zip())
    path = tmp_path / prepare.ARCHIVE_NAME
    path.write_bytes(b"existing user data")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare.assemble_archive(tmp_path, digest)
    assert path.read_bytes() == b"existing user data"


def test_path_traversal_zip_is_rejected_even_with_matching_digest(tmp_path):
    digest = write_parts(tmp_path, dataset_zip({"../escape.txt": "bad"}))
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        prepare.assemble_archive(tmp_path, digest)
    assert not (tmp_path / prepare.ARCHIVE_NAME).exists()
    assert not (tmp_path.parent / "escape.txt").exists()


def test_extraction_rejects_unvalidated_path_traversal_archive(tmp_path):
    archive_path = tmp_path / prepare.ARCHIVE_NAME
    archive_path.write_bytes(dataset_zip({"../escape.txt": "bad"}))
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        prepare.extract_archive(archive_path, num_classes=2, num_train=2, num_val=1)
    assert not (tmp_path / prepare.DATASET_NAME).exists()


def test_extract_complete_fixture_and_reuse_existing_directory(tmp_path):
    digest = write_parts(tmp_path, dataset_zip())
    archive_path = prepare.assemble_archive(tmp_path, digest)
    counts = dict(num_classes=2, num_train=2, num_val=1)
    root = prepare.extract_archive(archive_path, **counts)
    original = root / "train" / "class_a" / "images" / "a.JPEG"
    original.write_text("user touched this image")
    assert prepare.extract_archive(archive_path, **counts) == root
    assert original.read_text() == "user touched this image"


def test_incomplete_existing_dataset_is_not_overwritten(tmp_path):
    digest = write_parts(tmp_path, dataset_zip())
    archive_path = prepare.assemble_archive(tmp_path, digest)
    existing = tmp_path / prepare.DATASET_NAME
    existing.mkdir()
    marker = existing / "user-file.txt"
    marker.write_text("retain")
    with pytest.raises(ValueError, match="Incomplete dataset"):
        prepare.extract_archive(archive_path, num_classes=2, num_train=2, num_val=1)
    assert marker.read_text() == "retain"
