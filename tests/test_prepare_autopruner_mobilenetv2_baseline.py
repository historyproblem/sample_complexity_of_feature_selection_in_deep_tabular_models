from __future__ import annotations

import hashlib
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/prepare_autopruner_mobilenetv2_baseline.py"
)
ARCHIVE_MEMBER = (
    "outputs/runs/20261002_194727_baseline_pretrained_mobilenetv2_tinyimagenet200/"
    "checkpoints/best.pt"
)
BASELINE_BYTES = b"small verified checkpoint fixture\n"
BASELINE_SHA256 = hashlib.sha256(BASELINE_BYTES).hexdigest()


def run_cli(archive: Path, checkpoint: Path) -> subprocess.CompletedProcess[str]:
    # Substitute only the known checksum, keeping CLI parsing and file I/O
    # in a separate process without a 283 MB historical-checkpoint fixture.
    bootstrap = """
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("prepare_baseline", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.EXPECTED_SHA256 = sys.argv[2]
raise SystemExit(module.main(sys.argv[3:]))
"""
    return subprocess.run(
        [
            sys.executable,
            "-c",
            bootstrap,
            str(SCRIPT_PATH),
            BASELINE_SHA256,
            "--archive",
            str(archive),
            "--checkpoint",
            str(checkpoint),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def write_archive(path: Path, contents: bytes = BASELINE_BYTES) -> None:
    with tarfile.open(path, "w:gz") as bundle:
        member = tarfile.TarInfo(ARCHIVE_MEMBER)
        member.size = len(contents)
        bundle.addfile(member, io.BytesIO(contents))
        # Unselected archive members must never be extracted.
        ignored = tarfile.TarInfo("../unexpected.txt")
        ignored.size = 7
        bundle.addfile(ignored, io.BytesIO(b"ignored"))


def test_cli_reports_missing_archive_without_creating_checkpoint(tmp_path):
    archive = tmp_path / "mobilenet_baseline_partial.tar.gz"
    checkpoint = tmp_path / "destination" / "best.pt"

    result = run_cli(archive, checkpoint)

    assert result.returncode == 1
    assert "Upload mobilenet_baseline_partial.tar.gz" in result.stderr
    assert str(archive) in result.stderr
    assert not checkpoint.parent.exists()


def test_cli_refuses_to_overwrite_wrong_existing_checkpoint(tmp_path):
    archive = tmp_path / "baseline.tar.gz"
    write_archive(archive)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"different checkpoint")

    result = run_cli(archive, checkpoint)

    assert result.returncode == 1
    assert "Refusing to overwrite" in result.stderr
    assert checkpoint.read_bytes() == b"different checkpoint"


def test_cli_restores_only_verified_member_and_reuses_it_without_archive(tmp_path):
    archive = tmp_path / "baseline.tar.gz"
    write_archive(archive)
    checkpoint = tmp_path / "destination" / "best.pt"

    result = run_cli(archive, checkpoint)

    assert result.returncode == 0, result.stderr
    assert f"Verified baseline checkpoint: {checkpoint}" in result.stdout
    assert checkpoint.read_bytes() == BASELINE_BYTES
    assert list(checkpoint.parent.iterdir()) == [checkpoint]
    assert not (tmp_path / "unexpected.txt").exists()
    original_stat = checkpoint.stat()
    archive.unlink()

    reused = run_cli(archive, checkpoint)

    assert reused.returncode == 0, reused.stderr
    assert checkpoint.read_bytes() == BASELINE_BYTES
    assert checkpoint.stat().st_ino == original_stat.st_ino
    assert checkpoint.stat().st_mtime_ns == original_stat.st_mtime_ns


def test_cli_rejects_wrong_archived_checksum_and_cleans_temporary_file(tmp_path):
    archive = tmp_path / "baseline.tar.gz"
    write_archive(archive, b"different checkpoint")
    checkpoint = tmp_path / "destination" / "best.pt"

    result = run_cli(archive, checkpoint)

    assert result.returncode == 1
    assert "Archive best.pt SHA256 mismatch" in result.stderr
    assert not checkpoint.exists()
    assert list(checkpoint.parent.iterdir()) == []


@pytest.mark.parametrize("invalid_kind", ["symlink", "duplicate"])
def test_cli_rejects_nonregular_or_duplicate_target_member(tmp_path, invalid_kind):
    archive = tmp_path / "baseline.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        if invalid_kind == "symlink":
            member = tarfile.TarInfo(ARCHIVE_MEMBER)
            member.type = tarfile.SYMTYPE
            member.linkname = "../unexpected.txt"
            bundle.addfile(member)
        else:
            for _ in range(2):
                member = tarfile.TarInfo(ARCHIVE_MEMBER)
                member.size = len(BASELINE_BYTES)
                bundle.addfile(member, io.BytesIO(BASELINE_BYTES))
    checkpoint = tmp_path / "destination" / "best.pt"

    result = run_cli(archive, checkpoint)

    assert result.returncode == 1
    assert f"one regular file named {ARCHIVE_MEMBER}" in result.stderr
    assert not checkpoint.parent.exists()
