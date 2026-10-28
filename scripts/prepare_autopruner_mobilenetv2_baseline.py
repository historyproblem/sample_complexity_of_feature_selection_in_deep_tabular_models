"""Restore and verify the historical trained AutoPruner MobileNetV2 / TinyImageNet baseline."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_NAME = "mobilenet_baseline_partial.tar.gz"
EXPECTED_SHA256 = "a8230a3cc5ffd0263f8c7c6b90fc8b5a1c2c7f71207ee469b2444e74a1a5ef7a"
ARCHIVE_MEMBER = (
    "outputs/runs/20261002_194727_baseline_pretrained_mobilenetv2_tinyimagenet200/"
    "checkpoints/best.pt"
)
DEFAULT_CHECKPOINT = REPO_ROOT / ARCHIVE_MEMBER


def _verify_checkpoint(checkpoint: Path) -> None:
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint is not a regular file: {checkpoint}")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError(
            f"Checkpoint SHA256 mismatch: {checkpoint}. "
            "Refusing to overwrite the existing file."
        )


def prepare_baseline(archive: Path, checkpoint: Path) -> Path:
    """Return the verified historical checkpoint, restoring it only if absent."""

    if os.path.lexists(checkpoint):
        _verify_checkpoint(checkpoint)
        return checkpoint
    if not archive.is_file():
        raise FileNotFoundError(
            f"Baseline checkpoint is missing: {checkpoint}. "
            f"Upload {ARCHIVE_NAME} to {archive}, or pass --archive PATH."
        )

    with tarfile.open(archive, mode="r:gz") as bundle:
        candidates = [
            member for member in bundle.getmembers() if member.name == ARCHIVE_MEMBER
        ]
        if len(candidates) != 1 or not candidates[0].isfile():
            raise ValueError(
                f"Baseline archive must contain one regular file named {ARCHIVE_MEMBER}."
            )
        source = bundle.extractfile(candidates[0])
        if source is None:
            raise ValueError("Cannot read best.pt from the baseline archive.")

        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{checkpoint.name}.", suffix=".tmp", dir=checkpoint.parent
        )
        temporary = Path(temporary_name)
        try:
            digest = hashlib.sha256()
            with source, os.fdopen(descriptor, "wb") as destination:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(chunk)
                    digest.update(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            if digest.hexdigest() != EXPECTED_SHA256:
                raise ValueError(
                    "Archive best.pt SHA256 mismatch; checkpoint was not installed."
                )

            # Publish the fully written file atomically without replacing an
            # existing destination, including one created during verification.
            # The temporary file shares the destination directory/filesystem.
            try:
                os.link(temporary, checkpoint)
            except FileExistsError:
                _verify_checkpoint(checkpoint)
        finally:
            temporary.unlink(missing_ok=True)
    return checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=REPO_ROOT / ARCHIVE_NAME)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args(argv)
    archive = args.archive.expanduser().absolute()
    checkpoint = args.checkpoint.expanduser().absolute()
    try:
        prepared = prepare_baseline(archive, checkpoint)
    except (EOFError, OSError, tarfile.TarError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Verified baseline checkpoint: {prepared}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
