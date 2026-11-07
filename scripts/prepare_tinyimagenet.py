#!/usr/bin/env python3
"""Assemble the eight uploaded TinyImageNet ZIP parts and extract the dataset."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile


ARCHIVE_NAME = "tiny-imagenet-200.zip"
DATASET_NAME = "tiny-imagenet-200"
EXPECTED_SHA256 = "6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821"
PART_NAMES = tuple(
    f"{ARCHIVE_NAME}.part-{part}.sub-{subpart}"
    for part in "abcd"
    for subpart in "ab"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_member_paths(archive: zipfile.ZipFile) -> None:
    for entry in archive.infolist():
        member = PurePosixPath(entry.filename)
        if (
            member.is_absolute()
            or ".." in member.parts
            or not member.parts
            or member.parts[0] != DATASET_NAME
            or "\\" in entry.filename
            or stat.S_ISLNK(entry.external_attr >> 16)
        ):
            raise ValueError(f"Unsafe ZIP member: {entry.filename!r}")


def validate_archive(path: Path, expected_sha256: str = EXPECTED_SHA256) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA256 mismatch for {path}: {actual}; expected {expected_sha256}")
    with zipfile.ZipFile(path) as archive:
        validate_member_paths(archive)
        damaged = archive.testzip()
        if damaged is not None:
            raise ValueError(f"Damaged ZIP member: {damaged}")


def assemble_archive(data_dir: Path, expected_sha256: str = EXPECTED_SHA256) -> Path:
    archive_path = data_dir / ARCHIVE_NAME
    if archive_path.exists():
        validate_archive(archive_path, expected_sha256)
        print(f"Reusing verified archive: {archive_path}")
        return archive_path

    parts = [data_dir / name for name in PART_NAMES]
    missing = [str(path) for path in parts if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset parts:\n" + "\n".join(missing))

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{ARCHIVE_NAME}.", dir=data_dir, delete=False) as target:
            temporary_path = Path(target.name)
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        validate_archive(temporary_path, expected_sha256)
        # Refuse a concurrent destination rather than replacing an existing ZIP.
        os.link(temporary_path, archive_path)
        print(f"Assembled and verified: {archive_path}")
        return archive_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def validate_dataset(
    root: Path, *, num_classes: int = 200, num_train: int = 100000, num_val: int = 10000
) -> None:
    wnids_path = root / "wnids.txt"
    annotations_path = root / "val" / "val_annotations.txt"
    if not wnids_path.is_file() or not annotations_path.is_file():
        raise ValueError(f"Incomplete dataset at {root}: wnids.txt or val_annotations.txt is missing")
    wnids = wnids_path.read_text(encoding="utf-8").splitlines()
    classes = set(wnids)
    if len(wnids) != num_classes or len(classes) != num_classes:
        raise ValueError(f"Expected {num_classes} unique classes in {wnids_path}")
    train_paths = [path for path in (root / "train").glob("*/images/*.JPEG") if path.is_file()]
    val_paths = [path for path in (root / "val" / "images").glob("*.JPEG") if path.is_file()]
    train_classes = {path.parent.parent.name for path in train_paths}
    if len(train_paths) != num_train or train_classes != classes:
        raise ValueError(f"Expected {num_train} training images across {num_classes} classes at {root}")
    if len(val_paths) != num_val:
        raise ValueError(f"Expected {num_val} official validation images at {root}")
    annotations = [line.split("\t") for line in annotations_path.read_text(encoding="utf-8").splitlines()]
    if (
        len(annotations) != num_val
        or any(len(row) < 2 or row[1] not in classes for row in annotations)
        or {row[0] for row in annotations} != {path.name for path in val_paths}
    ):
        raise ValueError(f"Invalid official validation annotations at {annotations_path}")


def extract_archive(archive_path: Path, **layout_counts: int) -> Path:
    data_dir = archive_path.parent
    dataset_path = data_dir / DATASET_NAME
    if dataset_path.exists():
        validate_dataset(dataset_path, **layout_counts)
        print(f"Reusing complete dataset: {dataset_path}")
        return dataset_path
    # Extraction into a fresh directory prevents overwriting user data on failure.
    with tempfile.TemporaryDirectory(prefix=".tiny-imagenet-extract-", dir=data_dir) as temporary:
        with zipfile.ZipFile(archive_path) as archive:
            validate_member_paths(archive)
            archive.extractall(temporary)
        extracted = Path(temporary) / DATASET_NAME
        validate_dataset(extracted, **layout_counts)
        if dataset_path.exists():
            raise FileExistsError(f"Dataset appeared during extraction: {dataset_path}")
        extracted.rename(dataset_path)
    print(f"Extracted and validated: {dataset_path}")
    return dataset_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--assemble-only", action="store_true", help="Verify the combined ZIP without extracting it")
    args = parser.parse_args()
    archive_path = assemble_archive(args.data_dir)
    if not args.assemble_only:
        extract_archive(archive_path)


if __name__ == "__main__":
    main()
