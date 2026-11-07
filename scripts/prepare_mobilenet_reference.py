"""Import the accepted existing MobileNet validation reference without training.

Only history.csv and config_resolved.yaml are copied from the trusted archive.
Checkpoint files are never extracted or deserialized.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tarfile
import tempfile


EXPECTED_ARCHIVE_SHA256 = "7ada53ae4c56902dcfc48d3625273d66955dd8b9ca7702727d9854d342bf9cfd"
SOURCE_RUN = "outputs/runs/20261002_194727_baseline_pretrained_mobilenetv2_tinyimagenet200"
METADATA_FILES = ("history.csv", "config_resolved.yaml")
PROVENANCE_FILE = "reference_provenance.json"
OUTPUT_ROOT = Path("outputs/runs/mobilenet_onepass_thr05_gap05_1pp")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_history(content: bytes) -> None:
    """Require the full known 86-epoch validation curve, including epochs1–30."""
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
    try:
        epochs = [int(row["epoch"]) for row in rows]
        accuracy = [float(row["valid_accuracy"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Reference history needs integer epoch and valid_accuracy columns") from exc
    if epochs != list(range(1, 87)):
        raise ValueError("Reference history must contain exactly epochs1–86, in order")
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in accuracy):
        raise ValueError("Reference validation accuracy must be finite and between0and1")


def validate_existing(output_dir: Path, expected_sha256: str) -> None:
    try:
        provenance = json.loads((output_dir / PROVENANCE_FILE).read_text(encoding="utf-8"))
        if not isinstance(provenance, dict):
            raise ValueError("Reference provenance must be an object")
        required = {
            "archive_sha256": expected_sha256,
            "source_run": SOURCE_RUN,
            "actual_training_horizon_epochs": 150,
            "actual_cosine_T_max": 150,
            "available_history_epochs": 86,
            "selected_reference_epochs": [1, 30],
            "training_performed": False,
        }
        if any(provenance.get(key) != value for key, value in required.items()):
            raise ValueError("Reference provenance differs from the accepted source")
        for filename, key in (
            ("history.csv", "history_sha256"),
            ("config_resolved.yaml", "config_sha256"),
        ):
            if sha256_file(output_dir / filename) != provenance[key]:
                raise ValueError(f"Reference file changed: {filename}")
        validate_history((output_dir / "history.csv").read_bytes())
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Existing reference directory is incomplete or different: {output_dir}; "
            "refusing to replace it"
        ) from exc


def import_reference(
    archive_path: Path,
    output_dir: Path,
    expected_sha256: str = EXPECTED_ARCHIVE_SHA256,
) -> Path:
    """Publish metadata atomically; never replace a partial or foreign reference."""
    if output_dir.exists() or output_dir.is_symlink():
        validate_existing(output_dir, expected_sha256)
        print(f"Reusing verified existing reference: {output_dir}", flush=True)
        return output_dir
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Upload mobilenet_baseline_partial.tar.gz to the repository root first: {archive_path}. "
            "A new baseline will not be trained."
        )
    actual_sha256 = sha256_file(archive_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Reference archive SHA256 mismatch: {actual_sha256}; expected {expected_sha256}"
        )

    contents = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        for filename in METADATA_FILES:
            name = f"{SOURCE_RUN}/{filename}"
            matching = [member for member in members if member.name == name]
            if len(matching) != 1 or not matching[0].isfile():
                raise ValueError(f"Expected exactly one regular archive member: {name}")
            if matching[0].size > 2 * 1024 * 1024:
                raise ValueError(f"Unexpected reference metadata size: {name}")
            with archive.extractfile(matching[0]) as stream:
                contents[filename] = stream.read()
    validate_history(contents["history.csv"])
    if not contents["config_resolved.yaml"].strip():
        raise ValueError("Reference source config is empty")

    provenance = {
        "archive_sha256": actual_sha256,
        "source_archive_name": archive_path.name,
        "source_run": SOURCE_RUN,
        "actual_training_horizon_epochs": 150,
        "actual_cosine_T_max": 150,
        "available_history_epochs": 86,
        "selected_reference_epochs": [1, 30],
        "history_sha256": hashlib.sha256(contents["history.csv"]).hexdigest(),
        "config_sha256": hashlib.sha256(contents["config_resolved.yaml"]).hexdigest(),
        "training_performed": False,
        "checkpoint_extraction_or_deserialization": False,
        "note": "Existing 86-epoch history of a configured150-epoch/cosine150 run; "
        "epochs1–30 are selected as reference, not relabeled as cosine30 training.",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "reference"
        staging.mkdir()
        for filename, content in contents.items():
            (staging / filename).write_bytes(content)
        (staging / PROVENANCE_FILE).write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"Reference directory appeared during import: {output_dir}")
        staging.rename(output_dir)
    print(f"Imported existing reference metadata: {output_dir}", flush=True)
    print("Kept all86history rows; search uses epochs1–30. No baseline training performed.", flush=True)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=repo_root / "mobilenet_baseline_partial.tar.gz")
    parser.add_argument(
        "--output-dir", "--output", type=Path,
        default=repo_root / OUTPUT_ROOT / "imported_dense_reference_20261002",
    )
    args = parser.parse_args(argv)
    try:
        import_reference(args.archive, args.output_dir)
    except (OSError, ValueError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
