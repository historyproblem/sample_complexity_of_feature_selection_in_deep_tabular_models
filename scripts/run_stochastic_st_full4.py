"""Fresh 60+90 stochastic-ST models, bounded weights cache and official test."""
import argparse
from pathlib import Path
import shutil
import uuid

from launch_pruning_ablations import restore_reference
import run_stochastic_st_followup as queue
from net_complexity.training.one_shot_pruning_config import dense_source_paths


DEFAULT_CONFIG = queue.ROOT / "configs/experiment/pruning_v3/stochastic_st_full4.yaml"
REFERENCE_ARCHIVE = "epoch_split_dense_reference_seed42_fresh_upload.tar.gz"


def matching_reference(source, study):
    paths = dense_source_paths(source)
    return all(Path(path).is_file() and queue.digest(path) == study["reference_sha256"][key]
               for key, path in paths.items())


def prepare_reference(study, args):
    output = Path(args.output or queue.ROOT / study["output"]).resolve()
    candidates = [Path(args.dense_source or queue.ROOT / study["reference_output"]).resolve(), output / "reference_inputs"]
    for candidate in candidates:
        if matching_reference(candidate, study):
            args.dense_source = candidate
            return candidate
    archive = Path(args.reference_archive).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Immutable reference/initializer missing. Upload {REFERENCE_ARCHIVE} to the repository root "
                                "or pass --reference-archive PATH. No baseline or model training has started.")
    destination = output / "reference_inputs"
    queue.require(not destination.exists(), "Incomplete reference_inputs retained; choose a fresh --output directory")
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / (".reference_import_" + uuid.uuid4().hex)
    try:
        restore_reference(archive, temporary)
        queue.require(matching_reference(temporary, study), "Reference archive does not match the pinned input hashes")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)  # Only this invocation's newly created import staging directory.
    args.dense_source = destination
    print(f"Restored existing reference and zero-epoch initializer to {destination}; no baseline training", flush=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("run", "search", "inspect", "test"), default="run")
    parser.add_argument("--reference-archive", type=Path, default=queue.ROOT / REFERENCE_ARCHIVE)
    parser.add_argument("--dense-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.skip_wait, args.keep_epoch_checkpoints = True, False
    try:
        study = queue.load_study(args.config)
        queue.require(study["protocol"] in {queue.FULL4_PROTOCOL, queue.FULL2_PROTOCOL},
                      "This launcher requires a two- or four-full-model configuration")
        if not args.dry_run and args.stage != "test":
            prepare_reference(study, args)
        result = queue.execute(study, args)
        if args.stage == "inspect" and not args.dry_run:
            print(f"{len(study['searches'])} fresh-run recipes and immutable inputs verified; no model training started", flush=True)
        return result
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.exit(2, f"Fresh-run queue blocked: {exc}\n")


if __name__ == "__main__":
    main()
