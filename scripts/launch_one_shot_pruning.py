"""One shared adaptive search, then inherited/scratch compact training branches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from net_complexity.training.one_shot_pruning_config import (
    CONFIG_NAME, compose_config, output_paths, resolved_one_shot, validate_inputs,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=CONFIG_NAME)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dense-source", type=Path,
                        help="Existing dense run root or J1_dense_control directory; resolves reference metadata and shared zero-epoch initializer")
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths/contracts without training, CUDA, or dataset construction")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="Explicit configuration override; strict one-shot constraints still apply")
    args = parser.parse_args(argv)
    config = compose_config(args.config_name, args.override, dense_source=args.dense_source)
    if args.dry_run:
        report = resolved_one_shot(config, output_root=args.output)
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return report
    output = Path(output_paths(config, args.output)["root"])
    if output.exists():
        raise FileExistsError(f"Refusing existing one-shot output: {output}. Use a fresh --output directory.")
    try:
        validate_inputs(config)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    # Preview/preflight cannot import or accidentally invoke the training engine.
    from net_complexity.training.one_shot_pruning import run_one_shot_pruning
    return run_one_shot_pruning(config, output)


if __name__ == "__main__":
    main()
