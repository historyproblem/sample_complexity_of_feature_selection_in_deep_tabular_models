"""One opt-in accuracy-guided run. Preview does not train, download, or use CUDA."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from net_complexity.training.accuracy_guided_config import compose_config, resolved_v3, validate_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="accuracy_guided_gates_v3")
    parser.add_argument("--dry-run", action="store_true", help="Read-only resolved schema, paths and provenance; no CUDA/data loading")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="Explicit single-profile override; strict v3 schema still applies")
    args = parser.parse_args(argv)
    config = compose_config(args.config_name, args.override)
    report = resolved_v3(config)
    if args.dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return report
    validate_inputs(config)
    # Deliberately lazy: preview must not import the training engine or construct data.
    from net_complexity.training.accuracy_guided_pruning import run_accuracy_guided_pruning
    output = args.output or Path(config.run_history.root_dir) / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_accuracy_guided_gates_v3")
    return run_accuracy_guided_pruning(config, output.resolve(), resume_from=args.resume_from)


if __name__ == "__main__":
    main()
