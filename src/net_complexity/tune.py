from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from net_complexity.cli.tune import initialize_cli_flags, main

__all__ = ["main"]


if __name__ == "__main__":
    initialize_cli_flags()
    main()
