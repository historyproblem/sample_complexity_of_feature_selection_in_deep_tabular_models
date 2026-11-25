"""Twelve 60-epoch searches over warmup, initial lambda and soft tolerance; no recovery."""
from pathlib import Path

from run_stochastic_st_full4 import main as run_queue

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/experiment/pruning_v3/stochastic_st_timing12.yaml"


def main(argv=None):
    return run_queue(argv, default_config=DEFAULT_CONFIG, default_stage="search")


if __name__ == "__main__":
    main()
