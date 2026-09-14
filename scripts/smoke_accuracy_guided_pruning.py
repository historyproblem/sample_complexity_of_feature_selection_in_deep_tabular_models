"""One short CPU synthetic integration smoke; no dataset download or test access."""
import argparse
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run_smoke(destination):
    import torch
    from net_complexity.training.pruning_synthetic import make_synthetic_config
    from net_complexity.training.accuracy_guided_pruning import run_accuracy_guided_pruning
    from net_complexity.training.pruning_measurement import write_json
    torch.set_num_threads(1)
    destination = Path(destination).resolve()
    cfg = make_synthetic_config(destination / "inputs", learned_closed=True)
    result = run_accuracy_guided_pruning(cfg, destination / "run")
    assert result["status"] == "completed", result.get("error")
    assert result["ledger"] == {"global_training_epoch": 8, "search_epochs_consumed": 6,
                                 "optimizer_updates": 16, "consumed_training_examples": 64}
    assert result["learned_candidates_materialized"] == 2
    assert result["decisions"][1]["new_pruning"] is False
    assert len(result["transitions"]) == 1
    summary = {"status": "passed", "protocol": result["protocol"], "ledger": result["ledger"],
               "physical_removed_channels": result["learned_candidates_materialized"],
               "carry_rebase_transitions": len(result["transitions"]), "test_access": False,
               "quality_interpretation": "synthetic fixture only; no CIFAR quality claim"}
    write_json(destination / "smoke_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or Path(tempfile.mkdtemp(prefix="accuracy-guided-smoke-"))
    print(run_smoke(destination))
