from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from launch_one_shot_pruning import config_for
from net_complexity.models.channel_pruning import build_pruned_bottleneck_model


def test_bottleneck_log_counts_channels_not_boundary_keys(capsys):
    config = config_for("O1_internal_gap05_1")
    build_pruned_bottleneck_model(config, {"layer1.0": {"mid1": [0, 1, 2], "mid2": [0, 1]},
                                           "layer1.1": {"output": [0, 1, 2, 3]}})
    assert "2 blocks affected, 9 channels removed" in capsys.readouterr().out
