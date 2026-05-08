from omegaconf import OmegaConf

from net_complexity.training.engine import _is_mlflow_enabled


def test_mlflow_is_enabled_by_default_when_block_is_present():
    config = OmegaConf.create({"mlflow": {"run_name": "demo"}})

    assert _is_mlflow_enabled(config) is True


def test_mlflow_can_be_disabled_explicitly():
    config = OmegaConf.create({"mlflow": {"enabled": False, "run_name": "demo"}})

    assert _is_mlflow_enabled(config) is False
