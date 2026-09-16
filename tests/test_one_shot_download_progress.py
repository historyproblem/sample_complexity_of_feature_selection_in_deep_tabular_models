"""Download visibility is opt-in and does not alter data or random state."""

import pytest
import torch

from net_complexity.data import dataloaders as module
from net_complexity.training.one_shot_pruning_config import compose_config


@pytest.mark.parametrize("enabled", [True, False])
def test_download_progress_context_restores_tqdm_on_failure(monkeypatch, enabled):
    calls = []

    def fake_tqdm(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(module.torchvision_datasets_utils, "tqdm", fake_tqdm)
    with pytest.raises(RuntimeError, match="download failed"):
        with module._suppress_torchvision_download_progress(enabled=enabled):
            module.torchvision_datasets_utils.tqdm(total=10)
            raise RuntimeError("download failed")

    assert calls == [{"total": 10, **({"disable": True} if enabled else {})}]
    assert module.torchvision_datasets_utils.tqdm is fake_tqdm


def test_classic_download_progress_visible_only_when_requested(monkeypatch, tmp_path):
    calls = []

    def fake_tqdm(*args, **kwargs):
        calls.append(kwargs)

    class FakeCifar:
        def __init__(self, *, root, train, transform, download):
            assert train is True  # No official test construction in either mode.
            assert download is True
            self.transform = transform
            module.torchvision_datasets_utils.tqdm(total=10)

        def __len__(self):
            return 20

        def __getitem__(self, index):
            return torch.zeros(3, 32, 32), index % 10

    monkeypatch.setattr(module.torchvision_datasets_utils, "tqdm", fake_tqdm)
    monkeypatch.setattr(module.datasets, "CIFAR10", FakeCifar)
    options = dict(path_to_data=str(tmp_path), taskname="CIFAR10", num_workers=0,
                   pin_memory=False, include_test=False, seed=42, loader_seed=42)
    rng_before = torch.get_rng_state().clone()
    quiet = module.ClassicCVDataloaders(**options)
    assert calls == [{"total": 10, "disable": True}] * 2
    assert module.torchvision_datasets_utils.tqdm is fake_tqdm
    calls.clear()
    visible = module.ClassicCVDataloaders(**options, download_progress=True)
    assert calls == [{"total": 10}] * 2
    assert module.torchvision_datasets_utils.tqdm is fake_tqdm
    assert torch.equal(torch.get_rng_state(), rng_before)
    for name in ("train_dataloader", "valid_dataloader"):
        assert getattr(quiet, name).dataset.indices == getattr(visible, name).dataset.indices


def test_one_shot_config_enables_download_progress():
    config = compose_config()
    assert config.dataloaders.download_progress is True
