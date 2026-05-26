import warnings
from collections import Counter
from pathlib import Path

from PIL import Image

from net_complexity.data.dataloaders import (
    ClassicCVDataloaders,
    _resolve_num_workers,
    _resolve_pin_memory,
)


def test_resolve_num_workers_keeps_requested_value_when_shm_manager_is_available(monkeypatch):
    monkeypatch.setattr(
        "net_complexity.data.dataloaders._torch_shm_manager_available",
        lambda: True,
    )

    assert _resolve_num_workers(2) == 2


def test_resolve_num_workers_falls_back_to_zero_when_shm_manager_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "net_complexity.data.dataloaders._torch_shm_manager_available",
        lambda: False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = _resolve_num_workers(2)

    assert resolved == 0
    assert len(caught) == 1
    assert "falling back to num_workers=0" in str(caught[0].message)


def test_resolve_pin_memory_defaults_to_false_without_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    assert _resolve_pin_memory(None) is False


def test_resolve_pin_memory_respects_explicit_override():
    assert _resolve_pin_memory(True) is True
    assert _resolve_pin_memory(False) is False


def _write_rgb_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=color).save(path)


def _create_fake_tinyimagenet(root: Path) -> Path:
    tiny_root = root / "tiny-imagenet-200"
    tiny_root.mkdir(parents=True, exist_ok=True)
    wnids = ["n00000002", "n00000001"]
    (tiny_root / "wnids.txt").write_text("\n".join(wnids) + "\n", encoding="utf-8")
    (tiny_root / "words.txt").write_text(
        "n00000001 class one\nn00000002 class two\n",
        encoding="utf-8",
    )

    for class_idx, wnid in enumerate(wnids):
        for image_idx in range(4):
            _write_rgb_image(
                tiny_root / "train" / wnid / "images" / f"{wnid}_{image_idx}.JPEG",
                color=(class_idx * 80, image_idx * 40, 20),
            )

    _write_rgb_image(tiny_root / "val" / "images" / "val_0.JPEG", color=(10, 20, 30))
    _write_rgb_image(tiny_root / "val" / "images" / "val_1.JPEG", color=(40, 50, 60))
    (tiny_root / "val" / "val_annotations.txt").write_text(
        "val_0.JPEG n00000002 0 0 63 63\n"
        "val_1.JPEG n00000001 0 0 63 63\n",
        encoding="utf-8",
    )
    return tiny_root


def _build_tiny_loaders(tiny_root: Path) -> ClassicCVDataloaders:
    return ClassicCVDataloaders(
        path_to_data=str(tiny_root),
        batch_size=2,
        taskname="tinyimagenet200",
        num_workers=0,
        pin_memory=False,
        valid_per_class=1,
        seed=123,
        num_classes=2,
        image_size=64,
    )


def test_tinyimagenet_datasets_use_stratified_deterministic_train_valid_split(tmp_path):
    tiny_root = _create_fake_tinyimagenet(tmp_path)

    loaders = _build_tiny_loaders(tiny_root)
    repeated_loaders = _build_tiny_loaders(tiny_root)

    train_dataset = loaders.train_dataloader.dataset
    valid_dataset = loaders.valid_dataloader.dataset
    test_dataset = loaders.test_dataloader.dataset

    assert len(train_dataset) == 6
    assert len(valid_dataset) == 2
    assert len(test_dataset) == 2
    assert Counter(train_dataset.targets) == {0: 3, 1: 3}
    assert Counter(valid_dataset.targets) == {0: 1, 1: 1}
    assert Counter(test_dataset.targets) == {0: 1, 1: 1}

    valid_names = [path.name for path, _ in valid_dataset.samples]
    repeated_valid_names = [
        path.name
        for path, _ in repeated_loaders.valid_dataloader.dataset.samples
    ]
    assert valid_names == repeated_valid_names
    assert not set(path.name for path, _ in train_dataset.samples).intersection(valid_names)


def test_tinyimagenet_val_annotations_and_class_mapping_are_shared(tmp_path):
    tiny_root = _create_fake_tinyimagenet(tmp_path)
    loaders = _build_tiny_loaders(tiny_root)

    train_dataset = loaders.train_dataloader.dataset
    valid_dataset = loaders.valid_dataloader.dataset
    test_dataset = loaders.test_dataloader.dataset

    assert train_dataset.class_to_idx == {
        "n00000001": 0,
        "n00000002": 1,
    }
    assert valid_dataset.class_to_idx == train_dataset.class_to_idx
    assert test_dataset.class_to_idx == train_dataset.class_to_idx
    assert [(path.name, target) for path, target in test_dataset.samples] == [
        ("val_0.JPEG", 1),
        ("val_1.JPEG", 0),
    ]

    image, label = test_dataset[0]
    assert tuple(image.shape) == (3, 64, 64)
    assert label == 1
