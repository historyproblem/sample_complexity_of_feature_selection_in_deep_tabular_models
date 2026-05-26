from __future__ import annotations

import random
import subprocess
import warnings
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
TINYIMAGENET200_MEAN = (0.4802, 0.4481, 0.3975)
TINYIMAGENET200_STD = (0.2302, 0.2265, 0.2262)
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


class Dataloaders:
    train_dataloader: DataLoader = None
    valid_dataloader: DataLoader = None
    test_dataloader: DataLoader = None


@lru_cache(maxsize=1)
def _torch_shm_manager_available() -> bool:
    shm_manager = Path(torch.__file__).resolve().parent / "bin" / "torch_shm_manager"
    if not shm_manager.exists():
        return True

    try:
        probe = subprocess.run(
            [str(shm_manager), "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, PermissionError):
        return False
    probe_output = f"{probe.stdout}\n{probe.stderr}"
    if probe.returncode != 0 and "Operation not permitted" in probe_output:
        return False
    return True


def _resolve_num_workers(num_workers: int) -> int:
    requested_workers = max(0, int(num_workers))
    if requested_workers == 0:
        return 0
    if _torch_shm_manager_available():
        return requested_workers

    warnings.warn(
        "torch_shm_manager is unavailable in this environment; falling back to num_workers=0.",
        RuntimeWarning,
        stacklevel=2,
    )
    return 0


def _resolve_pin_memory(pin_memory: bool | None) -> bool:
    if pin_memory is not None:
        return bool(pin_memory)
    return torch.cuda.is_available()


def _normalize_taskname(taskname: str) -> str:
    return str(taskname).replace("-", "").replace("_", "").upper()


def _build_cifar_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
    return train_transform, test_transform


def _build_tinyimagenet_transforms(
    image_size: int = 64,
) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(image_size, padding=8),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=TINYIMAGENET200_MEAN, std=TINYIMAGENET200_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=TINYIMAGENET200_MEAN, std=TINYIMAGENET200_STD),
    ])
    return train_transform, test_transform


def _read_nonempty_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _image_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class TinyImageNetDataset(Dataset):
    def __init__(
        self,
        samples: Iterable[tuple[Path, int]],
        transform=None,
        class_to_idx: dict[str, int] | None = None,
    ):
        self.samples = list(samples)
        self.targets = [target for _, target in self.samples]
        self.transform = transform
        self.class_to_idx = dict(class_to_idx or {})
        self.classes = [
            class_name
            for class_name, _ in sorted(
                self.class_to_idx.items(),
                key=lambda item: item[1],
            )
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class ClassicCVDataloaders(Dataloaders):
    def __init__(
        self,
        path_to_data: str,
        train_val_ratio: tuple[float, float] | None = None,
        batch_size: int = 128,
        taskname: str = "MNIST",
        num_workers: int = 2,
        pin_memory: bool | None = None,
        valid_ratio: float | None = None,
        valid_per_class: int | None = None,
        seed: int = 42,
        num_classes: int | None = None,
        image_size: int | None = None,
    ):
        resolved_num_workers = _resolve_num_workers(num_workers)
        resolved_pin_memory = _resolve_pin_memory(pin_memory)

        normalized_taskname = _normalize_taskname(taskname)
        if normalized_taskname in {"TINYIMAGENET200", "TINYIMAGENET"}:
            self._init_tinyimagenet200(
                path_to_data=path_to_data,
                batch_size=batch_size,
                num_workers=resolved_num_workers,
                pin_memory=resolved_pin_memory,
                valid_ratio=valid_ratio,
                valid_per_class=valid_per_class,
                seed=seed,
                num_classes=num_classes,
                image_size=64 if image_size is None else image_size,
            )
            return

        if train_val_ratio is None:
            train_val_ratio = (0.9, 0.1)
        assert sum(train_val_ratio) <= 1.0001 and sum(
            train_val_ratio) >= 0.9999

        task2class = {
            "MNIST": datasets.MNIST,
            "CIFAR10": datasets.CIFAR10,
            "FASHIONMNIST": datasets.FashionMNIST,
        }
        if normalized_taskname not in task2class:
            known_tasks = ", ".join(sorted([*task2class, "TINYIMAGENET200"]))
            raise ValueError(f"Unknown taskname={taskname!r}. Known tasks: {known_tasks}.")
        train_transform, test_transform = _build_cifar_transforms()

        full_train_dataset = task2class[normalized_taskname](
            root=path_to_data,
            train=True,
            transform=test_transform,
            download=True
        )
        test_dataset = task2class[normalized_taskname](
            root=path_to_data,
            train=False,
            transform=test_transform,
            download=True
        )

        train_size = int(len(full_train_dataset))
        val_size = int(train_val_ratio[1]*train_size)
        train_size = train_size - val_size

        split_indices = torch.randperm(
            len(full_train_dataset),
            generator=torch.Generator().manual_seed(seed),
        ).tolist()
        train_indices = split_indices[:train_size]
        val_indices = split_indices[train_size:]

        val_dataset = Subset(full_train_dataset, val_indices)

        train_augmented_dataset = task2class[normalized_taskname](
            root=path_to_data,
            train=True,
            transform=train_transform,
            download=True
        )
        train_dataset = Subset(train_augmented_dataset, train_indices)

        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=resolved_num_workers,
            pin_memory=resolved_pin_memory,
            drop_last=True
        )

        self.valid_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=resolved_num_workers,
            pin_memory=resolved_pin_memory,
            drop_last=False
        )

        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=resolved_num_workers,
            pin_memory=resolved_pin_memory,
            drop_last=False
        )

    def _init_tinyimagenet200(
        self,
        *,
        path_to_data: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        valid_ratio: float | None,
        valid_per_class: int | None,
        seed: int,
        num_classes: int | None,
        image_size: int,
    ) -> None:
        tiny_root = Path(path_to_data).expanduser()
        wnids_path = tiny_root / "wnids.txt"
        train_root = tiny_root / "train"
        val_images_root = tiny_root / "val" / "images"
        val_annotations_path = tiny_root / "val" / "val_annotations.txt"
        required_paths = [wnids_path, train_root, val_images_root, val_annotations_path]
        missing_paths = [path for path in required_paths if not path.exists()]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(
                "TinyImageNet-200 was not found in the expected layout. "
                f"Missing: {missing}. Download it manually, for example: "
                "mkdir -p data && cd data && "
                "wget http://cs231n.stanford.edu/tiny-imagenet-200.zip && "
                "unzip tiny-imagenet-200.zip"
            )

        if valid_per_class is not None and valid_ratio is not None:
            raise ValueError("Set only one of valid_per_class or valid_ratio for TinyImageNet-200.")
        if valid_per_class is None and valid_ratio is None:
            valid_ratio = 0.1

        wnids = sorted(_read_nonempty_lines(wnids_path))
        if num_classes is not None and int(num_classes) != len(wnids):
            raise ValueError(
                f"TinyImageNet-200 config expected num_classes={num_classes}, "
                f"but wnids.txt contains {len(wnids)} classes."
            )
        class_to_idx = {wnid: class_idx for class_idx, wnid in enumerate(wnids)}
        train_transform, test_transform = _build_tinyimagenet_transforms(image_size=image_size)

        train_samples, valid_samples = self._build_tinyimagenet_train_valid_samples(
            train_root=train_root,
            wnids=wnids,
            class_to_idx=class_to_idx,
            valid_ratio=valid_ratio,
            valid_per_class=valid_per_class,
            seed=seed,
        )
        test_samples = self._build_tinyimagenet_val_samples(
            val_images_root=val_images_root,
            val_annotations_path=val_annotations_path,
            class_to_idx=class_to_idx,
        )

        train_dataset = TinyImageNetDataset(
            train_samples,
            transform=train_transform,
            class_to_idx=class_to_idx,
        )
        valid_dataset = TinyImageNetDataset(
            valid_samples,
            transform=test_transform,
            class_to_idx=class_to_idx,
        )
        test_dataset = TinyImageNetDataset(
            test_samples,
            transform=test_transform,
            class_to_idx=class_to_idx,
        )

        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        self.valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    @staticmethod
    def _build_tinyimagenet_train_valid_samples(
        *,
        train_root: Path,
        wnids: list[str],
        class_to_idx: dict[str, int],
        valid_ratio: float | None,
        valid_per_class: int | None,
        seed: int,
    ) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
        rng = random.Random(int(seed))
        train_samples: list[tuple[Path, int]] = []
        valid_samples: list[tuple[Path, int]] = []

        for wnid in wnids:
            class_image_root = train_root / wnid / "images"
            if not class_image_root.exists():
                raise FileNotFoundError(f"TinyImageNet train class directory is missing: {class_image_root}")

            class_samples = [
                (image_path, class_to_idx[wnid])
                for image_path in _image_paths(class_image_root)
            ]
            if not class_samples:
                raise ValueError(f"TinyImageNet class {wnid} does not contain any images.")

            shuffled_samples = list(class_samples)
            rng.shuffle(shuffled_samples)
            if valid_per_class is not None:
                valid_count = int(valid_per_class)
            else:
                assert valid_ratio is not None
                ratio = float(valid_ratio)
                if not 0.0 <= ratio < 1.0:
                    raise ValueError("valid_ratio must be within [0.0, 1.0).")
                valid_count = int(round(len(shuffled_samples) * ratio))
                if ratio > 0.0 and len(shuffled_samples) > 1:
                    valid_count = max(1, valid_count)

            if valid_count < 0:
                raise ValueError("valid_per_class must be >= 0.")
            if valid_count >= len(shuffled_samples):
                raise ValueError(
                    f"TinyImageNet class {wnid} has {len(shuffled_samples)} images, "
                    f"but valid_count={valid_count}; at least one train image per class is required."
                )

            valid_samples.extend(shuffled_samples[:valid_count])
            train_samples.extend(shuffled_samples[valid_count:])

        return train_samples, valid_samples

    @staticmethod
    def _build_tinyimagenet_val_samples(
        *,
        val_images_root: Path,
        val_annotations_path: Path,
        class_to_idx: dict[str, int],
    ) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        for line in _read_nonempty_lines(val_annotations_path):
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(
                    f"Malformed TinyImageNet val annotation line in {val_annotations_path}: {line!r}"
                )
            filename, wnid = fields[0], fields[1]
            if wnid not in class_to_idx:
                raise ValueError(
                    f"TinyImageNet val annotation references unknown class {wnid!r}."
                )
            image_path = val_images_root / filename
            if not image_path.exists():
                raise FileNotFoundError(f"TinyImageNet val image is missing: {image_path}")
            samples.append((image_path, class_to_idx[wnid]))
        return sorted(samples, key=lambda sample: sample[0].name)
