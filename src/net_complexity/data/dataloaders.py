import torch
import collections
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
import os
import json
import subprocess
import warnings
from functools import lru_cache
from pathlib import Path


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


class ClassicCVDataloaders(Dataloaders):
    def __init__(self, path_to_data: str,
                 train_val_ratio: tuple[float, float],
                 batch_size: int,
                 taskname: str = 'MNIST',
                 num_workers: int = 2,
                 pin_memory: bool | None = None):
        assert sum(train_val_ratio) <= 1.0001 and sum(
            train_val_ratio) >= 0.9999

        resolved_num_workers = _resolve_num_workers(num_workers)
        resolved_pin_memory = _resolve_pin_memory(pin_memory)

        task2class = {
            'MNIST': datasets.MNIST,
            'CIFAR10': datasets.CIFAR10,
            'FashionMNIST': datasets.FashionMNIST
        }
        # TODO:
        # change for other datasets
        # augmentations?
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])

        full_train_dataset = task2class[taskname](
            root=path_to_data,
            train=True,
            transform=test_transform,
            download=True
        )
        test_dataset = task2class[taskname](
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
            generator=torch.Generator().manual_seed(42),
        ).tolist()
        train_indices = split_indices[:train_size]
        val_indices = split_indices[train_size:]

        val_dataset = Subset(full_train_dataset, val_indices)

        train_augmented_dataset = task2class[taskname](
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
