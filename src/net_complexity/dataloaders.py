import torch
import collections
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import os
import json


class Dataloaders:
    train_dataloader: DataLoader = None
    valid_dataloader: DataLoader = None
    test_dataloader: DataLoader = None


class ClassicCVDataloaders(Dataloaders):
    def __init__(self, path_to_data: str,
                 train_val_ratio: tuple[float, float],
                 batch_size: int,
                 taskname: str = 'MNIST'):
        assert sum(train_val_ratio) <= 1.0001 and sum(
            train_val_ratio) >= 0.9999

        task2class = {
            'MNIST': datasets.MNIST,
            'CIFAR10': datasets.CIFAR10,
            'FashionMNIST': datasets.FashionMNIST
        }
        # TODO:
        # change for other datasets
        # augmentations?

        transform = transforms.Compose([
            lambda x: x.convert("RGB"),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = task2class[taskname](
            root=path_to_data,
            train=True,
            transform=transform,
            download=True
        )
        test_dataset = task2class[taskname](
            root=path_to_data,
            train=False,
            transform=transform,
            download=True
        )

        train_size = int(len(train_dataset))
        val_size = int(train_val_ratio[1]*train_size)
        train_size = train_size - val_size

        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size], torch.Generator().manual_seed(42))

        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )

        self.valid_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False
        )

        self.test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False
        )
