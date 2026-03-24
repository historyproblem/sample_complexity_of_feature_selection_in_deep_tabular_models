import collections
import torchvision.transforms as transforms
import os
import json

try:
    from IPython import embed
except:
    pass

_DATASETS = {}

Dataset = collections.namedtuple(
    'Dataset', ['trainset', 'testset'])


def _add_dataset(dataset_fn):
    _DATASETS[dataset_fn.__name__] = dataset_fn
    return dataset_fn


def _get_transforms(augment=True, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    basic_transform = [transforms.ToTensor(), normalize]

    transform_train = []
    if augment:
        transform_train += [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transform_train += [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    transform_train += basic_transform
    transform_train = transforms.Compose(transform_train)

    transform_test = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    transform_test += basic_transform
    transform_test = transforms.Compose(transform_test)

    return transform_train, transform_test


def _get_mnist_transforms(augment=True, invert=False, transpose=False):
    transform = [
        transforms.ToTensor(),
    ]
    if invert:
        transform += [transforms.Lambda(lambda x: 1. - x)]
    if transpose:
        transform += [transforms.Lambda(lambda x: x.transpose(2, 1))]
    transform += [
        transforms.Normalize((.5,), (.5,)),
        transforms.Lambda(lambda x: x.expand(3, 32, 32))
    ]

    transform_train = []
    transform_train += [transforms.Pad(padding=2)]
    if augment:
        transform_train += [transforms.RandomCrop(32, padding=4)]
    transform_train += transform
    transform_train = transforms.Compose(transform_train)

    transform_test = []
    transform_test += [transforms.Pad(padding=2)]
    transform_test += transform
    transform_test = transforms.Compose(transform_test)

    return transform_train, transform_test


def _get_cifar_transforms(augment=True):
    transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ]
    transform_train = []
    if augment:
        transform_train += [
            transforms.Pad(padding=4, fill=(125, 123, 113)),
            transforms.RandomCrop(32, padding=0),
            transforms.RandomHorizontalFlip()]
    transform_train += transform
    transform_train = transforms.Compose(transform_train)
    transform_test = []
    transform_test += transform
    transform_test = transforms.Compose(transform_test)
    return transform_train, transform_test


def set_metadata(trainset, testset, config, dataset_name):
    trainset.metadata = {
        'dataset': dataset_name,
        'task_id': config.task_id,
        'task_name': trainset.task_name,
    }
    testset.metadata = {
        'dataset': dataset_name,
        'task_id': config.task_id,
        'task_name': testset.task_name,
    }
    return trainset, testset

def load_tasks_map(tasks_map_file):
    assert os.path.exists(tasks_map_file), tasks_map_file
    with open(tasks_map_file, 'r') as f:
        tasks_map = json.load(f)
        tasks_map = {int(k): int(v) for k, v in tasks_map.items()}
    return tasks_map

@_add_dataset
def split_mnist(root, config):
    assert isinstance(config.task_id, tuple)
    from dataset.mnist import MNISTDataset, SplitMNISTTask
    transform_train, transform_test = _get_mnist_transforms()
    train_tasks = SplitMNISTTask(MNISTDataset(root, train=True))
    trainset = train_tasks.generate(
        classes=config.task_id, transform=transform_train)
    test_tasks = SplitMNISTTask(MNISTDataset(root, train=False))
    testset = test_tasks.generate(
        classes=config.task_id, transform=transform_test)
    trainset, testset = set_metadata(trainset, testset, config, 'split_mnist')
    return trainset, testset


@_add_dataset
def split_cifar(root, config):
    assert 0 <= config.task_id < 11
    from dataset.cifar import CIFAR10Dataset, CIFAR100Dataset, SplitCIFARTask
    transform_train, transform_test = _get_cifar_transforms()
    train_tasks = SplitCIFARTask(CIFAR10Dataset(
        root, train=True), CIFAR100Dataset(root, train=True))
    trainset = train_tasks.generate(
        task_id=config.task_id, transform=transform_train)
    test_tasks = SplitCIFARTask(CIFAR10Dataset(
        root, train=False), CIFAR100Dataset(root, train=False))
    testset = test_tasks.generate(
        task_id=config.task_id, transform=transform_test)
    trainset, testset = set_metadata(trainset, testset, config, 'split_cifar')
    return trainset, testset


@_add_dataset
def cifar10_mnist(root, config):
    from dataset.cifar import CIFAR10Dataset
    from dataset.mnist import MNISTDataset
    from dataset.expansion import UnionClassificationTaskExpander
    transform_train, transform_test = _get_cifar_transforms()
    trainset = UnionClassificationTaskExpander(merge_duplicate_images=False)(
        [CIFAR10Dataset(root, train=True), MNISTDataset(root, train=True, expand=True)], transform=transform_train)
    testset = UnionClassificationTaskExpander(merge_duplicate_images=False)(
        [CIFAR10Dataset(root, train=False), MNISTDataset(root, train=False, expand=True)], transform=transform_test)
    return trainset, testset


@_add_dataset
def cifar10(root):
    from torchvision.datasets import CIFAR10
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset = CIFAR10(root, train=True, transform=train_transform, download=True)
    testset = CIFAR10(root, train=False, transform=test_transform)
    return trainset, testset


@_add_dataset
def cifar100(root):
    from torchvision.datasets import CIFAR100
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    trainset = CIFAR100(root, train=True, transform=transform, download=True)
    testset = CIFAR100(root, train=False, transform=transform)
    return trainset, testset


@_add_dataset
def mnist(root):
    from torchvision.datasets import MNIST
    transform = transforms.Compose([
        lambda x: x.convert("RGB"),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = MNIST(root, train=True, transform=transform, download=True)
    testset = MNIST(root, train=False, transform=transform)
    return trainset, testset


@_add_dataset
def letters(root):
    from torchvision.datasets import EMNIST
    transform = transforms.Compose([
        lambda x: x.convert("RGB"),
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)),
    ])
    trainset = EMNIST(root, train=True, split='letters',
                      transform=transform, download=True)
    testset = EMNIST(root, train=False, split='letters', transform=transform)
    return trainset, testset


@_add_dataset
def kmnist(root):
    from torchvision.datasets import KMNIST
    transform = transforms.Compose([
        lambda x: x.convert("RGB"),
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    trainset = KMNIST(root, train=True, transform=transform, download=False)
    testset = KMNIST(root, train=False, transform=transform)
    return trainset, testset


@_add_dataset
def stl10(root):
    from torchvision.datasets import STL10
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    trainset = STL10(root, split='train', transform=transform, download=True)
    testset = STL10(root, split='test', transform=transform)
    trainset.targets = trainset.labels
    testset.targets = testset.labels
    return trainset, testset


def get_dataset(root, config=None):
    return _DATASETS[config.name](os.path.expanduser(root), config)
