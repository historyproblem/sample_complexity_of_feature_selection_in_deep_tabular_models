from pathlib import Path

import pytest
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from PIL import Image

from net_complexity.data.dataloaders import (
    _build_cifar_transforms,
    _build_tinyimagenet_transforms,
    _geometric_prefix,
    _resolve_transform,
)

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _op_types(compose):
    return [type(op).__name__ for op in compose.transforms]


def test_geometric_prefix_is_empty_without_resize_or_crop():
    assert _geometric_prefix(None, None) == []


def test_geometric_prefix_orders_resize_before_center_crop():
    ops = _geometric_prefix(256, 224)

    assert [type(op).__name__ for op in ops] == ["Resize", "CenterCrop"]
    assert ops[0].size == 256
    assert ops[1].size == (224, 224)


def test_cifar_transforms_default_recipe_is_unchanged():
    train, test = _build_cifar_transforms()

    assert _op_types(train) == ["RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"]
    assert _op_types(test) == ["ToTensor", "Normalize"]


def test_tinyimagenet_transforms_default_recipe_is_unchanged():
    train, test = _build_tinyimagenet_transforms(image_size=64)

    assert _op_types(train) == ["RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"]
    assert _op_types(test) == ["ToTensor", "Normalize"]


def test_tinyimagenet_transforms_apply_resize_and_center_crop():
    train, test = _build_tinyimagenet_transforms(image_size=64, resize=256, center_crop=224)

    # The geometric prefix replaces the native-size RandomCrop; the flip stays.
    assert _op_types(train) == [
        "Resize",
        "CenterCrop",
        "RandomHorizontalFlip",
        "ToTensor",
        "Normalize",
    ]
    assert _op_types(test) == ["Resize", "CenterCrop", "ToTensor", "Normalize"]

    image = Image.new("RGB", (64, 64))
    assert train(image).shape == (3, 224, 224)
    assert test(image).shape == (3, 224, 224)


def test_cifar_transforms_apply_resize_and_center_crop():
    train, test = _build_cifar_transforms(resize=256, center_crop=224)

    assert _op_types(test) == ["Resize", "CenterCrop", "ToTensor", "Normalize"]

    image = Image.new("RGB", (32, 32))
    assert train(image).shape == (3, 224, 224)
    assert test(image).shape == (3, 224, 224)


def test_resize_alone_sets_the_output_size():
    _, test = _build_tinyimagenet_transforms(image_size=64, resize=[128, 128])

    assert _op_types(test) == ["Resize", "ToTensor", "Normalize"]
    assert test(Image.new("RGB", (64, 64))).shape == (3, 128, 128)


def test_resolve_transform_falls_back_to_the_default():
    default = transforms.Compose([transforms.ToTensor()])

    assert _resolve_transform(None, default) is default


def test_resolve_transform_accepts_compose_list_and_callable():
    default = transforms.Compose([transforms.ToTensor()])
    override = transforms.Compose([transforms.Resize(8), transforms.ToTensor()])

    assert _resolve_transform(override, default) is override

    from_list = _resolve_transform([transforms.Resize(8), transforms.ToTensor()], default)
    assert isinstance(from_list, transforms.Compose)
    assert _op_types(from_list) == ["Resize", "ToTensor"]

    def custom(image):
        return torch.zeros(3, 4, 4)

    assert _resolve_transform(custom, default) is custom


def test_resolve_transform_rejects_unusable_override():
    with pytest.raises(TypeError, match="must be a torchvision Compose"):
        _resolve_transform(123, transforms.Compose([transforms.ToTensor()]))


def test_tinyimagenet_resize224_data_config_sets_the_imagenet_preset():
    cfg = OmegaConf.load(CONFIGS_DIR / "data" / "tinyimagenet200_resize224.yaml")

    assert cfg.dataloaders.resize == 256
    assert cfg.dataloaders.center_crop == 224
    assert cfg.dataloaders.taskname == "tinyimagenet200"


def test_native_tinyimagenet_data_config_stays_untouched():
    cfg = OmegaConf.load(CONFIGS_DIR / "data" / "tinyimagenet200_best_practice.yaml")

    # The pre-existing config must keep the native 64x64 pipeline.
    assert "resize" not in cfg.dataloaders
    assert "center_crop" not in cfg.dataloaders


def test_mobilenetv2_224_model_config_keeps_the_stock_stride_two_stem():
    cfg = OmegaConf.load(CONFIGS_DIR / "model" / "mobilenetv2_tinyimagenet200_224.yaml")

    assert cfg.model.backbone._target_ == "net_complexity.wrappers.MobileNetV2"
    assert cfg.model.backbone.stem_stride == 2
    assert cfg.model.backbone.num_classes == 200
