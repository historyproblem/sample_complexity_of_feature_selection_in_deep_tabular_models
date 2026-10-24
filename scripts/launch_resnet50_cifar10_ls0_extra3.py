"""Run three additional LS0 ResNet50/CIFAR10 models at the requested bands."""
from launch_resnet50_cifar10_ls0_curve7 import EXTRA3_CONFIG, main


if __name__ == "__main__":
    main(default_config=EXTRA3_CONFIG)
