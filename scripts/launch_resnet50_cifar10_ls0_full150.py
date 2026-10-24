"""Run one fresh LS0 ResNet50/CIFAR10 search60 + recovery90 model."""
from launch_resnet50_cifar10_ls0_curve7 import FULL150_CONFIG, main


if __name__ == "__main__":
    main(default_config=FULL150_CONFIG)
