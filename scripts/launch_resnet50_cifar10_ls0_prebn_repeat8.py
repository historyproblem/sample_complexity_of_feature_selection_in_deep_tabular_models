"""Repeat the strict pre-BN floor0.001 recipe eight times from the same initializer."""
from launch_resnet50_cifar10_ls0_curve7 import PREBN_REPEAT_CONFIG, main


if __name__ == "__main__":
    result = main(default_config=PREBN_REPEAT_CONFIG)
    if result.get("status") == "completed_with_failures":
        raise SystemExit(1)
