"""Run four historical pre-BN selection recipes with two channel floors."""
from launch_resnet50_cifar10_ls0_curve7 import PREBN_REPRO_CONFIG, main


if __name__ == "__main__":
    result = main(default_config=PREBN_REPRO_CONFIG)
    if result.get("status") == "completed_with_failures":
        raise SystemExit(1)
