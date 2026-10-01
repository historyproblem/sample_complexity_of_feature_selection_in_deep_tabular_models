# MobileNetV2/CIFAR-10 adaptive-lambda 60/90 curve

The checked overnight plan is
`configs/mobilenetv2_cifar10_parameter_curve_60_90_nightly.yaml`. It trains one
dense 150-epoch validation reference, then four independent models. Each model
uses 60 adaptive-lambda search epochs and one 90-epoch physical recovery, so no
model exceeds the 150-epoch budget.

The four validation-drop bands are 0.5/1.0, 0.75/1.0, 1.0/1.5 and 1.25/1.75
percentage points. The 1.66M, 1.13M, 0.71M and 0.58M sizes are reporting hints,
not forced quotas: validation-only checkpoint selection and learned gates decide
the actual physical parameter count. The dense CIFAR-10 model has 2,236,682
physical parameters and the configured 8% per-gate safety floor is 575,594.

Run the full server preflight first. Unlike TinyImageNet, this step downloads
and verifies the official CIFAR-10 train and test archives automatically:

```bash
.venv/bin/python scripts/launch_mobilenetv2_parameter_curve.py \
  --from-scratch --preflight-only
```

Then start the unattended run:

```bash
nohup .venv/bin/python scripts/launch_mobilenetv2_parameter_curve.py \
  --from-scratch > mobilenetv2_cifar10_parameter_curve.log 2>&1 &
```

Each finalized deployment is evaluated once on the frozen official CIFAR-10
test split. Search selection and adaptive-lambda control use only the fixed
45,000/5,000 train/validation split.
