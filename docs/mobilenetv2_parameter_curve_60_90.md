# MobileNetV2 four-point 60/90 pruning curve

The checked nightly plan is
`configs/mobilenetv2_parameter_curve_60_90_nightly.yaml`. It runs four
independent models. Each model uses one 60-epoch adaptive-lambda mask search,
then one 90-epoch physical MobileNetV2 recovery. The dense validation reference
is trained once and is external to every model's 150-epoch budget.

The ResNet50 landmarks 4M/9M/13M/18M cannot be literal MobileNetV2 targets:
the unpruned physical MobileNetV2 has only 2,480,072 parameters. The plan uses
feasible reporting hints 0.82M/0.95M/1.37M/1.90M. These are not quotas. The
four validation drop bands are the actual controls:

| Point | soft drop | hard drop | size hint |
|---|---:|---:|---:|
| target_1p90m | 0.50 pp | 1.00 pp | 1.90M |
| target_1p37m | 0.75 pp | 1.00 pp | 1.37M |
| target_0p95m | 1.00 pp | 1.50 pp | 0.95M |
| target_0p82m | 1.25 pp | 1.75 pp | 0.82M |

`min_keep_ratio=0.08` is only a safety floor. Exhaustively putting every
eligible internal width on that floor yields exactly 818,984 physical
parameters. The selected size is still learned and selected with validation.

## Server preparation

Run from the repository root and use the repository environment. TinyImageNet
must already be unpacked at `data/tiny-imagenet-200`; the launcher checks all
200 train class directories and the 10,000-row official validation annotation
file before spending GPU time.

Preview the complete 750-epoch clean-start plan without touching CUDA or data:

```bash
.venv/bin/python scripts/launch_mobilenetv2_parameter_curve.py \
  --from-scratch --dry-run
```

Run the server preflight only (CUDA forward/backward through every gate,
structural export/budget check, data layout, config checks, and reference
checks when an existing reference is supplied):

```bash
.venv/bin/python scripts/launch_mobilenetv2_parameter_curve.py \
  --from-scratch --preflight-only
```

For a clean server, train one dense150 reference and then all four points:

```bash
.venv/bin/python scripts/launch_mobilenetv2_parameter_curve.py --from-scratch
```

If a complete dense reference was downloaded and unpacked, reuse it instead:

```bash
.venv/bin/python scripts/launch_mobilenetv2_parameter_curve.py \
  --dense-source /absolute/path/to/mobilenetv2_dense_reference_seed42
```

The dense source must contain `J1_dense_control/pilot_state.json`,
`J1_dense_control/global_history.csv`, `J1_dense_control/resolved_config.yaml`,
and the sibling `shared_random_seed42.pt`. Trained baseline weights alone are
not enough: the method deliberately initializes all searches from the same
verified zero-epoch seed42 initializer while using the dense run only as an
immutable validation reference.

Progress is written atomically to `nightly_state.json`. Final validation,
physical parameter counts, target-hint deviations and frozen official test
metrics are collected in `nightly_summary.json`.
