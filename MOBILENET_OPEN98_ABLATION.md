# MobileNet: Gumbel noise versus deterministic STE

Two independent TinyImageNet-200 models, each exactly **30 search + 15 recovery = 45 new training epochs**. The sequential workflow uses90newepochs across the two models. Both start from ImageNet1K V2, a fresh200-class classifier, and fresh gates; neither inherits the other model's weights or controller.

Shared settings:

- Initial lambda0.001; adaptive controller enabled, warmup10 inside the30search epochs, three-epoch accuracy window, updates every epoch, no step boost. Automatic log-step is `log(10/0.001)/20 = 0.46051701859880917`.
- `gumbel_init_mode=paper_resnet50`: off/on logits0.1/4.0 plus the existing small initialization noise, nominal opening probability0.9801596943. This changes gate initialization, not the backbone architecture.
- Soft/hard accuracy-drop tolerances6/10percentage points. This intermediate band was already used in the previous grid and keeps the two arms comparable without picking a test winner.
- Internal expansion/depthwise/project channel masks enabled. Output masks use the existing residual-block semantics. Forward and removal thresholds both0.5; no parameter-removal cap.
- AdamW, classifier/gateLR1e-3 and ImageNet-weightLR1e-4, weight decay5e-4, cosine schedules30/15; label smoothing0, entropy disabled, batch128, workers0, seed42.
- Arm1 trains with `gumbel_hard` (beta1,temperature1); arm2 with `ste_hard` (deterministic forward, soft surrogate backward). Both evaluate with `deterministic_hard`.

The mask is read from the **best-validation search checkpoint's own channel metrics**, then those same trained Conv/BN/classifier values are sliced into the physical recovery model. This avoids combining the last epoch's mask with best.pt weights. Recovery has no gates/penalty, uses a fresh optimizer/cosine15, and retains the selected source controller state in `cyclic_summary.json:search_selections` for provenance. It does not reset a controller for a subsequent search because there is no subsequent search in this workflow.

The new checkpoint-selected path treats p_open=0.5 as closed, matching deterministic forward. The existing minimum-one-channel-per-group rule remains in force.

The data split is90,000train/10,000internal validation; the official labeled TinyImageNet val10,000images serve as test. Recovery best.pt is selected by internal validation and evaluated frozen on that official test. No evaluation BN updates or recalibration. Search retains the existing frozen-test evaluation behavior, but neither search test nor recovery test selects checkpoints/settings. Final comparison fields keep validation and official test separate; results are exploratory because prior test results have informed experimentation. One seed is not a multi-seed significance estimate.

## Reference curve

The launcher imports the same previously selected historical dense-reference archive from `outputs/baselines/mobilenet_baseline_partial.tar.gz`. It validates its checksum and copies the86available history epochs into the local output directory; the controller uses matching reference epochs1–30 from the historical150-epoch/cosine150 recipe. Reference history/metrics remain local input and are not included in the branch. Reference weights do not initialize either search and no new reference training occurs. The historical reference cost is recorded separately from45newepochs per model. Use `--baseline-archive PATH` if the archive is elsewhere.

## Server commands

Run from the repository root. Fetch/switch/pull must succeed first:

```bash
git fetch origin
git switch feature/mobilenet-open98-gumbel-ablation
git pull --ff-only
```

If the server environment/data have not been prepared:

```bash
bash scripts/setup_mobilenet_onepass.sh
```

The existing baseline archive must be present at `outputs/baselines/mobilenet_baseline_partial.tar.gz` or supplied with `--baseline-archive`. Dry-run prints the reference-import command plus exactly two training commands and uses only the standard library:

```bash
.venv/bin/python scripts/run_mobilenet_open98_ablation.py --dry-run
```

Launch sequentially:

```bash
mkdir -p outputs/logs
nohup .venv/bin/python -u scripts/run_mobilenet_open98_ablation.py \
  > outputs/logs/mobilenet_open98_gumbel_ablation.log 2>&1 < /dev/null &
tail -f outputs/logs/mobilenet_open98_gumbel_ablation.log
```

Outputs: `outputs/runs/mobilenet_open98_gumbel_ablation_20261120/{gumbel_hard,ste_hard}` and aggregate `comparison.csv`/`comparison.json` in that root. The comparison includes physical parameter count, internal/output removals, selected search/recovery epochs, validation accuracy and frozen official test accuracy.

Matching completed arms are reused. Missing/partial/foreign outputs or changed fingerprints fail before additional training; an interrupted arm is not automatically retrained. A root lock prevents two launchers from using the same output directory.

## Verification

150 affected checks passed, including19new ablation checks. They cover real config composition, identical native model initialization, stochastic versus deterministic hard samples and STE gradients, best-epoch mask/weight agreement, physical MobileNet survivor transfer/forward equivalence, held controller provenance, exact30+15cost, local reference validation, sequential execution, failure handling and completed-run reuse. Both real CLI configurations resolve; standard-library-only dry-run prints reference import plus two training commands. The real previously selected archive was also imported and validated locally without training.

An expanded legacy preset check has9pre-existing failures expecting older ImageNet V1/64px defaults. The same9failures reproduce on unmodified parent0c2ed7de (58other native MobileNet checks pass). The experiment uses the existing ImageNet V2/224px recipe. Local checks used Python3.9.6/Torch2.8CPU; the inherited server preflight checks the previously verified Torch2.6/torchvision0.21CUDA stack. These synthetic checks are not new TinyImageNet training or quality results.
