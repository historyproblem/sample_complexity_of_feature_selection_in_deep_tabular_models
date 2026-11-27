# Three complete stochastic-ST quality probes around the17M reference

The reference is `soft0025_hard050pp`:17,076,817 physical parameters,
93.90% selected recovery validation and93.59% official test. Its search uses
soft_drop0.00025/hard_drop0.005, warmup10 and lambda_init0.001.

Only two training parameters change: `training_arguments.adaptive_lambda.soft_drop`
and `training_arguments.adaptive_lambda.hard_drop`.

| Run | soft_drop | hard_drop | Soft/hard, percentage points | Purpose |
| --- | ---: | ---: | ---: | --- |
| Historical reference |0.00025 |0.005 |0.025 /0.5 | Measured17.077M reference; not rerun in this queue |
| tighter_hard |0.00025 |0.004 |0.025 /0.4 | Earlier lambda reduction when validation deteriorates |
| earlier_pressure |0.001 |0.005 |0.1 /0.5 | Permit lambda increases at slightly larger validation gaps |
| combined |0.001 |0.004 |0.1 /0.4 | Combine earlier increases and stricter protection |

These are hypotheses about search trajectories, not promises of17M parameters
or higher accuracy. Thresholds do not directly change a fixed physical recovery
model: each probe starts a fresh search from the same zero-epoch initializer.
They can alter both the learned mask and the inherited weights.

Warmup10, lambda_init0.001, automatic log-step (factor2.511886), lambda bounds
1e-8..80, floor0.08, internal gates, stochastic `gumbel_hard` training with noise
beta1/temperature1, deterministic-hard validation at probability threshold0.5,
label smoothing0, entropy0, seed42, backbone, augmentation, optimizer, learning
rates and schedulers are unchanged. Physical candidate selection remains
pre-BN on epochs31-60, including fallback. Recovery retains the same inherited
weights/mask policy, train-only BN200, fresh AdamW and cosine90; it has no gates.

Exactly three new full models run sequentially:
`search60 -> inherited recovery90 -> frozen official test`, then the next model.
The queue allocates450 training epochs total,150 per model. No old search weights
are required and no completed150-epoch deployment receives extra training.
Bounded checkpoints retain the online winners/best/last rather than every epoch.
The launcher extension only admits an exact three-model manifest; it does not
modify the training method or change old two-/four-model or timing12 queues.

## Server command

Run from the repository root. Dependencies use the standard `requirements.txt`.
The existing full4 reference cache is checked first, with fallback restoration
from `epoch_split_dense_reference_seed42_fresh_upload.tar.gz`; no new dense fit.

```sh
if git fetch origin &&
   git switch feature/pruning-ablations-search-recovery &&
   git pull --ff-only &&
   (test -x .venv/bin/python || python3 -m venv .venv) &&
   .venv/bin/python -m pip install --no-cache-dir -r requirements.txt &&
   .venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))' &&
   .venv/bin/python scripts/run_stochastic_st_full4.py \
     --config configs/experiment/pruning_v3/stochastic_st_quality17m3.yaml --stage inspect; then
  nohup .venv/bin/python -u scripts/run_stochastic_st_full4.py \
    --config configs/experiment/pruning_v3/stochastic_st_quality17m3.yaml --stage run \
    > pruning_stochastic_quality17m3.log 2>&1 < /dev/null &
  tail -f pruning_stochastic_quality17m3.log
fi
```

Output: `outputs/runs/pruning_stochastic_quality17m3_20261127`.
Use `--stage run --dry-run` to preview the three-model budget without training.
Selection uses validation only; test results are exploratory given prior runs.
