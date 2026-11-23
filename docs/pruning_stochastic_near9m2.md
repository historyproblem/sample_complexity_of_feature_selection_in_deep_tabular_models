# Two fresh stochastic-ST runs near the compact measured profiles

The wider .00125/.0025 and .0025/.005 probes retained 7552 and 7120 internal
channel positions. The earlier .00025/.0005 and .0003/.0006 profiles retained
3842 and 4386, with physical sizes 9.053M and 10.272M. These two new probes explore
closer tolerances. Their resulting sizes are unknown; `near9m2` is a planning
label, not a selection constraint or a promise of fewer than 3842 channels.

| ID | soft_drop | hard_drop | Soft / hard tolerance, percentage points |
| --- | ---: | ---: | ---: |
| gap_00275_0055pp | 0.000275 | 0.00055 | 0.0275 / 0.055 |
| gap_0035_007pp | 0.00035 | 0.0007 | 0.035 / 0.070 |

Exactly two new models run sequentially. Each consumes search60 -> inherited
recovery90 -> official test before the next search. Total new training: 300 epochs,
150 per model. Only adaptive-lambda soft_drop and hard_drop differ in the training
recipe. Internal gates, Gumbel-hard search training/noise beta1/temperature1,
deterministic-hard validation, adaptive lambda init.001/min1e-8/max80, floor.08,
label smoothing0/entropy0, seed42, optimizer/scheduler and selection epochs31-60/
physical pre-BN scoring are unchanged. Recovery uses inherited surviving weights,
train-only BN200, fresh AdamW and cosine90; physical recovery/deployment have no
gates. Bounded checkpoint retention and the pinned reference/initializer remain.

The reference path reuses the existing full4 `reference_inputs` cache; hashes are
unchanged. Both searches start from the shared zero-epoch initializer. Missing
cache can be restored from the original reference archive without fitting a new
baseline. The default full4 launcher runs exactly two models with this config.

## Server command

```sh
if git fetch origin &&
   git switch feature/pruning-ablations-search-recovery &&
   git pull --ff-only &&
   (test -x .venv/bin/python || python3 -m venv .venv) &&
   .venv/bin/python -m pip install --no-cache-dir -r requirements.txt &&
   .venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))' &&
   .venv/bin/python scripts/run_stochastic_st_full4.py \
     --config configs/experiment/pruning_v3/stochastic_st_near9m2.yaml --stage inspect; then
  nohup .venv/bin/python -u scripts/run_stochastic_st_full4.py \
    --config configs/experiment/pruning_v3/stochastic_st_near9m2.yaml \
    > pruning_stochastic_near9m2.log 2>&1 < /dev/null &
  tail -f pruning_stochastic_near9m2.log
fi
```

Output: `outputs/runs/pruning_stochastic_near9m2_20261123`. `--dry-run` previews the
two-model plan without training; `--stage test` retries missing test inference
after both fits complete. Selection uses validation; test comparisons remain
exploratory after earlier results informed the experiment series.

## Candidate records and the console log

The console `.log` contains progress and the selected model, not every candidate's
mask/physical size. In bounded-checkpoint mode, all search candidate records for
epochs31-60 are written separately to
`<id>/internal/search/shared_search/training/<run>/selection_candidates.jsonl`.
Each search record includes epoch, physical_cost, accuracy, ce_loss and
pruning_mask. This file can explain which candidates were rejected without
retaining every epoch's weights. It was not included in the supplied stronger2
console log. These new runs do not depend on uploading it.
