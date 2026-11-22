# Two fresh stochastic-ST runs with wider quality tolerances

Exactly two models run sequentially: search 60 epochs, inherited physical recovery
90 epochs, then official CIFAR-10 test, before the next model starts. Total new
training: 300 epochs. Each model consumes 150 epochs. The existing launcher accepts
the explicit pair configuration; its historical filename does not set the count.

| ID | soft_drop | hard_drop | Soft / hard tolerance, percentage points |
| --- | ---: | ---: | ---: |
| gap_0125_025pp | 0.00125 | 0.0025 | 0.125 / 0.25 |
| gap_025_050pp | 0.0025 | 0.005 | 0.25 / 0.50 |

The prior 9.053M point retained 3842 of 7552 internal channel positions, with
validation 94.24%. Its tolerances were 0.00025 / 0.0005. These new bands are five
and ten times wider. When the rolling validation gap is below soft_drop, adaptive
lambda increases; above hard_drop it decreases. Wider bands permit increasing
lambda for larger gaps. Feedback and checkpoint selection make the final number
of channels nonmonotonic in the tolerances; fewer than 3842 is an experiment goal,
not a guaranteed result or a new selection constraint.

Only soft_drop and hard_drop change in the resolved training configuration.
Internal gates, stochastic gumbel_hard search training (noise beta=1, temperature=1),
deterministic_hard search validation, adaptive lambda init=0.001/min=1e-8/max=80,
min_keep_ratio=0.08, label smoothing=0, entropy coefficient=0, seed=42, optimizer,
scheduler and selection epochs 31-60/pre-BN scoring remain unchanged. Recovery
inherits the selected surviving weights and uses train-only BN200, fresh AdamW
and a fresh 90-epoch cosine schedule. Recovery and deployment have no gates.

Both models start from the same pinned zero-epoch initializer. The reference input
path reuses the verified cache created by the completed full4 run:
`outputs/runs/pruning_stochastic_full4_20261121/reference_inputs`. Hashes are identical
to the full4 study. These are reference metadata and the zero-epoch initializer,
not the trained full4 model weights. If the cache was removed, upload the existing
`epoch_split_dense_reference_seed42_fresh_upload.tar.gz`; the launcher can restore
the four inputs without fitting a baseline. Bounded checkpoint storage remains on.

## Server command

```sh
if git fetch origin &&
   git switch feature/pruning-ablations-search-recovery &&
   git pull --ff-only &&
   (test -x .venv/bin/python || python3 -m venv .venv) &&
   .venv/bin/python -m pip install --no-cache-dir -r requirements.txt &&
   .venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))' &&
   .venv/bin/python scripts/run_stochastic_st_full4.py \
     --config configs/experiment/pruning_v3/stochastic_st_stronger2.yaml --stage inspect; then
  nohup .venv/bin/python -u scripts/run_stochastic_st_full4.py \
    --config configs/experiment/pruning_v3/stochastic_st_stronger2.yaml \
    > pruning_stochastic_stronger2.log 2>&1 < /dev/null &
  tail -f pruning_stochastic_stronger2.log
fi
```

Output: `outputs/runs/pruning_stochastic_stronger2_20261122`. Use `--dry-run` for
a two-model/300-epoch preview, or `--stage test` to retry missing official test
inference after both fits complete. Checkpoints and configuration selection use
validation only. Test comparisons informed by prior results remain exploratory.
