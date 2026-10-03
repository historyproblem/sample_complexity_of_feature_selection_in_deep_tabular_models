# Optimizer/scheduler handoff ablation

Config:
`configs/experiment/pruning_v3/optimizer_scheduler_handoff_60_90_repeats2.yaml`

The experiment performs one 60-epoch adaptive search, selects one checkpoint
and one pruning mask by validation, and constructs one compact model state.
Every comparison branch starts from byte-identical compact model parameters and
BatchNorm state. The only method axis is state inherited by the 90-epoch
physical stage:

| Method | AdamW per-parameter state | Cosine state |
| --- | --- | --- |
| `fresh_optimizer_fresh_scheduler` | fresh | new `T_max=90` |
| `mapped_optimizer_fresh_scheduler` | selected checkpoint moments and step | new `T_max=90` |
| `mapped_optimizer_resumed_scheduler` | selected checkpoint moments and step | selected checkpoint state continued |

The shared search uses `CosineAnnealingLR(T_max=150)`. This is required for the
third method: continuing a search-local `T_max=60` cosine after its minimum
would increase the learning rate again. The continued branch resumes the exact
selected checkpoint's `last_epoch`, step count and current base-group learning
rate. The obsolete gate scheduler-group entry is discarded with the gates.

Two paired recovery repeats use seeds 42 and 43. The train/validation split
seed remains 42; only recovery RNG and loader order change. Execution is:

1. all three methods with recovery seed 42;
2. all three methods with recovery seed 43.

Thus the first complete comparison is available after the first three physical
branches. Interrupting during repeat 2 preserves their finalized deployment and
validation artifacts. No BatchNorm recalibration, hidden warmup, test-based
selection, or training beyond 60 + 90 = 150 attributed epochs per branch is
performed.

Preview with an existing dense reference:

```bash
.venv/bin/python scripts/launch_one_shot_pruning.py \
  --config-name experiment/pruning_v3/optimizer_scheduler_handoff_60_90_repeats2 \
  --dense-source PATH_TO_DENSE_RUN \
  --output outputs/runs/optimizer_scheduler_handoff_60_90_repeats2 \
  --dry-run
```

Remove `--dry-run` to train. If all six branches finish, the launcher evaluates
their frozen selected deployments once on the official test set. Test results
are not used for checkpoint selection or further training.
