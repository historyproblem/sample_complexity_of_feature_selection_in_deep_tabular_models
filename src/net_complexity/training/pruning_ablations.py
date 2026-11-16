"""Physical forks for the size ablations; no training or test access here."""
from copy import deepcopy
import hashlib
import json

from .one_shot_pruning_config import to_v3_config
from .pruning_audit import build_structural
from .pruning_measurement import isolated_diagnostic_rng, mask_hash, state_hash
from net_complexity.models.pruning_budget import gates, make_matched_random_mask


def build_ablation_branches(config, carrier, learned_mask, identity, *, search_only=False):
    from .one_shot_pruning import build_physical_branches, physical_architecture_signature

    before = state_hash(carrier.state_dict())
    if identity["selected_model_state_hash"] != before or identity["mask_hash"] != mask_hash(learned_mask):
        raise ValueError("Selected checkpoint/mask identity differs before ablation fork")
    if search_only or list(config.one_shot.branches) == ["inherited"]:
        with isolated_diagnostic_rng():
            inherited = build_structural(to_v3_config(config), carrier, learned_mask).cpu()
        models = {"inherited": inherited}
        report = {}
    else:
        inherited, scratch, report = build_physical_branches(config, carrier, learned_mask, identity)
        random_mask = make_matched_random_mask(
            carrier, learned_mask, seed=int(config.one_shot.random_mask_seed),
            min_keep_ratio=float(config.accuracy_guided.eligibility.min_keep_ratio),
        )
        with isolated_diagnostic_rng():
            randomized = build_structural(to_v3_config(config), carrier, random_mask).cpu()
        if sum(p.numel() for p in randomized.parameters()) != sum(p.numel() for p in inherited.parameters()):
            raise AssertionError("Random and learned physical sizes differ")
        models = {"inherited": inherited, "scratch": scratch, "random": randomized}
    for name, model in models.items():
        if gates(model):
            raise AssertionError("Ablation recovery must be physically ungated")
        architecture_hash = hashlib.sha256(json.dumps(
            physical_architecture_signature(model), sort_keys=True,
        ).encode()).hexdigest()
        report[name] = {
            "initialization": ("pytorch_default_all_trainable_and_bn" if name == "scratch"
                               else "selected_surviving_state"),
            "initialization_state_hash": state_hash(model.state_dict()),
            "architecture_hash": architecture_hash,
            "pruning_mask": deepcopy(random_mask if name == "random" else learned_mask),
        }
    if state_hash(carrier.state_dict()) != before:
        raise AssertionError("Ablation fork mutated the selected search checkpoint")
    return models, {**identity, **report, "architecture_hash": report["inherited"]["architecture_hash"]}
