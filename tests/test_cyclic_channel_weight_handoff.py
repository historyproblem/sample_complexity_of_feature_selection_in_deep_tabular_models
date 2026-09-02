from pathlib import Path

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf

from net_complexity.models.channel_pruning import (
    apply_channel_mask,
    build_structurally_pruned_model_from_config,
    transfer_gated_weights_to_structural,
)
from net_complexity.models.feature_selection import get_gumbel_modules
from net_complexity.training.cyclic_channel_pruning import (
    _next_search_initializer,
    _search_to_structural_initializer,
)


def _weight_handoff_config():
    return OmegaConf.create(
        {
            "model": {
                "_target_": (
                    "net_complexity.wrappers."
                    "ClassificationFeatureSelectionWrapper"
                ),
                "lambda_coef": 0.0,
                "bypass_on_zero_lambda": True,
                "backbone": {
                    "_target_": "net_complexity.wrappers.ResNet50",
                    "num_classes": 5,
                    "in_channels": 3,
                    "stem_kernel_size": 3,
                    "stem_stride": 1,
                    "stem_padding": 1,
                    "use_maxpool": False,
                    "resnet_block": {
                        "_target_": (
                            "net_complexity.wrappers."
                            "MaskedGumbelBottleneckLayer"
                        ),
                        "_partial_": True,
                        "gate_internal_width": True,
                        "force_ones_mask": True,
                    },
                },
            },
            "cyclic_channel_pruning": {
                "weight_handoff": {
                    "enabled": True,
                    "checkpoint_name": "best.pt",
                    "initial_checkpoint": None,
                    "initial_run_dir": None,
                }
            },
        }
    )


def _structural_pruning_config(mask):
    return OmegaConf.create(
        {
            "enabled": True,
            "structural": True,
            "mode": "explicit",
            "mask": mask,
        }
    )


def _save_checkpoint(model: nn.Module, path: Path) -> None:
    torch.save({"model_state_dict": model.state_dict()}, path)


def _make_every_recovery_tensor_observable(model: nn.Module) -> None:
    """Make an omitted parameter or BN-buffer copy fail deterministically."""
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters(), start=1):
            parameter.add_(index * 1e-4)
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.add_(0.125)
                module.running_var.mul_(1.01)
                module.num_batches_tracked.add_(3)


def test_checkpoint_handoff_roundtrip_preserves_finetuned_survivors_and_gates(
    tmp_path,
):
    """Exercise the real checkpoint initializers used between two cycles.

    This covers the complete transition:
      gated search checkpoint -> structurally pruned recovery -> fine-tuning
      step -> next full-width gated carrier.

    Slicing that carrier back to the same structural topology must reproduce
    every recovery tensor exactly, while gate logits and disabled carrier
    positions must still come from the preceding search checkpoint.
    """
    torch.manual_seed(17)
    config = _weight_handoff_config()
    mask = {
        "backbone.layer2.0.gumbel_layer": [0, 7],
        "backbone.layer2.0.mid1_gumbel_layer": [2, 9],
        "backbone.layer2.0.mid2_gumbel_layer": [3, 11],
        "backbone.layer3.1.gumbel_layer": [5],
        "backbone.layer3.1.mid1_gumbel_layer": [6],
        "backbone.layer3.1.mid2_gumbel_layer": [8],
    }
    pruning_config = _structural_pruning_config(mask)

    search_model = instantiate(config.model)
    apply_channel_mask(search_model, mask)
    search_checkpoint = tmp_path / "search.pt"
    _save_checkpoint(search_model, search_checkpoint)
    search_state = {
        key: value.detach().clone() for key, value in search_model.state_dict().items()
    }

    recovery_model = build_structurally_pruned_model_from_config(
        config,
        pruning_config,
    )
    _search_to_structural_initializer(config, search_checkpoint)(recovery_model)

    search_model.eval()
    recovery_model.eval()
    sample = torch.randn(2, 3, 16, 16)
    targets = torch.tensor([0, 1])
    with torch.no_grad():
        search_logits = search_model(sample, targets).logits
        recovery_logits_before = recovery_model(sample, targets).logits
    torch.testing.assert_close(
        recovery_logits_before,
        search_logits,
        rtol=1e-5,
        atol=1e-6,
    )

    # Exercise a real fine-tuning update on the physically pruned model.
    recovered_fc_before = recovery_model.backbone.fc.weight.detach().clone()
    recovery_model.train()
    optimizer = torch.optim.SGD(recovery_model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    recovery_model(sample, targets).loss.backward()
    optimizer.step()
    assert not torch.equal(recovery_model.backbone.fc.weight, recovered_fc_before)

    # Then make every remaining tensor observably different so the exact
    # round-trip below catches any omitted Conv/BN parameter or BN buffer.
    _make_every_recovery_tensor_observable(recovery_model)
    recovery_checkpoint = tmp_path / "recovery.pt"
    _save_checkpoint(recovery_model, recovery_checkpoint)
    recovery_state = {
        key: value.detach().clone()
        for key, value in recovery_model.state_dict().items()
    }

    next_search_model = instantiate(config.model)
    _next_search_initializer(
        config,
        mask,
        search_checkpoint,
        recovery_checkpoint,
    )(next_search_model)
    # run_training() performs this immediately after model_initializer.
    apply_channel_mask(next_search_model, mask)

    # Gate state continues from search; recovery contains no Gumbel gates.
    search_gates = get_gumbel_modules(search_model.backbone)
    next_gates = get_gumbel_modules(next_search_model.backbone)
    assert search_gates.keys() == next_gates.keys()
    for gate_name in search_gates:
        torch.testing.assert_close(
            next_gates[gate_name].logits,
            search_gates[gate_name].logits,
            rtol=0,
            atol=0,
        )

    # A permanently disabled full-width carrier row is not overwritten by
    # the smaller recovery model; the mask keeps it inactive.
    torch.testing.assert_close(
        next_search_model.backbone.layer2[0].conv3.weight[0],
        search_state["backbone.layer2.0.conv3.weight"][0],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        next_search_model.backbone.layer2[0].conv2.weight[3],
        search_state["backbone.layer2.0.conv2.weight"][3],
        rtol=0,
        atol=0,
    )

    # Slice the next carrier back to structural form. Every parameter and BN
    # buffer must be exactly the fine-tuned recovery value, not a fresh init.
    roundtrip_recovery = build_structurally_pruned_model_from_config(
        config,
        pruning_config,
    )
    transfer_gated_weights_to_structural(
        next_search_model,
        roundtrip_recovery,
    )
    roundtrip_state = roundtrip_recovery.state_dict()
    assert recovery_state.keys() == roundtrip_state.keys()
    for key, expected in recovery_state.items():
        torch.testing.assert_close(
            roundtrip_state[key],
            expected,
            rtol=0,
            atol=0,
            msg=lambda message, key=key: f"{key}: {message}",
        )

    next_search_model.eval()
    roundtrip_recovery.eval()
    with torch.no_grad():
        next_logits = next_search_model(sample, targets).logits
        recovery_logits_after = roundtrip_recovery(sample, targets).logits
    torch.testing.assert_close(
        next_logits,
        recovery_logits_after,
        rtol=1e-5,
        atol=1e-6,
    )
