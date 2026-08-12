from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


ENTROPY_REGULARIZATION_MODES = {
    "disabled": 0.0,
    "plus_negative_entropy": 1.0,
    "minus_negative_entropy": -1.0,
    "bernoulli_kl": 1.0,
}

POSTERIOR_KL_REDUCTIONS = {"mean", "sum"}


def entropy_regularization_sign(mode: str) -> float:
    normalized = str(mode).strip().lower()
    if normalized not in ENTROPY_REGULARIZATION_MODES:
        allowed = ", ".join(sorted(ENTROPY_REGULARIZATION_MODES))
        raise ValueError(
            f"entropy_regularization must be one of: {allowed}. Got: {mode!r}"
        )
    return ENTROPY_REGULARIZATION_MODES[normalized]


def normalize_posterior_kl_reduction(reduction: str) -> str:
    normalized = str(reduction).strip().lower()
    if normalized not in POSTERIOR_KL_REDUCTIONS:
        allowed = ", ".join(sorted(POSTERIOR_KL_REDUCTIONS))
        raise ValueError(
            f"posterior_kl_reduction must be one of: {allowed}. Got: {reduction!r}"
        )
    return normalized


def bernoulli_kl_from_closed_open_log_odds(
    mean_p_open: torch.Tensor,
    negative_entropy: torch.Tensor,
    closed_open_log_odds: float,
) -> torch.Tensor:
    """Return mean Bernoulli KL for a shared prior over AIG gates.

    ``closed_open_log_odds`` is the coefficient from Eq. (4) of the
    variational reduction objective::

        alpha = log((1 - pi) / pi)

    where ``pi`` is the prior probability that a gate is open.  Expanding
    ``KL(Bernoulli(p) || Bernoulli(pi))`` gives

        negative_entropy + alpha * p + softplus(-alpha).

    The softplus term is constant with respect to the model parameters, but
    retaining it makes the reported regularization loss the actual KL value.
    """
    alpha = mean_p_open.new_tensor(float(closed_open_log_odds))
    return negative_entropy + alpha * mean_p_open + F.softplus(-alpha)


class AIGBlockGate(nn.Module):
    """
    Per-block AIG gate with explicit off/on logits.

    The forward pass returns a hard execution gate, while gradients flow through
    the two-state Gumbel-Softmax relaxation during training.
    """

    REGULARIZATION_ALIASES = {
        "l1": "l1_activation",
        "l1_activation": "l1_activation",
        "l1_activations": "l1_activation",
        "l1_gate": "l1_activation",
        "l1_probability": "l1_probability",
        "l1_prob": "l1_probability",
        "l1_probs": "l1_probability",
        "l2": "l2_activation",
        "l2_activation": "l2_activation",
        "l2_activations": "l2_activation",
        "l2_gate": "l2_activation",
        "l2_probability": "l2_probability",
        "l2_prob": "l2_probability",
        "l2_probs": "l2_probability",
    }

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 16,
        keep_prob_init: float | None = None,
        threshold: float = 0.5,
        temperature: float = 1.0,
        regularization: str = "l2_gate",
        init_logits: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        if keep_prob_init is not None and not 0.0 < keep_prob_init < 1.0:
            raise ValueError("keep_prob_init must be in (0, 1).")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1].")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")

        hidden_channels = max(1, int(hidden_channels))
        self.router = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )
        self.keep_prob_init = keep_prob_init
        self.init_logits = init_logits
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.regularization = self._normalize_regularization(regularization)
        self.bypass = False

        self.logits: torch.Tensor | None = None
        self.probabilities: torch.Tensor | None = None
        self.keep_probabilities: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self.reset_parameters()

    @classmethod
    def _normalize_regularization(cls, regularization: str) -> str:
        key = str(regularization).strip().lower()
        if key not in cls.REGULARIZATION_ALIASES:
            allowed = ", ".join(sorted(cls.REGULARIZATION_ALIASES))
            raise ValueError(
                f"AIG regularization must be one of: {allowed}. Got: {regularization!r}"
            )
        return cls.REGULARIZATION_ALIASES[key]

    def reset_parameters(self) -> None:
        final_conv = self.router[-1]
        assert isinstance(final_conv, nn.Conv2d)
        nn.init.zeros_(final_conv.weight)
        if self.init_logits is not None:
            off_logit, on_logit = self.init_logits
        else:
            keep_prob = 0.85 if self.keep_prob_init is None else float(self.keep_prob_init)
            off_logit = 0.0
            on_logit = math.log(keep_prob / (1.0 - keep_prob))
        final_conv.bias.data[0] = float(off_logit)
        final_conv.bias.data[1] = float(on_logit)

    def set_bypass(self, enabled: bool) -> None:
        self.bypass = bool(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(x, 1)
        logits = self.router(pooled)
        probabilities = F.softmax(logits, dim=1)
        keep_probabilities = probabilities[:, 1:2]

        if self.bypass:
            gate = torch.ones_like(keep_probabilities)
            keep_probabilities = torch.ones_like(keep_probabilities)
            probabilities = torch.cat(
                [torch.zeros_like(keep_probabilities), keep_probabilities],
                dim=1,
            )
        elif self.training:
            hard_sample = F.gumbel_softmax(
                logits.flatten(2).transpose(1, 2),
                tau=self.temperature,
                hard=True,
                dim=-1,
            ).transpose(1, 2).view_as(logits)
            gate = hard_sample[:, 1:2]
        else:
            gate = (keep_probabilities >= self.threshold).to(dtype=keep_probabilities.dtype)

        self.logits = logits
        self.probabilities = probabilities
        self.keep_probabilities = keep_probabilities
        self.activations = gate
        return gate

    def regularization_loss(self) -> torch.Tensor:
        if self.keep_probabilities is None:
            return next(self.parameters()).new_zeros(())
        if self.bypass:
            return self.keep_probabilities.new_zeros(())

        if self.regularization == "l1_probability":
            return self.keep_probabilities.mean()
        if self.regularization == "l2_probability":
            return self.keep_probabilities.mean() ** 2
        if self.activations is None:
            return next(self.parameters()).new_zeros(())
        if self.regularization == "l1_activation":
            return self.activations.mean()
        if self.regularization == "l2_activation":
            return self.activations.mean() ** 2
        raise AssertionError(f"Unhandled AIG regularization mode: {self.regularization}")

    def posterior_regularization_terms(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean p(open) and mean negative entropy of the soft posterior."""
        if self.logits is None or self.bypass:
            zero = next(self.parameters()).new_zeros(())
            return zero, zero

        # Move the two-state dimension last to match a Bernoulli posterior [..., 2].
        gate_logits = self.logits.movedim(1, -1)
        log_probs = F.log_softmax(gate_logits, dim=-1)
        probs = log_probs.exp()
        mean_p_open = probs[..., 1].mean()
        negative_entropy = (probs * log_probs).sum(dim=-1).mean()
        return mean_p_open, negative_entropy
