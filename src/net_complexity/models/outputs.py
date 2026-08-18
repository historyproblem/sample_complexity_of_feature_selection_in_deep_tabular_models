import torch
from dataclasses import dataclass


@dataclass
class ClassifModelOutput:
    ce_loss: torch.Tensor = None
    regularization_loss: torch.Tensor = None
    reg_loss: torch.Tensor = None
    mean_p_open: torch.Tensor = None
    negative_entropy: torch.Tensor = None
    loss: torch.Tensor = None
    logits: torch.Tensor = None
    mean_activations: list[torch.Tensor] = None
