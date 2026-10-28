from __future__ import annotations

import pytest
from torch import nn

from net_complexity.models.autopruner import load_autopruner_pretrained_backbone


def test_missing_pretrained_checkpoint_reports_path_in_exception_repr(tmp_path):
    checkpoint_path = tmp_path / "missing baseline" / "best.pt"

    with pytest.raises(FileNotFoundError) as error:
        load_autopruner_pretrained_backbone(nn.Linear(2, 2), checkpoint_path)

    # Hydra renders the exception repr, which omits OSError's filename field.
    assert str(checkpoint_path) in repr(error.value)
    assert "model.pretrained_checkpoint" in str(error.value)
