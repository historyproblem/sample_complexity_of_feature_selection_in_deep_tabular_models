import warnings

from net_complexity.data.dataloaders import _resolve_num_workers, _resolve_pin_memory


def test_resolve_num_workers_keeps_requested_value_when_shm_manager_is_available(monkeypatch):
    monkeypatch.setattr(
        "net_complexity.data.dataloaders._torch_shm_manager_available",
        lambda: True,
    )

    assert _resolve_num_workers(2) == 2


def test_resolve_num_workers_falls_back_to_zero_when_shm_manager_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "net_complexity.data.dataloaders._torch_shm_manager_available",
        lambda: False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = _resolve_num_workers(2)

    assert resolved == 0
    assert len(caught) == 1
    assert "falling back to num_workers=0" in str(caught[0].message)


def test_resolve_pin_memory_defaults_to_false_without_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    assert _resolve_pin_memory(None) is False


def test_resolve_pin_memory_respects_explicit_override():
    assert _resolve_pin_memory(True) is True
    assert _resolve_pin_memory(False) is False
