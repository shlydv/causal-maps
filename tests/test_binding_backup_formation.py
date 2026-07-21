"""Offline regression tests for multi-window binding edge specifications."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causal_maps.delta_binding_backup_formation import (  # noqa: E402
    FORMATION_WINDOWS, LATE_READ_LAYERS, _window_metrics)
from causal_maps.delta_binding_slot_broadcast import _forward_broadcast  # noqa: E402
from test_binding_slot_broadcast import DummyModel  # noqa: E402


def main():
    assert FORMATION_WINDOWS["early"] == (3, 4, 5, 6, 7, 8)
    assert FORMATION_WINDOWS["middle"] == (9, 10, 11, 12, 13, 14)
    assert FORMATION_WINDOWS["late"] == (15, 16, 17, 18, 19, 20)
    assert LATE_READ_LAYERS == (21, 22, 23, 24, 25, 26)

    model = DummyModel()
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    mask = torch.ones_like(ids)
    _forward_broadcast(
        model, ids, mask, inject_position=1,
        blocked_layer_keys=((3, 1), (21, 2)))
    minimum = torch.finfo(model.embedding.weight.dtype).min
    assert torch.all(model.model.layers[3].self_attn.seen_mask[:, :, 2:, 1] == minimum)
    assert torch.all(model.model.layers[21].self_attn.seen_mask[:, :, 3:, 2] == minimum)
    assert torch.all(model.model.layers[3].self_attn.seen_mask[:, :, 3:, 2] == 0)

    patched = {"natural_effect": 5.0, "add_effect": 5.0}
    own_without = {"natural_effect": 4.0, "add_effect": 4.0}
    own_with = {"natural_effect": 8.0, "add_effect": 8.0}
    other_without = {"natural_effect": 4.0, "add_effect": 4.0}
    other_with = {"natural_effect": 14.0, "add_effect": 14.0}
    metrics = _window_metrics(
        own_without, own_with, other_without, other_with)
    assert metrics["natural"]["shared_formation"]
    assert metrics["add"]["fraction_of_controlled_recovery_prevented"] == 0.6
    print("binding backup formation regression tests passed")


if __name__ == "__main__":
    main()
