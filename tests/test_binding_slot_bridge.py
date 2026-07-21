"""Offline regression tests for the bridge audit's patch-plus-mask path."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causal_maps.delta_binding_slot_broadcast import _forward_broadcast  # noqa: E402
from test_binding_slot_broadcast import DummyModel  # noqa: E402


def main():
    model = DummyModel()
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    mask = torch.ones_like(ids)
    logits = _forward_broadcast(
        model, ids, mask, inject_position=1, direction=torch.ones(8),
        patch=(20, 1, torch.zeros((2, 8))), blocked_layers=(21,),
        key_position=1)
    assert logits.shape == (2, 32)
    minimum = torch.finfo(model.embedding.weight.dtype).min
    assert torch.all(model.model.layers[21].self_attn.seen_mask[:, :, 2:, 1] == minimum)
    print("binding slot bridge patch-plus-mask regression tests passed")


if __name__ == "__main__":
    main()
