"""Offline regression tests for distributed binding broadcast masks."""
import os
import sys
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causal_maps.delta_binding_slot_broadcast import (  # noqa: E402
    BROADCAST_LAYERS, LAYOUT_OFFSETS, _forward_broadcast, _layout_trials)


class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_mask = None

    def forward(self, hidden, attention_mask=None, **_kwargs):
        self.seen_mask = attention_mask.detach().clone()
        return torch.zeros_like(hidden)


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummyAttention()

    def forward(self, hidden, attention_mask=None, **kwargs):
        return hidden + self.self_attn(hidden, attention_mask=attention_mask, **kwargs)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=2)
        self.model = SimpleNamespace()
        self.model.layers = nn.ModuleList([DummyLayer() for _ in range(27)])
        self.embedding = nn.Embedding(32, 8)
        self.head = nn.Linear(8, 32, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, use_cache=False, logits_to_keep=0):
        del use_cache, logits_to_keep
        hidden = self.embedding(input_ids)
        mask = attention_mask["full_attention"]
        for layer in self.model.layers:
            hidden = layer(hidden, attention_mask=mask)
        return SimpleNamespace(logits=self.head(hidden))


def main():
    rows = _layout_trials(list(range(10)))
    assert len(rows) == 80
    assert {row["offset"] for row in rows} == set(LAYOUT_OFFSETS)
    assert BROADCAST_LAYERS == (20, 21, 22, 23, 24, 25, 26)

    model = DummyModel()
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    mask = torch.ones_like(ids)
    logits = _forward_broadcast(
        model, ids, mask, inject_position=1, direction=torch.ones(8),
        blocked_layers=(20,), key_position=1)
    assert logits.shape == (2, 32)
    minimum = torch.finfo(model.embedding.weight.dtype).min
    blocked = model.model.layers[20].self_attn.seen_mask
    untouched = model.model.layers[21].self_attn.seen_mask
    assert blocked.shape[1] == 2
    assert torch.all(blocked[:, :, 2:, 1] == minimum)
    # Query 0 cannot normally attend to future key 1; query 1 may attend to
    # itself and must remain untouched by the broadcast intervention.
    assert torch.all(blocked[:, :, 0, 1] == minimum)
    assert torch.all(blocked[:, :, 1, 1] == 0)
    assert torch.all(untouched[:, :, 2:, 1] == 0)
    print("binding slot broadcast regression tests passed")


if __name__ == "__main__":
    main()
