"""Offline regression tests for binding slot-to-readout attention masks."""
import os
import sys
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causal_maps.delta_binding_slot_transport import (  # noqa: E402
    FRESH_OFFSETS, _forward_edge, _fresh_trials)


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
    rows = _fresh_trials(list(range(10)))
    assert len(rows) == 40
    assert {row["offset"] for row in rows} == set(FRESH_OFFSETS)
    for source in range(10):
        offsets = [
            (row["target"] - row["source"]) % 10
            for row in rows if row["source"] == source]
        assert offsets == list(FRESH_OFFSETS)

    model = DummyModel()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    mask = torch.ones_like(ids)
    logits = _forward_edge(
        model, ids, mask, inject_position=1, direction=torch.ones(8),
        blocked_layers=(20,), key_position=1)
    assert logits.shape == (2, 32)
    minimum = torch.finfo(model.embedding.weight.dtype).min
    blocked = model.model.layers[20].self_attn.seen_mask
    untouched = model.model.layers[21].self_attn.seen_mask
    assert blocked.shape[1] == 2
    assert torch.all(blocked[:, :, 3, 1] == minimum)
    assert torch.all(untouched[:, :, 3, 1] == 0)
    assert torch.all(blocked[:, :, 1, 1] == 0)
    print("binding slot transport regression tests passed")


if __name__ == "__main__":
    main()
