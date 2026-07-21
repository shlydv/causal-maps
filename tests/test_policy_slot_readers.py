"""Regression test for policy-slot attention-edge hooks."""
import os
import sys
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causal_maps.delta_policy_slot_readers import (
    TEMPLATE_B_POSITION, _forward_reader)  # noqa: E402


class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_mask = None

    def forward(self, hidden_states, attention_mask=None, **_kwargs):
        self.seen_mask = attention_mask.detach().clone()
        return torch.zeros_like(hidden_states), None


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummyAttention()

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        update, _ = self.self_attn(
            hidden_states, attention_mask=attention_mask, **kwargs)
        return hidden_states + update


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace()
        self.model.layers = nn.ModuleList(
            [DummyLayer() for _ in range(28)])
        self.embedding = nn.Embedding(256, 8)
        self.head = nn.Linear(8, 16, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, use_cache=False):
        del use_cache
        hidden = self.embedding(input_ids)
        masks = attention_mask
        for layer in self.model.layers:
            hidden = layer(
                hidden,
                attention_mask=masks["full_attention"])
        return SimpleNamespace(logits=self.head(hidden))


def main():
    model = DummyModel()
    seq_len = TEMPLATE_B_POSITION + 4
    ids = torch.zeros((2, seq_len), dtype=torch.long)
    am = torch.ones_like(ids)
    direction = torch.ones(8)
    logits = _forward_reader(
        model, ids, am, direction, blocked_layers=(3,))
    assert logits.shape == (2, 16)
    minimum = torch.finfo(model.embedding.weight.dtype).min
    blocked = model.model.layers[3].self_attn.seen_mask
    untouched = model.model.layers[4].self_attn.seen_mask
    assert torch.all(
        blocked[..., TEMPLATE_B_POSITION + 1:, TEMPLATE_B_POSITION]
        == minimum)
    assert torch.all(
        untouched[..., TEMPLATE_B_POSITION + 1:, TEMPLATE_B_POSITION]
        == 0)
    assert torch.all(
        blocked[..., TEMPLATE_B_POSITION, TEMPLATE_B_POSITION] == 0)
    print("policy slot reader hook regression test passed")


if __name__ == "__main__":
    main()
