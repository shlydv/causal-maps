"""Offline regression tests for head-specific policy-edge masks."""
import hashlib
import os
import sys
from collections import Counter
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causal_maps.delta_policy_head_routing import (  # noqa: E402
    NUM_HEADS, PROTOCOL_DOCUMENT_SHA256, TEMPLATE_B_POSITION,
    _forward_heads, _fresh_rows, _matched_random_sets, _paired_heads)


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

    def forward(
            self, input_ids, attention_mask, use_cache=False,
            logits_to_keep=0):
        del use_cache, logits_to_keep
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(
                hidden,
                attention_mask=attention_mask["full_attention"])
        return SimpleNamespace(logits=self.head(hidden))


def main():
    protocol_path = os.path.join(
        os.path.dirname(__file__), "..", "POLICY_HEAD_ROUTING_PROTOCOL.md")
    with open(protocol_path, "rb") as handle:
        assert hashlib.sha256(
            handle.read()).hexdigest() == PROTOCOL_DOCUMENT_SHA256

    model = DummyModel()
    seq_len = TEMPLATE_B_POSITION + 4
    ids = torch.zeros((2, seq_len), dtype=torch.long)
    am = torch.ones_like(ids)
    direction = torch.ones(8)
    blocked_heads = ((4, 3), (4, 7), (10, 2))
    logits = _forward_heads(
        model, ids, am, direction, blocked_heads)
    assert logits.shape == (2, 16)
    minimum = torch.finfo(model.embedding.weight.dtype).min
    layer4 = model.model.layers[4].self_attn.seen_mask
    layer5 = model.model.layers[5].self_attn.seen_mask
    layer10 = model.model.layers[10].self_attn.seen_mask
    assert layer4.shape[1] == NUM_HEADS
    assert layer5.shape[1] == 1
    assert layer10.shape[1] == NUM_HEADS
    assert torch.all(
        layer4[:, (3, 7), TEMPLATE_B_POSITION + 1:,
               TEMPLATE_B_POSITION] == minimum)
    assert torch.all(
        layer4[:, 0, TEMPLATE_B_POSITION + 1:,
               TEMPLATE_B_POSITION] == 0)
    assert torch.all(
        layer10[:, 2, TEMPLATE_B_POSITION + 1:,
                TEMPLATE_B_POSITION] == minimum)
    assert torch.all(
        layer4[:, (3, 7), TEMPLATE_B_POSITION,
               TEMPLATE_B_POSITION] == 0)

    generator = torch.Generator().manual_seed(3)
    head_dim = 4
    query = torch.randn(
        (1, NUM_HEADS, seq_len, head_dim), generator=generator)
    key_grouped = torch.randn(
        (1, 4, seq_len, head_dim), generator=generator)
    value_grouped = torch.randn(
        (1, 4, seq_len, head_dim), generator=generator)
    key = key_grouped.repeat_interleave(NUM_HEADS // 4, dim=1)
    value = value_grouped.repeat_interleave(NUM_HEADS // 4, dim=1)
    causal = torch.zeros((1, NUM_HEADS, seq_len, seq_len))
    future = torch.triu(torch.ones(
        (seq_len, seq_len), dtype=torch.bool), diagonal=1)
    causal.masked_fill_(
        future.unsqueeze(0).unsqueeze(0), torch.finfo(causal.dtype).min)
    head_mask = causal.clone()
    head_mask[
        :, 3, TEMPLATE_B_POSITION + 1:, TEMPLATE_B_POSITION
    ] = torch.finfo(head_mask.dtype).min
    baseline_output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=causal)
    blocked_output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=head_mask)
    assert torch.allclose(
        baseline_output[:, 0], blocked_output[:, 0])
    assert not torch.allclose(
        baseline_output[:, 3, -1], blocked_output[:, 3, -1])

    clean, steered = _paired_heads(
        model, ids, am, direction, blocked_heads)
    assert clean.shape == steered.shape == (2, 16)

    reference = (
        (4, 1), (4, 2), (10, 3), (15, 4),
        (15, 5), (15, 6), (18, 7), (18, 8))
    random_sets = _matched_random_sets(reference, 17)
    expected_counts = Counter(layer for layer, _ in reference)
    assert len(random_sets) == 100
    assert all(
        Counter(layer for layer, _ in item) == expected_counts
        and len(item) == len(set(item)) == 8
        and item != reference
        for item in random_sets)

    rows = _fresh_rows()
    assert len(rows) == 10
    assert all(
        row["a"] < row["b"]
        and row["a"] + row["b"] != row["database_value"]
        for row in rows)
    print("policy head routing regression tests passed")


if __name__ == "__main__":
    main()
