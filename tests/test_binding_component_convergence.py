"""Offline regression tests for binding component-convergence plumbing."""
import os
import sys
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causal_maps.delta_binding_component_convergence import (  # noqa: E402
    HEAD_KIND, MLP_KIND, _candidate_components, _forward_components,
    _jaccard, _matched_random_sets, _split_trials, _top_components)
from causal_maps.delta_operator import INJECT_LAYER, _trials  # noqa: E402


class DummyAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.o_proj = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.o_proj.weight)

    def forward(self, hidden):
        return self.o_proj(hidden)


class DummyLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self_attn = DummyAttention(width)
        self.mlp = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.mlp.weight)

    def forward(self, hidden):
        return hidden + self.self_attn(hidden) + self.mlp(hidden)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=2, hidden_size=8)
        self.model = SimpleNamespace()
        self.model.layers = nn.ModuleList([DummyLayer(8) for _ in range(9)])
        self.embedding = nn.Embedding(32, 8)
        self.head = nn.Linear(8, 32, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.head(hidden))


def main():
    rows = _trials(list(range(10)))
    discovery, heldout = _split_trials(rows)
    assert len(discovery) == len(heldout) == 20
    for source in range(10):
        discovery_offsets = [
            (row["target"] - row["source"]) % 10
            for row in discovery if row["source"] == source]
        heldout_offsets = [
            (row["target"] - row["source"]) % 10
            for row in heldout if row["source"] == source]
        assert discovery_offsets == [1, 3]
        assert heldout_offsets == [5, 7]

    candidates = _candidate_components(3)
    assert candidates == [
        (HEAD_KIND, 0), (HEAD_KIND, 1), (HEAD_KIND, 2), (MLP_KIND, None)]
    assert _top_components([0.1, 0.7, 0.4, 0.6], candidates, 2) == (
        (HEAD_KIND, 1), (MLP_KIND, None))
    assert _jaccard(((HEAD_KIND, 1),), ((HEAD_KIND, 1), (MLP_KIND, None))) == .5

    reference = ((HEAD_KIND, 1), (HEAD_KIND, 3), (MLP_KIND, None))
    random_sets = _matched_random_sets(reference, 30, seed=2, n_null=100)
    assert len(random_sets) == 100
    assert all(
        item != tuple(sorted(reference))
        and sum(kind == HEAD_KIND for kind, _ in item) == 2
        and (MLP_KIND, None) in item
        for item in random_sets)

    model = DummyModel()
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.ones_like(ids)
    direction = torch.ones(8)
    baseline, cache = _forward_components(
        model, ids, mask, inject_position=1, direction=direction,
        capture_positions=(1, 2))
    head_ablated, _ = _forward_components(
        model, ids, mask, inject_position=1, direction=direction,
        components=((HEAD_KIND, 0),), component_position=2)
    mlp_ablated, _ = _forward_components(
        model, ids, mask, inject_position=1, direction=direction,
        components=((MLP_KIND, None),), component_position=2)
    assert baseline.shape == head_ablated.shape == mlp_ablated.shape == (2, 32)
    assert cache[8].shape == (2, 2, 8)
    assert not torch.equal(baseline, head_ablated)
    assert not torch.equal(baseline, mlp_ablated)
    print("binding component convergence regression tests passed")


if __name__ == "__main__":
    main()
