"""Offline regression tests for matched full-state binding timeline hooks."""
import os
import sys
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causal_maps.delta_binding_causal_state_timeline import (  # noqa: E402
    _capture_conditions, _safe_ratio, _site_metrics)


class DummyLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, hidden):
        return self.proj(hidden)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace()
        self.model.layers = nn.ModuleList([DummyLayer(8) for _ in range(27)])
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
    assert _safe_ratio(3, 2) == 1.5
    assert _safe_ratio(1, 0) is None
    model = DummyModel()
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    natural_ids = torch.tensor([[1, 7, 3], [4, 8, 6]])
    group = {
        "clean_ids": ids,
        "clean_am": torch.ones_like(ids),
        "natural_ids": natural_ids,
        "natural_am": torch.ones_like(natural_ids),
        "slot": 1,
        "last": 2,
        "direction": torch.ones((2, 8)),
        "pos_ids": torch.tensor([1, 1]),
        "neg_ids": torch.tensor([2, 2]),
    }
    captured = _capture_conditions(model, [group])
    assert len(captured["observations"]) == 1
    assert captured["observations"][0]["clean_cache"][26].shape == (2, 2, 8)
    base = {key: captured[key] for key in ("natural_effect", "add_effect")}
    metrics = _site_metrics(model, captured["observations"], 2, 0, base)
    assert set(metrics) == {
        "geometry", "sufficiency", "swaps", "necessity",
        "causally_interchangeable"}
    assert "natural_state_into_add_effect" in metrics["swaps"]
    assert "clean_state_into_natural_effect" in metrics["necessity"]
    metrics_last = _site_metrics(model, captured["observations"], 26, 1, base)
    assert isinstance(metrics_last["causally_interchangeable"], bool)
    print("binding causal state timeline regression tests passed")


if __name__ == "__main__":
    main()
