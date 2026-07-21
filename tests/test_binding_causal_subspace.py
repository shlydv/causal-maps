"""Offline regression tests for distributed binding-subspace interventions."""
import os
import sys
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causal_maps.delta_binding_causal_subspace import (  # noqa: E402
    _basis_energy, _fit_basis, _forward_remove_subspace, _random_basis)
from causal_maps.delta_trajectory import _forward  # noqa: E402


class DummyLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, hidden):
        return hidden + self.proj(hidden)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
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
    generator = torch.Generator().manual_seed(7)
    natural = torch.randn((6, 8), generator=generator)
    add = natural + .01 * torch.randn((6, 8), generator=generator)
    basis, singular = _fit_basis([
        {"natural_disp": natural[:3], "add_disp": add[:3]},
        {"natural_disp": natural[3:], "add_disp": add[3:]},
    ], max_rank=4)
    assert basis.shape == (8, 4)
    assert singular.shape == (8,)
    assert torch.allclose(basis.T @ basis, torch.eye(4), atol=1e-5)
    energy = _basis_energy([
        {"natural_disp": natural, "add_disp": add}], basis[:, :2])
    assert 0 <= energy["natural_energy"] <= 1
    assert 0 <= energy["add_energy"] <= 1
    random = _random_basis(8, 3, __import__("numpy").random.default_rng(3))
    assert random.shape == (8, 3)
    assert torch.allclose(random.T @ random, torch.eye(3), atol=1e-5)

    model = DummyModel()
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.ones_like(ids)
    ordinary, cache = _forward(model, ids, mask, (1, 2), (8,))
    clean_last = cache[8][:, 1]
    removed = _forward_remove_subspace(
        model, ids, mask, inject_position=1, clean_state=clean_last,
        basis=basis[:, :2], position=2)
    assert ordinary.shape == removed.shape == (2, 32)
    # The matched CLEAN state has zero displacement and is therefore unchanged.
    assert torch.allclose(ordinary, removed, atol=1e-5)
    changed = _forward_remove_subspace(
        model, ids, mask, inject_position=1, clean_state=torch.zeros_like(clean_last),
        basis=basis[:, :2], position=2)
    assert not torch.equal(ordinary, changed)
    print("binding causal subspace regression tests passed")


if __name__ == "__main__":
    main()
