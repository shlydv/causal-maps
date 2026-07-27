import torch

from causal_maps.delta_controller_matrix import _fresh_domain_rows
from causal_maps.delta_cross_domain_controller import DOMAIN_SPECS, _domain_rows
from causal_maps.delta_endogenous_controller_factorial import (
    _equalization_patch_pair_axes,
    _fresh_color_rows_v4,
)
from causal_maps.delta_leave_color_out_shared import _fresh_color_rows
from causal_maps.delta_residual_only_confirmation import _fresh_color_rows_v3
from causal_maps.delta_shared_adapter_decomposition import _fresh_color_rows_v2


def _prompt_signatures(rows):
    return {
        (state, row["d1"], row["d2"])
        for row in rows
        for state in (row["source"], row["target"])
    }


def test_v4_color_rows_are_balanced_and_disjoint():
    values = DOMAIN_SPECS["color_state"]["values"]
    prior = (
        _domain_rows(values)
        + _fresh_domain_rows(values)
        + _fresh_color_rows()
        + _fresh_color_rows_v2()
        + _fresh_color_rows_v3()
    )
    rows = _fresh_color_rows_v4()
    assert len(rows) == 30
    assert len({(row["source"], row["target"]) for row in rows}) == 30
    assert not (_prompt_signatures(rows) & _prompt_signatures(prior))
    for field in ("source", "target"):
        counts = {
            value: sum(row[field] == value for row in rows)
            for value in values
        }
        assert set(counts.values()) <= {3, 4}


def test_equalization_removes_sequence_level_axes_and_preserves_midpoint():
    generator = torch.Generator().manual_seed(901)
    p = torch.randn(3, 11, generator=generator)
    r = torch.randn(3, 11, generator=generator)
    r = r - (
        torch.dot(r.flatten(), p.flatten())
        / torch.dot(p.flatten(), p.flatten())
    ) * p
    belief = torch.randn(7, 19, 11, generator=generator)
    search = torch.randn(7, 19, 11, generator=generator)
    positions = [13, 14, 15]

    belief_patches, search_patches, invariants = (
        _equalization_patch_pair_axes(
            [belief], [search], positions, {"P": p, "R": r}))
    patched_belief = belief_patches[0][1].double()
    patched_search = search_patches[0][1].double()
    original_belief = belief[:, positions, :].double()
    original_search = search[:, positions, :].double()

    assert invariants["pass"]
    assert torch.allclose(
        0.5 * (patched_belief + patched_search),
        0.5 * (original_belief + original_search),
        atol=1e-5,
        rtol=0.0,
    )
    remaining = (patched_belief - patched_search).flatten(start_dim=1)
    p_flat = p.double().flatten()
    r_flat = r.double().flatten()
    p_coordinate = (remaining @ p_flat) / p_flat.square().sum()
    r_coordinate = (remaining @ r_flat) / r_flat.square().sum()
    assert float(p_coordinate.abs().max()) < 1e-5
    assert float(r_coordinate.abs().max()) < 1e-5
