import pytest
import torch

from causal_maps.delta_heldout_inverse_control import (
    DIRECTIONS,
    HELDOUT_FAMILIES,
    _PredictionBoundary,
    _coefficients_to_delta,
    _overall_adjudication,
    _pilot_rows,
    _ridge_coefficients,
)


def test_pilot_rows_are_pair_disjoint():
    rows = _pilot_rows()
    pairs = {
        name: {
            (row["source"], row["target"])
            for row in split
        }
        for name, split in rows.items()
    }
    names = list(pairs)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            assert not pairs[left] & pairs[right]


def test_prediction_boundary_prevents_early_target_access():
    boundary = _PredictionBoundary()
    with pytest.raises(RuntimeError):
        boundary.require_target_access()
    boundary.freeze("json-hash", "npz-hash")
    boundary.require_target_access()


def test_ridge_inverse_recovers_reachable_target_and_caps_norm():
    response = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.2]],
        [[0.9, 0.1], [0.1, 1.1], [0.4, 0.3]],
    ])
    target = torch.tensor([1.0, 0.5, 0.6])
    coefficients, metadata = _ridge_coefficients(
        response, target, norm_cap=10.0)
    predicted = torch.einsum(
        "bhr,br->bh", response, coefficients)
    progress = (
        predicted * target
    ).sum(dim=-1) / target.square().sum()
    assert min(progress.tolist()) >= 0.95
    assert min(metadata["predicted_target_cosine"]) >= 0.95

    capped, cap_metadata = _ridge_coefficients(
        response, 100.0 * target, norm_cap=0.25)
    assert bool((capped.norm(dim=-1) <= 0.25 + 1e-6).all())
    assert all(cap_metadata["cap_applied"])


def test_coefficient_sign_is_preserved_in_patch_conversion():
    basis = torch.eye(6)[:2]
    delta = _coefficients_to_delta(
        torch.tensor([[2.0, -1.0]]), basis, hidden_size=2)
    assert torch.allclose(
        delta.flatten(),
        torch.tensor([2.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
    )


def test_overall_gate_requires_family_and_direction_breadth():
    cells = {}
    for family in HELDOUT_FAMILIES:
        for direction in DIRECTIONS:
            cells[f"{family}/{direction}"] = {
                "exact_oracle_pass": True,
                "local_map_capacity_pass": True,
                "local_level_pass": True,
                "control_pass": True,
                "pass": True,
                "shared_mean_progress": 0.0,
            }
    result = _overall_adjudication(cells)
    assert result["verdict"] == (
        "PROSPECTIVE_HELDOUT_INVERSE_STATE_CONTROL"
    )

    cells[f"{HELDOUT_FAMILIES[1]}/{DIRECTIONS[1]}"]["pass"] = False
    cells[f"{HELDOUT_FAMILIES[1]}/{DIRECTIONS[0]}"]["pass"] = False
    result = _overall_adjudication(cells)
    assert result["verdict"] == "SPECIFICITY_OR_BREADTH_FAILED"
