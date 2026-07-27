import math

import torch

from causal_maps.delta_prospective_causal_sensitivity import (
    _axis_prediction,
    _prospective_rows,
    _rankdata,
    _scale_per_position,
    _spearman,
)


def test_prospective_rows_have_disjoint_split_histories():
    values = tuple("abcdefgh")
    rows = _prospective_rows(values)
    assert len(rows) == 50
    assert len({
        (row["source"], row["target"]) for row in rows
    }) == 50
    assert len({
        (state, row["d1"], row["d2"])
        for row in rows
        for state in (row["source"], row["target"])
    }) == 100


def test_axis_prediction_uses_bidirectional_minimum():
    components = {
        1: torch.tensor([[1.0, 0.0]]),
        2: torch.tensor([[0.0, 1.0]]),
        3: torch.tensor([[1.0, 1.0]]),
    }
    gradients = {
        "belief": torch.tensor([[[2.0, 0.5]]]),
        "search": torch.tensor([[[0.1, 1.5]]]),
    }
    rows = _axis_prediction(components, gradients)
    assert [row["axis"] for row in rows] == [3, 2, 1]
    assert math.isclose(
        rows[0]["bidirectional_sensitivity"], 1.6)
    assert all(row["coherent_positive"] for row in rows)


def test_rank_helpers_and_position_norm_matching():
    assert _rankdata([10.0, 20.0, 20.0, 40.0]) == [
        1.0, 2.5, 2.5, 4.0]
    assert math.isclose(
        _spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]),
        1.0)

    source = torch.tensor([
        [3.0, 4.0],
        [1.0, 0.0],
        [0.0, 2.0],
    ])
    reference = torch.tensor([
        [6.0, 8.0],
        [0.0, 7.0],
        [5.0, 12.0],
    ])
    scaled = _scale_per_position(source, reference)
    assert torch.allclose(
        scaled.norm(dim=-1), reference.norm(dim=-1))
