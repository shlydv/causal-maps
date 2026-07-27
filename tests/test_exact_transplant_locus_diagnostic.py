import torch

from causal_maps.delta_exact_transplant_locus_diagnostic import (
    _direction_pass,
    _row_transport,
    _unused_rows,
    _verdict,
)
from causal_maps.delta_prospective_causal_sensitivity import (
    _prospective_rows,
)
from causal_maps.delta_heterogeneous_family_screen import VALUES


def test_unused_pairs_are_outside_previous_fifty():
    previous = {
        (row["source"], row["target"])
        for row in _prospective_rows(VALUES)
    }
    rows = _unused_rows()
    pairs = {(row["source"], row["target"]) for row in rows}
    assert len(rows) == 12
    assert len(pairs) == 6
    assert previous.isdisjoint(pairs)
    assert all(sum(
        (row["source"], row["target"]) == pair
        for row in rows) == 2 for pair in pairs)


def test_transport_progress_and_distance_have_direct_meaning():
    origin = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[2.0, 0.0]])
    halfway = torch.tensor([[1.0, 0.0]])
    progress, distance = _row_transport(origin, target, halfway)
    assert progress == [0.5]
    assert distance == [0.5]


def test_direction_gate_requires_broad_progress_and_accuracy():
    passing = {
        "mean_progress": 0.30,
        "median_distance_ratio": 0.80,
        "positive_rows": 20,
        "minimum_answer_accuracy": 1.0,
    }
    assert _direction_pass(passing)
    assert not _direction_pass({**passing, "positive_rows": 17})
    assert not _direction_pass({
        **passing, "minimum_answer_accuracy": 0.79})


def test_verdict_distinguishes_late_and_directional_diagnostics():
    base = {
        "pass_cells": [{
            "layer": 21,
            "checkpoint": 24,
            "position_group": "answer_prefix_3",
            "score": 0.4,
        }],
        "control_pass_cells": [],
        "one_direction_cell_count": 0,
        "best_cell": {
            "layer": 21,
            "position_group": "answer_prefix_3",
        },
    }
    late = {
        **base,
        "pass_cells": [{
            "layer": 23,
            "checkpoint": 27,
            "position_group": "answer_prefix_3",
            "score": 0.3,
        }],
        "best_cell": {
            "layer": 23,
            "position_group": "answer_prefix_3",
        },
    }
    assert _verdict({
        "maximum_score": base,
        "two_hop_pointer": late,
    }) == "L24_ASSAY_INCOMPLETE"

    directional = {
        **base,
        "pass_cells": [],
        "one_direction_cell_count": 3,
    }
    assert _verdict({
        "maximum_score": base,
        "two_hop_pointer": directional,
    }) == "DIRECTION_ASYMMETRIC_CONTROL"
