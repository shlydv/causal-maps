from causal_maps.delta_cross_domain_controller import _domain_rows
from causal_maps.delta_cross_family_causal_subspace import (
    FAMILY_ORDER,
    VALUES,
    _geometry_adjudication,
    _orientation_score,
    _split_rows,
)


def _task():
    return {
        "eligible": True,
        "source_intervention": {
            "sufficient": True,
            "forward_effect_rows": [1.0, 1.0],
            "reverse_effect_rows": [1.0, 1.0],
        },
    }


def _cell(mediation):
    blocked = 1.0 - mediation
    return {
        "blocked_intervention": {
            "forward_effect_rows": [blocked, blocked],
            "reverse_effect_rows": [blocked, blocked],
        },
    }


def _arm(belief, search):
    return {
        "values": {"belief": belief, "search": search},
        "tasks": {"belief": _task(), "search": _task()},
        "cells": {
            "belief": _cell(belief),
            "search": _cell(search),
        },
    }


def test_orientation_score_accepts_either_frozen_gap_sign():
    negative = _orientation_score(
        _arm(0.50, 0.50), _arm(0.60, 0.40), -1)
    positive = _orientation_score(
        _arm(0.50, 0.50), _arm(0.40, 0.60), +1)
    assert negative["bidirectional_score"] > 0.09
    assert positive["bidirectional_score"] > 0.09
    assert negative["positive_worlds"] == 2
    assert positive["positive_worlds"] == 2


def test_geometry_gate_uses_heldout_projection_energy():
    folds = {
        family: {
            "rank_metadata": {
                "3": {"fraction_of_controller_energy": 0.30},
                "7": {"fraction_of_controller_energy": 0.60},
            },
        }
        for family in FAMILY_ORDER
    }
    result = _geometry_adjudication(folds)
    assert result["pass"]
    folds[FAMILY_ORDER[0]]["rank_metadata"]["3"][
        "fraction_of_controller_energy"] = 0.0
    folds[FAMILY_ORDER[1]]["rank_metadata"]["3"][
        "fraction_of_controller_energy"] = 0.0
    folds[FAMILY_ORDER[2]]["rank_metadata"]["3"][
        "fraction_of_controller_energy"] = 0.0
    folds[FAMILY_ORDER[3]]["rank_metadata"]["3"][
        "fraction_of_controller_energy"] = 0.0
    folds[FAMILY_ORDER[4]]["rank_metadata"]["3"][
        "fraction_of_controller_energy"] = 0.0
    assert not _geometry_adjudication(folds)["pass"]


def test_tested_pairs_are_disjoint_from_screen_pairs():
    screen_pairs = {
        (row["source"], row["target"])
        for row in _domain_rows(VALUES)
    }
    splits = _split_rows()
    unseen_pairs = {
        (row["source"], row["target"])
        for rows in splits.values()
        for row in rows
    }
    assert len(unseen_pairs) == 35
    assert screen_pairs.isdisjoint(unseen_pairs)
