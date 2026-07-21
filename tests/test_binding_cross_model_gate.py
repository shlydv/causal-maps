from causal_maps.delta_binding_cross_model_gate import (
    _candidate_layers, _discovery_viable, _heldout_confirmed, _split_rows)
from causal_maps.delta_operator import _trials


def test_normalized_early_candidates_are_distinct_and_early():
    assert _candidate_layers(32) == (2, 4, 6)
    assert _candidate_layers(26) == (2, 3, 5)


def test_split_rows_is_balanced_for_any_valid_value_count():
    rows = _trials([str(i) for i in range(8)])
    discovery, heldout = _split_rows(rows)
    assert len(discovery) == len(heldout) == 16
    assert {row["query"] for row in discovery} == {"X", "Y"}


def test_selection_and_confirmation_are_distinct_gates():
    metrics = {"clean_acc": .8, "natural_acc": .8, "natural_effect": 1.,
               "add_effect": .6, "wrong_direction_effect": .1,
               "other_slot_effect": .55, "positive_add_fraction": .8,
               "effect_ratio": .6}
    assert _discovery_viable(metrics)
    assert not _heldout_confirmed(metrics)  # ratio and slot control fail.
