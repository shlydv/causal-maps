from causal_maps.delta_binding_cross_surface_transfer import (
    _literal_transfer_confirmed, _mapping_baseline_confirmed)


def _metrics():
    return {
        "clean_acc": 1., "natural_acc": 1., "natural_effect": 10.,
        "mapping_native_add_effect": 9.8,
        "literal_transfer_add_effect": 9.5,
        "literal_wrong_direction_effect": 4.,
        "literal_other_slot_effect": .3,
        "positive_literal_transfer_fraction": 1.,
        "literal_to_natural_ratio": .95,
        "literal_to_native_ratio": .97,
    }


def test_literal_transfer_requires_native_baseline_and_both_controls():
    metrics = _metrics()
    assert _mapping_baseline_confirmed(metrics)
    assert _literal_transfer_confirmed(metrics)
    metrics["literal_other_slot_effect"] = 9.45
    assert not _literal_transfer_confirmed(metrics)
    metrics = _metrics()
    metrics["literal_to_native_ratio"] = .6
    assert not _literal_transfer_confirmed(metrics)
