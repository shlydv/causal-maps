from causal_maps.delta_mistral_binding_backup_port import (
    EARLY_FORMATION_LAYERS, LATE_READ_LAYERS, PATCH_LAYER,
    _formation_metrics, _mistral_port_layers)


def test_frozen_mistral_normalized_windows():
    config = _mistral_port_layers(32)
    assert config["patch_layer"] == PATCH_LAYER == 23
    assert config["late_read_layers"] == LATE_READ_LAYERS == (24, 25, 26, 27, 28, 29, 30)
    assert config["early_formation_layers"] == EARLY_FORMATION_LAYERS == (3, 4, 5, 6, 7, 8, 9)


def test_formation_difference_in_differences_contract():
    own_without = {"natural_effect": 4.0, "add_effect": 4.0}
    own_with = {"natural_effect": 8.0, "add_effect": 8.0}
    other_without = {"natural_effect": 4.0, "add_effect": 4.0}
    other_with = {"natural_effect": 14.0, "add_effect": 14.0}
    metrics = _formation_metrics(own_without, own_with, other_without, other_with)
    assert metrics["natural"]["shared_early_formation"]
    assert metrics["add"]["fraction_of_controlled_recovery_prevented"] == .6
