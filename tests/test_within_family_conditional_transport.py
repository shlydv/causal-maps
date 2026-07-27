from causal_maps.delta_within_family_conditional_transport import (
    PROTOCOL,
    VERDICT_MAP,
)


def test_protocol_is_branch_closing_and_functionally_gated():
    assert PROTOCOL["training_scope"] == "within_family_only"
    assert PROTOCOL["per_family_gate"][
        "minimum_value_accuracy_each_direction"] == 0.80
    assert "nonlinear-model" in PROTOCOL["stopping_rule"]


def test_positive_and_negative_verdicts_are_explicit():
    assert VERDICT_MAP[
        "PREDICTABLE_STATE_CONDITIONED_TRANSPORT"
    ] == "FAMILY_SPECIFIC_CONTROL_LAWS"
    assert VERDICT_MAP[
        "ORACLE_ONLY_STATE_TRANSPORT"
    ] == "NO_LEARNABLE_WITHIN_FAMILY_CONTROL"
