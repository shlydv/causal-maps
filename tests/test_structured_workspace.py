from causal_maps.delta_structured_workspace import (
    MARKER, _composition_rows, _counterfactual, _expected, _rows, _user)


def test_single_counterfactual_changes_only_one_bound_relation():
    clean = _rows("Paris", "Rome", "ac", "test")
    natural = _counterfactual(clean, {"ac": "Rome"})
    assert len(clean) == len(natural) == 5
    for left, right in zip(clean, natural):
        changed = {key for key in left if left[key] != right[key]}
        assert changed == {"ac"}
        assert left["as"] == left["bc"] == "Paris"
        assert left["bs"] == "Rome"
        assert right["as"] == right["bc"] == "Paris"
        assert right["bs"] == "Rome"


def test_wrong_address_counterfactual_is_positive_and_specific():
    clean = _rows("Paris", "Rome", "as", "train")
    natural = _counterfactual(clean, {"as": "Rome"})
    for left, right in zip(clean, natural):
        assert left["as"] == "Paris" and right["as"] == "Rome"
        assert all(left[key] == right[key]
                   for key in left if key != "as")


def test_joint_counterfactual_changes_exactly_two_registers():
    clean = _composition_rows("test")
    joint = _counterfactual(clean, {"ac": "Rome", "bs": "Delhi"})
    for left, right in zip(clean, joint):
        changed = {key for key in left if left[key] != right[key]}
        assert changed == {"ac", "bs"}


def test_checkpoint_precedes_question_and_state_is_not_printed():
    row = _rows("Paris", "Rome", "ac", "train")[0]
    text = _user(row, "search_ac", "ledger")
    assert text.index(MARKER) < text.index("Question:")
    assert "belief(Alice" not in text
    assert "SEARCH" in text


def test_expected_outputs_are_semantically_distinct_readouts():
    row = _rows("Paris", "Rome", "ac", "test")[0]
    assert _expected(row, "belief_ac") == "BELIEF Paris"
    assert _expected(row, "search_ac") == "SEARCH Paris"
    assert _expected(row, "tell_ac") == "TELL Paris"
    assert _expected(row, "truth_cube").startswith("TRUTH ")
