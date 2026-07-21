from causal_maps.delta_compositional_agent_control import (
    _conflict_rows, _exact_score, _expected, _values)


def test_frozen_rows_are_balanced_conflicts():
    rows = _conflict_rows()
    assert len(rows) == 16
    assert len(rows[::2]) == len(rows[1::2]) == 8
    for family in ("database", "calculator"):
        assert all(_values(row, family)[1] != _values(row, family)[2]
                   for row in rows)


def test_factorial_outputs_separate_phase_and_source():
    for row in _conflict_rows():
        for family in ("database", "calculator"):
            call, internal, external = _values(row, family)
            assert _expected(row, family, "A", "I") == call
            assert _expected(row, family, "A", "E") == call
            assert _expected(row, family, "B", "I") == f"ANSWER {internal}"
            assert _expected(row, family, "B", "E") == f"ANSWER {external}"


def test_exact_score_is_strict_after_special_token_normalization():
    score = _exact_score(
        [" ANSWER 1</s>", "ANSWER 3"], ["ANSWER 1", "ANSWER 2"])
    assert score["accuracy"] == .5
    assert score["rows"] == [True, False]
