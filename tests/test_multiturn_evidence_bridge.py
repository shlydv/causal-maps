from causal_maps.delta_multiturn_evidence_bridge import (
    _score_conflicts, _zero_rows)


def test_zero_rows_are_all_diagnostic_conflicts():
    rows = _zero_rows()
    assert len(rows) == 20
    assert all(row["key"] == "D" and row["database_value"] == 0
               for row in rows)
    assert all(row["a"] + row["b"] != 0 for row in rows)
    assert len(rows[::2]) == len(rows[1::2]) == 10


def test_all_conflict_scorer_does_not_require_collision_rows():
    score = _score_conflicts(["0", "7"], ["0", "0"], ["5", "7"])
    assert score["diagnostic_target_acc"] == .5
    assert score["diagnostic_source_acc"] == .5
    assert score["collision_exact_acc"] is None
    assert score["collision_answers"] == []
