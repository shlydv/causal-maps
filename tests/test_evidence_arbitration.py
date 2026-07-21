from causal_maps.delta_orchestration_screen import _rows


def test_frozen_evidence_conflicts_include_zero_failures():
    train, test = _rows()[::2], _rows()[1::2]
    train_diag = [r["database_value"] != r["a"] + r["b"] for r in train]
    test_diag = [r["database_value"] != r["a"] + r["b"] for r in test]
    assert len(train) == len(test) == 10
    assert sum(train_diag) >= 8
    assert sum(test_diag) == 8
    zeros = [r for r in test if r["key"] == "D"]
    assert len(zeros) == 2
    assert [r["a"] + r["b"] for r in zeros] == [4, 9]
    assert all(r["database_value"] == 0 for r in zeros)
