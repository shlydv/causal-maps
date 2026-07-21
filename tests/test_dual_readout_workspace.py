from causal_maps.delta_dual_readout_workspace import _answer, _rows


def test_splits_are_disjoint_and_answers_are_one_digit():
    donor = _rows("donor")
    test = _rows("test")
    assert {(x["a"], x["b"]) for x in donor}.isdisjoint(
        {(x["a"], x["b"]) for x in test})
    assert len(donor) == len(test) == 8
    for row in donor + test:
        assert row["target_a"] == row["a"] + 1
        assert len(_answer(row, "sum", False)) == 1
        assert len(_answer(row, "sum", True)) == 1
        assert _answer(row, "parity", False) != _answer(row, "parity", True)
