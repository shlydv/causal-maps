from causal_maps.delta_addressed_arithmetic_state import _rows


def test_rows_have_fixed_program_and_single_digit_answers():
    rows = _rows()
    assert len(rows) == 12
    assert all(row["target"] == row["source"] + 2 for row in rows)
    assert all(row["target"] + row["b"] <= 9 for row in rows)
