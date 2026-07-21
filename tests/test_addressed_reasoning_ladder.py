from causal_maps.delta_addressed_reasoning_ladder import _prompt, _rows


class _Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "<chat>" + messages[0]["content"]


def test_rows_form_three_disjoint_two_hop_chains():
    rows = _rows(list("abcdefghij"))
    assert len(rows) == 10
    for row in rows:
        assert len(set(row.values())) == 9


def test_prompt_exposes_start_anchor():
    row = _rows(list("abcdefghij"))[0]
    text, offset = _prompt(_Tokenizer(), row, row["source_start"])
    assert text[offset] == row["source_start"]
