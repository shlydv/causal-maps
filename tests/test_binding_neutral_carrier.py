from causal_maps.delta_binding_neutral_carrier import _neutral_matches, _neutral_text


class _Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "<chat>" + messages[0]["content"]


def test_neutral_carrier_exposes_value_anchor():
    text, offset = _neutral_text(_Tokenizer(), "blue")
    assert text[offset:offset + 4] == "blue"


def test_neutral_match_requires_both_controls():
    metrics = {"clean_acc": 1., "natural_acc": 1., "let_controller_effect": 10.,
               "neutral_carrier_effect": 9.8, "positive_neutral_fraction": 1.,
               "neutral_to_natural_ratio": .98, "neutral_to_let_ratio": .98,
               "neutral_wrong_direction_effect": 4., "neutral_other_slot_effect": .2}
    assert _neutral_matches(metrics)
    metrics["neutral_other_slot_effect"] = 9.75
    assert not _neutral_matches(metrics)
