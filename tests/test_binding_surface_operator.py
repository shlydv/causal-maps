from causal_maps.delta_binding_surface_operator import (
    _confirmed, _surface_single_text, _surface_two_text)


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "<chat>" + messages[0]["content"]


def test_mapping_prompts_expose_the_value_anchors():
    tok = _FakeTokenizer()
    text, offset = _surface_single_text(tok, "X", "blue")
    assert text[offset:offset + 4] == "blue"
    text, x_offset, y_offset = _surface_two_text(tok, "X", "Y", "blue", "cold", "X")
    assert text[x_offset:x_offset + 4] == "blue"
    assert text[y_offset:y_offset + 4] == "cold"


def test_confirmation_requires_both_specificity_controls():
    metrics = {"clean_acc": 1., "natural_acc": 1., "natural_effect": 10.,
               "add_effect": 9.5, "positive_add_fraction": 1.,
               "effect_ratio": .95, "wrong_direction_effect": 4.,
               "other_slot_effect": .2}
    assert _confirmed(metrics)
    metrics["other_slot_effect"] = 9.45
    assert not _confirmed(metrics)
