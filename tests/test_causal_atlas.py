from causal_maps.delta_causal_atlas import (
    N_RANDOM,
    PROTOCOL,
    _middle_frame,
    _self_check,
    _tail_probability,
    _verdict_from_flags,
)


def test_causal_atlas_self_check():
    assert _self_check()["pass"]


def test_random_tail_is_add_one_exact():
    probability, exceed = _tail_probability(1.0, [0.0] * N_RANDOM)
    assert probability == 0.05
    assert exceed == 0


def test_middle_frame_is_unique():
    assert _middle_frame("epistemic", "communication") == "search"
    assert _middle_frame("search", "epistemic") == "communication"


def test_all_preregistered_verdicts_are_reachable():
    observed = {
        _verdict_from_flags(False, False, False, False),
        _verdict_from_flags(True, True, True, True),
        _verdict_from_flags(True, True, False, False),
        _verdict_from_flags(True, False, True, False),
        _verdict_from_flags(True, False, False, False),
    }
    assert observed == set(PROTOCOL["verdicts"])
