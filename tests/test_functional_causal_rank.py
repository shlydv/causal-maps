from causal_maps.delta_functional_causal_rank import (
    N_RANDOM,
    PROTOCOL,
    _self_check,
    _tail_probability,
    _verdict_from_counts,
)


def test_functional_causal_rank_self_check():
    assert _self_check()["pass"]


def test_add_one_random_tail():
    probability, exceed = _tail_probability(1.0, [0.0] * N_RANDOM)
    assert probability == 0.05
    assert exceed == 0


def test_every_frozen_verdict_is_reachable():
    observed = {
        _verdict_from_counts(0, 0, 0, 1.0, 0),
        _verdict_from_counts(4, 3, 0, 1.0, 0),
        _verdict_from_counts(4, 4, 4, 0.05, 0),
        _verdict_from_counts(4, 4, 2, 0.05, 0),
        _verdict_from_counts(4, 4, 0, 1.0, 1),
        _verdict_from_counts(4, 4, 0, 1.0, 0),
    }
    assert observed == set(PROTOCOL["verdicts"])
