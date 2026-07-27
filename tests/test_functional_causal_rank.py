from causal_maps.delta_functional_causal_rank import (
    N_RANDOM,
    PATCH_WIDTH,
    PRIMARY_CHECKPOINT,
    PROTOCOL,
    _processed,
    _self_check,
    _tail_probability,
    _verdict_from_counts,
)
import torch


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


def test_processed_state_accepts_sequence_or_final_token():
    cache = {
        f"checkpoint_{PRIMARY_CHECKPOINT}": torch.randn(3, 7)
    }
    direct = torch.randn(3, PATCH_WIDTH, 7)
    assert torch.equal(
        _processed(cache, PRIMARY_CHECKPOINT, direct),
        _processed(cache, PRIMARY_CHECKPOINT, direct[:, -1, :]),
    )
