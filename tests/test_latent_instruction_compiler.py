import torch
import pytest

from causal_maps.delta_latent_instruction_compiler import (
    EARLY_LAYERS,
    N_RANDOM,
    PROGRAM_WIDTH,
    PROTOCOL,
    _context_match,
    _derangements,
    _random_programs,
    _self_check,
    _select_layer,
    _verdict,
)


def test_latent_instruction_compiler_self_check():
    assert _self_check()["pass"]


def test_layer_selection_is_prospective_and_deterministic():
    cells = lambda value: [  # noqa: E731
        {
            "l21": {"mean_progress": value},
            "l27": {"mean_progress": value + 0.1},
            "minimum_value_accuracy": 1.0,
            "baseline_minimum_value_accuracy": 1.0,
        }
        for _ in range(4)
    ]
    calibration = {
        str(layer): cells(0.1) for layer in EARLY_LAYERS
    }
    calibration[str(EARLY_LAYERS[2])] = cells(0.4)
    rows, selected = _select_layer(calibration)
    assert len(rows) == len(EARLY_LAYERS)
    assert selected["layer"] == EARLY_LAYERS[2]
    assert selected["score"] == 0.4


def test_random_programs_are_exactly_norm_matched():
    program = torch.arange(
        PROGRAM_WIDTH * 7, dtype=torch.float32
    ).reshape(PROGRAM_WIDTH, 7)
    randoms = _random_programs({"a_to_b": program})["a_to_b"]
    assert len(randoms) == N_RANDOM
    for value in randoms:
        assert torch.allclose(
            value.norm(), program.norm(), atol=1e-4, rtol=1e-6)


def test_receiver_specific_context_match():
    source = torch.randn(24, 3, 9)
    target = torch.randn(24, 3, 9)
    result = _context_match(
        target.clone(), target, source, _derangements(24))
    assert result["pass"]
    assert result["add_one_p"] == 0.05


def test_derangement_guard_rejects_impossible_request():
    with pytest.raises(ValueError):
        _derangements(N_RANDOM)


def test_all_frozen_verdicts_are_reachable():
    observed = {
        _verdict(3, 4, 4, 4, 4, 4, True),
        _verdict(4, 3, 2, 2, 2, 2, False),
        _verdict(4, 3, 3, 3, 2, 2, False),
        _verdict(4, 4, 2, 2, 2, 2, False),
        _verdict(4, 4, 4, 4, 4, 4, True),
        "CALIBRATION_CODE_NULL",
    }
    assert observed == set(PROTOCOL["verdicts"])
