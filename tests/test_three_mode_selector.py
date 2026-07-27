import torch

from causal_maps.delta_three_mode_selector import (
    EARLY_LAYERS,
    FRAMES,
    N_CELLS,
    N_RANDOM,
    PROGRAM_WIDTH,
    PROTOCOL,
    _decoy,
    _random_programs,
    _select_layer,
    _self_check,
    _transitions,
    _verdict,
)


def test_three_mode_selector_self_check():
    assert _self_check()["pass"]


def test_every_transition_has_one_distinct_decoy():
    assert len(FRAMES) == 3
    assert len(_transitions()) == 6
    for source, target in _transitions():
        assert _decoy(source, target) not in (source, target)


def test_random_programs_are_exactly_norm_matched_per_destination():
    programs = {
        frame: torch.arange(
            PROGRAM_WIDTH * 9, dtype=torch.float32
        ).reshape(PROGRAM_WIDTH, 9) + index
        for index, frame in enumerate(FRAMES)
    }
    randoms = _random_programs(programs)
    for target in FRAMES:
        assert len(randoms[target]) == N_RANDOM
        for value in randoms[target]:
            assert torch.allclose(
                value.norm(), programs[target].norm(),
                atol=1e-4, rtol=1e-6)


def test_layer_selection_uses_minimum_destination_specific_score():
    calibration = {
        str(layer): [
            {
                "selector_score": 0.01,
                "primary": {"minimum_value_accuracy": 1.0},
                "baseline_minimum_value_accuracy": 1.0,
            }
            for _ in range(12)
        ]
        for layer in EARLY_LAYERS
    }
    for cell in calibration[str(EARLY_LAYERS[3])]:
        cell["selector_score"] = 0.25
    rows, selected = _select_layer(calibration)
    assert len(rows) == len(EARLY_LAYERS)
    assert selected["layer"] == EARLY_LAYERS[3]
    assert selected["score"] == 0.25


def test_all_frozen_verdicts_are_reachable():
    observed = {
        _verdict(
            N_CELLS - 1, N_CELLS, N_CELLS, N_CELLS,
            N_CELLS, N_CELLS, True),
        _verdict(
            N_CELLS, N_CELLS - 1, N_CELLS, N_CELLS,
            N_CELLS, N_CELLS, True),
        _verdict(
            N_CELLS, N_CELLS, N_CELLS, N_CELLS,
            N_CELLS, N_CELLS, True),
        _verdict(
            N_CELLS, N_CELLS, N_CELLS, N_CELLS,
            N_CELLS - 1, N_CELLS, False),
        _verdict(
            N_CELLS, N_CELLS, 8, N_CELLS - 2,
            8, N_CELLS, False),
        _verdict(
            N_CELLS, N_CELLS, 5, 5, 5, N_CELLS, False),
        "CALIBRATION_SELECTOR_NULL",
    }
    assert observed == set(PROTOCOL["verdicts"])
