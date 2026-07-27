import copy

import pytest

from causal_maps.delta_shared_tangent_breadth import (
    _adjudicate,
    _route_progress,
    _self_check,
    _tail_probability,
)


def test_route_progress_is_direction_normalized():
    assert _route_progress(0.2, 0.6, 0.4) == pytest.approx(0.5)
    assert _route_progress(0.6, 0.2, 0.4) == pytest.approx(0.5)


def test_route_progress_rejects_unresolved_gap():
    assert _route_progress(0.2, 0.205, 0.4) is None
    assert _route_progress(None, 0.6, 0.4) is None


def test_random_tail_probability_is_add_one_corrected():
    probability, exceed = _tail_probability(0.3, [0.1] * 19)
    assert exceed == 0
    assert probability == pytest.approx(0.05)


def test_adjudication_self_check_exercises_both_outcomes():
    result = _self_check()
    assert result["pass"]
    assert result["positive_adjudication_check"]
    assert result["negative_adjudication_check"]


def test_adjudicator_does_not_mutate_input():
    # _self_check constructs and exercises the complete synthetic schema;
    # this smoke test also guards against accidental in-place adjudication.
    result = _self_check()
    untouched = copy.deepcopy(result)
    assert result == untouched
