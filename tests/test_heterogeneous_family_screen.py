import re

from causal_maps.delta_cross_domain_controller import _domain_rows
from causal_maps.delta_heterogeneous_family_screen import (
    FAMILY_ORDER,
    FAMILY_SPECS,
    VALUES,
    _failure_reasons,
    _family_user,
)


def test_each_family_changes_the_counterfactual_anchor_once_in_raw_text():
    row = _domain_rows(VALUES)[0]
    for family in FAMILY_ORDER:
        prompt = _family_user(
            row, FAMILY_SPECS[family], "BELIEF")
        assert len(re.findall(
            rf"\b{re.escape(row['state'])}\b", prompt)) == 1
        assert "Reply with exactly BELIEF" in prompt


def test_failure_reasons_keep_accuracy_source_and_gap_separate():
    task = {
        "eligible": True,
        "source_intervention": {"sufficient": True},
    }
    tasks = {"belief": dict(task), "search": dict(task)}
    assert _failure_reasons(
        tasks, {"belief": 0.60, "search": 0.55}) == []
    assert _failure_reasons(
        tasks, {"belief": 0.55, "search": 0.60}) == []
    assert _failure_reasons(
        tasks, {"belief": 0.60, "search": 0.58}) == [
            "ORIGINAL_ROUTE_GAP_BELOW_0.03"]

    tasks["search"]["source_intervention"]["sufficient"] = False
    assert _failure_reasons(
        tasks, {"belief": 0.60, "search": 0.55}) == [
            "SEARCH_SOURCE_INELIGIBLE"]
