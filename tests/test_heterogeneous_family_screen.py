import re

import pytest
import torch

from causal_maps.delta_cross_domain_controller import _domain_rows
from causal_maps.delta_heterogeneous_family_screen import (
    FAMILY_ORDER,
    FAMILY_SPECS,
    VALUES,
    _failure_reasons,
    _family_user,
    _validate_history_change,
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


def test_history_change_allows_repeated_state_mentions():
    clean = {"ids": torch.tensor([[1, 2, 3, 2], [4, 5, 6, 5]])}
    natural = {"ids": torch.tensor([[1, 7, 3, 7], [4, 8, 6, 8]])}
    _validate_history_change(clean, natural)


def test_history_change_rejects_identical_or_misaligned_rows():
    clean = {"ids": torch.tensor([[1, 2, 3], [4, 5, 6]])}
    partly_unchanged = {
        "ids": torch.tensor([[1, 7, 3], [4, 5, 6]])}
    with pytest.raises(ValueError, match="identical for rows"):
        _validate_history_change(clean, partly_unchanged)

    misaligned = {"ids": torch.tensor([[1, 7], [4, 8]])}
    with pytest.raises(ValueError, match="shapes differ"):
        _validate_history_change(clean, misaligned)
