import inspect

import torch

from causal_maps.delta_predictive_conditional_transport import (
    FAMILY_ORDER,
    _derange_rows,
    _family_adjudication,
    _fit_low_rank,
    _overall_adjudication,
    _predict_low_rank,
    _row_splits,
    _select_predictor,
    _training_families,
)


def test_frozen_splits_are_pair_disjoint():
    splits = _row_splits()
    pairs = {
        name: {
            (row["source"], row["target"])
            for row in rows
        }
        for name, rows in splits.items()
    }
    assert len(splits["train"]) == 24
    assert len(splits["validation"]) == 8
    assert len(splits["test"]) == 12
    assert pairs["train"].isdisjoint(pairs["validation"])
    assert pairs["train"].isdisjoint(pairs["test"])
    assert pairs["validation"].isdisjoint(pairs["test"])


def test_predictor_interface_cannot_receive_counterpart_state():
    assert list(inspect.signature(_predict_low_rank).parameters) == [
        "predictor", "origin_states"]


def test_training_scope_is_exact_and_unambiguous():
    target = FAMILY_ORDER[3]
    within = _training_families(target, "within_family")
    cross = _training_families(target, "leave_one_family_out")
    assert within == [target]
    assert target not in cross
    assert set(cross) == set(FAMILY_ORDER) - {target}


def test_reduced_rank_predictor_recovers_state_conditioned_displacement():
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(80, 12, generator=generator)
    weights = torch.randn(12, 9, generator=generator)
    y = x @ weights + 0.01 * torch.randn(
        80, 9, generator=generator)
    predictor = _fit_low_rank(x[:60], y[:60], rank=12, ridge=0.01)
    predicted = _predict_low_rank(predictor, x[60:])
    mean_error = (y[60:] - y[:60].mean(0)).square().mean()
    predicted_error = (y[60:] - predicted).square().mean()
    assert predicted_error < 0.01 * mean_error


def test_model_selection_uses_validation_not_test_data():
    generator = torch.Generator().manual_seed(23)
    x = torch.randn(100, 10, generator=generator)
    y = x[:, :3] @ torch.randn(3, 7, generator=generator)
    predictor, metadata = _select_predictor(
        x[:70], y[:70], x[70:], y[70:])
    assert predictor["rank"] >= 3
    assert metadata["selected"]["normalized_mse"] < 0.05


def test_derangement_has_no_row_fixed_points():
    tensor = torch.arange(30).reshape(10, 3)
    shifted = _derange_rows(tensor)
    assert not torch.any(torch.all(tensor == shifted, dim=1))


def test_overall_verdict_requires_six_predictive_families():
    base = {
        "exact_pass": True,
        "predicted_pass": True,
        "exemplar_pass": False,
        "global_pass": False,
        "locus_specific": True,
    }
    families = {name: dict(base) for name in FAMILY_ORDER}
    assert _overall_adjudication(families)["verdict"] == (
        "PREDICTABLE_STATE_CONDITIONED_TRANSPORT")
    for name in FAMILY_ORDER[:3]:
        families[name]["predicted_pass"] = False
    assert _overall_adjudication(families)["verdict"] == (
        "PARTIAL_STATE_CONDITIONED_TRANSPORT")


def test_global_template_cannot_pass_by_destroying_the_answer():
    def direction(progress, accuracy=1.0):
        return {
            "mean_progress": progress,
            "median_distance_ratio": 0.5,
            "positive_rows": 24,
            "minimum_answer_accuracy": accuracy,
        }

    arms = {}
    for arm in (
            "exact", "conditional", "mean_displacement",
            "target_centroid", "nearest_neighbor", "row_shuffled",
            "instruction_control", "identical_control"):
        progress = {
            "exact": 0.8,
            "conditional": 0.45,
            "mean_displacement": 0.2,
            "target_centroid": 0.6,
            "nearest_neighbor": 0.2,
            "row_shuffled": 0.2,
            "instruction_control": 0.0,
            "identical_control": 0.0,
        }[arm]
        accuracy = 0.0 if arm == "target_centroid" else 1.0
        arms[arm] = {
            name: direction(progress, accuracy)
            for name in ("belief_to_search", "search_to_belief")
        }
    result = _family_adjudication(arms)
    assert not result["global_pass"]
    assert not result["arm_direction_pass"]["target_centroid"][
        "belief_to_search"]
