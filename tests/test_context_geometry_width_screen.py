import torch

from causal_maps.delta_context_geometry_width_screen import (
    FAMILIES,
    _dose_shape,
    _overall_decision,
    _processed_central_derivative,
    _response_geometry,
    _screen_rows,
)


def test_rows_are_pair_disjoint():
    rows = _screen_rows()
    calibration = {
        (row["source"], row["target"])
        for row in rows["calibration"]
    }
    test = {
        (row["source"], row["target"])
        for row in rows["test"]
    }
    assert not calibration & test


def test_dose_classifier_separates_smooth_and_gated():
    linear = {
        str(alpha): {
            "mean_progress": alpha,
            "minimum_answer_accuracy": 1.0,
        }
        for alpha in (0.25, 0.50, 0.75, 1.00)
    }
    gated = {
        "0.25": {
            "mean_progress": 0.0,
            "minimum_answer_accuracy": 1.0,
        },
        "0.5": {
            "mean_progress": 0.02,
            "minimum_answer_accuracy": 1.0,
        },
        "0.75": {
            "mean_progress": 0.10,
            "minimum_answer_accuracy": 1.0,
        },
        "1.0": {
            "mean_progress": 1.0,
            "minimum_answer_accuracy": 1.0,
        },
    }
    assert _dose_shape(linear)["smooth"]
    assert _dose_shape(gated)["gated"]
    assert not _dose_shape(gated)["smooth"]


def test_context_specific_response_maps_license_local_geometry():
    generator = torch.Generator().manual_seed(9)
    fingerprints = {}
    for index, family in enumerate(FAMILIES):
        center = torch.zeros(32)
        center[index * 4:(index + 1) * 4] = 1.0
        fingerprints[family] = {
            operation: (
                center[None, :].repeat(10, 1)
                + 0.005 * torch.randn(
                    10, 32, generator=generator))
            for operation in ("belief", "search")
        }
    result = _response_geometry(fingerprints, 1.0)
    assert result["verdict"] == "STABLE_CONTEXT_SPECIFIC_MAPS"
    assert _overall_decision(
        "SMOOTH_LOCAL_RESPONSE", result["verdict"]
    ) == "LOCAL_GEOMETRY_PILOT_LICENSED"


def test_direct_residual_identity_is_removed_exactly():
    direction = torch.randn(3, 12)
    dynamic = torch.randn(5, 12)
    baseline = torch.randn(5, 12)
    step = 0.2
    plus = baseline + step * (
        direction[-1][None, :] + dynamic)
    minus = baseline - step * (
        direction[-1][None, :] + dynamic)
    recovered = _processed_central_derivative(
        plus, minus, step, direction)
    assert torch.allclose(recovered, dynamic, atol=1e-5)
