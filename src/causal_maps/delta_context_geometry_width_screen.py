"""Width-first screen of smooth, gated, and template-only causal control."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from .delta_anchor_write import _resolve
from .delta_cross_domain_controller import _generic_accuracy
from .delta_exact_transplant_locus_diagnostic import (
    CHECKPOINT_LAYERS,
    _capture_baseline,
    _direction_summary,
    _row_transport,
    _run_exact_patch,
)
from .delta_heterogeneous_family_screen import (
    FAMILY_SPECS,
    VALUES,
    _family_alignment,
)
from .logutil import Heartbeat, log
from .model_utils import (
    get_decoder_layers,
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)
from .patching import _split_output


PROTOCOL_VERSION = "2026-07-27-p2-context-geometry-width-screen-v1"
FAMILIES = (
    "private_belief",
    "two_hop_pointer",
    "maximum_score",
    "constraint_elimination",
)
SOURCE_LAYER = 21
PRIMARY_CHECKPOINT = 27
CALIBRATION_N = 4
TEST_N = 8
BASIS_RANK = 4
PROBE_FRACTION = 0.10
DOSE_ALPHAS = (-0.50, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
DIRECTIONS = ("belief_to_search", "search_to_belief")

MINIMUM_EXACT_PROGRESS = 0.25
MINIMUM_VALUE_ACCURACY = 0.80
MINIMUM_DOSE_SPEARMAN = 0.90
MINIMUM_DOSE_LINEAR_R2 = 0.80
MAXIMUM_SMOOTH_STEP_FRACTION = 0.55
MINIMUM_GATED_STEP_FRACTION = 0.65
MINIMUM_SMOOTH_FAMILIES = 3
MINIMUM_GATED_FAMILIES = 2
MINIMUM_PROCESSED_TARGET_RATIO = 0.05

MINIMUM_WITHIN_MAP_COSINE = 0.75
MINIMUM_CONTEXT_SEPARATION = 0.15
MINIMUM_SHARED_MAP_COSINE = 0.75
MINIMUM_PROBE_ACCURACY = 0.80
MINIMUM_PROCESSED_RESPONSE_NORM = 1e-4
RANDOM_SEED = 781233


PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "purpose": (
        "Cheaply distinguish smooth context-conditioned local geometry, "
        "discrete gating, shared response geometry, and unstable or "
        "template-only behavior before a held-out covariant transport run."
    ),
    "status": "discovery screen; no paper claim is licensed by this run",
    "model": "Qwen2.5-7B-Instruct, 8-bit",
    "families": list(FAMILIES),
    "rows": {
        "basis_calibration": CALIBRATION_N,
        "screen_test": TEST_N,
        "directed_pairs_disjoint_between_splits": True,
        "historical_pair_reuse": (
            "Allowed for this width screen; any positive branch requires "
            "fresh prospective histories."
        ),
    },
    "locus": {
        "source_layer": SOURCE_LAYER,
        "positions": "three answer-prefix command tokens",
        "checkpoint_layers": list(CHECKPOINT_LAYERS),
        "primary_checkpoint": PRIMARY_CHECKPOINT,
    },
    "dose_screen": {
        "intervention": (
            "origin plus alpha times the exact same-row opposite-operation "
            "answer-prefix displacement"
        ),
        "alphas": list(DOSE_ALPHAS),
        "smooth_gate": {
            "minimum_exact_progress_each_direction":
                MINIMUM_EXACT_PROGRESS,
            "minimum_spearman_alpha_to_progress":
                MINIMUM_DOSE_SPEARMAN,
            "minimum_linear_r2_on_alpha_0_to_1":
                MINIMUM_DOSE_LINEAR_R2,
            "maximum_largest_step_fraction":
                MAXIMUM_SMOOTH_STEP_FRACTION,
            "minimum_answer_accuracy": MINIMUM_VALUE_ACCURACY,
            "minimum_processed_to_raw_target_norm_ratio":
                MINIMUM_PROCESSED_TARGET_RATIO,
        },
        "gating_candidate": {
            "minimum_exact_progress_each_direction":
                MINIMUM_EXACT_PROGRESS,
            "minimum_largest_step_fraction":
                MINIMUM_GATED_STEP_FRACTION,
            "minimum_answer_accuracy": MINIMUM_VALUE_ACCURACY,
            "minimum_processed_to_raw_target_norm_ratio":
                MINIMUM_PROCESSED_TARGET_RATIO,
        },
    },
    "response_map_screen": {
        "basis": (
            "rank-4 uncentered SVD basis of four family mean exact "
            "BELIEF-to-SEARCH displacements on calibration rows only"
        ),
        "probe": (
            "central finite differences using plus/minus 0.10 times the "
            "median family-template norm along each unit basis direction"
        ),
        "outputs": (
            "complete final-token residual response at L24 and L27 after "
            "subtracting the exact residual identity carry-through from "
            "the patched final answer-prefix position; no one-dimensional "
            "route-score gradient"
        ),
        "primary_stability_gate": {
            "minimum_within_family_cosine":
                MINIMUM_WITHIN_MAP_COSINE,
            "minimum_within_minus_cross_cosine":
                MINIMUM_CONTEXT_SEPARATION,
            "shared_map_cross_family_cosine":
                MINIMUM_SHARED_MAP_COSINE,
            "minimum_probe_answer_accuracy":
                MINIMUM_PROBE_ACCURACY,
            "minimum_processed_response_norm":
                MINIMUM_PROCESSED_RESPONSE_NORM,
        },
    },
    "interpretation": {
        "smooth_and_context_specific": (
            "license a held-out covariant-transport pilot"),
        "gated_and_stable": (
            "license a selector-mediation factorial"),
        "unstable_or_mixed": (
            "license no deeper branch; favor family templates or a "
            "different representation level"),
    },
    "stopping_rule": (
        "Do not tune prompts, alpha values, basis rank, probe scale, "
        "thresholds, source layer, or checkpoints after output. This "
        "screen selects a branch; it cannot establish the final mechanism."
    ),
    "random_seed": RANDOM_SEED,
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest().upper()


def _screen_rows():
    calibration = []
    test = []
    for source_index in range(CALIBRATION_N):
        target_index = (source_index + 2) % len(VALUES)
        remaining = [
            VALUES[(source_index + offset) % len(VALUES)]
            for offset in range(1, len(VALUES) + 1)
            if VALUES[(source_index + offset) % len(VALUES)]
            not in (VALUES[source_index], VALUES[target_index])
        ]
        calibration.append({
            "row_index": source_index,
            "split": "calibration",
            "source": VALUES[source_index],
            "target": VALUES[target_index],
            "state": VALUES[source_index],
            "d1": remaining[2],
            "d2": remaining[4],
        })
    for source_index in range(TEST_N):
        target_index = (source_index + 3) % len(VALUES)
        remaining = [
            VALUES[(source_index + offset) % len(VALUES)]
            for offset in range(1, len(VALUES) + 1)
            if VALUES[(source_index + offset) % len(VALUES)]
            not in (VALUES[source_index], VALUES[target_index])
        ]
        test.append({
            "row_index": source_index,
            "split": "test",
            "source": VALUES[source_index],
            "target": VALUES[target_index],
            "state": VALUES[source_index],
            "d1": remaining[1],
            "d2": remaining[3],
        })
    calibration_pairs = {
        (row["source"], row["target"]) for row in calibration
    }
    test_pairs = {(row["source"], row["target"]) for row in test}
    if calibration_pairs & test_pairs:
        raise AssertionError("calibration and test pairs overlap")
    if len(calibration) != CALIBRATION_N or len(test) != TEST_N:
        raise AssertionError("frozen row counts changed")
    return {"calibration": calibration, "test": test}


@torch.no_grad()
def _capture_answer_states(model, batch, positions):
    cache = {}

    def hook(_module, _args, output):
        states, _rebuild = _split_output(output)
        cache["states"] = (
            states[:, positions, :].detach().float().cpu())

    handle = get_decoder_layers(model)[SOURCE_LAYER].register_forward_hook(
        hook)
    try:
        model(
            input_ids=batch["ids"], attention_mask=batch["am"],
            use_cache=False)
    finally:
        handle.remove()
    return cache["states"]


def _template_basis(templates):
    matrix = torch.stack([
        templates[family].flatten().float()
        for family in FAMILIES
    ])
    _left, singular, right = torch.linalg.svd(
        matrix, full_matrices=False)
    basis = right[:BASIS_RANK]
    if basis.shape[0] != BASIS_RANK:
        raise AssertionError("frozen basis rank is unavailable")
    reconstruction = matrix @ basis.T @ basis
    explained = float(
        reconstruction.square().sum()
        / matrix.square().sum().clamp_min(1e-8))
    return basis, {
        "singular_values": singular.tolist(),
        "rank4_reconstruction_energy": explained,
        "family_template_norms": {
            family: float(templates[family].norm())
            for family in FAMILIES
        },
    }


def _processed_checkpoint_state(checkpoint_state, source_answer_state):
    """Remove the exact residual identity path at the final patched token."""
    return checkpoint_state.float() - source_answer_state[:, -1, :].float()


def _processed_central_derivative(
        plus_state, minus_state, step, basis_direction):
    """Central derivative with the known direct identity derivative removed."""
    raw = (
        plus_state.float() - minus_state.float()
    ) / (2.0 * float(step))
    return raw - basis_direction[-1, :].float()[None, :]


def _rankdata(values):
    order = np.argsort(np.asarray(values), kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(x, y):
    if len(x) < 2:
        return 0.0
    xr = _rankdata(x)
    yr = _rankdata(y)
    if float(np.std(xr)) <= 1e-12 or float(np.std(yr)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def _dose_shape(alpha_summaries, processed_target_ratio=1.0):
    fit_alphas = [0.0, 0.25, 0.50, 0.75, 1.00]
    progress = [
        0.0 if alpha == 0.0
        else float(alpha_summaries[str(alpha)]["mean_progress"])
        for alpha in fit_alphas
    ]
    exact = float(progress[-1])
    correlation = _spearman(fit_alphas, progress)
    design = np.asarray(fit_alphas, dtype=np.float64)
    outcome = np.asarray(progress, dtype=np.float64)
    coefficients = np.polyfit(design, outcome, deg=1)
    predicted = coefficients[0] * design + coefficients[1]
    residual = float(np.square(outcome - predicted).sum())
    total = float(np.square(outcome - outcome.mean()).sum())
    r2 = 1.0 - residual / max(total, 1e-12)
    increments = np.diff(outcome)
    largest_step_fraction = float(
        max(0.0, float(increments.max())) / max(abs(exact), 1e-8))
    minimum_accuracy = min(
        float(alpha_summaries[str(alpha)]["minimum_answer_accuracy"])
        for alpha in fit_alphas[1:])
    smooth = bool(
        exact >= MINIMUM_EXACT_PROGRESS - 1e-9
        and correlation >= MINIMUM_DOSE_SPEARMAN - 1e-9
        and r2 >= MINIMUM_DOSE_LINEAR_R2 - 1e-9
        and largest_step_fraction
        <= MAXIMUM_SMOOTH_STEP_FRACTION + 1e-9
        and minimum_accuracy >= MINIMUM_VALUE_ACCURACY - 1e-9
        and processed_target_ratio
        >= MINIMUM_PROCESSED_TARGET_RATIO - 1e-9)
    gated = bool(
        exact >= MINIMUM_EXACT_PROGRESS - 1e-9
        and correlation >= 0.80 - 1e-9
        and largest_step_fraction
        >= MINIMUM_GATED_STEP_FRACTION - 1e-9
        and minimum_accuracy >= MINIMUM_VALUE_ACCURACY - 1e-9
        and processed_target_ratio
        >= MINIMUM_PROCESSED_TARGET_RATIO - 1e-9)
    return {
        "fit_alphas": fit_alphas,
        "mean_progress": progress,
        "spearman": correlation,
        "linear_r2": float(r2),
        "linear_slope": float(coefficients[0]),
        "largest_step_fraction": largest_step_fraction,
        "minimum_answer_accuracy_0_to_1": minimum_accuracy,
        "processed_to_raw_target_norm_ratio":
            float(processed_target_ratio),
        "smooth": smooth,
        "gated": gated,
    }


def _cosine_rows(left, right):
    left = left.float()
    right = right.float()
    left = left / left.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    right = right / right.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return left @ right.T


def _off_diagonal_median(matrix):
    if matrix.shape[0] < 2:
        return 0.0
    mask = ~torch.eye(
        matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return float(matrix[mask].median())


def _response_geometry(fingerprints, minimum_accuracy):
    """Adjudicate primary-checkpoint response-map reliability."""
    within = {}
    family_means = {}
    operation_cosines = {}
    for family in FAMILIES:
        within[family] = {}
        family_means[family] = {}
        for operation in ("belief", "search"):
            values = fingerprints[family][operation].float()
            within[family][operation] = _off_diagonal_median(
                _cosine_rows(values, values))
            family_means[family][operation] = values.mean(dim=0)
        operation_cosines[family] = float(_cosine_rows(
            family_means[family]["belief"][None, :],
            family_means[family]["search"][None, :])[0, 0])

    cross_values = []
    cross_by_operation = {}
    for operation in ("belief", "search"):
        values = torch.stack([
            family_means[family][operation] for family in FAMILIES
        ])
        cosine = _cosine_rows(values, values)
        cross_by_operation[operation] = []
        for left in range(len(FAMILIES)):
            for right in range(left + 1, len(FAMILIES)):
                value = float(cosine[left, right])
                cross_values.append(value)
                cross_by_operation[operation].append(value)

    within_values = [
        value for family in within.values() for value in family.values()
    ]
    response_norms = [
        float(value.norm(dim=1).median())
        for family in FAMILIES
        for value in fingerprints[family].values()
    ]
    minimum_response_norm = min(response_norms)
    median_within = float(np.median(within_values))
    median_cross = float(np.median(cross_values))
    separation = median_within - median_cross
    stable = bool(
        median_within >= MINIMUM_WITHIN_MAP_COSINE - 1e-9
        and minimum_accuracy >= MINIMUM_PROBE_ACCURACY - 1e-9
        and minimum_response_norm
        >= MINIMUM_PROCESSED_RESPONSE_NORM - 1e-12)
    context_specific = bool(
        stable
        and separation >= MINIMUM_CONTEXT_SEPARATION - 1e-9)
    shared = bool(
        stable
        and median_cross >= MINIMUM_SHARED_MAP_COSINE - 1e-9
        and separation < MINIMUM_CONTEXT_SEPARATION - 1e-9)
    if context_specific:
        verdict = "STABLE_CONTEXT_SPECIFIC_MAPS"
    elif shared:
        verdict = "STABLE_SHARED_MAPS"
    elif (
            median_within < 0.50
            or minimum_accuracy < MINIMUM_PROBE_ACCURACY
            or minimum_response_norm < MINIMUM_PROCESSED_RESPONSE_NORM):
        verdict = "UNSTABLE_OR_INELIGIBLE_MAPS"
    else:
        verdict = "MIXED_RESPONSE_MAPS"
    return {
        "within_family_operation_cosines": within,
        "cross_family_cosines_by_operation": cross_by_operation,
        "belief_search_map_cosine_by_family": operation_cosines,
        "median_within_family_cosine": median_within,
        "median_cross_family_cosine": median_cross,
        "within_minus_cross": separation,
        "minimum_probe_answer_accuracy": float(minimum_accuracy),
        "minimum_processed_response_norm": minimum_response_norm,
        "stable": stable,
        "context_specific": context_specific,
        "shared": shared,
        "verdict": verdict,
    }


def _overall_decision(dose_verdict, map_verdict):
    if (
            dose_verdict == "SMOOTH_LOCAL_RESPONSE"
            and map_verdict == "STABLE_CONTEXT_SPECIFIC_MAPS"):
        return "LOCAL_GEOMETRY_PILOT_LICENSED"
    if (
            dose_verdict == "GATED_RESPONSE_CANDIDATE"
            and map_verdict in (
                "STABLE_CONTEXT_SPECIFIC_MAPS",
                "STABLE_SHARED_MAPS")):
        return "SELECTOR_FACTORIAL_LICENSED"
    if map_verdict == "STABLE_SHARED_MAPS":
        return "SHARED_RESPONSE_GEOMETRY_CONFLICT"
    return "NO_DEEPER_BRANCH_LICENSED"


def _self_check():
    rows = _screen_rows()
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
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    fingerprints = {}
    for family_index, family in enumerate(FAMILIES):
        fingerprints[family] = {}
        center = torch.zeros(32)
        center[family_index * 4:(family_index + 1) * 4] = 1.0
        for operation in ("belief", "search"):
            fingerprints[family][operation] = (
                center[None, :].repeat(10, 1)
                + 0.005 * torch.randn(
                    10, 32, generator=generator))
    response = _response_geometry(fingerprints, 1.0)
    identity_direction = torch.randn(
        3, 12, generator=generator)
    dynamic = torch.randn(5, 12, generator=generator)
    baseline = torch.randn(5, 12, generator=generator)
    step = 0.2
    plus = baseline + step * (
        identity_direction[-1][None, :] + dynamic)
    minus = baseline - step * (
        identity_direction[-1][None, :] + dynamic)
    recovered_dynamic = _processed_central_derivative(
        plus, minus, step, identity_direction)
    passed = bool(
        len(rows["calibration"]) == CALIBRATION_N
        and len(rows["test"]) == TEST_N
        and _dose_shape(linear)["smooth"]
        and _dose_shape(gated)["gated"]
        and response["context_specific"]
        and torch.allclose(
            recovered_dynamic, dynamic, atol=1e-5)
        and _overall_decision(
            "SMOOTH_LOCAL_RESPONSE",
            response["verdict"])
        == "LOCAL_GEOMETRY_PILOT_LICENSED")
    if not passed:
        raise AssertionError("context geometry self-check failed")
    return {
        "row_split_check": True,
        "smooth_classifier_check": True,
        "gated_classifier_check": True,
        "context_specific_map_check": True,
        "direct_identity_removal_check": True,
        "pass": True,
    }


@torch.no_grad()
def run_delta_context_geometry_width_screen(
        model_path, out_dir,
        model_key="qwen7b_context_geometry_width_screen",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=8, self_test_only=False):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_world) != TEST_N:
        raise ValueError("v1 is frozen to exactly eight test rows")
    self_check = _self_check()
    if self_test_only:
        result = {
            "stage": "delta_context_geometry_width_screen",
            "protocol_sha256": PROTOCOL_SHA256,
            "self_check": self_check,
            "verdict": "SELF_CHECK_PASS",
        }
        path = os.path.join(
            out_dir, "delta_context_geometry_width_screen_self_check.json")
        with open(path, "w") as handle:
            json.dump(result, handle, indent=2)
        log(
            "CONTEXT-GEOMETRY-WIDTH-SCREEN self-check pass "
            f"protocol={PROTOCOL_SHA256}")
        return result

    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    if PRIMARY_CHECKPOINT >= model_num_hidden_layers(model):
        raise ValueError("frozen checkpoint layer is absent")
    rows = _screen_rows()

    alignments = {"calibration": {}, "test": {}}
    for split in ("calibration", "test"):
        for family in FAMILIES:
            alignments[split][family] = _family_alignment(
                tok, dev, rows[split], FAMILY_SPECS[family])

    calibration_hb = Heartbeat(
        len(FAMILIES) * 2 * 2,
        "context_geometry_calibration_capture",
        every_sec=30, out_dir=out_dir)
    templates = {}
    for family in FAMILIES:
        alignment = alignments["calibration"][family]
        differences = []
        for history, (belief, search) in enumerate(alignment["batches"]):
            belief_state = _capture_answer_states(
                model, belief, alignment["answer_positions"])
            calibration_hb.step(
                extra=f"{family}/h{history}/belief")
            search_state = _capture_answer_states(
                model, search, alignment["answer_positions"])
            calibration_hb.step(
                extra=f"{family}/h{history}/search")
            differences.append(search_state - belief_state)
        templates[family] = torch.cat(differences).mean(dim=0)
    calibration_hb.done()
    basis, basis_metadata = _template_basis(templates)
    template_norms = [
        float(templates[family].norm()) for family in FAMILIES
    ]
    probe_step = PROBE_FRACTION * float(np.median(template_norms))
    if probe_step <= 1e-8:
        raise ValueError("frozen probe step is zero")

    baseline_hb = Heartbeat(
        len(FAMILIES) * 2 * 2,
        "context_geometry_test_baselines",
        every_sec=30, out_dir=out_dir)
    baselines = {}
    for family in FAMILIES:
        alignment = alignments["test"][family]
        baselines[family] = {}
        positions = alignment["answer_positions"]
        for history, (belief, search) in enumerate(alignment["batches"]):
            for operation, batch in (
                    ("belief", belief), ("search", search)):
                baselines[family][(history, operation)] = (
                    _capture_baseline(
                        model, batch, SOURCE_LAYER, positions))
                baseline_hb.step(
                    extra=f"{family}/h{history}/{operation}")
    baseline_hb.done()

    dose_total = (
        len(FAMILIES) * len(DIRECTIONS) * 2 * len(DOSE_ALPHAS))
    dose_hb = Heartbeat(
        dose_total, "context_geometry_dose_screen",
        every_sec=30, out_dir=out_dir)
    dose_rows = {
        family: {
            direction: {
                alpha: {
                    checkpoint: {
                        "raw_progress": [],
                        "raw_distance": [],
                        "processed_progress": [],
                        "processed_distance": [],
                    }
                    for checkpoint in CHECKPOINT_LAYERS
                }
                for alpha in DOSE_ALPHAS
            }
            for direction in DIRECTIONS
        }
        for family in FAMILIES
    }
    dose_accuracy = {
        family: {
            direction: {alpha: [] for alpha in DOSE_ALPHAS}
            for direction in DIRECTIONS
        }
        for family in FAMILIES
    }
    dose_target_ratio = {
        family: {
            direction: {
                checkpoint: []
                for checkpoint in CHECKPOINT_LAYERS
            }
            for direction in DIRECTIONS
        }
        for family in FAMILIES
    }
    for family in FAMILIES:
        alignment = alignments["test"][family]
        for direction in DIRECTIONS:
            origin_operation, target_operation = (
                ("belief", "search")
                if direction == "belief_to_search"
                else ("search", "belief"))
            for history, pair in enumerate(alignment["batches"]):
                batch = (
                    pair[0] if origin_operation == "belief" else pair[1])
                expected = (
                    [row["source"] for row in rows["test"]]
                    if history == 0
                    else [row["target"] for row in rows["test"]])
                origin = baselines[family][
                    (history, origin_operation)]
                target = baselines[family][
                    (history, target_operation)]
                origin_answer = origin["source"]
                exact_delta = target["source"] - origin_answer
                origin_direct = origin_answer[:, -1, :]
                target_direct = target["source"][:, -1, :]
                for checkpoint in CHECKPOINT_LAYERS:
                    raw_target = (
                        target[f"checkpoint_{checkpoint}"]
                        - origin[f"checkpoint_{checkpoint}"])
                    processed_target = (
                        _processed_checkpoint_state(
                            target[f"checkpoint_{checkpoint}"],
                            target["source"])
                        - _processed_checkpoint_state(
                            origin[f"checkpoint_{checkpoint}"],
                            origin["source"]))
                    ratio = (
                        processed_target.norm(dim=-1)
                        / raw_target.norm(dim=-1).clamp_min(1e-8))
                    dose_target_ratio[family][direction][
                        checkpoint].extend(ratio.tolist())
                for alpha in DOSE_ALPHAS:
                    target_values = (
                        origin_answer + float(alpha) * exact_delta)
                    patched = _run_exact_patch(
                        model, batch, SOURCE_LAYER,
                        alignment["answer_positions"], target_values)
                    dose_accuracy[family][direction][alpha].append(
                        _generic_accuracy(
                            patched["logits"], batch, expected, VALUES))
                    for checkpoint in CHECKPOINT_LAYERS:
                        raw_progress, raw_distance = _row_transport(
                            origin[f"checkpoint_{checkpoint}"],
                            target[f"checkpoint_{checkpoint}"],
                            patched[f"checkpoint_{checkpoint}"])
                        patched_direct = (
                            origin_direct
                            + float(alpha)
                            * (target_direct - origin_direct))
                        processed_progress, processed_distance = (
                            _row_transport(
                                _processed_checkpoint_state(
                                    origin[f"checkpoint_{checkpoint}"],
                                    origin["source"]),
                                _processed_checkpoint_state(
                                    target[f"checkpoint_{checkpoint}"],
                                    target["source"]),
                                patched[f"checkpoint_{checkpoint}"]
                                - patched_direct))
                        dose_rows[family][direction][alpha][checkpoint][
                            "raw_progress"].extend(raw_progress)
                        dose_rows[family][direction][alpha][checkpoint][
                            "raw_distance"].extend(raw_distance)
                        dose_rows[family][direction][alpha][checkpoint][
                            "processed_progress"].extend(
                                processed_progress)
                        dose_rows[family][direction][alpha][checkpoint][
                            "processed_distance"].extend(
                                processed_distance)
                    dose_hb.step(
                        extra=(
                            f"{family}/{direction}/h{history}/"
                            f"alpha_{alpha:+.2f}"))
    dose_hb.done()

    dose_results = {}
    smooth_families = []
    gated_families = []
    for family in FAMILIES:
        dose_results[family] = {}
        family_smooth = True
        family_gated = True
        for direction in DIRECTIONS:
            by_checkpoint = {}
            for checkpoint in CHECKPOINT_LAYERS:
                by_alpha = {}
                for alpha in DOSE_ALPHAS:
                    row = dose_rows[family][direction][alpha][checkpoint]
                    by_alpha[str(alpha)] = {
                        "raw": _direction_summary(
                            row["raw_progress"], row["raw_distance"],
                            dose_accuracy[family][direction][alpha]),
                        "processed": _direction_summary(
                            row["processed_progress"],
                            row["processed_distance"],
                            dose_accuracy[family][direction][alpha]),
                    }
                by_checkpoint[str(checkpoint)] = by_alpha
            primary_processed = {
                alpha: value["processed"]
                for alpha, value in by_checkpoint[
                    str(PRIMARY_CHECKPOINT)].items()
            }
            target_ratios = {
                str(checkpoint): float(np.median(
                    dose_target_ratio[family][direction][checkpoint]))
                for checkpoint in CHECKPOINT_LAYERS
            }
            shape = _dose_shape(
                primary_processed,
                target_ratios[str(PRIMARY_CHECKPOINT)])
            dose_results[family][direction] = {
                "by_checkpoint": by_checkpoint,
                "processed_to_raw_target_norm_ratio_by_checkpoint":
                    target_ratios,
                "shape": shape,
            }
            family_smooth = family_smooth and shape["smooth"]
            family_gated = family_gated and shape["gated"]
        dose_results[family]["smooth_both_directions"] = family_smooth
        dose_results[family]["gated_both_directions"] = family_gated
        if family_smooth:
            smooth_families.append(family)
        if family_gated:
            gated_families.append(family)
        log(
            f"DOSE-SCREEN {family} smooth={family_smooth} "
            f"gated={family_gated}")
    if len(smooth_families) >= MINIMUM_SMOOTH_FAMILIES:
        dose_verdict = "SMOOTH_LOCAL_RESPONSE"
    elif len(gated_families) >= MINIMUM_GATED_FAMILIES:
        dose_verdict = "GATED_RESPONSE_CANDIDATE"
    else:
        dose_verdict = "MIXED_OR_IRREGULAR_RESPONSE"

    probe_total = (
        len(FAMILIES) * 2 * 2 * BASIS_RANK * 2)
    probe_hb = Heartbeat(
        probe_total, "context_geometry_response_maps",
        every_sec=30, out_dir=out_dir)
    fingerprints = {
        checkpoint: {
            family: {"belief": [], "search": []}
            for family in FAMILIES
        }
        for checkpoint in CHECKPOINT_LAYERS
    }
    response_arrays = {}
    probe_accuracy = []
    basis_shaped = basis.reshape(
        BASIS_RANK, 3, -1).float()
    for family in FAMILIES:
        alignment = alignments["test"][family]
        family_maps = {
            checkpoint: {"belief": [], "search": []}
            for checkpoint in CHECKPOINT_LAYERS
        }
        for operation in ("belief", "search"):
            for history, pair in enumerate(alignment["batches"]):
                batch = pair[0] if operation == "belief" else pair[1]
                expected = (
                    [row["source"] for row in rows["test"]]
                    if history == 0
                    else [row["target"] for row in rows["test"]])
                origin = baselines[family][(history, operation)]
                axis_responses = {
                    checkpoint: [] for checkpoint in CHECKPOINT_LAYERS
                }
                for axis in range(BASIS_RANK):
                    step = probe_step * basis_shaped[axis]
                    outputs = {}
                    for sign in (-1, 1):
                        values = origin["source"] + float(sign) * step
                        outputs[sign] = _run_exact_patch(
                            model, batch, SOURCE_LAYER,
                            alignment["answer_positions"], values)
                        accuracy = _generic_accuracy(
                            outputs[sign]["logits"], batch,
                            expected, VALUES)
                        probe_accuracy.append(float(accuracy))
                        probe_hb.step(
                            extra=(
                                f"{family}/{operation}/h{history}/"
                                f"axis_{axis + 1}/sign_{sign:+d}"))
                    for checkpoint in CHECKPOINT_LAYERS:
                        processed_derivative = (
                            _processed_central_derivative(
                                outputs[1][f"checkpoint_{checkpoint}"],
                                outputs[-1][f"checkpoint_{checkpoint}"],
                                probe_step,
                                basis_shaped[axis]))
                        axis_responses[checkpoint].append(
                            processed_derivative)
                for checkpoint in CHECKPOINT_LAYERS:
                    # [row, axis, downstream_hidden]
                    value = torch.stack(
                        axis_responses[checkpoint], dim=1)
                    family_maps[checkpoint][operation].append(value)
        for checkpoint in CHECKPOINT_LAYERS:
            for operation in ("belief", "search"):
                # [history, row, axis, downstream_hidden]
                value = torch.stack(
                    family_maps[checkpoint][operation], dim=0)
                response_arrays[
                    f"{family}_{operation}_L{checkpoint}"] = (
                        value.numpy().astype(np.float16))
                fingerprints[checkpoint][family][operation] = (
                    value.flatten(0, 1).flatten(1))
    probe_hb.done()

    response_npz = os.path.join(
        out_dir,
        f"context_geometry_response_maps_{model_key}.npz")
    np.savez_compressed(response_npz, **response_arrays)
    with open(response_npz, "rb") as handle:
        response_npz_sha = hashlib.sha256(
            handle.read()).hexdigest().upper()
    minimum_probe_accuracy = float(min(probe_accuracy))
    map_results = {
        str(checkpoint): _response_geometry(
            fingerprints[checkpoint], minimum_probe_accuracy)
        for checkpoint in CHECKPOINT_LAYERS
    }
    map_verdict = map_results[str(PRIMARY_CHECKPOINT)]["verdict"]
    overall = _overall_decision(dose_verdict, map_verdict)

    result = {
        "stage": "delta_context_geometry_width_screen",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "self_check": self_check,
        "rows": rows,
        "basis_metadata": basis_metadata,
        "probe_step": probe_step,
        "response_map_npz_sha256": response_npz_sha,
        "dose_results": dose_results,
        "dose_adjudication": {
            "smooth_families": smooth_families,
            "gated_families": gated_families,
            "verdict": dose_verdict,
        },
        "response_map_adjudication": map_results,
        "primary_map_verdict": map_verdict,
        "verdict": overall,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_context_geometry_width_screen_{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"CONTEXT-GEOMETRY-WIDTH-SCREEN verdict={overall} "
        f"dose={dose_verdict} maps={map_verdict} "
        f"smooth={smooth_families} gated={gated_families} "
        f"artifact={path}")
    return result
