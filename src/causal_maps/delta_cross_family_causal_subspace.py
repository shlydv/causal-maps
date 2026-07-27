"""Prospective held-out-family test of a shared causal control subspace."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from .delta_anchor_write import _resolve
from .delta_causal_rank_spectrum import (
    _component,
    _controller_basis,
    _natural_patch_pair,
)
from .delta_content_cancelled_controller import (
    _fixed_patch,
    _functional_score,
    _norm_matched_directions,
    _world_mediation,
)
from .delta_controller_matrix import (
    _controller_from_alignment,
    _evaluation_states,
)
from .delta_heterogeneous_family_screen import (
    FAMILY_ORDER,
    FAMILY_SPECS,
    VALUES,
    _family_alignment,
)
from .delta_operation_handoff_depth import _full_sites
from .delta_prospective_causal_sensitivity import (
    CONTROL_LAYERS,
    _correlation,
    _evaluate_arm,
    _prospective_rows,
    _scale_per_position,
    _spearman,
)
from .delta_sparse_transport import _attention_geometry
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)


PROTOCOL_VERSION = "2026-07-26-p2-cross-family-causal-subspace-v1"
SCREEN_N = 15
DONOR_N = 10
CALIBRATION_N = 10
TEST_N = 15
TOTAL_N = SCREEN_N + DONOR_N + CALIBRATION_N + TEST_N
RECONSTRUCTION_RANKS = (1, 2, 3, 7)
N_RANDOM = 3
RANDOM_SEED = 127031

# Geometry is allowed to stop the expensive causal phase, but no behavioral
# or route-gap threshold is used to select families.
MINIMUM_MEDIAN_RANK3_ENERGY = 0.25
MINIMUM_MEDIAN_RANK7_ENERGY = 0.50
MINIMUM_RANK7_FOLDS = 6
MINIMUM_PER_FOLD_RANK7_ENERGY = 0.25

MINIMUM_ORIENTATION_GAP = 0.010
MINIMUM_CAUSAL_SCORE = 0.015
MINIMUM_RECOVERY = 0.60
MINIMUM_POSITIVE_WORLDS = 10
MINIMUM_SUCCESSFUL_FAMILIES = 6
MINIMUM_POOLED_SPEARMAN = 0.50


PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "hypothesis": (
        "Natural answer-prefix controllers from heterogeneous computations "
        "span a shared low-dimensional causal subspace. A held-out family's "
        "coordinate, estimated on non-interventional calibration rows, "
        "predicts the sign and magnitude of route steering on independent "
        "test rows."),
    "families": list(FAMILY_ORDER),
    "rows": {
        "screen_rows_excluded": SCREEN_N,
        "donor_rows": DONOR_N,
        "calibration_rows": CALIBRATION_N,
        "causal_test_rows": TEST_N,
        "total_unique_directed_pairs": TOTAL_N,
    },
    "leave_one_family_out": True,
    "controller_locus": {
        "layer": 21,
        "positions": "three matched answer-prefix command positions",
        "contrast": "BELIEF minus X X SEARCH",
    },
    "route_assay": {
        "layer": 24,
        "orientation": (
            "Sign of SEARCH-minus-BELIEF route gap on calibration rows; "
            "frozen before test interventions."),
        "outcome": (
            "Bidirectional movement in that frozen orientation, reported "
            "continuously with no test-set route-gap eligibility cutoff."),
    },
    "geometry_gate": {
        "median_rank3_energy": MINIMUM_MEDIAN_RANK3_ENERGY,
        "median_rank7_energy": MINIMUM_MEDIAN_RANK7_ENERGY,
        "rank7_folds_at_or_above": {
            "energy": MINIMUM_PER_FOLD_RANK7_ENERGY,
            "count": MINIMUM_RANK7_FOLDS,
        },
    },
    "causal_arms": [
        "original",
        "exact natural answer-prefix interchange",
        "within-family calibration controller",
        *[
            f"held-out projection into donor rank {rank}"
            for rank in RECONSTRUCTION_RANKS
        ],
        "norm-matched donor mean",
        "orthogonal residual after donor rank 7",
        "rank-3 component at instruction positions",
        f"{N_RANDOM} norm-matched random directions",
    ],
    "decision": {
        "minimum_calibration_orientation_gap": MINIMUM_ORIENTATION_GAP,
        "minimum_causal_score": MINIMUM_CAUSAL_SCORE,
        "minimum_recovery_of_within_family": MINIMUM_RECOVERY,
        "minimum_positive_worlds": MINIMUM_POSITIVE_WORLDS,
        "minimum_successful_families": MINIMUM_SUCCESSFUL_FAMILIES,
        "minimum_pooled_geometry_effect_spearman":
            MINIMUM_POOLED_SPEARMAN,
        "reserved_random_null_confirmation": 19,
    },
    "random_seed": RANDOM_SEED,
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest().upper()


def _split_rows():
    """Use only directed pairs not seen by the eligibility screen."""
    rows = _prospective_rows(VALUES)
    if len(rows) != TOTAL_N:
        raise AssertionError("unexpected prospective row count")
    unseen = rows[SCREEN_N:]
    return {
        "donor": unseen[:DONOR_N],
        "calibration": unseen[DONOR_N:DONOR_N + CALIBRATION_N],
        "test": unseen[DONOR_N + CALIBRATION_N:],
    }


def _patch_pair(states, positions, displacement):
    return {
        "belief": _fixed_patch(
            states["belief"], positions, displacement, -1.0),
        "search": _fixed_patch(
            states["search"], positions, displacement, +1.0),
    }


def _public_geometry(fold):
    return {
        "donor_names": fold["donor_names"],
        "basis_geometry": fold["basis_geometry"],
        "rank_components": fold["rank_metadata"],
        "donor_mean": fold["donor_mean_metadata"],
        "orthogonal_residual": fold["residual_metadata"],
    }


def _geometry_adjudication(folds):
    rank3 = [
        fold["rank_metadata"]["3"]["fraction_of_controller_energy"]
        for fold in folds.values()
    ]
    rank7 = [
        fold["rank_metadata"]["7"]["fraction_of_controller_energy"]
        for fold in folds.values()
    ]
    passing_rank7 = sum(
        value >= MINIMUM_PER_FOLD_RANK7_ENERGY - 1e-9
        for value in rank7
    )
    median3 = float(np.median(rank3))
    median7 = float(np.median(rank7))
    passed = bool(
        median3 >= MINIMUM_MEDIAN_RANK3_ENERGY - 1e-9
        and median7 >= MINIMUM_MEDIAN_RANK7_ENERGY - 1e-9
        and passing_rank7 >= MINIMUM_RANK7_FOLDS)
    return {
        "rank3_energy_by_family": {
            name: float(fold["rank_metadata"]["3"][
                "fraction_of_controller_energy"])
            for name, fold in folds.items()
        },
        "rank7_energy_by_family": {
            name: float(fold["rank_metadata"]["7"][
                "fraction_of_controller_energy"])
            for name, fold in folds.items()
        },
        "median_rank3_energy": median3,
        "median_rank7_energy": median7,
        "rank7_folds_at_or_above_0.25": passing_rank7,
        "pass": passed,
    }


def _orientation_score(arm, original, predicted_search_minus_belief_sign):
    """Score movement toward the opposite operation using a frozen sign."""
    sign = int(predicted_search_minus_belief_sign)
    original_belief = float(original["values"]["belief"])
    original_search = float(original["values"]["search"])
    patched_belief = float(arm["values"]["belief"])
    patched_search = float(arm["values"]["search"])
    belief_to_search = sign * (patched_belief - original_belief)
    search_to_belief = -sign * (patched_search - original_search)

    original_world = {
        operation: _world_mediation(
            original["tasks"][operation], original["cells"][operation])
        for operation in ("belief", "search")
    }
    patched_world = {
        operation: _world_mediation(
            arm["tasks"][operation], arm["cells"][operation])
        for operation in ("belief", "search")
    }
    rows = []
    for index, (ob, os_, pb, ps) in enumerate(zip(
            original_world["belief"], original_world["search"],
            patched_world["belief"], patched_world["search"])):
        if any(value is None for value in (ob, os_, pb, ps)):
            rows.append({
                "world_offset": index,
                "belief_to_search": None,
                "search_to_belief": None,
                "both_predicted": False,
            })
            continue
        down = sign * (float(pb) - float(ob))
        up = -sign * (float(ps) - float(os_))
        rows.append({
            "world_offset": index,
            "belief_to_search": down,
            "search_to_belief": up,
            "both_predicted": bool(down > 0.0 and up > 0.0),
        })
    positive_worlds = sum(row["both_predicted"] for row in rows)
    functional = _functional_score(
        {
            "belief_to_search_movement": belief_to_search,
            "search_to_belief_movement": search_to_belief,
            "bidirectional_score": min(
                belief_to_search, search_to_belief),
        },
        [*original["tasks"].values(), *arm["tasks"].values()],
    )
    return {
        "predicted_search_minus_belief_sign": sign,
        "belief_to_search_movement": belief_to_search,
        "search_to_belief_movement": search_to_belief,
        "bidirectional_score": min(
            belief_to_search, search_to_belief),
        "positive_worlds": int(positive_worlds),
        "per_world": rows,
        "functional": functional["functional"],
        "functional_bidirectional_score":
            functional["functional_bidirectional_score"],
    }


def _family_adjudication(name, fold, calibration, causal):
    calibration_gap = float(
        calibration["values"]["search"]
        - calibration["values"]["belief"])
    predicted_sign = int(np.sign(calibration_gap))
    original = causal["arms"]["original"]
    test_gap = float(
        original["values"]["search"]
        - original["values"]["belief"])
    test_sign = int(np.sign(test_gap))
    scores = causal["scores"]
    reference = scores["natural_prefix_interchange"]
    within = scores["within_family_controller"]
    rank3 = scores["donor_rank_03"]
    rank7 = scores["donor_rank_07"]
    residual = scores["orthogonal_residual"]
    instruction = scores["instruction_position_rank3"]
    random_scores = [
        scores[f"random_direction_{index:02d}"][
            "functional_bidirectional_score"]
        for index in range(N_RANDOM)
        if scores[f"random_direction_{index:02d}"]["functional"]
    ]
    random_max = max(random_scores) if random_scores else None
    within_score = within["functional_bidirectional_score"]

    def recovery(score):
        return (
            score / within_score
            if within_score > 1e-8 else None)

    orientation_stable = bool(
        abs(calibration_gap) >= MINIMUM_ORIENTATION_GAP - 1e-9
        and predicted_sign != 0
        and predicted_sign == test_sign)
    reference_pass = bool(
        reference["functional"]
        and reference["functional_bidirectional_score"]
        >= MINIMUM_CAUSAL_SCORE - 1e-9
        and reference["positive_worlds"] >= MINIMUM_POSITIVE_WORLDS)
    within_pass = bool(
        within["functional"]
        and within_score >= MINIMUM_CAUSAL_SCORE - 1e-9
        and within["positive_worlds"] >= MINIMUM_POSITIVE_WORLDS)
    locus_specific = bool(
        instruction["functional"]
        and instruction["functional_bidirectional_score"]
        < max(0.01, 0.5 * rank3[
            "functional_bidirectional_score"]))
    random_specific = bool(
        len(random_scores) == N_RANDOM
        and random_max < rank3["functional_bidirectional_score"])
    residual_weaker = bool(
        residual["functional"]
        and residual["functional_bidirectional_score"]
        < rank3["functional_bidirectional_score"])
    source_state_unchanged = bool(
        scores["donor_rank_03"]["source_state_max_abs_change"] <= 1e-8
        and scores["donor_rank_07"][
            "source_state_max_abs_change"] <= 1e-8)

    rank3_recovery = recovery(rank3["functional_bidirectional_score"])
    rank7_recovery = recovery(rank7["functional_bidirectional_score"])
    rank3_pass = bool(
        orientation_stable and reference_pass and within_pass
        and rank3["functional"]
        and rank3["functional_bidirectional_score"]
        >= MINIMUM_CAUSAL_SCORE - 1e-9
        and rank3["positive_worlds"] >= MINIMUM_POSITIVE_WORLDS
        and rank3_recovery is not None
        and rank3_recovery >= MINIMUM_RECOVERY - 1e-9
        and locus_specific and random_specific and residual_weaker
        and source_state_unchanged)
    rank7_pass = bool(
        orientation_stable and reference_pass and within_pass
        and rank7["functional"]
        and rank7["functional_bidirectional_score"]
        >= MINIMUM_CAUSAL_SCORE - 1e-9
        and rank7["positive_worlds"] >= MINIMUM_POSITIVE_WORLDS
        and rank7_recovery is not None
        and rank7_recovery >= MINIMUM_RECOVERY - 1e-9
        and source_state_unchanged)
    return {
        "family": name,
        "calibration_search_minus_belief_gap": calibration_gap,
        "predicted_orientation_sign": predicted_sign,
        "test_search_minus_belief_gap": test_gap,
        "test_orientation_sign": test_sign,
        "orientation_stable": orientation_stable,
        "reference_pass": reference_pass,
        "within_family_pass": within_pass,
        "rank3_pass": rank3_pass,
        "rank7_pass": rank7_pass,
        "rank3_recovery_of_within_family": rank3_recovery,
        "rank7_recovery_of_within_family": rank7_recovery,
        "rank3_energy_fraction": fold["rank_metadata"]["3"][
            "fraction_of_controller_energy"],
        "rank7_energy_fraction": fold["rank_metadata"]["7"][
            "fraction_of_controller_energy"],
        "rank3_score": rank3["functional_bidirectional_score"],
        "rank7_score": rank7["functional_bidirectional_score"],
        "within_family_score": within_score,
        "residual_score":
            residual["functional_bidirectional_score"],
        "instruction_score":
            instruction["functional_bidirectional_score"],
        "random_scores": random_scores,
        "random_max": random_max,
        "locus_specific": locus_specific,
        "random_specific": random_specific,
        "residual_weaker_than_rank3": residual_weaker,
        "source_state_unchanged": source_state_unchanged,
    }


def _overall_adjudication(folds, family_results):
    rank3_passes = [
        name for name, value in family_results.items()
        if value["rank3_pass"]
    ]
    rank7_passes = [
        name for name, value in family_results.items()
        if value["rank7_pass"]
    ]
    assay_resolved = [
        name for name, value in family_results.items()
        if value["orientation_stable"]
        and value["reference_pass"]
        and value["within_family_pass"]
    ]
    residual_dominant = [
        name for name, value in family_results.items()
        if value["residual_score"] >= max(
            MINIMUM_CAUSAL_SCORE, value["rank3_score"])
    ]
    control_failures = [
        name for name, value in family_results.items()
        if not value["locus_specific"]
        or not value["random_specific"]
    ]
    predicted = []
    actual = []
    for name, fold in folds.items():
        for rank in RECONSTRUCTION_RANKS:
            predicted.append(float(
                fold["rank_metadata"][str(rank)][
                    "fraction_of_controller_energy"]))
            actual.append(float(
                family_results[name][
                    f"rank{rank}_score"]
                if rank in (3, 7)
                else family_results[name]["all_rank_scores"][str(rank)]))
    pooled_spearman = _spearman(predicted, actual)
    pooled_pearson = _correlation(predicted, actual)

    if len(assay_resolved) < MINIMUM_SUCCESSFUL_FAMILIES:
        verdict = "CAUSAL_ASSAY_OR_ORIENTATION_UNRESOLVED"
    elif (
            len(rank3_passes) >= MINIMUM_SUCCESSFUL_FAMILIES
            and pooled_spearman is not None
            and pooled_spearman >= MINIMUM_POOLED_SPEARMAN - 1e-9):
        verdict = "SHARED_LOW_RANK_CAUSAL_CONTROL"
    elif len(rank7_passes) >= MINIMUM_SUCCESSFUL_FAMILIES:
        verdict = "SHARED_HIGHER_RANK_CAUSAL_CONTROL"
    elif len(residual_dominant) >= 4:
        verdict = "TASK_CONDITIONED_CAUSAL_RESIDUALS"
    elif len(control_failures) >= 3:
        verdict = "LOCUS_OR_RANDOM_CONTROL_FAILURE"
    else:
        verdict = "NO_SHARED_CAUSAL_SUBSPACE"
    return {
        "assay_resolved_families": assay_resolved,
        "rank3_pass_families": rank3_passes,
        "rank7_pass_families": rank7_passes,
        "residual_dominant_families": residual_dominant,
        "control_failure_families": control_failures,
        "pooled_geometry_effect_spearman": pooled_spearman,
        "pooled_geometry_effect_pearson": pooled_pearson,
        "verdict": verdict,
    }


@torch.no_grad()
def run_delta_cross_family_causal_subspace(
        model_path, out_dir,
        model_key="qwen7b_cross_family_causal_subspace",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=50):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_world) != TOTAL_N:
        raise ValueError("v1 is frozen to exactly 50 unique row pairs")
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    if max(CONTROL_LAYERS) >= model_num_hidden_layers(model):
        raise ValueError("frozen control layers are absent")
    n_heads, head_dim = _attention_geometry(model)
    l24_sites = _full_sites((22, 23, 24), n_heads)
    rows = _split_rows()

    prepared = {}
    alignment_errors = {}
    for family in FAMILY_ORDER:
        spec = {**FAMILY_SPECS[family], "values": VALUES}
        try:
            alignments = {
                split: _family_alignment(tok, dev, split_rows, spec)
                for split, split_rows in rows.items()
            }
        except (AssertionError, ValueError) as exc:
            alignment_errors[family] = str(exc)
            continue
        if len({
                tuple(value["answer_positions"])
                for value in alignments.values()}) != 1:
            alignment_errors[family] = (
                "answer-prefix positions differ across splits")
            continue
        donor_controller, donor_samples = _controller_from_alignment(
            model, alignments["donor"])
        calibration_controller, calibration_samples = (
            _controller_from_alignment(
                model, alignments["calibration"]))
        prepared[family] = {
            "spec": spec,
            "alignments": alignments,
            "donor_controller": donor_controller,
            "donor_samples": donor_samples,
            "calibration_controller": calibration_controller,
            "calibration_samples": calibration_samples,
        }
        log(
            f"CROSS-FAMILY-GEOMETRY captured {family} "
            f"({len(prepared)}/{len(FAMILY_ORDER)})")

    if alignment_errors or len(prepared) != len(FAMILY_ORDER):
        result = {
            "stage": "delta_cross_family_causal_subspace",
            "protocol": PROTOCOL,
            "protocol_sha256": PROTOCOL_SHA256,
            "alignment_errors": alignment_errors,
            "prepared_families": list(prepared),
            "verdict": "TOKENIZATION_OR_ALIGNMENT_FAILURE",
        }
        path = os.path.join(
            out_dir,
            f"results_delta_cross_family_causal_subspace_"
            f"{model_key}.json")
        with open(path, "w") as handle:
            json.dump(result, handle, indent=2, default=float)
        return result

    folds = {}
    basis_arrays = {}
    for target_name, target in prepared.items():
        donors = {
            name: value["donor_controller"]
            for name, value in prepared.items()
            if name != target_name
        }
        donor_names, singular, basis, coefficients, geometry = (
            _controller_basis(donors))
        target_controller = target["calibration_controller"]
        rank_components = {}
        rank_metadata = {}
        for rank in RECONSTRUCTION_RANKS:
            component, metadata = _component(
                target_controller, basis, range(1, rank + 1))
            rank_components[rank] = component
            rank_metadata[str(rank)] = metadata
        donor_mean = torch.stack(list(donors.values())).mean(dim=0)
        donor_mean = _scale_per_position(
            donor_mean, target_controller)
        rank7 = rank_components[7]
        residual = target_controller.float() - rank7.float()
        folds[target_name] = {
            "donor_names": donor_names,
            "basis": basis,
            "basis_geometry": geometry,
            "rank_components": rank_components,
            "rank_metadata": rank_metadata,
            "donor_mean": donor_mean,
            "donor_mean_metadata": {
                "frobenius_norm": float(donor_mean.norm()),
                "cosine_with_target": float(
                    torch.dot(
                        donor_mean.flatten(),
                        target_controller.flatten().float())
                    / (
                        donor_mean.norm()
                        * target_controller.float().norm()
                    ).clamp_min(1e-8)),
            },
            "residual": residual,
            "residual_metadata": {
                "frobenius_norm": float(residual.norm()),
                "fraction_of_target_energy": float(
                    residual.square().sum()
                    / target_controller.float().square().sum().clamp_min(
                        1e-8)),
            },
        }
        basis_arrays[f"basis_{target_name}"] = basis.numpy()
        basis_arrays[f"singular_{target_name}"] = singular.numpy()
        basis_arrays[f"coefficients_{target_name}"] = coefficients.numpy()
        basis_arrays[f"target_{target_name}"] = (
            target_controller.numpy())

    geometry = _geometry_adjudication(folds)
    basis_path = os.path.join(
        out_dir, f"cross_family_basis_{model_key}.npz")
    np.savez(basis_path, **basis_arrays)
    with open(basis_path, "rb") as handle:
        basis_sha = hashlib.sha256(handle.read()).hexdigest().upper()
    geometry_path = os.path.join(
        out_dir, f"cross_family_geometry_{model_key}.json")
    geometry_artifact = {
        "protocol_sha256": PROTOCOL_SHA256,
        "statement": "Frozen before any test-row causal intervention.",
        "geometry": {
            name: _public_geometry(fold)
            for name, fold in folds.items()
        },
        "geometry_adjudication": geometry,
        "basis_npz_sha256": basis_sha,
    }
    with open(geometry_path, "w") as handle:
        json.dump(geometry_artifact, handle, indent=2, default=float)
    with open(geometry_path, "rb") as handle:
        geometry_sha = hashlib.sha256(
            handle.read()).hexdigest().upper()
    log(
        f"FROZEN cross-family geometry sha256={geometry_sha} "
        f"median_rank3={geometry['median_rank3_energy']:.3f} "
        f"median_rank7={geometry['median_rank7_energy']:.3f} "
        f"pass={geometry['pass']}")

    if not geometry["pass"]:
        result = {
            "stage": "delta_cross_family_causal_subspace",
            "protocol": PROTOCOL,
            "protocol_sha256": PROTOCOL_SHA256,
            "model_key": model_key,
            "model_path": model_path,
            "quantization": quantization,
            "geometry_artifact_sha256": geometry_sha,
            "basis_npz_sha256": basis_sha,
            "geometry": geometry_artifact["geometry"],
            "geometry_adjudication": geometry,
            "verdict": "GEOMETRY_GATE_FAILED",
        }
        path = os.path.join(
            out_dir,
            f"results_delta_cross_family_causal_subspace_"
            f"{model_key}.json")
        with open(path, "w") as handle:
            json.dump(result, handle, indent=2, default=float)
        log(f"CROSS-FAMILY-SUBSPACE verdict={result['verdict']}")
        return result

    # Freeze route orientation on calibration rows before test intervention.
    calibration_hb = Heartbeat(
        len(FAMILY_ORDER) * 2 * 4,
        "cross_family_calibration",
        every_sec=30, out_dir=out_dir)
    calibration_results = {}
    for family, data in prepared.items():
        alignment = data["alignments"]["calibration"]
        target = {
            "name": family,
            "spec": data["spec"],
            "alignment": alignment,
            "source": [row["source"] for row in rows["calibration"]],
            "target": [row["target"] for row in rows["calibration"]],
            "n_world": CALIBRATION_N,
        }
        arm = _evaluate_arm(
            model, target, None, l24_sites, head_dim,
            calibration_hb, "orientation")
        arm.pop("_private_contexts")
        calibration_results[family] = arm
    calibration_hb.done()

    predictions = {}
    for family, calibration in calibration_results.items():
        gap = float(
            calibration["values"]["search"]
            - calibration["values"]["belief"])
        predictions[family] = {
            "calibration_search_minus_belief_gap": gap,
            "predicted_orientation_sign": int(np.sign(gap)),
            "rank_energy": {
                str(rank): folds[family]["rank_metadata"][str(rank)][
                    "fraction_of_controller_energy"]
                for rank in RECONSTRUCTION_RANKS
            },
        }
    prediction_path = os.path.join(
        out_dir, f"cross_family_predictions_{model_key}.json")
    with open(prediction_path, "w") as handle:
        json.dump({
            "protocol_sha256": PROTOCOL_SHA256,
            "geometry_artifact_sha256": geometry_sha,
            "statement": (
                "Frozen before any test-row causal intervention."),
            "predictions": predictions,
        }, handle, indent=2, default=float)
    with open(prediction_path, "rb") as handle:
        prediction_sha = hashlib.sha256(
            handle.read()).hexdigest().upper()
    log(
        f"FROZEN cross-family predictions sha256={prediction_sha} "
        + " ".join(
            f"{name}={value['predicted_orientation_sign']:+d}"
            for name, value in predictions.items()))

    arm_count = 10 + N_RANDOM
    causal_hb = Heartbeat(
        len(FAMILY_ORDER) * arm_count * 2 * 4,
        "cross_family_causal_test",
        every_sec=30, out_dir=out_dir)
    causal_results = {}
    for family_index, (family, data) in enumerate(prepared.items()):
        fold = folds[family]
        alignment = data["alignments"]["test"]
        states = _evaluation_states(model, alignment)
        positions = alignment["answer_positions"]
        rank3 = fold["rank_components"][3]
        random_directions = _norm_matched_directions(
            rank3, n_random=N_RANDOM,
            seed=RANDOM_SEED + family_index * 1009)
        arm_patches = {
            "original": None,
            "natural_prefix_interchange": _natural_patch_pair(
                states, positions),
            "within_family_controller": _patch_pair(
                states, positions, data["calibration_controller"]),
            **{
                f"donor_rank_{rank:02d}": _patch_pair(
                    states, positions, fold["rank_components"][rank])
                for rank in RECONSTRUCTION_RANKS
            },
            "donor_mean": _patch_pair(
                states, positions, fold["donor_mean"]),
            "orthogonal_residual": _patch_pair(
                states, positions, fold["residual"]),
            "instruction_position_rank3": _patch_pair(
                states, alignment["instruction_positions"], rank3),
        }
        for index, direction in enumerate(random_directions):
            arm_patches[f"random_direction_{index:02d}"] = _patch_pair(
                states, positions, direction)
        if len(arm_patches) != arm_count:
            raise AssertionError("frozen causal arm count changed")

        target = {
            "name": family,
            "spec": data["spec"],
            "alignment": alignment,
            "source": [row["source"] for row in rows["test"]],
            "target": [row["target"] for row in rows["test"]],
            "n_world": TEST_N,
        }
        arm_results = {}
        original_sources = None
        for arm_name, patches in arm_patches.items():
            arm = _evaluate_arm(
                model, target, patches, l24_sites, head_dim,
                causal_hb, arm_name)
            private = arm.pop("_private_contexts")
            if arm_name == "original":
                original_sources = {
                    operation: {
                        "clean": private[operation][
                            "clean_source"].clone(),
                        "natural": private[operation][
                            "natural_source"].clone(),
                    }
                    for operation in ("belief", "search")
                }
                arm["source_state_max_abs_change"] = 0.0
            else:
                maximum = 0.0
                for operation in ("belief", "search"):
                    maximum = max(
                        maximum,
                        float((
                            private[operation]["clean_source"]
                            - original_sources[operation]["clean"]
                        ).abs().max()),
                        float((
                            private[operation]["natural_source"]
                            - original_sources[operation]["natural"]
                        ).abs().max()),
                    )
                arm["source_state_max_abs_change"] = maximum
            arm_results[arm_name] = arm

        original = arm_results["original"]
        predicted_sign = predictions[family][
            "predicted_orientation_sign"]
        scores = {}
        for arm_name, arm in arm_results.items():
            if arm_name == "original":
                continue
            scores[arm_name] = _orientation_score(
                arm, original, predicted_sign)
            scores[arm_name]["source_state_max_abs_change"] = arm[
                "source_state_max_abs_change"]
        causal_results[family] = {
            "test_rows": rows["test"],
            "positions": {
                "answer_prefix": positions,
                "instruction": alignment["instruction_positions"],
            },
            "arms": arm_results,
            "scores": scores,
        }
        log(
            f"CROSS-FAMILY {family} "
            f"rank3={scores['donor_rank_03']['functional_bidirectional_score']:+.5f} "
            f"rank7={scores['donor_rank_07']['functional_bidirectional_score']:+.5f} "
            f"within={scores['within_family_controller']['functional_bidirectional_score']:+.5f}")
    causal_hb.done()

    family_adjudication = {}
    for family in FAMILY_ORDER:
        result = _family_adjudication(
            family, folds[family],
            calibration_results[family],
            causal_results[family])
        result["all_rank_scores"] = {
            str(rank): causal_results[family]["scores"][
                f"donor_rank_{rank:02d}"][
                    "functional_bidirectional_score"]
            for rank in RECONSTRUCTION_RANKS
        }
        family_adjudication[family] = result
    overall = _overall_adjudication(
        folds, family_adjudication)

    result = {
        "stage": "delta_cross_family_causal_subspace",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "geometry_artifact_sha256": geometry_sha,
        "prediction_artifact_sha256": prediction_sha,
        "basis_npz_sha256": basis_sha,
        "geometry": {
            name: _public_geometry(fold)
            for name, fold in folds.items()
        },
        "geometry_adjudication": geometry,
        "calibration_results": calibration_results,
        "predictions": predictions,
        "causal_results": causal_results,
        "family_adjudication": family_adjudication,
        "overall_adjudication": overall,
        "verdict": overall["verdict"],
    }
    path = os.path.join(
        out_dir,
        f"results_delta_cross_family_causal_subspace_"
        f"{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"CROSS-FAMILY-CAUSAL-SUBSPACE verdict={result['verdict']} "
        f"rank3={overall['rank3_pass_families']} "
        f"rank7={overall['rank7_pass_families']} "
        f"artifact={path}")
    return result
