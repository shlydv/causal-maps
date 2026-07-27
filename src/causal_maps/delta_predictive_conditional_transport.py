"""Prospective test of state-conditioned, non-oracle causal transport.

The held-out target operation state is never an input to prediction.  Each
leave-one-family-out predictor receives only the origin L21 answer-prefix
state and is frozen before target-family counterpart states or interventions
are evaluated.
"""
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
    _direction_pass,
    _direction_summary,
    _position_groups,
    _row_transport,
    _run_exact_patch,
    _unused_rows,
)
from .delta_heterogeneous_family_screen import (
    FAMILY_ORDER,
    FAMILY_SPECS,
    VALUES,
    _family_alignment,
)
from .delta_prospective_causal_sensitivity import _prospective_rows
from .logutil import Heartbeat, log
from .model_utils import (
    get_decoder_layers,
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)
from .patching import _split_output


PROTOCOL_VERSION = "2026-07-27-p2-predictive-conditional-transport-v1"
LAYER = 21
TRAIN_N = 24
VALIDATION_N = 8
TEST_N = 12
DIRECTIONS = ("belief_to_search", "search_to_belief")
RANKS = (1, 3, 8, 16, 32)
RIDGES = (0.01, 0.1, 1.0)
RANDOM_SEED = 330071

MINIMUM_EXACT_FAMILIES = 6
MINIMUM_PREDICTED_FAMILIES = 6
MINIMUM_PREDICTED_PROGRESS = 0.40
MINIMUM_RECOVERY_OF_EXACT = 0.50
MINIMUM_CONDITIONAL_GAIN = 0.10
MINIMUM_POSITIVE_ROWS = 18
MAXIMUM_MEDIAN_DISTANCE = 0.85
MINIMUM_VALUE_ACCURACY = 0.80


PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "hypothesis": (
        "The BELIEF/SEARCH operation displacement is not a global vector. "
        "It is a low-complexity function of the current hidden state. A "
        "donor-only reduced-rank predictor can therefore infer a causal "
        "operation displacement for a wholly held-out computation family "
        "without observing that row's opposite-operation activation."
    ),
    "model": "Qwen2.5-7B-Instruct, 8-bit",
    "families": list(FAMILY_ORDER),
    "leave_one_family_out": True,
    "locus": {
        "layer": LAYER,
        "positions": "three answer-prefix command tokens",
        "checkpoints": list(CHECKPOINT_LAYERS),
    },
    "rows": {
        "donor_train": TRAIN_N,
        "donor_validation": VALIDATION_N,
        "heldout_test": TEST_N,
        "heldout_test_pairs_disjoint_from_donor_pairs": True,
        "two_test_distractor_variants_per_directed_pair": True,
    },
    "predictor": {
        "input": "origin-operation L21 state only",
        "target": "opposite-minus-origin L21 displacement",
        "family_target_data_used_for_fit_or_selection": False,
        "model": "reduced-rank linear regression on origin-state PCs",
        "ranks": list(RANKS),
        "ridge_values": list(RIDGES),
        "selection": (
            "minimum donor-validation normalized displacement MSE, with "
            "rank then ridge as deterministic tie-breakers; the selected "
            "model is refit on donor train plus validation rows"
        ),
        "predictions_hashed_before_target_counterparts_and_interventions":
            True,
    },
    "causal_arms": [
        "exact counterpart state (oracle upper bound)",
        "selected conditional predictor",
        "global mean displacement",
        "global target-state centroid",
        "nearest donor in predictor feature space",
        "row-shuffled conditional prediction",
        "conditional displacement at instruction positions",
        "conditional displacement at matched identical-token positions",
    ],
    "per_family_primary_gate": {
        "exact_reference_passes_both_directions": True,
        "minimum_predicted_progress_each_direction":
            MINIMUM_PREDICTED_PROGRESS,
        "minimum_recovery_of_exact_each_direction":
            MINIMUM_RECOVERY_OF_EXACT,
        "minimum_gain_over_mean_and_centroid":
            MINIMUM_CONDITIONAL_GAIN,
        "minimum_gain_over_row_shuffle":
            MINIMUM_CONDITIONAL_GAIN,
        "minimum_positive_rows_each_direction": MINIMUM_POSITIVE_ROWS,
        "maximum_median_target_distance_each_direction":
            MAXIMUM_MEDIAN_DISTANCE,
        "minimum_value_accuracy_each_direction":
            MINIMUM_VALUE_ACCURACY,
        "instruction_and_identical_controls": (
            "each below max(0.10, half the predicted score)"
        ),
    },
    "overall_gate": {
        "minimum_exact_reference_families": MINIMUM_EXACT_FAMILIES,
        "minimum_predicted_families": MINIMUM_PREDICTED_FAMILIES,
    },
    "stopping_rule": (
        "Adjudicate this frozen test without prompt, split, rank, ridge, "
        "threshold, or locus rescue. Cross-wording/model replication is "
        "licensed only by a positive or clearly partial predictive result."
    ),
    "random_seed": RANDOM_SEED,
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest().upper()


def _row_splits():
    donor = _prospective_rows(VALUES)
    if len(donor) < TRAIN_N + VALIDATION_N:
        raise AssertionError("insufficient frozen donor rows")
    train = donor[:TRAIN_N]
    validation = donor[TRAIN_N:TRAIN_N + VALIDATION_N]
    test = _unused_rows()
    train_pairs = {(row["source"], row["target"]) for row in train}
    validation_pairs = {
        (row["source"], row["target"]) for row in validation
    }
    test_pairs = {(row["source"], row["target"]) for row in test}
    if train_pairs & validation_pairs:
        raise AssertionError("train and validation directed pairs overlap")
    if (train_pairs | validation_pairs) & test_pairs:
        raise AssertionError("donor and test directed pairs overlap")
    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


@torch.no_grad()
def _capture_positions(model, batch, positions):
    cache = {}

    def hook(_module, _args, output):
        states, _rebuild = _split_output(output)
        cache["states"] = (
            states[:, positions, :].detach().float().cpu())

    handle = get_decoder_layers(model)[LAYER].register_forward_hook(hook)
    try:
        model(
            input_ids=batch["ids"], attention_mask=batch["am"],
            use_cache=False)
    finally:
        handle.remove()
    return cache["states"]


def _direction_xy(captured, direction):
    belief = torch.cat([
        captured[history]["belief"] for history in range(2)
    ], dim=0).flatten(1).float()
    search = torch.cat([
        captured[history]["search"] for history in range(2)
    ], dim=0).flatten(1).float()
    if direction == "belief_to_search":
        return belief, search - belief, search
    if direction == "search_to_belief":
        return search, belief - search, belief
    raise KeyError(direction)


def _fit_low_rank(x, y, rank, ridge):
    """Fit a deterministic reduced-rank linear displacement predictor."""
    x = x.float()
    y = y.float()
    x_mean = x.mean(dim=0)
    y_mean = y.mean(dim=0)
    xc = x - x_mean
    yc = y - y_mean
    gram = xc @ xc.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    usable = int(min(
        int(rank), x.shape[0] - 1,
        int((eigenvalues > 1e-8).sum())))
    if usable < 1:
        raise ValueError("origin-state design has zero usable rank")
    singular = eigenvalues[:usable].clamp_min(1e-8).sqrt()
    left = eigenvectors[:, :usable]
    basis = (left.T @ xc) / singular[:, None]
    scale = singular / max(1.0, float(np.sqrt(x.shape[0] - 1)))
    z = (xc @ basis.T) / scale.clamp_min(1e-8)
    eye = torch.eye(usable, dtype=z.dtype)
    beta = torch.linalg.solve(
        z.T @ z + float(ridge) * eye,
        z.T @ yc)
    return {
        "rank": usable,
        "ridge": float(ridge),
        "x_mean": x_mean,
        "x_basis": basis,
        "x_scale": scale,
        "y_mean": y_mean,
        "beta": beta,
    }


def _input_decomposition(x, maximum_rank):
    """Compute the expensive origin-state eigensystem once."""
    x = x.float()
    x_mean = x.mean(dim=0)
    xc = x - x_mean
    gram = xc @ xc.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    usable = int(min(
        int(maximum_rank), x.shape[0] - 1,
        int((eigenvalues > 1e-8).sum())))
    if usable < 1:
        raise ValueError("origin-state design has zero usable rank")
    singular = eigenvalues[:usable].clamp_min(1e-8).sqrt()
    basis = (
        eigenvectors[:, :usable].T @ xc
        / singular[:, None])
    scale = singular / max(
        1.0, float(np.sqrt(x.shape[0] - 1)))
    return {
        "x_mean": x_mean,
        "xc": xc,
        "basis": basis,
        "scale": scale,
        "usable": usable,
    }


def _features(predictor, x):
    return (
        (x.float() - predictor["x_mean"])
        @ predictor["x_basis"].T
        / predictor["x_scale"].clamp_min(1e-8)
    )


def _predict_low_rank(predictor, origin_states):
    """Predict from origin states only; no counterpart argument exists."""
    return predictor["y_mean"] + _features(
        predictor, origin_states) @ predictor["beta"]


def _prediction_metrics(predicted, target, mean_baseline):
    residual = target.float() - predicted.float()
    centered = target.float() - mean_baseline.float()
    nmse = float(
        residual.square().sum()
        / centered.square().sum().clamp_min(1e-8))
    row_cosine = torch.nn.functional.cosine_similarity(
        predicted.float(), target.float(), dim=1)
    return {
        "normalized_mse": nmse,
        "mean_row_cosine": float(row_cosine.mean()),
        "median_row_cosine": float(row_cosine.median()),
    }


def _select_predictor(train_x, train_y, validation_x, validation_y):
    mean = train_y.mean(dim=0, keepdim=True)
    decomposition = _input_decomposition(train_x, max(RANKS))
    y_mean = train_y.float().mean(dim=0)
    yc = train_y.float() - y_mean
    candidates = []
    models = {}
    for rank in RANKS:
        for ridge in RIDGES:
            usable = min(int(rank), decomposition["usable"])
            basis = decomposition["basis"][:usable]
            scale = decomposition["scale"][:usable]
            z = (
                decomposition["xc"] @ basis.T
                / scale.clamp_min(1e-8))
            beta = torch.linalg.solve(
                z.T @ z + float(ridge) * torch.eye(
                    usable, dtype=z.dtype),
                z.T @ yc)
            predictor = {
                "rank": usable,
                "ridge": float(ridge),
                "x_mean": decomposition["x_mean"],
                "x_basis": basis,
                "x_scale": scale,
                "y_mean": y_mean,
                "beta": beta,
            }
            predicted = _predict_low_rank(predictor, validation_x)
            metrics = _prediction_metrics(
                predicted, validation_y, mean)
            key = (int(rank), float(ridge))
            models[key] = predictor
            candidates.append({
                "requested_rank": int(rank),
                "effective_rank": int(predictor["rank"]),
                "ridge": float(ridge),
                **metrics,
            })
    selected = min(
        candidates,
        key=lambda value: (
            value["normalized_mse"],
            value["requested_rank"],
            value["ridge"],
        ))
    predictor = models[(
        selected["requested_rank"], selected["ridge"])]
    return predictor, {
        "selected": selected,
        "candidates": candidates,
        "mean_baseline_validation": _prediction_metrics(
            mean.expand_as(validation_y), validation_y, mean),
    }


def _nearest_displacement(predictor, train_x, train_y, test_x):
    train_features = _features(predictor, train_x)
    test_features = _features(predictor, test_x)
    indices = torch.cdist(
        test_features, train_features).argmin(dim=1)
    return train_y[indices], indices.tolist()


def _derange_rows(tensor):
    if tensor.shape[0] < 2:
        raise ValueError("cannot derange fewer than two predictions")
    return torch.roll(tensor, shifts=1, dims=0)


def _predictor_public_metadata(predictor, selection, train_x, train_y):
    return {
        "selection": selection,
        "n_training_rows": int(train_x.shape[0]),
        "input_dimension": int(train_x.shape[1]),
        "output_dimension": int(train_y.shape[1]),
        "effective_rank": int(predictor["rank"]),
        "ridge": float(predictor["ridge"]),
        "mean_displacement_norm": float(
            predictor["y_mean"].norm()),
    }


def _tensor_digest(tensor):
    array = tensor.detach().contiguous().float().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest().upper()


def _predictor_digest(predictor):
    digest = hashlib.sha256()
    for key in ("x_mean", "x_basis", "x_scale", "y_mean", "beta"):
        array = (
            predictor[key].detach().contiguous().float().numpy())
        digest.update(key.encode())
        digest.update(array.tobytes())
    digest.update(str(predictor["rank"]).encode())
    digest.update(str(predictor["ridge"]).encode())
    return digest.hexdigest().upper()


def _predicted_direction_pass(summary, exact_summary):
    recovery = (
        summary["mean_progress"]
        / max(1e-8, exact_summary["mean_progress"]))
    passed = bool(
        summary["mean_progress"]
        >= MINIMUM_PREDICTED_PROGRESS - 1e-9
        and recovery >= MINIMUM_RECOVERY_OF_EXACT - 1e-9
        and summary["positive_rows"] >= MINIMUM_POSITIVE_ROWS
        and summary["median_distance_ratio"]
        <= MAXIMUM_MEDIAN_DISTANCE + 1e-9
        and summary["minimum_answer_accuracy"]
        >= MINIMUM_VALUE_ACCURACY - 1e-9)
    return passed, float(recovery)


def _arm_score(arm):
    return min(
        arm["belief_to_search"]["mean_progress"],
        arm["search_to_belief"]["mean_progress"])


def _family_adjudication(arms):
    exact_pass = all(
        _direction_pass(arms["exact"][direction])
        for direction in DIRECTIONS)
    predicted_direction = {}
    recovery = {}
    for direction in DIRECTIONS:
        passed, value = _predicted_direction_pass(
            arms["conditional"][direction],
            arms["exact"][direction])
        predicted_direction[direction] = passed
        recovery[direction] = value
    scores = {
        arm: _arm_score(value)
        for arm, value in arms.items()
    }
    conditional = scores["conditional"]
    global_baseline = max(
        scores["mean_displacement"],
        scores["target_centroid"])
    conditional_gain = conditional - global_baseline
    shuffle_gain = conditional - scores["row_shuffled"]
    locus_limit = max(0.10, 0.5 * conditional)
    locus_specific = bool(
        scores["instruction_control"] < locus_limit
        and scores["identical_control"] < locus_limit)
    predicted_pass = bool(
        exact_pass
        and all(predicted_direction.values())
        and conditional_gain >= MINIMUM_CONDITIONAL_GAIN - 1e-9
        and shuffle_gain >= MINIMUM_CONDITIONAL_GAIN - 1e-9
        and locus_specific)
    exemplar_pass = bool(
        exact_pass
        and scores["nearest_neighbor"]
        >= MINIMUM_PREDICTED_PROGRESS - 1e-9
        and scores["nearest_neighbor"] - global_baseline
        >= MINIMUM_CONDITIONAL_GAIN - 1e-9)
    global_pass = bool(
        exact_pass
        and global_baseline
        >= MINIMUM_PREDICTED_PROGRESS - 1e-9)
    return {
        "exact_pass": exact_pass,
        "predicted_direction_pass": predicted_direction,
        "recovery_of_exact": recovery,
        "scores": scores,
        "conditional_gain_over_best_global": conditional_gain,
        "conditional_gain_over_row_shuffle": shuffle_gain,
        "locus_specific": locus_specific,
        "predicted_pass": predicted_pass,
        "exemplar_pass": exemplar_pass,
        "global_pass": global_pass,
    }


def _overall_adjudication(families):
    exact = [
        name for name, value in families.items()
        if value["exact_pass"]
    ]
    predicted = [
        name for name, value in families.items()
        if value["predicted_pass"]
    ]
    exemplar = [
        name for name, value in families.items()
        if value["exemplar_pass"]
    ]
    global_template = [
        name for name, value in families.items()
        if value["global_pass"]
    ]
    controls_failed = [
        name for name, value in families.items()
        if not value["locus_specific"]
    ]
    if len(exact) < MINIMUM_EXACT_FAMILIES:
        verdict = "EXACT_REFERENCE_NOT_GENERAL"
    elif controls_failed:
        verdict = "NONSPECIFIC_PREDICTED_TRANSPORT"
    elif len(predicted) >= MINIMUM_PREDICTED_FAMILIES:
        verdict = "PREDICTABLE_STATE_CONDITIONED_TRANSPORT"
    elif len(predicted) >= 3:
        verdict = "PARTIAL_STATE_CONDITIONED_TRANSPORT"
    elif len(exemplar) >= MINIMUM_PREDICTED_FAMILIES:
        verdict = "EXEMPLAR_CONDITIONAL_TRANSPORT"
    elif len(global_template) >= MINIMUM_PREDICTED_FAMILIES:
        verdict = "GLOBAL_TEMPLATE_TRANSPORT"
    else:
        verdict = "ORACLE_ONLY_STATE_TRANSPORT"
    return {
        "exact_reference_families": exact,
        "predicted_families": predicted,
        "exemplar_families": exemplar,
        "global_template_families": global_template,
        "control_failure_families": controls_failed,
        "verdict": verdict,
    }


def _self_check():
    splits = _row_splits()
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    x = torch.randn(48, 12, generator=generator)
    weights = torch.randn(12, 9, generator=generator)
    y = x @ weights
    predictor = _fit_low_rank(
        x[:36], y[:36], rank=12, ridge=0.01)
    predicted = _predict_low_rank(predictor, x[36:])
    relative_error = float(
        (predicted - y[36:]).square().sum()
        / y[36:].square().sum().clamp_min(1e-8))
    deranged = _derange_rows(torch.arange(30).reshape(10, 3))
    fixed_rows = int(torch.all(
        deranged == torch.arange(30).reshape(10, 3), dim=1).sum())
    passed = bool(
        relative_error < 1e-4
        and fixed_rows == 0
        and len(splits["test"]) == TEST_N)
    if not passed:
        raise AssertionError(
            "predictive conditional transport self-check failed")
    return {
        "relative_linear_recovery_error": relative_error,
        "derangement_fixed_rows": fixed_rows,
        "test_rows": len(splits["test"]),
        "pass": passed,
    }


@torch.no_grad()
def run_delta_predictive_conditional_transport(
        model_path, out_dir,
        model_key="qwen7b_predictive_conditional_transport",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=12, self_test_only=False):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_world) != TEST_N:
        raise ValueError("v1 is frozen to exactly 12 held-out histories")
    self_check = _self_check()
    if self_test_only:
        result = {
            "stage": "delta_predictive_conditional_transport",
            "protocol_sha256": PROTOCOL_SHA256,
            "self_check": self_check,
            "verdict": "SELF_CHECK_PASS",
        }
        path = os.path.join(
            out_dir, "predictive_conditional_transport_self_check.json")
        with open(path, "w") as handle:
            json.dump(result, handle, indent=2)
        log(
            f"PREDICTIVE-CONDITIONAL-TRANSPORT self-check pass "
            f"protocol={PROTOCOL_SHA256}")
        return result
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    if max(CHECKPOINT_LAYERS) >= model_num_hidden_layers(model):
        raise ValueError("frozen checkpoint layer is absent")
    rows = _row_splits()

    # Donor capture uses no held-out test histories.
    donor_data = {}
    donor_hb = Heartbeat(
        len(FAMILY_ORDER) * 2 * 2 * 2,
        "predictive_transport_donor_capture",
        every_sec=30, out_dir=out_dir)
    for family in FAMILY_ORDER:
        spec = {**FAMILY_SPECS[family], "values": VALUES}
        donor_data[family] = {"spec": spec}
        for split in ("train", "validation"):
            alignment = _family_alignment(
                tok, dev, rows[split], spec)
            captured = {}
            for history, (belief, search) in enumerate(
                    alignment["batches"]):
                captured[history] = {}
                for operation, batch in (
                        ("belief", belief), ("search", search)):
                    captured[history][operation] = _capture_positions(
                        model, batch, alignment["answer_positions"])
                    donor_hb.step(
                        extra=f"{family}/{split}/{history}/{operation}")
            donor_data[family][split] = {
                "alignment": alignment,
                "captured": captured,
            }
    donor_hb.done()

    predictors = {}
    predictor_metadata = {}
    for target_family in FAMILY_ORDER:
        donors = [
            family for family in FAMILY_ORDER
            if family != target_family
        ]
        predictors[target_family] = {}
        predictor_metadata[target_family] = {
            "donor_families": donors,
            "target_family_excluded": target_family not in donors,
        }
        for direction in DIRECTIONS:
            train_sets = [
                _direction_xy(
                    donor_data[family]["train"]["captured"],
                    direction)
                for family in donors
            ]
            validation_sets = [
                _direction_xy(
                    donor_data[family]["validation"]["captured"],
                    direction)
                for family in donors
            ]
            train_x = torch.cat([value[0] for value in train_sets])
            train_y = torch.cat([value[1] for value in train_sets])
            train_target = torch.cat(
                [value[2] for value in train_sets])
            validation_x = torch.cat(
                [value[0] for value in validation_sets])
            validation_y = torch.cat(
                [value[1] for value in validation_sets])
            _selection_model, selection = _select_predictor(
                train_x, train_y, validation_x, validation_y)
            validation_target = torch.cat(
                [value[2] for value in validation_sets])
            final_x = torch.cat([train_x, validation_x])
            final_y = torch.cat([train_y, validation_y])
            final_target = torch.cat(
                [train_target, validation_target])
            predictor = _fit_low_rank(
                final_x, final_y,
                selection["selected"]["requested_rank"],
                selection["selected"]["ridge"])
            predictors[target_family][direction] = {
                "model": predictor,
                "train_x": final_x,
                "train_y": final_y,
                "mean_displacement": final_y.mean(dim=0),
                "target_centroid": final_target.mean(dim=0),
            }
            predictor_metadata[target_family][direction] = (
                _predictor_public_metadata(
                    predictor, selection, final_x, final_y))
            predictor_metadata[target_family][direction][
                "frozen_predictor_sha256"] = _predictor_digest(predictor)

    # Predict each direction sequentially. The opposite target is not
    # captured or present in this call path.
    test_alignments = {}
    predictions = {}
    prediction_arrays = {}
    prediction_hb = Heartbeat(
        len(FAMILY_ORDER) * len(DIRECTIONS) * 2,
        "predictive_transport_source_only_prediction",
        every_sec=30, out_dir=out_dir)
    for family_index, family in enumerate(FAMILY_ORDER):
        spec = donor_data[family]["spec"]
        alignment = _family_alignment(
            tok, dev, rows["test"], spec)
        test_alignments[family] = alignment
        predictions[family] = {}
        for direction in DIRECTIONS:
            operation = (
                "belief" if direction == "belief_to_search"
                else "search")
            origin_parts = []
            for history, pair in enumerate(alignment["batches"]):
                batch = pair[0] if operation == "belief" else pair[1]
                origin_parts.append(_capture_positions(
                    model, batch, alignment["answer_positions"]))
                prediction_hb.step(
                    extra=f"{family}/{direction}/history_{history}")
            origin = torch.cat(origin_parts).flatten(1).float()
            fit = predictors[family][direction]
            conditional = _predict_low_rank(
                fit["model"], origin)
            nearest, nearest_indices = _nearest_displacement(
                fit["model"], fit["train_x"], fit["train_y"], origin)
            mean = fit["mean_displacement"][None, :].expand_as(
                conditional)
            centroid_delta = (
                fit["target_centroid"][None, :] - origin)
            shuffled = _derange_rows(conditional)
            shape = (
                origin.shape[0], len(alignment["answer_positions"]), -1)
            predictions[family][direction] = {
                "conditional": conditional.reshape(shape),
                "mean_displacement": mean.reshape(shape),
                "target_centroid": centroid_delta.reshape(shape),
                "nearest_neighbor": nearest.reshape(shape),
                "row_shuffled": shuffled.reshape(shape),
                "nearest_indices": nearest_indices,
            }
            prediction_arrays[f"{family}_{direction}_conditional"] = (
                conditional.reshape(shape).numpy().astype(np.float16))
            predictor_metadata[family][direction][
                "test_prediction_sha256"] = _tensor_digest(conditional)
            predictor_metadata[family][direction][
                "test_prediction_norm_mean"] = float(
                    conditional.norm(dim=1).mean())
            del origin, origin_parts
    prediction_hb.done()

    prediction_npz = os.path.join(
        out_dir, f"predictive_transport_predictions_{model_key}.npz")
    np.savez_compressed(prediction_npz, **prediction_arrays)
    with open(prediction_npz, "rb") as handle:
        prediction_npz_sha = hashlib.sha256(
            handle.read()).hexdigest().upper()
    prediction_json = os.path.join(
        out_dir, f"predictive_transport_freeze_{model_key}.json")
    with open(prediction_json, "w") as handle:
        json.dump({
            "protocol_sha256": PROTOCOL_SHA256,
            "statement": (
                "Frozen before counterpart test-state capture and before "
                "any test-family causal intervention."),
            "row_splits": rows,
            "predictors": predictor_metadata,
            "prediction_npz_sha256": prediction_npz_sha,
        }, handle, indent=2, default=float)
    with open(prediction_json, "rb") as handle:
        prediction_json_sha = hashlib.sha256(
            handle.read()).hexdigest().upper()
    log(
        f"FROZEN predictive transport sha256={prediction_json_sha} "
        f"arrays={prediction_npz_sha}")

    # Counterpart states are captured only after the prediction freeze.
    arm_names = (
        "exact",
        "conditional",
        "mean_displacement",
        "target_centroid",
        "nearest_neighbor",
        "row_shuffled",
        "instruction_control",
        "identical_control",
    )
    causal_hb = Heartbeat(
        len(FAMILY_ORDER) * len(arm_names)
        * len(DIRECTIONS) * 2,
        "predictive_conditional_transport_causal_test",
        every_sec=30, out_dir=out_dir)
    family_results = {}
    for family_index, family in enumerate(FAMILY_ORDER):
        alignment = test_alignments[family]
        groups = _position_groups(
            alignment, RANDOM_SEED + 1009 * family_index)
        union_positions = sorted({
            position for positions in groups.values()
            for position in positions
        })
        union_index = {
            position: index
            for index, position in enumerate(union_positions)
        }
        answer_offsets = [
            union_index[position]
            for position in groups["answer_prefix_3"]
        ]
        instruction_offsets = [
            union_index[position]
            for position in groups["instruction_3"]
        ]
        identical_positions = groups["identical_control_3"]
        identical_offsets = [
            union_index[position] for position in identical_positions
        ]
        baselines = {}
        for history, (belief, search) in enumerate(
                alignment["batches"]):
            for operation, batch in (
                    ("belief", belief), ("search", search)):
                baselines[(history, operation)] = _capture_baseline(
                    model, batch, LAYER, union_positions)

        arm_rows = {
            arm: {
                direction: {
                    "progress": {checkpoint: [] for checkpoint
                                 in CHECKPOINT_LAYERS},
                    "distance": {checkpoint: [] for checkpoint
                                 in CHECKPOINT_LAYERS},
                    "accuracy": [],
                }
                for direction in DIRECTIONS
            }
            for arm in arm_names
        }
        for direction in DIRECTIONS:
            origin_operation, target_operation = (
                ("belief", "search")
                if direction == "belief_to_search"
                else ("search", "belief"))
            for history, pair in enumerate(alignment["batches"]):
                batch = pair[0] if origin_operation == "belief" else pair[1]
                expected = (
                    [row["source"] for row in rows["test"]]
                    if history == 0
                    else [row["target"] for row in rows["test"]])
                start = history * TEST_N
                stop = start + TEST_N
                origin_answer = baselines[
                    (history, origin_operation)
                ]["source"][:, answer_offsets, :]
                target_answer = baselines[
                    (history, target_operation)
                ]["source"][:, answer_offsets, :]
                origin_instruction = baselines[
                    (history, origin_operation)
                ]["source"][:, instruction_offsets, :]
                origin_identical = baselines[
                    (history, origin_operation)
                ]["source"][:, identical_offsets, :]
                conditional_delta = predictions[family][direction][
                    "conditional"][start:stop]
                patches = {
                    "exact": (
                        groups["answer_prefix_3"], target_answer),
                    "conditional": (
                        groups["answer_prefix_3"],
                        origin_answer + conditional_delta),
                    "mean_displacement": (
                        groups["answer_prefix_3"],
                        origin_answer + predictions[family][direction][
                            "mean_displacement"][start:stop]),
                    "target_centroid": (
                        groups["answer_prefix_3"],
                        origin_answer + predictions[family][direction][
                            "target_centroid"][start:stop]),
                    "nearest_neighbor": (
                        groups["answer_prefix_3"],
                        origin_answer + predictions[family][direction][
                            "nearest_neighbor"][start:stop]),
                    "row_shuffled": (
                        groups["answer_prefix_3"],
                        origin_answer + predictions[family][direction][
                            "row_shuffled"][start:stop]),
                    "instruction_control": (
                        groups["instruction_3"],
                        origin_instruction + conditional_delta),
                    "identical_control": (
                        identical_positions,
                        origin_identical + conditional_delta),
                }
                for arm, (positions, target_values) in patches.items():
                    patched = _run_exact_patch(
                        model, batch, LAYER, positions, target_values)
                    arm_rows[arm][direction]["accuracy"].append(
                        _generic_accuracy(
                            patched["logits"], batch, expected, VALUES))
                    for checkpoint in CHECKPOINT_LAYERS:
                        progress, distance = _row_transport(
                            baselines[(history, origin_operation)][
                                f"checkpoint_{checkpoint}"],
                            baselines[(history, target_operation)][
                                f"checkpoint_{checkpoint}"],
                            patched[f"checkpoint_{checkpoint}"])
                        arm_rows[arm][direction]["progress"][
                            checkpoint].extend(progress)
                        arm_rows[arm][direction]["distance"][
                            checkpoint].extend(distance)
                    causal_hb.step(
                        extra=f"{family}/{arm}/{direction}/h{history}")

        arms = {}
        for arm in arm_names:
            arms[arm] = {}
            for direction in DIRECTIONS:
                by_checkpoint = {}
                for checkpoint in CHECKPOINT_LAYERS:
                    by_checkpoint[str(checkpoint)] = _direction_summary(
                        arm_rows[arm][direction]["progress"][checkpoint],
                        arm_rows[arm][direction]["distance"][checkpoint],
                        arm_rows[arm][direction]["accuracy"])
                # L27 is the direct terminal-state primary outcome; L24 is
                # retained to test whether transport appears earlier.
                arms[arm][direction] = {
                    **by_checkpoint["27"],
                    "by_checkpoint": by_checkpoint,
                }
        adjudication = _family_adjudication(arms)
        family_results[family] = {
            "positions": {
                "answer": groups["answer_prefix_3"],
                "instruction": groups["instruction_3"],
                "identical": identical_positions,
            },
            "arms": arms,
            "adjudication": adjudication,
        }
        log(
            f"PREDICTIVE-TRANSPORT {family} "
            f"exact={adjudication['scores']['exact']:.3f} "
            f"conditional={adjudication['scores']['conditional']:.3f} "
            f"global={max(adjudication['scores']['mean_displacement'], adjudication['scores']['target_centroid']):.3f} "
            f"shuffle={adjudication['scores']['row_shuffled']:.3f} "
            f"pass={adjudication['predicted_pass']}")
    causal_hb.done()

    family_adjudication = {
        family: value["adjudication"]
        for family, value in family_results.items()
    }
    overall = _overall_adjudication(family_adjudication)
    result = {
        "stage": "delta_predictive_conditional_transport",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "self_check": self_check,
        "prediction_freeze_sha256": prediction_json_sha,
        "prediction_npz_sha256": prediction_npz_sha,
        "row_splits": rows,
        "predictors": predictor_metadata,
        "family_results": family_results,
        "overall_adjudication": overall,
        "verdict": overall["verdict"],
    }
    path = os.path.join(
        out_dir,
        f"results_delta_predictive_conditional_transport_"
        f"{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"PREDICTIVE-CONDITIONAL-TRANSPORT verdict={result['verdict']} "
        f"exact={overall['exact_reference_families']} "
        f"predicted={overall['predicted_families']} "
        f"artifact={path}")
    return result
