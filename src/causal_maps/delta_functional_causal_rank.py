"""Prospective functional causal-rank experiment.

This experiment does not try to reconstruct a target hidden state.  It asks
whether the downstream effect of an exact state difference is compressible in
a basis ordered by local downstream response rather than activation energy.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from .delta_anchor_write import _resolve
from .delta_causal_atlas import (
    _PredictionBoundary,
    _commands,
    _frame_alignment,
    _row_splits,
    _tail_probability,
)
from .delta_cross_domain_controller import _generic_accuracy
from .delta_exact_transplant_locus_diagnostic import (
    CHECKPOINT_LAYERS,
    _capture_baseline,
    _run_exact_patch,
)
from .delta_heterogeneous_family_screen import FAMILY_SPECS, VALUES
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)


PROTOCOL_VERSION = "2026-07-27-p2-functional-causal-rank-v1"
SOURCE_LAYER = 21
PRIMARY_CHECKPOINT = 27
PATCH_WIDTH = 3
FAMILIES = ("maximum_score", "two_hop_pointer")
FRAMES = ("epistemic", "search")
TRAIN_PANEL = "anchor"
SELECTION_PANEL = "synonym_a"
TEST_PANEL = "synonym_b"
TRAIN_N = 24
CALIBRATION_N = 4
SELECTION_N = 4
TEST_N = 12
MAX_BASIS_RANK = 47
RANK_GRID = (1, 2, 4, 8, 16, 32, 47)
MAXIMUM_LOW_RANK = 16
N_RANDOM = 19
RANDOM_SEED = 271828

MINIMUM_EXACT_PROGRESS = 0.45
MINIMUM_FULL_SPAN_PROGRESS = 0.35
MINIMUM_FULL_RECOVERY_OF_EXACT = 0.55
MINIMUM_CAUSAL_PROGRESS = 0.35
MINIMUM_CAUSAL_RECOVERY_OF_FULL = 0.80
MINIMUM_CAUSAL_RECOVERY_OF_EXACT = 0.50
MINIMUM_CAUSAL_MARGIN = 0.08
MINIMUM_POSITIVE_ROWS = 20
MINIMUM_ANSWER_ACCURACY = 0.80
MAXIMUM_POSITION_CONTROL_FRACTION = 0.50
MAXIMUM_CALIBRATION_EVEN_ODD_RATIO = 0.35
MAXIMUM_RANDOM_P = 0.05


PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "status": "prospective capacity test; target-informed test projections",
    "question": (
        "Is the downstream effect of an exact operation-state difference "
        "low-dimensional under a causal response metric even when ordinary "
        "activation geometry does not reconstruct the target state?"
    ),
    "interpretation_boundary": (
        "This is a functional-compression capacity test. Test projections "
        "may use the exact test difference after rank and causal modes are "
        "frozen. It does not claim source-only synthesis."
    ),
    "model": "Qwen2.5-7B-Instruct, 8-bit, Tesla T4",
    "families": list(FAMILIES),
    "family_choice": (
        "The two previously established exact-transplant diagnostic "
        "families, bracketing strong and weak causal-atlas outcomes."
    ),
    "frames": list(FRAMES),
    "panels": {
        "basis_and_metric": "BELIEF/SEARCH",
        "rank_selection": "THINK/FIND",
        "heldout_test": "KNOW/LOOK",
    },
    "world_split": {
        "basis_train_directed_pairs": TRAIN_N,
        "metric_calibration_directed_pairs": CALIBRATION_N,
        "rank_selection_directed_pairs": SELECTION_N,
        "heldout_test_histories": TEST_N,
        "all_directed_pair_sets_disjoint": True,
    },
    "locus": {
        "source_layer": SOURCE_LAYER,
        "positions": "final three position-matched answer-command tokens",
        "checkpoints": list(CHECKPOINT_LAYERS),
        "primary_checkpoint": PRIMARY_CHECKPOINT,
        "direct_identity_removed": True,
    },
    "basis": {
        "construction": (
            "Uncentered orthonormal basis of exact SEARCH-minus-BELIEF "
            "L21 answer-prefix differences on basis-training rows."
        ),
        "maximum_rank": MAX_BASIS_RANK,
        "rank_grid": list(RANK_GRID),
    },
    "causal_metric": {
        "construction": (
            "For every activation-basis axis, apply plus and minus one "
            "training-RMS coefficient on calibration rows. Concatenate the "
            "odd, target-normalized processed-L27 responses. The response "
            "Gram eigenvectors order standardized activation coordinates."
        ),
        "central_difference": True,
        "nonlinearity_check": (
            "median even-response norm divided by odd-response norm"
        ),
        "maximum_even_odd_ratio":
            MAXIMUM_CALIBRATION_EVEN_ODD_RATIO,
    },
    "selection": (
        "On the disjoint THINK/FIND panel, select the smallest causal rank "
        "recovering at least 80% of full train-span progress. If none does, "
        "freeze the rank with greatest progress; no post-test rank change."
    ),
    "test_arms": [
        "exact matched target state",
        "full train-difference span projection",
        "selected causal-order projection",
        "equal-rank activation-PCA projection",
        "training mean difference",
        "training mean projected into selected causal modes",
        "19 seeded equal-rank random standardized-coordinate subspaces",
        "selected projection at instruction positions",
        "selected projection at identical-token positions",
    ],
    "primary_gates": {
        "minimum_exact_progress": MINIMUM_EXACT_PROGRESS,
        "minimum_full_span_progress": MINIMUM_FULL_SPAN_PROGRESS,
        "minimum_full_recovery_of_exact":
            MINIMUM_FULL_RECOVERY_OF_EXACT,
        "maximum_selected_rank": MAXIMUM_LOW_RANK,
        "minimum_causal_progress": MINIMUM_CAUSAL_PROGRESS,
        "minimum_causal_recovery_of_full":
            MINIMUM_CAUSAL_RECOVERY_OF_FULL,
        "minimum_causal_recovery_of_exact":
            MINIMUM_CAUSAL_RECOVERY_OF_EXACT,
        "minimum_margin_over_equal_rank_pca_and_mean":
            MINIMUM_CAUSAL_MARGIN,
        "minimum_positive_rows": MINIMUM_POSITIVE_ROWS,
        "minimum_answer_accuracy": MINIMUM_ANSWER_ACCURACY,
        "maximum_position_control_fraction":
            MAXIMUM_POSITION_CONTROL_FRACTION,
        "maximum_add_one_random_p": MAXIMUM_RANDOM_P,
    },
    "verdicts": [
        "LOW_RANK_CAUSAL_EFFECT_SUBSPACE",
        "CONTEXT_DEPENDENT_FUNCTIONAL_COMPRESSION",
        "NO_CAUSAL_ORDER_ADVANTAGE",
        "HIGH_DIMENSIONAL_FUNCTIONAL_CONTROL",
        "TRAIN_DIFFERENCE_SPAN_INSUFFICIENT",
        "ASSAY_INELIGIBLE",
    ],
    "stopping_rule": (
        "No prompt, family, panel, split, layer, rank, scale, metric, "
        "threshold, random seed, or control rescue follows the result."
    ),
    "random_seed": RANDOM_SEED,
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest().upper()


def _transition_key(source, target):
    return f"{source}_to_{target}"


def _directions():
    return (
        ("epistemic", "search"),
        ("search", "epistemic"),
    )


def _functional_rows():
    split = _row_splits()
    if len(split["train"]) != TRAIN_N:
        raise AssertionError("basis-training split changed")
    if len(split["validation"]) != CALIBRATION_N + SELECTION_N:
        raise AssertionError("calibration/selection split changed")
    return {
        "train": split["train"],
        "calibration": split["validation"][:CALIBRATION_N],
        "selection": split["validation"][CALIBRATION_N:],
        "test": split["test"],
    }


def _flatten(value):
    return value.detach().float().cpu().flatten(1)


def _basis_from_differences(differences):
    """Uncentered deterministic orthonormal activation basis."""
    # The sample Gram is tiny (48 x 48), so solve it in float64.  Float32
    # otherwise promotes numerical null axes into apparent causal candidates.
    values = differences.double()
    gram = values @ values.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    tolerance = max(
        1e-8, float(eigenvalues[0]) * 1e-10)
    usable = int(min(
        MAX_BASIS_RANK,
        int((eigenvalues > tolerance).sum())))
    if usable < 2:
        raise ValueError("exact-difference basis has fewer than two axes")
    singular = eigenvalues[:usable].clamp_min(1e-8).sqrt()
    basis = (
        eigenvectors[:, :usable].T @ values
        / singular[:, None])
    # Eigensystem signs are arbitrary.  Fix each by its largest loading.
    for index in range(usable):
        pivot = int(torch.argmax(basis[index].abs()))
        if float(basis[index, pivot]) < 0.0:
            basis[index] *= -1.0
    coefficients = values @ basis.T
    scales = coefficients.square().mean(dim=0).sqrt().clamp_min(1e-6)
    return {
        "basis": basis.float(),
        "scales": scales.float(),
        "singular_values": singular.float(),
        "rank": usable,
        "training_energy": singular.square().float(),
    }


def _available_ranks(rank):
    return tuple(sorted(set(
        min(int(value), int(rank))
        for value in RANK_GRID
    )))


def _activation_projection(model, differences, rank):
    basis = model["basis"][:int(rank)]
    coefficients = differences.float() @ basis.T
    return coefficients @ basis


def _causal_projection(model, differences, rotation, rank):
    basis = model["basis"]
    scales = model["scales"]
    coefficients = differences.float() @ basis.T
    standardized = coefficients / scales
    modes = rotation[:, :int(rank)]
    projected = standardized @ modes @ modes.T
    return (projected * scales) @ basis


def _random_rotation(rank, selected_rank, seed):
    generator = torch.Generator().manual_seed(int(seed))
    raw = torch.randn(
        int(rank), int(selected_rank), generator=generator)
    rotation, _r = torch.linalg.qr(raw, mode="reduced")
    return rotation


def _response_rotation(responses):
    """Return causal modes in standardized activation-coordinate space."""
    values = responses.double()
    gram = values @ values.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    rotation = eigenvectors[:, order]
    for index in range(rotation.shape[1]):
        pivot = int(torch.argmax(rotation[:, index].abs()))
        if float(rotation[pivot, index]) < 0.0:
            rotation[:, index] *= -1.0
    return rotation.float(), eigenvalues.float()


def _processed(cache, checkpoint, direct):
    return (
        cache[f"checkpoint_{checkpoint}"].float()
        - direct[:, -1, :].float()
    )


def _transport_rows(origin, target, patched):
    direction = target.float() - origin.float()
    displacement = patched.float() - origin.float()
    norm2 = direction.square().sum(dim=-1).clamp_min(1e-8)
    progress = (displacement * direction).sum(dim=-1) / norm2
    distance = (
        (patched.float() - target.float()).norm(dim=-1)
        / direction.norm(dim=-1).clamp_min(1e-8))
    return progress, distance


def _empty_accumulator():
    return {
        "accuracies": [],
        **{
            checkpoint: {"progress": [], "distance": []}
            for checkpoint in CHECKPOINT_LAYERS
        },
    }


def _append_result(
        accumulator, origin, target, patched, patched_direct,
        accuracy):
    accumulator["accuracies"].append(float(accuracy))
    for checkpoint in CHECKPOINT_LAYERS:
        origin_value = _processed(
            origin, checkpoint, origin["answer_source"])
        target_value = _processed(
            target, checkpoint, target["answer_source"])
        patched_value = _processed(
            patched, checkpoint, patched_direct)
        progress, distance = _transport_rows(
            origin_value, target_value, patched_value)
        accumulator[checkpoint]["progress"].extend(
            progress.tolist())
        accumulator[checkpoint]["distance"].extend(
            distance.tolist())


def _summarize_accumulator(accumulator):
    result = {}
    for checkpoint in CHECKPOINT_LAYERS:
        progress = accumulator[checkpoint]["progress"]
        distance = accumulator[checkpoint]["distance"]
        result[str(checkpoint)] = {
            "n_rows": len(progress),
            "mean_progress": float(np.mean(progress)),
            "median_progress": float(np.median(progress)),
            "positive_rows": int(sum(value > 0.0 for value in progress)),
            "progress_rows": [float(value) for value in progress],
            "mean_distance_ratio": float(np.mean(distance)),
            "median_distance_ratio": float(np.median(distance)),
            "distance_ratio_rows": [float(value) for value in distance],
            "minimum_answer_accuracy": float(
                min(accumulator["accuracies"])),
            "answer_accuracy_by_history": [
                float(value) for value in accumulator["accuracies"]
            ],
        }
    return result


def _expected(rows, history):
    key = "source" if int(history) == 0 else "target"
    return [row[key] for row in rows]


def _capture_answer_states(model, alignment, heartbeat, label):
    from .delta_predictive_conditional_transport import _capture_positions

    result = {frame: [] for frame in FRAMES}
    for history, batches in enumerate(alignment["batches"]):
        for frame in FRAMES:
            result[frame].append(_capture_positions(
                model, batches[frame],
                alignment["answer_positions"]))
            heartbeat.step(
                extra=f"{label}/h{history}/{frame}/basis_capture")
    return {
        frame: _flatten(torch.cat(result[frame], dim=0))
        for frame in FRAMES
    }


def _capture_alignment_baselines(
        model, alignment, heartbeat, label):
    union = sorted(set(
        alignment["answer_positions"]
        + alignment["instruction_positions"]
        + alignment["identical_positions"]))
    offsets = {
        name: [
            union.index(position)
            for position in alignment[f"{name}_positions"]
        ]
        for name in ("answer", "instruction", "identical")
    }
    result = []
    for history, batches in enumerate(alignment["batches"]):
        row = {}
        for frame in FRAMES:
            cache = _capture_baseline(
                model, batches[frame], SOURCE_LAYER, union)
            for name in ("answer", "instruction", "identical"):
                cache[f"{name}_source"] = (
                    cache["source"][:, offsets[name], :])
            row[frame] = cache
            heartbeat.step(
                extra=f"{label}/h{history}/{frame}/baseline")
        result.append(row)
    return result


def _exact_flat(baselines, source, target):
    return torch.cat([
        _flatten(
            baselines[history][target]["answer_source"]
            - baselines[history][source]["answer_source"])
        for history in range(2)
    ], dim=0)


def _history_delta(flat, history, n_rows, template):
    return flat[
        history * int(n_rows):(history + 1) * int(n_rows)
    ].reshape_as(template)


def _patch_delta(
        model, batch, origin, target, positions, delta,
        expected, accumulator):
    target_values = origin["answer_source"] + delta
    patched_direct = target_values[:, -1, :]
    patched = _run_exact_patch(
        model, batch, SOURCE_LAYER, positions, target_values)
    accuracy = _generic_accuracy(
        patched["logits"], batch, expected, VALUES)
    _append_result(
        accumulator, origin, target, patched,
        patched_direct, accuracy)


def _patch_position_delta(
        model, batch, origin, target, positions, base, delta,
        expected, accumulator):
    target_values = base + delta
    patched = _run_exact_patch(
        model, batch, SOURCE_LAYER, positions, target_values)
    accuracy = _generic_accuracy(
        patched["logits"], batch, expected, VALUES)
    _append_result(
        accumulator, origin, target, patched,
        origin["answer_source"], accuracy)


def _calibrate_causal_modes(
        model, alignment, baselines, basis_model,
        source, target, heartbeat, label):
    """Estimate a central-difference pullback metric at processed L27."""
    responses = []
    even_odd_rows = []
    rank = int(basis_model["rank"])
    hidden_size = int(
        baselines[0][source]["answer_source"].shape[-1])
    for axis in range(rank):
        axis_delta = (
            basis_model["scales"][axis]
            * basis_model["basis"][axis]
        ).reshape(1, PATCH_WIDTH, hidden_size)
        axis_responses = []
        for history in range(2):
            origin = baselines[history][source]
            target_cache = baselines[history][target]
            batch = alignment["batches"][history][source]
            delta = axis_delta.expand_as(
                origin["answer_source"])
            plus_values = origin["answer_source"] + delta
            minus_values = origin["answer_source"] - delta
            plus = _run_exact_patch(
                model, batch, SOURCE_LAYER,
                alignment["answer_positions"], plus_values)
            heartbeat.step(
                extra=f"{label}/{source}_to_{target}/axis{axis}/plus")
            minus = _run_exact_patch(
                model, batch, SOURCE_LAYER,
                alignment["answer_positions"], minus_values)
            heartbeat.step(
                extra=f"{label}/{source}_to_{target}/axis{axis}/minus")
            origin_value = _processed(
                origin, PRIMARY_CHECKPOINT,
                origin["answer_source"])
            target_value = _processed(
                target_cache, PRIMARY_CHECKPOINT,
                target_cache["answer_source"])
            plus_value = _processed(
                plus, PRIMARY_CHECKPOINT, plus_values)
            minus_value = _processed(
                minus, PRIMARY_CHECKPOINT, minus_values)
            odd = 0.5 * (plus_value - minus_value)
            even = (
                0.5 * (plus_value + minus_value)
                - origin_value)
            target_scale = (
                target_value - origin_value
            ).norm(dim=-1, keepdim=True).clamp_min(1e-8)
            axis_responses.append(
                (odd / target_scale).flatten())
            ratio = (
                even.norm(dim=-1)
                / odd.norm(dim=-1).clamp_min(1e-8))
            even_odd_rows.extend(ratio.tolist())
        responses.append(torch.cat(axis_responses))
    response_matrix = torch.stack(responses)
    rotation, eigenvalues = _response_rotation(response_matrix)
    total = eigenvalues.sum().clamp_min(1e-8)
    return {
        "rotation": rotation,
        "eigenvalues": eigenvalues,
        "cumulative_response_fraction": (
            eigenvalues.cumsum(dim=0) / total),
        "median_even_odd_ratio": float(np.median(even_odd_rows)),
        "mean_even_odd_ratio": float(np.mean(even_odd_rows)),
        "maximum_even_odd_ratio": float(max(even_odd_rows)),
        "even_odd_rows": [float(value) for value in even_odd_rows],
        "response_sha256": hashlib.sha256(
            response_matrix.numpy().tobytes()
        ).hexdigest().upper(),
    }


def _evaluate_flat_deltas(
        model, alignment, baselines, rows,
        source, target, flat_deltas, heartbeat, label,
        position_control=None):
    accumulator = _empty_accumulator()
    n_rows = len(rows)
    for history in range(2):
        origin = baselines[history][source]
        target_cache = baselines[history][target]
        batch = alignment["batches"][history][source]
        delta = _history_delta(
            flat_deltas, history, n_rows,
            origin["answer_source"])
        expected = _expected(rows, history)
        if position_control is None:
            _patch_delta(
                model, batch, origin, target_cache,
                alignment["answer_positions"], delta,
                expected, accumulator)
        else:
            positions = alignment[f"{position_control}_positions"]
            base = origin[f"{position_control}_source"]
            _patch_position_delta(
                model, batch, origin, target_cache,
                positions, base, delta, expected, accumulator)
        heartbeat.step(
            extra=(
                f"{label}/{source}_to_{target}/h{history}/"
                f"{position_control or 'answer'}"))
    return _summarize_accumulator(accumulator)


def _selection_for_direction(
        model, alignment, baselines, rows, basis_model,
        causal_model, source, target, heartbeat, label):
    exact = _exact_flat(baselines, source, target)
    full = _activation_projection(
        basis_model, exact, basis_model["rank"])
    exact_result = _evaluate_flat_deltas(
        model, alignment, baselines, rows,
        source, target, exact, heartbeat, f"{label}/exact")
    full_result = _evaluate_flat_deltas(
        model, alignment, baselines, rows,
        source, target, full, heartbeat, f"{label}/full")
    candidates = []
    for rank in _available_ranks(basis_model["rank"]):
        predicted = _causal_projection(
            basis_model, exact,
            causal_model["rotation"], rank)
        result = _evaluate_flat_deltas(
            model, alignment, baselines, rows,
            source, target, predicted, heartbeat,
            f"{label}/causal_rank_{rank}")
        primary = result[str(PRIMARY_CHECKPOINT)]
        full_progress = full_result[str(PRIMARY_CHECKPOINT)][
            "mean_progress"]
        recovery = (
            primary["mean_progress"] / full_progress
            if full_progress > 1e-8 else None)
        candidates.append({
            "rank": int(rank),
            "result": result,
            "recovery_of_full": recovery,
            "selection_pass": bool(
                recovery is not None
                and recovery >= MINIMUM_CAUSAL_RECOVERY_OF_FULL - 1e-9
                and primary["positive_rows"]
                >= 2 * len(rows) - 1
                and primary["minimum_answer_accuracy"]
                >= MINIMUM_ANSWER_ACCURACY - 1e-9),
        })
    passing = [
        row for row in candidates if row["selection_pass"]
    ]
    if passing:
        selected = min(passing, key=lambda row: row["rank"])
    else:
        selected = max(
            candidates,
            key=lambda row: (
                row["result"][str(PRIMARY_CHECKPOINT)][
                    "mean_progress"],
                -row["rank"],
            ))
    return {
        "exact": exact_result,
        "full_span": full_result,
        "candidates": candidates,
        "selected_rank": int(selected["rank"]),
        "selection_gate_pass": bool(selected["selection_pass"]),
    }


def _random_seed(family_index, direction_index, random_index):
    return (
        RANDOM_SEED
        + int(family_index) * 100003
        + int(direction_index) * 1009
        + int(random_index) * 17
    )


def _build_test_predictions(
        basis_model, causal_model, selection,
        exact, mean_delta, family_index, direction_index):
    selected_rank = int(selection["selected_rank"])
    predictions = {
        "exact_state_oracle": exact,
        "full_span_oracle": _activation_projection(
            basis_model, exact, basis_model["rank"]),
        "causal_selected_oracle": _causal_projection(
            basis_model, exact,
            causal_model["rotation"], selected_rank),
        "pca_same_rank_oracle": _activation_projection(
            basis_model, exact, selected_rank),
        "mean_template": mean_delta.expand(
            exact.shape[0], -1).clone(),
        "causal_mean_template": _causal_projection(
            basis_model,
            mean_delta.expand(exact.shape[0], -1),
            causal_model["rotation"], selected_rank),
    }
    random_rotations = {}
    for random_index in range(N_RANDOM):
        rotation = _random_rotation(
            basis_model["rank"], selected_rank,
            _random_seed(
                family_index, direction_index, random_index))
        name = f"random_{random_index:02d}"
        random_rotations[name] = {
            "seed": _random_seed(
                family_index, direction_index, random_index),
            "sha256": hashlib.sha256(
                rotation.numpy().tobytes()
            ).hexdigest().upper(),
        }
    return predictions, random_rotations


def _tensor_sha256(value):
    return hashlib.sha256(
        value.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest().upper()


def _write_prediction_freeze(
        out_dir, public_models, predictions, random_metadata):
    metadata = {
        "protocol_sha256": PROTOCOL_SHA256,
        "models": public_models,
        "random_subspaces": random_metadata,
        "predictions": {
            family: {
                direction: {
                    arm: {
                        "shape": list(value.shape),
                        "sha256": _tensor_sha256(value),
                    }
                    for arm, value in arms.items()
                }
                for direction, arms in family_values.items()
            }
            for family, family_values in predictions.items()
        },
    }
    json_path = os.path.join(
        out_dir, "functional_causal_rank_freeze.json")
    with open(json_path, "w") as handle:
        json.dump(metadata, handle, indent=2)
    arrays = {
        f"{family}__{direction}__{arm}": value.numpy()
        for family, family_values in predictions.items()
        for direction, arms in family_values.items()
        for arm, value in arms.items()
    }
    npz_path = os.path.join(
        out_dir, "functional_causal_rank_predictions.npz")
    np.savez_compressed(npz_path, **arrays)
    with open(json_path, "rb") as handle:
        json_hash = hashlib.sha256(
            handle.read()).hexdigest().upper()
    with open(npz_path, "rb") as handle:
        npz_hash = hashlib.sha256(
            handle.read()).hexdigest().upper()
    return {
        "json_path": json_path,
        "npz_path": npz_path,
        "json_sha256": json_hash,
        "npz_sha256": npz_hash,
        "metadata": metadata,
    }


def _reload_predictions(freeze, predictions):
    with np.load(freeze["npz_path"]) as archive:
        for family, family_values in predictions.items():
            for direction, arms in family_values.items():
                for arm in arms:
                    key = f"{family}__{direction}__{arm}"
                    value = torch.from_numpy(
                        np.array(archive[key])).float()
                    expected = freeze["metadata"]["predictions"][
                        family][direction][arm]["sha256"]
                    if _tensor_sha256(value) != expected:
                        raise AssertionError(
                            f"frozen prediction changed: {key}")
                    predictions[family][direction][arm] = value


def _basis_public(model):
    energy = model["training_energy"]
    return {
        "rank": int(model["rank"]),
        "basis_sha256": _tensor_sha256(model["basis"]),
        "scales_sha256": _tensor_sha256(model["scales"]),
        "singular_values": model["singular_values"].tolist(),
        "cumulative_activation_energy": (
            energy.cumsum(dim=0)
            / energy.sum().clamp_min(1e-8)
        ).tolist(),
    }


def _causal_public(model):
    return {
        "rotation_sha256": _tensor_sha256(model["rotation"]),
        "response_sha256": model["response_sha256"],
        "eigenvalues": model["eigenvalues"].tolist(),
        "cumulative_response_fraction":
            model["cumulative_response_fraction"].tolist(),
        "median_even_odd_ratio": model["median_even_odd_ratio"],
        "mean_even_odd_ratio": model["mean_even_odd_ratio"],
        "maximum_even_odd_ratio": model["maximum_even_odd_ratio"],
        "even_odd_rows": model["even_odd_rows"],
    }


def _primary(result):
    return result[str(PRIMARY_CHECKPOINT)]


def _cell_adjudication(
        results, selected_rank, selection_gate_pass,
        calibration_even_odd):
    exact = _primary(results["exact_state_oracle"])
    full = _primary(results["full_span_oracle"])
    causal = _primary(results["causal_selected_oracle"])
    pca = _primary(results["pca_same_rank_oracle"])
    mean = _primary(results["mean_template"])
    causal_mean = _primary(results["causal_mean_template"])
    instruction = _primary(results["instruction"])
    identical = _primary(results["identical"])
    exact_progress = exact["mean_progress"]
    full_progress = full["mean_progress"]
    causal_progress = causal["mean_progress"]
    full_recovery = (
        full_progress / exact_progress
        if exact_progress > 1e-8 else None)
    causal_full_recovery = (
        causal_progress / full_progress
        if full_progress > 1e-8 else None)
    causal_exact_recovery = (
        causal_progress / exact_progress
        if exact_progress > 1e-8 else None)
    exact_pass = bool(
        exact_progress >= MINIMUM_EXACT_PROGRESS - 1e-9
        and exact["positive_rows"] >= MINIMUM_POSITIVE_ROWS
        and exact["minimum_answer_accuracy"]
        >= MINIMUM_ANSWER_ACCURACY - 1e-9)
    full_pass = bool(
        exact_pass
        and full_progress >= MINIMUM_FULL_SPAN_PROGRESS - 1e-9
        and full_recovery is not None
        and full_recovery
        >= MINIMUM_FULL_RECOVERY_OF_EXACT - 1e-9
        and full["positive_rows"] >= MINIMUM_POSITIVE_ROWS
        and full["minimum_answer_accuracy"]
        >= MINIMUM_ANSWER_ACCURACY - 1e-9)
    linearity_pass = bool(
        calibration_even_odd
        <= MAXIMUM_CALIBRATION_EVEN_ODD_RATIO + 1e-9)
    compression_pass = bool(
        full_pass
        and selection_gate_pass
        and int(selected_rank) <= MAXIMUM_LOW_RANK
        and causal_progress >= MINIMUM_CAUSAL_PROGRESS - 1e-9
        and causal_full_recovery is not None
        and causal_full_recovery
        >= MINIMUM_CAUSAL_RECOVERY_OF_FULL - 1e-9
        and causal_exact_recovery is not None
        and causal_exact_recovery
        >= MINIMUM_CAUSAL_RECOVERY_OF_EXACT - 1e-9
        and causal["positive_rows"] >= MINIMUM_POSITIVE_ROWS
        and causal["minimum_answer_accuracy"]
        >= MINIMUM_ANSWER_ACCURACY - 1e-9
        and causal_progress - max(
            pca["mean_progress"],
            mean["mean_progress"],
            causal_mean["mean_progress"])
        >= MINIMUM_CAUSAL_MARGIN - 1e-9
        and instruction["mean_progress"] < max(
            0.10,
            MAXIMUM_POSITION_CONTROL_FRACTION * causal_progress)
        and identical["mean_progress"] < max(
            0.10,
            MAXIMUM_POSITION_CONTROL_FRACTION * causal_progress)
        and linearity_pass)
    return {
        "selected_rank": int(selected_rank),
        "exact_progress": exact_progress,
        "full_span_progress": full_progress,
        "causal_progress": causal_progress,
        "equal_rank_pca_progress": pca["mean_progress"],
        "mean_template_progress": mean["mean_progress"],
        "causal_mean_template_progress":
            causal_mean["mean_progress"],
        "instruction_progress": instruction["mean_progress"],
        "identical_progress": identical["mean_progress"],
        "full_recovery_of_exact": full_recovery,
        "causal_recovery_of_full": causal_full_recovery,
        "causal_recovery_of_exact": causal_exact_recovery,
        "calibration_even_odd_ratio": calibration_even_odd,
        "linearity_pass": linearity_pass,
        "selection_gate_pass": bool(selection_gate_pass),
        "exact_pass": exact_pass,
        "full_span_pass": full_pass,
        "compression_pass_before_random": compression_pass,
    }


def _verdict_from_counts(
        exact_count, full_count, compression_count,
        random_p, high_rank_count):
    total = len(FAMILIES) * len(_directions())
    if exact_count < total:
        return "ASSAY_INELIGIBLE"
    if full_count < total:
        return "TRAIN_DIFFERENCE_SPAN_INSUFFICIENT"
    if compression_count == total and random_p <= MAXIMUM_RANDOM_P:
        return "LOW_RANK_CAUSAL_EFFECT_SUBSPACE"
    if compression_count >= len(_directions()) and (
            random_p <= MAXIMUM_RANDOM_P):
        return "CONTEXT_DEPENDENT_FUNCTIONAL_COMPRESSION"
    if high_rank_count > 0:
        return "HIGH_DIMENSIONAL_FUNCTIONAL_CONTROL"
    return "NO_CAUSAL_ORDER_ADVANTAGE"


def _overall_adjudication(families):
    cells = [
        value
        for family in families.values()
        for value in family["cells"].values()
    ]
    exact_count = sum(cell["exact_pass"] for cell in cells)
    full_count = sum(cell["full_span_pass"] for cell in cells)
    compression_count = sum(
        cell["compression_pass_before_random"] for cell in cells)
    high_rank_count = sum(
        cell["selection_gate_pass"]
        and cell["selected_rank"] > MAXIMUM_LOW_RANK
        for cell in cells)
    selected_score = min(
        cell["causal_progress"] for cell in cells)
    random_scores = []
    for random_index in range(N_RANDOM):
        arm = f"random_{random_index:02d}"
        random_scores.append(min(
            _primary(direction["results"][arm])["mean_progress"]
            for family in families.values()
            for direction in family["directions"].values()
        ))
    random_p, exceed = _tail_probability(
        selected_score, random_scores)
    verdict = _verdict_from_counts(
        exact_count, full_count, compression_count,
        random_p, high_rank_count)
    return {
        "cell_count": len(cells),
        "exact_count": int(exact_count),
        "full_span_count": int(full_count),
        "compression_count_before_random":
            int(compression_count),
        "high_rank_count": int(high_rank_count),
        "selected_breadth_score": selected_score,
        "random_breadth_scores": random_scores,
        "random_empirical_p": random_p,
        "random_exceed_count": int(exceed),
        "pass": verdict == "LOW_RANK_CAUSAL_EFFECT_SUBSPACE",
        "verdict": verdict,
    }


def _self_check():
    generator = torch.Generator().manual_seed(1234)
    raw = torch.randn(12, 6, generator=generator)
    left, _singular, right_t = torch.linalg.svd(
        raw, full_matrices=False)
    orthogonal = left @ right_t
    coefficients = torch.randn(48, 6, generator=generator)
    coefficients[:, 0] *= 4.0
    coefficients[:, 5] *= 0.20
    differences = coefficients @ orthogonal.T
    basis_model = _basis_from_differences(differences)
    rank = basis_model["rank"]
    responses = torch.zeros(rank, rank)
    responses[0, 0] = 0.10
    responses[-1, -1] = 10.0
    rotation, _eigenvalues = _response_rotation(responses)
    test = basis_model["basis"][-1:].clone()
    causal = _causal_projection(
        basis_model, test, rotation, 1)
    pca = _activation_projection(basis_model, test, 1)
    full = _activation_projection(
        basis_model, test, basis_model["rank"])
    probability, exceed = _tail_probability(
        1.0, [0.0] * N_RANDOM)
    boundary = _PredictionBoundary()
    blocked = False
    try:
        boundary.require_evaluation()
    except RuntimeError:
        blocked = True
    boundary.freeze("json", "npz")
    boundary.require_evaluation()
    verdicts = {
        _verdict_from_counts(0, 0, 0, 1.0, 0),
        _verdict_from_counts(4, 3, 0, 1.0, 0),
        _verdict_from_counts(4, 4, 4, 0.05, 0),
        _verdict_from_counts(4, 4, 2, 0.05, 0),
        _verdict_from_counts(4, 4, 0, 1.0, 1),
        _verdict_from_counts(4, 4, 0, 1.0, 0),
    }
    rows = _functional_rows()
    checks = {
        "causal_recovery": bool(torch.allclose(
            causal, test, atol=1e-4, rtol=1e-4)),
        "pca_rejection": bool(float(pca.norm()) < 0.1),
        "full_recovery": bool(torch.allclose(
            full, test, atol=1e-4, rtol=1e-4)),
        "random_tail": bool(probability == 0.05 and exceed == 0),
        "prediction_boundary": blocked,
        "verdict_taxonomy": verdicts == set(PROTOCOL["verdicts"]),
        "row_splits": bool(
            len(rows["calibration"]) == CALIBRATION_N
            and len(rows["selection"]) == SELECTION_N
            and len(rows["test"]) == TEST_N),
    }
    passed = all(checks.values())
    if not passed:
        diagnostics = {
            "checks": checks,
            "causal_error": float((causal - test).norm()),
            "pca_norm": float(pca.norm()),
            "full_error": float((full - test).norm()),
            "probability": probability,
            "exceed": exceed,
            "observed_verdicts": sorted(verdicts),
            "expected_verdicts": sorted(PROTOCOL["verdicts"]),
            "basis_rank": int(basis_model["rank"]),
        }
        raise AssertionError(
            "functional causal-rank self-check failed: "
            + json.dumps(diagnostics, sort_keys=True))
    return {
        "causal_order_recovers_low_energy_axis": True,
        "equal_rank_pca_fails_low_energy_axis": True,
        "full_projection_invariant": True,
        "prediction_boundary_check": True,
        "random_tail_check": True,
        "verdict_taxonomy_check": True,
        "row_split_check": True,
        "pass": True,
    }


@torch.no_grad()
def run_delta_functional_causal_rank(
        model_path, out_dir,
        model_key="qwen7b_functional_causal_rank",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=TEST_N, self_test_only=False):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_world) != TEST_N:
        raise ValueError(
            "functional causal-rank v1 requires twelve test histories")
    self_check = _self_check()
    if self_test_only:
        result = {
            "stage": "delta_functional_causal_rank",
            "protocol": PROTOCOL,
            "protocol_sha256": PROTOCOL_SHA256,
            "self_check": self_check,
            "verdict": "SELF_CHECK_PASS",
        }
        path = os.path.join(
            out_dir, "functional_causal_rank_self_check.json")
        with open(path, "w") as handle:
            json.dump(result, handle, indent=2)
        log(
            "FUNCTIONAL-CAUSAL-RANK self-check pass "
            f"protocol={PROTOCOL_SHA256}")
        return result

    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    if PRIMARY_CHECKPOINT >= model_num_hidden_layers(model):
        raise ValueError("functional causal-rank checkpoint is absent")
    rows = _functional_rows()
    commands, padding_plan, tokenization_tables = _commands(tok)
    capture_hb = Heartbeat(
        len(FAMILIES) * 4 * 4,
        "functional_causal_rank_capture",
        every_sec=30, out_dir=out_dir)

    family_data = {}
    for family in FAMILIES:
        spec = FAMILY_SPECS[family]
        train_alignment = _frame_alignment(
            tok, dev, rows["train"], spec,
            commands[TRAIN_PANEL])
        train_states = _capture_answer_states(
            model, train_alignment, capture_hb,
            f"{family}/{TRAIN_PANEL}")
        train_difference = (
            train_states["search"] - train_states["epistemic"])
        basis_model = _basis_from_differences(train_difference)
        mean_forward = train_difference.mean(
            dim=0, keepdim=True)

        calibration_alignment = _frame_alignment(
            tok, dev, rows["calibration"], spec,
            commands[TRAIN_PANEL])
        calibration_baselines = _capture_alignment_baselines(
            model, calibration_alignment, capture_hb,
            f"{family}/{TRAIN_PANEL}/calibration")

        selection_alignment = _frame_alignment(
            tok, dev, rows["selection"], spec,
            commands[SELECTION_PANEL])
        selection_baselines = _capture_alignment_baselines(
            model, selection_alignment, capture_hb,
            f"{family}/{SELECTION_PANEL}/selection")

        test_alignment = _frame_alignment(
            tok, dev, rows["test"], spec,
            commands[TEST_PANEL])
        test_baselines = _capture_alignment_baselines(
            model, test_alignment, capture_hb,
            f"{family}/{TEST_PANEL}/test")
        family_data[family] = {
            "basis_model": basis_model,
            "mean_forward": mean_forward,
            "alignments": {
                "train": train_alignment,
                "calibration": calibration_alignment,
                "selection": selection_alignment,
                "test": test_alignment,
            },
            "calibration_baselines": calibration_baselines,
            "selection_baselines": selection_baselines,
            "test_baselines": test_baselines,
        }
    capture_hb.done()

    metric_total = sum(
        int(family_data[family]["basis_model"]["rank"])
        * len(_directions()) * 2 * 2
        for family in FAMILIES
    )
    metric_hb = Heartbeat(
        metric_total, "functional_causal_metric",
        every_sec=30, out_dir=out_dir)
    for family in FAMILIES:
        data = family_data[family]
        data["causal_models"] = {}
        for source, target in _directions():
            key = _transition_key(source, target)
            data["causal_models"][key] = _calibrate_causal_modes(
                model, data["alignments"]["calibration"],
                data["calibration_baselines"],
                data["basis_model"], source, target,
                metric_hb, family)
    metric_hb.done()

    selection_total = sum(
        len(_directions()) * 2 * (
            2 + len(_available_ranks(
                family_data[family]["basis_model"]["rank"])))
        for family in FAMILIES
    )
    selection_hb = Heartbeat(
        selection_total, "functional_causal_rank_selection",
        every_sec=30, out_dir=out_dir)
    for family in FAMILIES:
        data = family_data[family]
        data["selection"] = {}
        for source, target in _directions():
            key = _transition_key(source, target)
            data["selection"][key] = _selection_for_direction(
                model, data["alignments"]["selection"],
                data["selection_baselines"], rows["selection"],
                data["basis_model"], data["causal_models"][key],
                source, target, selection_hb, family)
            log(
                f"FUNCTIONAL-CAUSAL-RANK select family={family} "
                f"direction={key} rank="
                f"{data['selection'][key]['selected_rank']} "
                f"gate={data['selection'][key]['selection_gate_pass']}")
    selection_hb.done()

    predictions = {}
    random_metadata = {}
    public_models = {}
    for family_index, family in enumerate(FAMILIES):
        data = family_data[family]
        predictions[family] = {}
        random_metadata[family] = {}
        public_models[family] = {
            "basis": _basis_public(data["basis_model"]),
            "directions": {},
        }
        for direction_index, (source, target) in enumerate(_directions()):
            key = _transition_key(source, target)
            exact = _exact_flat(
                data["test_baselines"], source, target)
            mean_delta = (
                data["mean_forward"]
                if source == "epistemic"
                else -data["mean_forward"])
            prediction, random_rows = _build_test_predictions(
                data["basis_model"], data["causal_models"][key],
                data["selection"][key], exact, mean_delta,
                family_index, direction_index)
            predictions[family][key] = prediction
            random_metadata[family][key] = random_rows
            public_models[family]["directions"][key] = {
                "causal_metric": _causal_public(
                    data["causal_models"][key]),
                "selection": data["selection"][key],
            }

    freeze = _write_prediction_freeze(
        out_dir, public_models, predictions, random_metadata)
    boundary = _PredictionBoundary()
    boundary.freeze(
        freeze["json_sha256"], freeze["npz_sha256"])
    _reload_predictions(freeze, predictions)
    boundary.require_evaluation()
    log(
        "FROZEN functional causal-rank projections "
        f"json={freeze['json_sha256']} "
        f"npz={freeze['npz_sha256']}")

    test_arm_count = 6 + N_RANDOM + 2
    test_hb = Heartbeat(
        len(FAMILIES) * len(_directions()) * 2 * test_arm_count,
        "functional_causal_rank_test",
        every_sec=30, out_dir=out_dir)
    families = {}
    for family_index, family in enumerate(FAMILIES):
        data = family_data[family]
        direction_results = {}
        cell_results = {}
        for direction_index, (source, target) in enumerate(_directions()):
            key = _transition_key(source, target)
            results = {}
            for arm, flat_delta in predictions[family][key].items():
                results[arm] = _evaluate_flat_deltas(
                    model, data["alignments"]["test"],
                    data["test_baselines"], rows["test"],
                    source, target, flat_delta,
                    test_hb, f"{family}/{key}/{arm}")
            exact = predictions[family][key][
                "exact_state_oracle"]
            selected_rank = data["selection"][key]["selected_rank"]
            for random_index in range(N_RANDOM):
                arm = f"random_{random_index:02d}"
                seed = _random_seed(
                    family_index, direction_index, random_index)
                rotation = _random_rotation(
                    data["basis_model"]["rank"],
                    selected_rank, seed)
                expected_hash = random_metadata[family][key][arm][
                    "sha256"]
                if _tensor_sha256(rotation) != expected_hash:
                    raise AssertionError(
                        f"random rotation changed: {family}/{key}/{arm}")
                delta = _causal_projection(
                    data["basis_model"], exact,
                    rotation, selected_rank)
                results[arm] = _evaluate_flat_deltas(
                    model, data["alignments"]["test"],
                    data["test_baselines"], rows["test"],
                    source, target, delta,
                    test_hb, f"{family}/{key}/{arm}")
            selected_delta = predictions[family][key][
                "causal_selected_oracle"]
            for control in ("instruction", "identical"):
                results[control] = _evaluate_flat_deltas(
                    model, data["alignments"]["test"],
                    data["test_baselines"], rows["test"],
                    source, target, selected_delta,
                    test_hb, f"{family}/{key}/{control}",
                    position_control=control)
            cell = _cell_adjudication(
                results, selected_rank,
                data["selection"][key]["selection_gate_pass"],
                data["causal_models"][key][
                    "median_even_odd_ratio"])
            direction_results[key] = {
                "source": source,
                "target": target,
                "results": results,
                "adjudication": cell,
            }
            cell_results[key] = cell
            log(
                f"FUNCTIONAL-CAUSAL-RANK cell={family}/{key} "
                f"exact={cell['exact_progress']:+.3f} "
                f"full={cell['full_span_progress']:+.3f} "
                f"causal={cell['causal_progress']:+.3f} "
                f"pca={cell['equal_rank_pca_progress']:+.3f} "
                f"rank={selected_rank}")
        families[family] = {
            "model": public_models[family],
            "directions": direction_results,
            "cells": cell_results,
        }
    test_hb.done()

    overall = _overall_adjudication(families)
    result = {
        "stage": "delta_functional_causal_rank",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "self_check": self_check,
        "prediction_freeze": {
            key: freeze[key]
            for key in (
                "json_path", "npz_path",
                "json_sha256", "npz_sha256")
        },
        "commands": commands,
        "padding_plan": padding_plan,
        "tokenization_tables": tokenization_tables,
        "rows": rows,
        "families": families,
        "overall": overall,
        "verdict": overall["verdict"],
    }
    path = os.path.join(
        out_dir,
        "results_delta_functional_causal_rank_"
        f"{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"FUNCTIONAL-CAUSAL-RANK verdict={result['verdict']} "
        f"exact={overall['exact_count']}/4 "
        f"full={overall['full_span_count']}/4 "
        f"compressed={overall['compression_count_before_random']}/4 "
        f"random_p={overall['random_empirical_p']:.3f} "
        f"artifact={path}")
    return result
