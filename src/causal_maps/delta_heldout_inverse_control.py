"""Prospective target-blind inverse control on held-out computations."""
from __future__ import annotations

import hashlib
import itertools
import json
import os

import numpy as np
import torch

from .delta_anchor_write import _resolve
from .delta_context_geometry_width_screen import (
    _processed_central_derivative,
    _processed_checkpoint_state,
)
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
    _family_batch,
)
from .delta_prospective_causal_sensitivity import _prospective_rows
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)


PROTOCOL_VERSION = "2026-07-27-p2-heldout-inverse-control-v1"
TRAIN_FAMILIES = (
    "private_belief",
    "two_hop_pointer",
    "maximum_score",
    "constraint_elimination",
)
HELDOUT_FAMILIES = ("minimum_score", "set_intersection")
HELDOUT_SPECS = {
    "minimum_score": {
        "preamble": (
            "Compare the numerical scores and return the lowest-scoring "
            "label."
        ),
        "rule": "The selected label is the one with the smallest score.",
        "story": (
            "{state} scored one point. {d1} scored five points. {d2} "
            "scored nine points."
        ),
        "question": "Which label had the lowest score?",
        "answer_description": "color word",
    },
    "set_intersection": {
        "preamble": (
            "Find the unique label that occurs in both stated groups."
        ),
        "rule": "Return the sole member of the set intersection.",
        "story": (
            "GROUP_ONE contains {state} and {d1}. GROUP_TWO contains "
            "{state} and {d2}."
        ),
        "question": "Which label appears in both groups?",
        "answer_description": "color word",
    },
}
DIRECTIONS = ("belief_to_search", "search_to_belief")
SOURCE_LAYER = 21
PRIMARY_CHECKPOINT = 27
BASIS_RANK = 4
TRAIN_N = 6
IDENTIFICATION_N = 6
TEST_N = 6
PATCH_WIDTH = 3
PROBE_FRACTION = 0.10
RIDGE_FRACTION = 0.01
MAXIMUM_NORM_MULTIPLIER = 1.25
N_RANDOM = 3
RANDOM_SEED = 947321

MINIMUM_TRAIN_TARGET_COSINE = 0.20
MINIMUM_EXACT_PROGRESS = 0.50
MINIMUM_TARGET_ORACLE_PROGRESS = 0.40
MINIMUM_LOCAL_PROGRESS = 0.25
MINIMUM_POSITIVE_ROWS = 9
MAXIMUM_DISTANCE_RATIO = 0.95
MINIMUM_VALUE_ACCURACY = 0.80
MINIMUM_ORACLE_RECOVERY = 0.60
MINIMUM_CONTROL_MARGIN = 0.10
MINIMUM_PASSING_CELLS = 3


PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "claim_status": (
        "prospective hidden-state pilot; route confirmation required"),
    "hypothesis": (
        "A globally nonlinear LLM has a locally identifiable, "
        "context-conditioned response map at the L21 answer-prefix causal "
        "interface. Generic target-blind probes can identify that map well "
        "enough to synthesize an unseen operation-state transition."),
    "model": "Qwen2.5-7B-Instruct, 8-bit",
    "training_families": list(TRAIN_FAMILIES),
    "heldout_families": {
        name: HELDOUT_SPECS[name] for name in HELDOUT_FAMILIES
    },
    "rows": {
        "training_pairs": TRAIN_N,
        "heldout_identification_pairs_per_direction":
            IDENTIFICATION_N,
        "heldout_test_pairs_per_direction": TEST_N,
        "all_identification_and_test_pairs_are_disjoint": True,
        "histories_per_pair": 2,
    },
    "locus": {
        "source_layer": SOURCE_LAYER,
        "positions": "three final answer-prefix command tokens",
        "primary_checkpoint": PRIMARY_CHECKPOINT,
        "replication_checkpoints": list(CHECKPOINT_LAYERS),
        "remove_exact_final_position_identity_carry": True,
    },
    "training_only_construction": {
        "basis": "rank-4 SVD of four family mean L21 controllers",
        "shared_target": (
            "normalized mean of normalized family processed L27 "
            "BELIEF-to-SEARCH targets, restored to median target norm"),
        "probe_fraction_of_median_controller_norm": PROBE_FRACTION,
        "minimum_target_pairwise_cosine":
            MINIMUM_TRAIN_TARGET_COSINE,
    },
    "inverse": {
        "ridge_fraction_of_mean_gram_diagonal": RIDGE_FRACTION,
        "maximum_norm_multiple_of_training_controller":
            MAXIMUM_NORM_MULTIPLIER,
    },
    "pre_target_controls": [
        "shared response map",
        "other held-out family response map, norm matched",
        "same held-out family opposite-origin response map, norm matched",
        "negative local inverse",
        f"{N_RANDOM} random rank-four directions, norm matched",
    ],
    "post_freeze_oracles": [
        "training-basis projection of exact L21 target",
        "local inverse toward exact held-out L27 target",
        "exact L21 target state",
    ],
    "cell_gate": {
        "minimum_exact_progress": MINIMUM_EXACT_PROGRESS,
        "minimum_target_oracle_progress":
            MINIMUM_TARGET_ORACLE_PROGRESS,
        "minimum_local_progress": MINIMUM_LOCAL_PROGRESS,
        "minimum_positive_rows": MINIMUM_POSITIVE_ROWS,
        "maximum_distance_ratio": MAXIMUM_DISTANCE_RATIO,
        "minimum_value_accuracy": MINIMUM_VALUE_ACCURACY,
        "minimum_target_oracle_recovery":
            MINIMUM_ORACLE_RECOVERY,
        "minimum_margin_over_best_control":
            MINIMUM_CONTROL_MARGIN,
        "negative_sign_maximum_progress": 0.0,
    },
    "overall_gate": {
        "minimum_passing_cells": MINIMUM_PASSING_CELLS,
        "both_families_represented": True,
        "both_directions_represented": True,
    },
    "stopping_rule": (
        "No prompt, family, row, layer, rank, probe scale, inverse "
        "regularization, norm cap, control or threshold changes follow "
        "the output."),
    "random_seed": RANDOM_SEED,
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest().upper()


class _PredictionBoundary:
    """Enforce that target access follows a durable prediction hash."""

    def __init__(self):
        self._frozen = False
        self.json_sha256 = None
        self.npz_sha256 = None

    def freeze(self, json_sha256, npz_sha256):
        if not json_sha256 or not npz_sha256:
            raise ValueError("both prediction hashes are required")
        self.json_sha256 = str(json_sha256)
        self.npz_sha256 = str(npz_sha256)
        self._frozen = True

    def require_target_access(self):
        if not self._frozen:
            raise RuntimeError(
                "held-out target access attempted before prediction freeze")


def _pilot_rows():
    rows = _prospective_rows(VALUES)
    selected = {
        "training": rows[20:20 + TRAIN_N],
        "identify_belief_to_search":
            rows[26:26 + IDENTIFICATION_N],
        "test_belief_to_search": rows[32:32 + TEST_N],
        "identify_search_to_belief":
            rows[38:38 + IDENTIFICATION_N],
        "test_search_to_belief": rows[44:44 + TEST_N],
    }
    pair_sets = {
        split: {
            (row["source"], row["target"]) for row in split_rows
        }
        for split, split_rows in selected.items()
    }
    for left, right in itertools.combinations(pair_sets, 2):
        if pair_sets[left] & pair_sets[right]:
            raise AssertionError(f"{left}/{right} directed pairs overlap")
    expected = {
        "training": TRAIN_N,
        "identify_belief_to_search": IDENTIFICATION_N,
        "test_belief_to_search": TEST_N,
        "identify_search_to_belief": IDENTIFICATION_N,
        "test_search_to_belief": TEST_N,
    }
    if any(len(selected[key]) != count
           for key, count in expected.items()):
        raise AssertionError("frozen pilot row counts changed")
    return selected


def _origin_alignment(tok, dev, rows, spec, operation):
    """Render one operation only; the opposite command is never tokenized."""
    command = "BELIEF" if operation == "belief" else "X X SEARCH"
    histories = (
        rows,
        [{**row, "state": row["target"]} for row in rows],
    )
    batches = [
        _family_batch(tok, history_rows, spec, command, dev)
        for history_rows in histories
    ]
    lengths = {int(batch["ids"].shape[1]) for batch in batches}
    if len(lengths) != 1:
        raise ValueError("origin-only clean/natural lengths differ")
    prefix_lengths = {
        int(batch["ids"].shape[1])
        - len(tok.encode(text, add_special_tokens=False))
        for batch in batches
        for text in batch["texts"]
    }
    if prefix_lengths != {PATCH_WIDTH}:
        raise ValueError(
            f"origin command prefix is not width three: {prefix_lengths}")
    positions = list(range(
        int(batches[0]["ids"].shape[1]) - PATCH_WIDTH,
        int(batches[0]["ids"].shape[1])))
    if positions[-1] != int(batches[0]["ids"].shape[1]) - 1:
        raise AssertionError("origin-only answer positions are not final")
    return {"batches": batches, "answer_positions": positions}


def _template_basis(templates):
    matrix = torch.stack([
        templates[family].flatten().float()
        for family in TRAIN_FAMILIES
    ])
    _left, singular, right = torch.linalg.svd(
        matrix, full_matrices=False)
    basis = right[:BASIS_RANK]
    if basis.shape[0] != BASIS_RANK:
        raise AssertionError("rank-four training basis is unavailable")
    reconstruction = matrix @ basis.T @ basis
    energy = float(
        reconstruction.square().sum()
        / matrix.square().sum().clamp_min(1e-8))
    return basis, {
        "singular_values": singular.tolist(),
        "rank4_reconstruction_energy": energy,
        "controller_norms": {
            family: float(templates[family].norm())
            for family in TRAIN_FAMILIES
        },
    }


def _cosine(left, right):
    left = left.flatten().float()
    right = right.flatten().float()
    return float(
        torch.dot(left, right)
        / (left.norm() * right.norm()).clamp_min(1e-8))


def _shared_training_target(family_targets):
    vectors = torch.stack([
        family_targets[family].flatten().float()
        for family in TRAIN_FAMILIES
    ])
    norms = vectors.norm(dim=1)
    units = vectors / norms[:, None].clamp_min(1e-8)
    unit_mean = units.mean(dim=0)
    unit_mean = unit_mean / unit_mean.norm().clamp_min(1e-8)
    target = unit_mean * norms.median()
    pairwise = [
        _cosine(vectors[left], vectors[right])
        for left in range(len(TRAIN_FAMILIES))
        for right in range(left + 1, len(TRAIN_FAMILIES))
    ]
    return target, {
        "family_target_norms": {
            family: float(family_targets[family].norm())
            for family in TRAIN_FAMILIES
        },
        "pairwise_cosines": pairwise,
        "median_pairwise_cosine": float(np.median(pairwise)),
        "shared_target_norm": float(target.norm()),
    }


def _ridge_coefficients(response_map, target, norm_cap):
    """Solve a batched ridge inverse for [batch, hidden, rank] maps."""
    response = response_map.float()
    if response.dim() == 2:
        response = response.unsqueeze(0)
    target = target.float()
    if target.dim() == 1:
        target = target.unsqueeze(0).expand(response.shape[0], -1)
    if target.shape[:2] != response.shape[:2]:
        raise ValueError("response-map and target shapes differ")
    gram = torch.einsum("bhr,bhs->brs", response, response)
    rhs = torch.einsum("bhr,bh->br", response, target)
    rank = int(response.shape[-1])
    ridge = (
        RIDGE_FRACTION
        * gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    ).clamp_min(1e-8)
    eye = torch.eye(rank, dtype=gram.dtype)[None, :, :]
    coefficients = torch.linalg.solve(
        gram + ridge[:, None, None] * eye,
        rhs.unsqueeze(-1),
    ).squeeze(-1)
    uncapped = coefficients.norm(dim=-1)
    scale = torch.clamp(
        float(norm_cap) / uncapped.clamp_min(1e-8), max=1.0)
    coefficients = coefficients * scale[:, None]
    predicted = torch.einsum(
        "bhr,br->bh", response, coefficients)
    target_norm2 = target.square().sum(dim=-1).clamp_min(1e-8)
    progress = (
        predicted * target
    ).sum(dim=-1) / target_norm2
    cosine = (
        predicted * target
    ).sum(dim=-1) / (
        predicted.norm(dim=-1) * target.norm(dim=-1)
    ).clamp_min(1e-8)
    return coefficients, {
        "ridge": ridge.tolist(),
        "uncapped_norm": uncapped.tolist(),
        "capped_norm": coefficients.norm(dim=-1).tolist(),
        "cap_applied": (scale < 1.0 - 1e-9).tolist(),
        "predicted_target_progress": progress.tolist(),
        "predicted_target_cosine": cosine.tolist(),
    }


def _coefficients_to_delta(coefficients, basis, hidden_size):
    flat = coefficients.float() @ basis.float()
    return flat.reshape(-1, PATCH_WIDTH, int(hidden_size))


def _norm_match_coefficients(control, reference):
    control = control.float()
    reference = reference.float()
    scale = (
        reference.norm(dim=-1)
        / control.norm(dim=-1).clamp_min(1e-8)
    )
    return control * scale[:, None]


def _random_coefficients(reference, seed):
    generator = torch.Generator().manual_seed(int(seed))
    value = torch.randn(
        reference.shape, generator=generator, dtype=torch.float32)
    return _norm_match_coefficients(value, reference)


def _training_processed(baseline):
    return _processed_checkpoint_state(
        baseline[f"checkpoint_{PRIMARY_CHECKPOINT}"],
        baseline["source"])


def _capture_response_map(
        model, batch, positions, origin, basis_shaped, probe_step,
        expected, heartbeat, label):
    derivatives = []
    accuracies = []
    for axis in range(BASIS_RANK):
        outputs = {}
        direction = basis_shaped[axis]
        for sign in (-1, 1):
            values = (
                origin["source"]
                + float(sign) * float(probe_step) * direction)
            outputs[sign] = _run_exact_patch(
                model, batch, SOURCE_LAYER, positions, values)
            accuracies.append(float(_generic_accuracy(
                outputs[sign]["logits"], batch, expected, VALUES)))
            heartbeat.step(
                extra=f"{label}/axis_{axis + 1}/sign_{sign:+d}")
        derivative = _processed_central_derivative(
            outputs[1][f"checkpoint_{PRIMARY_CHECKPOINT}"],
            outputs[-1][f"checkpoint_{PRIMARY_CHECKPOINT}"],
            probe_step,
            direction,
        )
        derivatives.append(derivative)
    # [row, downstream_hidden, rank]
    return torch.stack(derivatives, dim=-1), accuracies


def _artifact_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest().upper()


def _tensor_digest(value):
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest().upper()


def _input_digest(batch):
    ids = batch["ids"].detach().cpu().contiguous().numpy()
    return hashlib.sha256(ids.tobytes()).hexdigest().upper()


def _self_check():
    boundary = _PredictionBoundary()
    boundary_blocked = False
    try:
        boundary.require_target_access()
    except RuntimeError:
        boundary_blocked = True
    boundary.freeze("A", "B")
    boundary.require_target_access()

    response_local = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.2]],
        [[0.9, 0.1], [0.1, 1.1], [0.4, 0.3]],
    ])
    target = torch.tensor([1.0, 0.5, 0.6])
    coefficients, metadata = _ridge_coefficients(
        response_local, target, norm_cap=10.0)
    predicted = torch.einsum(
        "bhr,br->bh", response_local, coefficients)
    progress = (
        predicted * target
    ).sum(dim=-1) / target.square().sum()

    capped, cap_metadata = _ridge_coefficients(
        response_local, 100.0 * target, norm_cap=0.25)
    cap_ok = bool(
        (capped.norm(dim=-1) <= 0.25 + 1e-6).all()
        and all(cap_metadata["cap_applied"]))

    basis = torch.eye(6)[:2]
    delta = _coefficients_to_delta(
        torch.tensor([[2.0, -1.0]]), basis, hidden_size=2)
    sign_ok = bool(
        torch.allclose(delta.flatten(), torch.tensor(
            [2.0, -1.0, 0.0, 0.0, 0.0, 0.0])))

    identity_direction = torch.randn(3, 7)
    dynamic = torch.randn(5, 7)
    baseline = torch.randn(5, 7)
    step = 0.2
    plus = baseline + step * (
        identity_direction[-1][None, :] + dynamic)
    minus = baseline - step * (
        identity_direction[-1][None, :] + dynamic)
    recovered = _processed_central_derivative(
        plus, minus, step, identity_direction)
    identity_ok = bool(torch.allclose(
        recovered, dynamic, atol=1e-5))

    rows = _pilot_rows()
    passed = bool(
        boundary_blocked
        and min(progress.tolist()) >= 0.95
        and min(metadata["predicted_target_cosine"]) >= 0.95
        and cap_ok and sign_ok and identity_ok
        and len(rows["training"]) == TRAIN_N
        and all(
            len(rows[f"identify_{direction}"]) == IDENTIFICATION_N
            and len(rows[f"test_{direction}"]) == TEST_N
            for direction in DIRECTIONS))
    if not passed:
        raise AssertionError("held-out inverse-control self-check failed")
    return {
        "prediction_boundary_blocks_target": boundary_blocked,
        "ridge_inverse_check": True,
        "norm_cap_check": cap_ok,
        "coefficient_to_patch_sign_check": sign_ok,
        "direct_identity_removal_check": identity_ok,
        "row_split_check": True,
        "pass": True,
    }


def _arm_summary(accumulator):
    result = {}
    for checkpoint in CHECKPOINT_LAYERS:
        row = accumulator[checkpoint]
        result[str(checkpoint)] = {
            "raw": _direction_summary(
                row["raw_progress"], row["raw_distance"],
                accumulator["accuracies"]),
            "processed": _direction_summary(
                row["processed_progress"],
                row["processed_distance"],
                accumulator["accuracies"]),
        }
    return result


def _cell_adjudication(arms):
    def processed(arm):
        return arms[arm][str(PRIMARY_CHECKPOINT)]["processed"]

    exact = processed("exact_state_oracle")
    target_oracle = processed("target_informed_inverse_oracle")
    local = processed("local_inverse")
    shared = processed("shared_inverse")
    wrong = processed("wrong_context_inverse")
    opposite = processed("opposite_origin_inverse")
    negative = processed("negative_local_inverse")
    random_values = [
        processed(f"random_{index:02d}")["mean_progress"]
        for index in range(N_RANDOM)
    ]
    best_control = max(
        shared["mean_progress"],
        wrong["mean_progress"],
        opposite["mean_progress"],
        *random_values,
    )
    recovery = (
        local["mean_progress"]
        / target_oracle["mean_progress"]
        if target_oracle["mean_progress"] > 1e-8 else None
    )
    exact_pass = bool(
        exact["mean_progress"] >= MINIMUM_EXACT_PROGRESS - 1e-9
        and exact["minimum_answer_accuracy"]
        >= MINIMUM_VALUE_ACCURACY - 1e-9)
    capacity_pass = bool(
        target_oracle["mean_progress"]
        >= MINIMUM_TARGET_ORACLE_PROGRESS - 1e-9
        and target_oracle["minimum_answer_accuracy"]
        >= MINIMUM_VALUE_ACCURACY - 1e-9)
    local_level_pass = bool(
        local["mean_progress"] >= MINIMUM_LOCAL_PROGRESS - 1e-9
        and local["positive_rows"] >= MINIMUM_POSITIVE_ROWS
        and local["median_distance_ratio"]
        <= MAXIMUM_DISTANCE_RATIO + 1e-9
        and local["minimum_answer_accuracy"]
        >= MINIMUM_VALUE_ACCURACY - 1e-9
        and recovery is not None
        and recovery >= MINIMUM_ORACLE_RECOVERY - 1e-9)
    control_pass = bool(
        local["mean_progress"] - best_control
        >= MINIMUM_CONTROL_MARGIN - 1e-9
        and negative["mean_progress"] <= 1e-9)
    return {
        "exact_oracle_pass": exact_pass,
        "local_map_capacity_pass": capacity_pass,
        "local_level_pass": local_level_pass,
        "control_pass": control_pass,
        "pass": bool(
            exact_pass and capacity_pass
            and local_level_pass and control_pass),
        "local_mean_progress": local["mean_progress"],
        "target_oracle_mean_progress": target_oracle["mean_progress"],
        "exact_mean_progress": exact["mean_progress"],
        "shared_mean_progress": shared["mean_progress"],
        "wrong_context_mean_progress": wrong["mean_progress"],
        "opposite_origin_mean_progress": opposite["mean_progress"],
        "random_mean_progress": random_values,
        "negative_mean_progress": negative["mean_progress"],
        "best_control_mean_progress": best_control,
        "margin_over_best_control":
            local["mean_progress"] - best_control,
        "recovery_of_target_informed_oracle": recovery,
    }


def _overall_adjudication(cells):
    exact_cells = [
        name for name, value in cells.items()
        if value["exact_oracle_pass"]
    ]
    capacity_cells = [
        name for name, value in cells.items()
        if value["local_map_capacity_pass"]
    ]
    local_level_cells = [
        name for name, value in cells.items()
        if value["local_level_pass"]
    ]
    passing = [
        name for name, value in cells.items() if value["pass"]
    ]
    shared_level_cells = [
        name for name, value in cells.items()
        if value["shared_mean_progress"]
        >= MINIMUM_LOCAL_PROGRESS - 1e-9
    ]
    represented_families = {
        name.split("/")[0] for name in passing
    }
    represented_directions = {
        name.split("/")[1] for name in passing
    }
    broad_pass = bool(
        len(passing) >= MINIMUM_PASSING_CELLS
        and represented_families == set(HELDOUT_FAMILIES)
        and represented_directions == set(DIRECTIONS))
    if len(exact_cells) < len(cells):
        verdict = "EXACT_OR_ASSAY_INELIGIBLE"
    elif len(capacity_cells) < MINIMUM_PASSING_CELLS:
        verdict = "LOCAL_MAP_OR_RANK4_CAPACITY_FAILED"
    elif len(local_level_cells) < MINIMUM_PASSING_CELLS:
        verdict = "TRAINING_ONLY_TARGET_FAILED"
    elif (
            len(shared_level_cells) >= MINIMUM_PASSING_CELLS
            and len(passing) < MINIMUM_PASSING_CELLS):
        verdict = "SHARED_INVERSE_SUFFICIENT_OR_CONTEXT_GAIN_ABSENT"
    elif broad_pass:
        verdict = "PROSPECTIVE_HELDOUT_INVERSE_STATE_CONTROL"
    else:
        verdict = "SPECIFICITY_OR_BREADTH_FAILED"
    return {
        "exact_oracle_pass_cells": exact_cells,
        "local_map_capacity_pass_cells": capacity_cells,
        "local_level_pass_cells": local_level_cells,
        "shared_level_cells": shared_level_cells,
        "passing_cells": passing,
        "represented_families": sorted(represented_families),
        "represented_directions": sorted(represented_directions),
        "broad_pass": broad_pass,
        "verdict": verdict,
    }


@torch.no_grad()
def run_delta_heldout_inverse_control(
        model_path, out_dir,
        model_key="qwen7b_heldout_inverse_control",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=6, self_test_only=False):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_world) != TEST_N:
        raise ValueError(
            "v1 is frozen to six identification and six test rows "
            "per direction")
    self_check = _self_check()
    if self_test_only:
        result = {
            "stage": "delta_heldout_inverse_control",
            "protocol_sha256": PROTOCOL_SHA256,
            "self_check": self_check,
            "verdict": "SELF_CHECK_PASS",
        }
        path = os.path.join(
            out_dir, "delta_heldout_inverse_control_self_check.json")
        with open(path, "w") as handle:
            json.dump(result, handle, indent=2)
        log(
            "HELDOUT-INVERSE-CONTROL self-check pass "
            f"protocol={PROTOCOL_SHA256}")
        return result

    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    if PRIMARY_CHECKPOINT >= model_num_hidden_layers(model):
        raise ValueError("frozen checkpoint is absent")
    rows = _pilot_rows()

    training_alignments = {}
    training_baselines = {}
    templates = {}
    family_targets = {}
    training_capture_hb = Heartbeat(
        len(TRAIN_FAMILIES) * 2 * 2,
        "inverse_training_capture",
        every_sec=30, out_dir=out_dir)
    for family in TRAIN_FAMILIES:
        alignment = _family_alignment(
            tok, dev, rows["training"], FAMILY_SPECS[family])
        training_alignments[family] = alignment
        training_baselines[family] = {}
        template_samples = []
        target_samples = []
        for history, (belief_batch, search_batch) in enumerate(
                alignment["batches"]):
            belief = _capture_baseline(
                model, belief_batch, SOURCE_LAYER,
                alignment["answer_positions"])
            training_capture_hb.step(
                extra=f"{family}/h{history}/belief")
            search = _capture_baseline(
                model, search_batch, SOURCE_LAYER,
                alignment["answer_positions"])
            training_capture_hb.step(
                extra=f"{family}/h{history}/search")
            training_baselines[family][(history, "belief")] = belief
            training_baselines[family][(history, "search")] = search
            template_samples.append(search["source"] - belief["source"])
            target_samples.append(
                _training_processed(search) - _training_processed(belief))
        templates[family] = torch.cat(template_samples).mean(dim=0)
        family_targets[family] = torch.cat(target_samples).mean(dim=0)
    training_capture_hb.done()

    basis, basis_metadata = _template_basis(templates)
    hidden_size = int(basis.shape[1] // PATCH_WIDTH)
    basis_shaped = basis.reshape(
        BASIS_RANK, PATCH_WIDTH, hidden_size)
    controller_norms = [
        float(templates[family].norm()) for family in TRAIN_FAMILIES
    ]
    median_controller_norm = float(np.median(controller_norms))
    probe_step = PROBE_FRACTION * median_controller_norm
    norm_cap = MAXIMUM_NORM_MULTIPLIER * median_controller_norm
    shared_target, target_metadata = _shared_training_target(
        family_targets)
    if (
            target_metadata["median_pairwise_cosine"]
            < MINIMUM_TRAIN_TARGET_COSINE - 1e-9):
        result = {
            "stage": "delta_heldout_inverse_control",
            "protocol": PROTOCOL,
            "protocol_sha256": PROTOCOL_SHA256,
            "model_key": model_key,
            "self_check": self_check,
            "basis_metadata": basis_metadata,
            "training_target_metadata": target_metadata,
            "verdict": "TRAINING_TARGET_UNRESOLVED",
        }
        path = os.path.join(
            out_dir,
            f"results_delta_heldout_inverse_control_{model_key}.json")
        with open(path, "w") as handle:
            json.dump(result, handle, indent=2, default=float)
        log(
            "HELDOUT-INVERSE-CONTROL "
            f"verdict={result['verdict']} "
            f"target_cos={target_metadata['median_pairwise_cosine']:.3f}")
        return result

    training_probe_hb = Heartbeat(
        len(TRAIN_FAMILIES) * 2 * 2 * BASIS_RANK * 2,
        "inverse_training_response_maps",
        every_sec=30, out_dir=out_dir)
    training_maps = {"belief": [], "search": []}
    training_probe_accuracy = []
    for family in TRAIN_FAMILIES:
        alignment = training_alignments[family]
        for history, pair in enumerate(alignment["batches"]):
            expected = (
                [row["source"] for row in rows["training"]]
                if history == 0
                else [row["target"] for row in rows["training"]])
            for operation, batch in (
                    ("belief", pair[0]), ("search", pair[1])):
                response, accuracies = _capture_response_map(
                    model, batch, alignment["answer_positions"],
                    training_baselines[family][(history, operation)],
                    basis_shaped, probe_step, expected,
                    training_probe_hb,
                    f"{family}/{operation}/h{history}")
                training_maps[operation].append(response)
                training_probe_accuracy.extend(accuracies)
    training_probe_hb.done()
    shared_maps = {
        "belief_to_search": torch.cat(
            training_maps["belief"]).mean(dim=0),
        "search_to_belief": torch.cat(
            training_maps["search"]).mean(dim=0),
    }

    # Target-blind held-out phase. Identification and causal test rows are
    # disjoint, and only each split's origin command is rendered.
    boundary = _PredictionBoundary()
    test_origin_alignments = {}
    test_origin_baselines = {}
    local_maps = {}
    local_map_reliability = {}
    heldout_probe_accuracy = []
    origin_capture_hb = Heartbeat(
        len(HELDOUT_FAMILIES) * len(DIRECTIONS) * 2
        * (2 + BASIS_RANK * 2),
        "inverse_target_blind_identification",
        every_sec=30, out_dir=out_dir)
    for family in HELDOUT_FAMILIES:
        spec = HELDOUT_SPECS[family]
        test_origin_alignments[family] = {}
        test_origin_baselines[family] = {}
        local_maps[family] = {}
        local_map_reliability[family] = {}
        for direction in DIRECTIONS:
            origin_operation = (
                "belief" if direction == "belief_to_search" else "search")
            identify_rows = rows[f"identify_{direction}"]
            identify_alignment = _origin_alignment(
                tok, dev, identify_rows, spec, origin_operation)
            history_maps = []
            for history, batch in enumerate(
                    identify_alignment["batches"]):
                expected = (
                    [row["source"] for row in identify_rows]
                    if history == 0
                    else [row["target"] for row in identify_rows])
                origin = _capture_baseline(
                    model, batch, SOURCE_LAYER,
                    identify_alignment["answer_positions"])
                origin_capture_hb.step(
                    extra=(
                        f"{family}/{direction}/identify/h{history}/"
                        "baseline"))
                response, accuracies = _capture_response_map(
                    model, batch, identify_alignment["answer_positions"],
                    origin, basis_shaped, probe_step, expected,
                    origin_capture_hb,
                    f"{family}/{direction}/identify/h{history}")
                history_maps.append(response)
                heldout_probe_accuracy.extend(accuracies)
            local_maps[family][direction] = torch.cat(
                history_maps).mean(dim=0)
            history_means = [
                value.mean(dim=0) for value in history_maps
            ]
            local_map_reliability[family][direction] = {
                "clean_natural_map_cosine": _cosine(
                    history_means[0], history_means[1]),
                "identification_map_norm": float(
                    local_maps[family][direction].norm()),
            }

            test_rows = rows[f"test_{direction}"]
            test_alignment = _origin_alignment(
                tok, dev, test_rows, spec, origin_operation)
            test_origin_alignments[family][direction] = test_alignment
            test_origin_baselines[family][direction] = {}
            for history, batch in enumerate(test_alignment["batches"]):
                origin = _capture_baseline(
                    model, batch, SOURCE_LAYER,
                    test_alignment["answer_positions"])
                test_origin_baselines[family][direction][history] = origin
                origin_capture_hb.step(
                    extra=(
                        f"{family}/{direction}/test/h{history}/"
                        "origin"))
    origin_capture_hb.done()

    predictions = {}
    prediction_arrays = {
        "basis": basis.numpy().astype(np.float32),
        "shared_target_belief_to_search":
            shared_target.numpy().astype(np.float32),
        "shared_target_search_to_belief":
            (-shared_target).numpy().astype(np.float32),
        "shared_map_belief_to_search":
            shared_maps["belief_to_search"].numpy().astype(np.float32),
        "shared_map_search_to_belief":
            shared_maps["search_to_belief"].numpy().astype(np.float32),
    }
    for family_index, family in enumerate(HELDOUT_FAMILIES):
        other_family = next(
            value for value in HELDOUT_FAMILIES if value != family)
        predictions[family] = {}
        for direction in DIRECTIONS:
            target = (
                shared_target
                if direction == "belief_to_search"
                else -shared_target)
            opposite_direction = (
                "search_to_belief"
                if direction == "belief_to_search"
                else "belief_to_search")
            local_map = local_maps[family][direction]
            local_z, local_meta = _ridge_coefficients(
                local_map, target, norm_cap)
            shared_z, shared_meta = _ridge_coefficients(
                shared_maps[direction], target, norm_cap)
            wrong_z, wrong_meta = _ridge_coefficients(
                local_maps[other_family][direction],
                target, norm_cap)
            wrong_z = _norm_match_coefficients(wrong_z, local_z)
            opposite_z, opposite_meta = _ridge_coefficients(
                local_maps[family][opposite_direction],
                target, norm_cap)
            opposite_z = _norm_match_coefficients(
                opposite_z, local_z)
            arm_coefficients = {
                "local_inverse": local_z,
                "shared_inverse": shared_z,
                "wrong_context_inverse": wrong_z,
                "opposite_origin_inverse": opposite_z,
                "negative_local_inverse": -local_z,
            }
            for random_index in range(N_RANDOM):
                arm_coefficients[f"random_{random_index:02d}"] = (
                    _random_coefficients(
                        local_z,
                        RANDOM_SEED
                        + family_index * 100003
                        + DIRECTIONS.index(direction) * 1009
                        + random_index))
            predictions[family][direction] = {
                "test_origin_input_sha256": {
                    str(history): _input_digest(
                        test_origin_alignments[family][direction][
                            "batches"][history])
                    for history in range(2)
                },
                "local_map_sha256": _tensor_digest(local_map),
                "local_map_reliability":
                    local_map_reliability[family][direction],
                "coefficient_sha256": {
                    name: _tensor_digest(value)
                    for name, value in arm_coefficients.items()
                },
                "local_inverse_metadata": local_meta,
                "shared_inverse_metadata": shared_meta,
                "wrong_context_inverse_metadata": wrong_meta,
                "opposite_origin_inverse_metadata": opposite_meta,
            }
            prediction_arrays[
                f"{family}_{direction}_local_map"] = (
                    local_map.numpy().astype(np.float32))
            for name, value in arm_coefficients.items():
                prediction_arrays[
                    f"{family}_{direction}_{name}"] = (
                        value.numpy().astype(np.float32))

    prediction_npz_path = os.path.join(
        out_dir, f"heldout_inverse_predictions_{model_key}.npz")
    np.savez_compressed(prediction_npz_path, **prediction_arrays)
    prediction_npz_sha = _artifact_sha256(prediction_npz_path)
    prediction_json_path = os.path.join(
        out_dir, f"heldout_inverse_prediction_freeze_{model_key}.json")
    prediction_artifact = {
        "protocol_sha256": PROTOCOL_SHA256,
        "statement": (
            "Frozen before rendering or evaluating any held-out "
            "opposite-operation batch."),
        "basis_metadata": basis_metadata,
        "training_target_metadata": target_metadata,
        "probe_step": probe_step,
        "norm_cap": norm_cap,
        "minimum_training_probe_accuracy":
            float(min(training_probe_accuracy)),
        "minimum_heldout_probe_accuracy":
            float(min(heldout_probe_accuracy)),
        "prediction_npz_sha256": prediction_npz_sha,
        "predictions": predictions,
    }
    with open(prediction_json_path, "w") as handle:
        json.dump(prediction_artifact, handle, indent=2, default=float)
    prediction_json_sha = _artifact_sha256(prediction_json_path)
    boundary.freeze(prediction_json_sha, prediction_npz_sha)
    log(
        "FROZEN held-out inverse predictions "
        f"json={prediction_json_sha} npz={prediction_npz_sha}")

    # Opposite-operation access starts here and is mechanically guarded.
    boundary.require_target_access()
    if _artifact_sha256(prediction_json_path) != prediction_json_sha:
        raise AssertionError(
            "frozen prediction JSON changed before evaluation")
    if _artifact_sha256(prediction_npz_path) != prediction_npz_sha:
        raise AssertionError("frozen prediction NPZ changed before evaluation")
    with np.load(prediction_npz_path) as frozen_archive:
        frozen_prediction_arrays = {
            key: np.array(frozen_archive[key])
            for key in frozen_archive.files
        }
    predicted_arm_names = [
        "local_inverse",
        "shared_inverse",
        "wrong_context_inverse",
        "opposite_origin_inverse",
        "negative_local_inverse",
        *[f"random_{index:02d}" for index in range(N_RANDOM)],
    ]
    oracle_arm_names = [
        "basis_projection_oracle",
        "target_informed_inverse_oracle",
        "exact_state_oracle",
    ]
    arm_names = predicted_arm_names + oracle_arm_names
    evaluation_hb = Heartbeat(
        len(HELDOUT_FAMILIES) * len(DIRECTIONS)
        * 2 * len(arm_names),
        "inverse_post_freeze_evaluation",
        every_sec=30, out_dir=out_dir)
    cell_results = {}
    oracle_metadata = {}
    for family in HELDOUT_FAMILIES:
        spec = HELDOUT_SPECS[family]
        oracle_metadata[family] = {}
        for direction in DIRECTIONS:
            test_rows = rows[f"test_{direction}"]
            full_alignment = _family_alignment(
                tok, dev, test_rows, spec)
            origin_operation, target_operation = (
                ("belief", "search")
                if direction == "belief_to_search"
                else ("search", "belief"))
            origin_index = 0 if origin_operation == "belief" else 1
            target_index = 0 if target_operation == "belief" else 1
            if (
                    full_alignment["answer_positions"]
                    != test_origin_alignments[family][direction][
                        "answer_positions"]):
                raise AssertionError(
                    "post-freeze answer positions changed")
            accumulators = {
                arm: {
                    checkpoint: {
                        "raw_progress": [],
                        "raw_distance": [],
                        "processed_progress": [],
                        "processed_distance": [],
                    }
                    for checkpoint in CHECKPOINT_LAYERS
                }
                for arm in arm_names
            }
            for arm in arm_names:
                accumulators[arm]["accuracies"] = []

            target_baselines = {}
            exact_processed_targets = []
            for history, pair in enumerate(full_alignment["batches"]):
                origin_batch = pair[origin_index]
                target_batch = pair[target_index]
                if (
                        _input_digest(origin_batch)
                        != predictions[family][direction][
                            "test_origin_input_sha256"][str(history)]):
                    raise AssertionError(
                        "post-freeze origin batch differs from frozen input")
                origin = test_origin_baselines[
                    family][direction][history]
                target_baseline = _capture_baseline(
                    model, target_batch, SOURCE_LAYER,
                    full_alignment["answer_positions"])
                target_baselines[history] = target_baseline
                exact_processed_targets.append(
                    _training_processed(target_baseline)
                    - _training_processed(origin))
            exact_processed_target_mean = torch.cat(
                exact_processed_targets).mean(dim=0)
            target_oracle_z, target_oracle_meta = (
                _ridge_coefficients(
                    local_maps[family][direction],
                    exact_processed_target_mean,
                    norm_cap))
            oracle_metadata[family][direction] = {
                "target_informed_inverse": target_oracle_meta,
                "basis_projection_energy_fraction": {},
            }

            for history, pair in enumerate(full_alignment["batches"]):
                origin_batch = pair[origin_index]
                origin = test_origin_baselines[
                    family][direction][history]
                target_baseline = target_baselines[history]
                exact_delta = (
                    target_baseline["source"] - origin["source"])
                exact_flat = exact_delta.flatten(1)
                basis_projection_z = exact_flat @ basis.T
                oracle_metadata[family][direction][
                    "basis_projection_energy_fraction"][str(history)] = (
                        basis_projection_z.square().sum(dim=-1)
                        / exact_flat.square().sum(
                            dim=-1).clamp_min(1e-8)
                    ).tolist()
                coefficients = {
                    name: torch.from_numpy(
                        frozen_prediction_arrays[
                            f"{family}_{direction}_{name}"]
                    ).expand(TEST_N, -1).clone()
                    for name in predicted_arm_names
                }
                coefficients["basis_projection_oracle"] = (
                    basis_projection_z)
                coefficients["target_informed_inverse_oracle"] = (
                    target_oracle_z.expand(TEST_N, -1).clone())
                expected = (
                    [row["source"] for row in test_rows]
                    if history == 0
                    else [row["target"] for row in test_rows])
                for arm_name in arm_names:
                    if arm_name == "exact_state_oracle":
                        delta = exact_delta
                    else:
                        delta = _coefficients_to_delta(
                            coefficients[arm_name],
                            basis, hidden_size)
                    patched = _run_exact_patch(
                        model, origin_batch, SOURCE_LAYER,
                        full_alignment["answer_positions"],
                        origin["source"] + delta)
                    accumulators[arm_name]["accuracies"].append(
                        float(_generic_accuracy(
                            patched["logits"], origin_batch,
                            expected, VALUES)))
                    patched_direct = (
                        origin["source"][:, -1, :]
                        + delta[:, -1, :])
                    for checkpoint in CHECKPOINT_LAYERS:
                        raw_progress, raw_distance = _row_transport(
                            origin[f"checkpoint_{checkpoint}"],
                            target_baseline[
                                f"checkpoint_{checkpoint}"],
                            patched[f"checkpoint_{checkpoint}"])
                        origin_processed = _processed_checkpoint_state(
                            origin[f"checkpoint_{checkpoint}"],
                            origin["source"])
                        target_processed = _processed_checkpoint_state(
                            target_baseline[
                                f"checkpoint_{checkpoint}"],
                            target_baseline["source"])
                        patched_processed = (
                            patched[f"checkpoint_{checkpoint}"]
                            - patched_direct)
                        processed_progress, processed_distance = (
                            _row_transport(
                                origin_processed,
                                target_processed,
                                patched_processed))
                        row = accumulators[arm_name][checkpoint]
                        row["raw_progress"].extend(raw_progress)
                        row["raw_distance"].extend(raw_distance)
                        row["processed_progress"].extend(
                            processed_progress)
                        row["processed_distance"].extend(
                            processed_distance)
                    evaluation_hb.step(
                        extra=(
                            f"{family}/{direction}/h{history}/"
                            f"{arm_name}"))
            arms = {
                name: _arm_summary(accumulators[name])
                for name in arm_names
            }
            key = f"{family}/{direction}"
            cell_results[key] = {
                "family": family,
                "direction": direction,
                "rows": test_rows,
                "arms": arms,
                "adjudication": _cell_adjudication(arms),
            }
            local = cell_results[key]["adjudication"]
            log(
                f"HELDOUT-INVERSE {key} "
                f"local={local['local_mean_progress']:+.3f} "
                f"target_oracle="
                f"{local['target_oracle_mean_progress']:+.3f} "
                f"best_control="
                f"{local['best_control_mean_progress']:+.3f} "
                f"pass={local['pass']}")
    evaluation_hb.done()

    adjudications = {
        name: value["adjudication"]
        for name, value in cell_results.items()
    }
    overall = _overall_adjudication(adjudications)
    result = {
        "stage": "delta_heldout_inverse_control",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "self_check": self_check,
        "basis_metadata": basis_metadata,
        "training_target_metadata": target_metadata,
        "probe_step": probe_step,
        "norm_cap": norm_cap,
        "minimum_training_probe_accuracy":
            float(min(training_probe_accuracy)),
        "minimum_heldout_probe_accuracy":
            float(min(heldout_probe_accuracy)),
        "prediction_freeze_sha256": prediction_json_sha,
        "prediction_npz_sha256": prediction_npz_sha,
        "cell_results": cell_results,
        "oracle_metadata": oracle_metadata,
        "overall_adjudication": overall,
        "verdict": overall["verdict"],
    }
    result_path = os.path.join(
        out_dir,
        f"results_delta_heldout_inverse_control_{model_key}.json")
    with open(result_path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"HELDOUT-INVERSE-CONTROL verdict={result['verdict']} "
        f"passes={overall['passing_cells']} artifact={result_path}")
    return result
