"""Prospective causal-atlas and parallel-transport experiment."""
from __future__ import annotations

import hashlib
import itertools
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
    _unused_rows,
)
from .delta_heterogeneous_family_screen import (
    FAMILY_ORDER,
    FAMILY_SPECS,
    VALUES,
    _family_batch,
    _validate_history_change,
)
from .delta_lexical_class import TASK_GRID, _padding_plan
from .delta_predictive_conditional_transport import (
    _capture_positions,
    _fit_low_rank,
    _predict_low_rank,
    _select_predictor,
)
from .delta_preprint_battery import _compatible_world_rows
from .delta_prospective_causal_sensitivity import _prospective_rows
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)


PROTOCOL_VERSION = "2026-07-27-p2-causal-atlas-v1"
SOURCE_LAYER = 21
PRIMARY_CHECKPOINT = 27
PATCH_WIDTH = 3
TRAIN_N = 24
VALIDATION_N = 8
TEST_N = 12
ATLAS_RANKS = (2, 4, 8, 16, 32)
FRAME_NAMES = ("epistemic", "communication", "search")
TRAIN_PANELS = ("anchor", "synonym_a")
TEST_PANEL = "synonym_b"
PANEL_LABELS = {
    "anchor": {
        "epistemic": "BELIEF",
        "communication": "TELL",
        "search": "SEARCH",
    },
    "synonym_a": {
        "epistemic": "THINK",
        "communication": "REPORT",
        "search": "FIND",
    },
    "synonym_b": {
        "epistemic": "KNOW",
        "communication": "SAY",
        "search": "LOOK",
    },
}
TRANSITIONS = tuple(
    (source, target)
    for source in FRAME_NAMES
    for target in FRAME_NAMES
    if source != target
)
N_RANDOM = 19
CAUSAL_RANDOM_INDICES = (0, 1, 2)
RANDOM_SEED = 991337

MINIMUM_EXACT_MEAN_PROGRESS = 0.45
MINIMUM_EXACT_POSITIVE_TRANSITIONS = 6
MINIMUM_ATLAS_MEAN_PROGRESS = 0.35
MINIMUM_ATLAS_POSITIVE_TRANSITIONS = 5
MINIMUM_RECOVERY_OF_EXACT = 0.55
MINIMUM_VALUE_ACCURACY = 0.80
MINIMUM_BASELINE_MARGIN = 0.08
MINIMUM_SPECIFICITY_MARGIN = 0.08
MINIMUM_COMPOSED_PROGRESS = 0.30
MINIMUM_COMPOSED_POSITIVE_TRANSITIONS = 5
MAXIMUM_COMPOSITION_AGREEMENT_ERROR = 0.35
MAXIMUM_INVERSE_LOOP_ERROR = 0.50
MINIMUM_CONTENT_EQUIVARIANCE_COSINE = 0.60
MAXIMUM_CONTENT_EQUIVARIANCE_ERROR = 0.85
MAXIMUM_POSITION_CONTROL_FRACTION = 0.50
MINIMUM_PASSING_FAMILIES = 6
MAXIMUM_RANDOM_P = 0.05


PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "status": "prospective structural test; a pass licenses replication",
    "hypothesis": (
        "Within heterogeneous computations, position-matched pretrained "
        "readout frames are local coordinate charts over a shared content "
        "state. A low-rank atlas learned from two lexical panels should "
        "transport a held-out lexical panel causally and obey approximate "
        "inverse, composition, and content-equivariance laws."
    ),
    "model": "Qwen2.5-7B-Instruct, 8-bit, Tesla T4",
    "families": list(FAMILY_ORDER),
    "frames": list(FRAME_NAMES),
    "panels": {
        panel: PANEL_LABELS[panel] for panel in PANEL_LABELS
    },
    "lexical_split": {
        "train": list(TRAIN_PANELS),
        "test": TEST_PANEL,
        "test_panel_never_used_for_fit_or_rank_selection": True,
    },
    "world_split": {
        "train_directed_pairs": TRAIN_N,
        "validation_directed_pairs": VALIDATION_N,
        "test_histories": TEST_N,
        "all_pair_sets_disjoint": True,
    },
    "locus": {
        "source_layer": SOURCE_LAYER,
        "positions": "last three position-matched answer-command tokens",
        "checkpoints": list(CHECKPOINT_LAYERS),
        "primary_checkpoint": PRIMARY_CHECKPOINT,
        "direct_identity_removed": True,
    },
    "atlas": {
        "construction": (
            "Per-family, per-frame PCA charts aligned by paired orthogonal "
            "Procrustes; off-chart source residual is preserved."
        ),
        "candidate_ranks": list(ATLAS_RANKS),
        "selection": (
            "minimum mean validation normalized transition MSE; lower rank "
            "is the deterministic tie-breaker"
        ),
        "refit": "training plus validation after rank selection",
        "source_only_prediction_hash_before_target_evaluation": True,
    },
    "primary_controls": [
        "exact matched target state",
        "independent pairwise reduced-rank predictor",
        "per-transition mean displacement",
        "wrong target frame",
        f"{N_RANDOM} row-shuffled atlas displacements",
        "atlas displacement at instruction positions",
        "atlas displacement at matched identical-token positions",
    ],
    "structural_predictions": {
        "inverse": "T_ba(T_ab(h)) approximately returns h",
        "composition": "T_bc(T_ab(h)) approximately equals T_ac(h)",
        "content_equivariance": (
            "T(h_natural)-T(h_clean) matches the target-frame content update"
        ),
    },
    "per_family_gates": {
        "minimum_exact_mean_progress": MINIMUM_EXACT_MEAN_PROGRESS,
        "minimum_exact_positive_transitions":
            MINIMUM_EXACT_POSITIVE_TRANSITIONS,
        "minimum_atlas_mean_progress": MINIMUM_ATLAS_MEAN_PROGRESS,
        "minimum_atlas_positive_transitions":
            MINIMUM_ATLAS_POSITIVE_TRANSITIONS,
        "minimum_recovery_of_exact": MINIMUM_RECOVERY_OF_EXACT,
        "minimum_value_accuracy": MINIMUM_VALUE_ACCURACY,
        "minimum_margin_over_mean_and_pairwise":
            MINIMUM_BASELINE_MARGIN,
        "minimum_margin_over_wrong_frame_and_causal_random":
            MINIMUM_SPECIFICITY_MARGIN,
        "maximum_instruction_or_identical_fraction":
            MAXIMUM_POSITION_CONTROL_FRACTION,
        "minimum_composed_progress": MINIMUM_COMPOSED_PROGRESS,
        "minimum_composed_positive_transitions":
            MINIMUM_COMPOSED_POSITIVE_TRANSITIONS,
        "maximum_composition_agreement_error":
            MAXIMUM_COMPOSITION_AGREEMENT_ERROR,
        "maximum_inverse_loop_error": MAXIMUM_INVERSE_LOOP_ERROR,
        "minimum_content_equivariance_cosine":
            MINIMUM_CONTENT_EQUIVARIANCE_COSINE,
        "maximum_content_equivariance_error":
            MAXIMUM_CONTENT_EQUIVARIANCE_ERROR,
    },
    "overall_gate": {
        "minimum_exact_eligible_families": MINIMUM_PASSING_FAMILIES,
        "minimum_full_atlas_families": MINIMUM_PASSING_FAMILIES,
        "maximum_add_one_random_p": MAXIMUM_RANDOM_P,
    },
    "verdicts": {
        "CAUSAL_ATLAS_WITH_COMPOSITION": (
            "Transport and all algebraic laws pass in at least six families."
        ),
        "CAUSAL_TRANSPORT_WITHOUT_ALGEBRA": (
            "Causal transport passes broadly but inverse/composition/content "
            "laws do not."
        ),
        "ALGEBRAIC_FIT_WITHOUT_CAUSAL_CONTROL": (
            "Offline atlas laws fit but causal transport does not."
        ),
        "NO_LOW_COMPLEXITY_CAUSAL_ATLAS": (
            "Exact references are eligible but the atlas does not pass."
        ),
        "ASSAY_INELIGIBLE": (
            "Exact matched-state references fail in too many families."
        ),
    },
    "stopping_rule": (
        "No prompt, split, panel, family, rank, metric, threshold, position, "
        "control, seed, or model-class rescue follows the output. A full pass "
        "licenses cross-model replication; any other verdict is interpreted "
        "as written."
    ),
    "random_seed": RANDOM_SEED,
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest().upper()


class _PredictionBoundary:
    def __init__(self):
        self._frozen = False
        self.json_sha256 = None
        self.npz_sha256 = None

    def freeze(self, json_sha256, npz_sha256):
        if not json_sha256 or not npz_sha256:
            raise ValueError("prediction hashes are required")
        self.json_sha256 = str(json_sha256)
        self.npz_sha256 = str(npz_sha256)
        self._frozen = True

    def require_evaluation(self):
        if not self._frozen:
            raise RuntimeError(
                "target evaluation attempted before prediction freeze")


def _artifact_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest().upper()


def _tensor_sha256(value):
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest().upper()


def _transition_key(source, target):
    return f"{source}_to_{target}"


def _middle_frame(source, target):
    return next(
        frame for frame in FRAME_NAMES
        if frame not in (source, target))


def _row_splits():
    prospective = _prospective_rows(VALUES)
    train = prospective[:TRAIN_N]
    validation = prospective[TRAIN_N:TRAIN_N + VALIDATION_N]
    test = _unused_rows()
    if len(train) != TRAIN_N or len(validation) != VALIDATION_N:
        raise AssertionError("atlas train/validation counts changed")
    if len(test) != TEST_N:
        raise AssertionError("atlas test count changed")
    pair_sets = {
        "train": {(row["source"], row["target"]) for row in train},
        "validation": {
            (row["source"], row["target"]) for row in validation},
        "test": {(row["source"], row["target"]) for row in test},
    }
    for left, right in itertools.combinations(pair_sets, 2):
        if pair_sets[left] & pair_sets[right]:
            raise AssertionError(f"{left}/{right} directed pairs overlap")
    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


def _commands(tok):
    compatible, _indices = _compatible_world_rows(
        tok, torch.device("cpu"), 30)
    _target, plan, tables = _padding_plan(tok, compatible[0])
    if plan is None:
        raise ValueError("nine-label position-matching plan is absent")
    commands = {}
    for panel, labels in PANEL_LABELS.items():
        commands[panel] = {
            frame: plan[TASK_GRID[frame][label]]["command"]
            for frame, label in labels.items()
        }
    return commands, plan, tables


def _contiguous_groups(positions):
    groups = []
    for position in positions:
        if not groups or position != groups[-1][-1] + 1:
            groups.append([position])
        else:
            groups[-1].append(position)
    return groups


def _frame_alignment(tok, dev, rows, spec, commands):
    histories = (
        rows,
        [{**row, "state": row["target"]} for row in rows],
    )
    batches = []
    reference_masks = None
    marker = None
    for history_rows in histories:
        frame_batches = {
            frame: _family_batch(
                tok, history_rows, spec, commands[frame], dev)
            for frame in FRAME_NAMES
        }
        shapes = {tuple(batch["ids"].shape)
                  for batch in frame_batches.values()}
        if len(shapes) != 1:
            raise ValueError("three frame shapes differ")
        markers = {int(batch["marker"])
                   for batch in frame_batches.values()}
        if len(markers) != 1:
            raise ValueError("three frame marker positions differ")
        current_marker = next(iter(markers))
        if marker is not None and marker != current_marker:
            raise ValueError("history marker positions differ")
        marker = current_marker
        masks = []
        for left, right in itertools.combinations(FRAME_NAMES, 2):
            difference = (
                frame_batches[left]["ids"] != frame_batches[right]["ids"])
            if not bool((difference == difference[0:1]).all()):
                raise ValueError("frame difference mask varies by row")
            masks.append(difference[0].detach().cpu())
        if reference_masks is not None:
            if any(not torch.equal(left, right)
                   for left, right in zip(reference_masks, masks)):
                raise ValueError("frame masks vary by history")
        reference_masks = masks
        batches.append(frame_batches)
    for frame in FRAME_NAMES:
        _validate_history_change(
            batches[0][frame], batches[1][frame])

    union_difference = torch.stack(reference_masks).any(dim=0)
    groups = _contiguous_groups(torch.nonzero(
        union_difference, as_tuple=False).flatten().tolist())
    if len(groups) < 2:
        raise ValueError(f"expected two command occurrences, got {groups}")
    readout = int(batches[0][FRAME_NAMES[0]]["ids"].shape[1] - 1)
    answer_positions = list(range(
        groups[-1][-1] - PATCH_WIDTH + 1,
        groups[-1][-1] + 1))
    instruction_positions = list(range(
        groups[0][-1] - PATCH_WIDTH + 1,
        groups[0][-1] + 1))
    if answer_positions != list(range(
            readout - PATCH_WIDTH + 1, readout + 1)):
        raise ValueError("answer command does not end at readout")
    excluded = set(answer_positions + instruction_positions)
    common = torch.stack([
        batches[0][frame]["ids"][0].detach().cpu()
        for frame in FRAME_NAMES
    ])
    identical = (common == common[0:1]).all(dim=0)
    candidates = [
        int(position)
        for position in torch.nonzero(
            identical, as_tuple=False).flatten().tolist()
        if position not in excluded and position < instruction_positions[0]
    ]
    identical_groups = _contiguous_groups(candidates)
    eligible = [group for group in identical_groups
                if len(group) >= PATCH_WIDTH]
    if not eligible:
        raise ValueError("no matched identical-token control window")
    identical_positions = eligible[-1][-PATCH_WIDTH:]
    return {
        "batches": batches,
        "marker": marker,
        "readout": readout,
        "answer_positions": answer_positions,
        "instruction_positions": instruction_positions,
        "identical_positions": identical_positions,
        "difference_groups": groups,
    }


def _flatten(value):
    return value.detach().float().cpu().flatten(1)


def _pca_chart(states, rank):
    states = states.float()
    mean = states.mean(dim=0)
    centered = states - mean
    gram = centered @ centered.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    usable = int(min(
        int(rank), states.shape[0] - 1,
        int((eigenvalues > 1e-8).sum())))
    if usable < 1:
        raise ValueError("atlas chart has zero usable rank")
    singular = eigenvalues[:usable].clamp_min(1e-8).sqrt()
    basis = (
        eigenvectors[:, :usable].T @ centered
        / singular[:, None])
    coordinates = centered @ basis.T
    return {
        "mean": mean,
        "basis": basis,
        "coordinates": coordinates,
        "rank": usable,
        "singular_values": singular,
    }


def _fit_atlas(states_by_frame, rank):
    charts = {
        frame: _pca_chart(states_by_frame[frame], rank)
        for frame in FRAME_NAMES
    }
    common_rank = min(chart["rank"] for chart in charts.values())
    reference = charts[FRAME_NAMES[0]]["coordinates"][:, :common_rank]
    for frame in FRAME_NAMES:
        chart = charts[frame]
        chart["basis"] = chart["basis"][:common_rank]
        chart["coordinates"] = chart["coordinates"][:, :common_rank]
        chart["singular_values"] = chart["singular_values"][:common_rank]
        cross = chart["coordinates"].T @ reference
        left, _singular, right_t = torch.linalg.svd(
            cross, full_matrices=False)
        chart["rotation"] = left @ right_t
        chart["rank"] = common_rank
    return {
        "rank": common_rank,
        "charts": charts,
    }


def _atlas_transport(atlas, source, target, states):
    source_chart = atlas["charts"][source]
    target_chart = atlas["charts"][target]
    value = states.float()
    source_centered = value - source_chart["mean"]
    source_coordinates = source_centered @ source_chart["basis"].T
    common = source_coordinates @ source_chart["rotation"]
    target_coordinates = common @ target_chart["rotation"].T
    source_reconstruction = (
        source_chart["mean"]
        + source_coordinates @ source_chart["basis"])
    target_reconstruction = (
        target_chart["mean"]
        + target_coordinates @ target_chart["basis"])
    residual = value - source_reconstruction
    return target_reconstruction + residual


def _transport_metrics(origin, target, predicted):
    direction = target.float() - origin.float()
    displacement = predicted.float() - origin.float()
    norm2 = direction.square().sum(dim=-1).clamp_min(1e-8)
    progress = (displacement * direction).sum(dim=-1) / norm2
    distance = (
        (predicted.float() - target.float()).norm(dim=-1)
        / direction.norm(dim=-1).clamp_min(1e-8))
    return {
        "mean_progress": float(progress.mean()),
        "median_progress": float(progress.median()),
        "positive_rows": int((progress > 0.0).sum()),
        "positive_fraction": float((progress > 0.0).float().mean()),
        "mean_distance_ratio": float(distance.mean()),
        "median_distance_ratio": float(distance.median()),
        "progress_rows": progress.tolist(),
        "distance_ratio_rows": distance.tolist(),
    }


def _validation_score(atlas, states_by_frame):
    values = []
    per_transition = {}
    for source, target in TRANSITIONS:
        origin = states_by_frame[source]
        exact = states_by_frame[target]
        predicted = _atlas_transport(
            atlas, source, target, origin)
        ratio = float(
            (predicted - exact).square().sum()
            / (exact - origin).square().sum().clamp_min(1e-8))
        per_transition[_transition_key(source, target)] = ratio
        values.append(ratio)
    return float(np.mean(values)), per_transition


def _select_atlas(train_states, validation_states):
    candidates = []
    models = {}
    for rank in ATLAS_RANKS:
        atlas = _fit_atlas(train_states, rank)
        score, per_transition = _validation_score(
            atlas, validation_states)
        candidates.append({
            "requested_rank": int(rank),
            "effective_rank": int(atlas["rank"]),
            "validation_normalized_mse": score,
            "per_transition": per_transition,
        })
        models[int(rank)] = atlas
    selected = min(
        candidates,
        key=lambda row: (
            row["validation_normalized_mse"],
            row["requested_rank"],
        ))
    return models[selected["requested_rank"]], {
        "selected": selected,
        "candidates": candidates,
    }


def _refit_pairwise(train_states, validation_states):
    models = {}
    metadata = {}
    for source, target in TRANSITIONS:
        key = _transition_key(source, target)
        train_x = train_states[source]
        train_y = train_states[target] - train_x
        validation_x = validation_states[source]
        validation_y = validation_states[target] - validation_x
        _selected_model, selection = _select_predictor(
            train_x, train_y, validation_x, validation_y)
        chosen = selection["selected"]
        all_x = torch.cat([train_x, validation_x], dim=0)
        all_y = torch.cat([train_y, validation_y], dim=0)
        models[key] = _fit_low_rank(
            all_x, all_y,
            chosen["effective_rank"], chosen["ridge"])
        metadata[key] = selection
    return models, metadata


def _pairwise_transport(models, source, target, states):
    key = _transition_key(source, target)
    return states + _predict_low_rank(models[key], states)


def _mean_transport(atlas, source, target, states):
    return (
        states
        + atlas["charts"][target]["mean"]
        - atlas["charts"][source]["mean"])


def _atlas_public(atlas):
    return {
        "rank": int(atlas["rank"]),
        "charts": {
            frame: {
                "mean_sha256": _tensor_sha256(
                    atlas["charts"][frame]["mean"]),
                "basis_sha256": _tensor_sha256(
                    atlas["charts"][frame]["basis"]),
                "rotation_sha256": _tensor_sha256(
                    atlas["charts"][frame]["rotation"]),
                "singular_values": atlas["charts"][frame][
                    "singular_values"].tolist(),
            }
            for frame in FRAME_NAMES
        },
    }


def _tail_probability(observed, random_values):
    exceed = sum(
        float(value) >= float(observed)
        for value in random_values)
    return (1.0 + exceed) / (1.0 + len(random_values)), exceed


def _content_equivariance(origin, target, predicted):
    if origin.shape[0] != 2 * TEST_N:
        raise ValueError("content equivariance expects paired histories")
    source_update = (
        predicted[TEST_N:] - predicted[:TEST_N])
    target_update = target[TEST_N:] - target[:TEST_N]
    cosine = torch.nn.functional.cosine_similarity(
        source_update, target_update, dim=-1)
    relative_error = (
        (source_update - target_update).norm(dim=-1)
        / target_update.norm(dim=-1).clamp_min(1e-8))
    progress = (
        (source_update * target_update).sum(dim=-1)
        / target_update.square().sum(dim=-1).clamp_min(1e-8))
    return {
        "mean_cosine": float(cosine.mean()),
        "median_cosine": float(cosine.median()),
        "mean_relative_error": float(relative_error.mean()),
        "median_relative_error": float(relative_error.median()),
        "mean_progress": float(progress.mean()),
        "cosine_rows": cosine.tolist(),
        "relative_error_rows": relative_error.tolist(),
    }


def _inverse_metrics(atlas, states_by_frame):
    errors = []
    per_transition = {}
    for source, target in TRANSITIONS:
        origin = states_by_frame[source]
        target_state = states_by_frame[target]
        forward = _atlas_transport(
            atlas, source, target, origin)
        loop = _atlas_transport(
            atlas, target, source, forward)
        error = (
            (loop - origin).norm(dim=-1)
            / (target_state - origin).norm(dim=-1).clamp_min(1e-8))
        key = _transition_key(source, target)
        per_transition[key] = {
            "mean_relative_error": float(error.mean()),
            "median_relative_error": float(error.median()),
            "rows": error.tolist(),
        }
        errors.extend(error.tolist())
    return {
        "mean_relative_error": float(np.mean(errors)),
        "median_relative_error": float(np.median(errors)),
        "per_transition": per_transition,
    }


def _composition_metrics(atlas, states_by_frame):
    errors = []
    progress = []
    per_transition = {}
    for source, target in TRANSITIONS:
        middle = _middle_frame(source, target)
        origin = states_by_frame[source]
        target_state = states_by_frame[target]
        direct = _atlas_transport(
            atlas, source, target, origin)
        composed = _atlas_transport(
            atlas, middle, target,
            _atlas_transport(atlas, source, middle, origin))
        scale = (target_state - origin).norm(
            dim=-1).clamp_min(1e-8)
        agreement = (composed - direct).norm(dim=-1) / scale
        metric = _transport_metrics(origin, target_state, composed)
        key = _transition_key(source, target)
        per_transition[key] = {
            **metric,
            "mean_agreement_error": float(agreement.mean()),
            "median_agreement_error": float(agreement.median()),
            "agreement_error_rows": agreement.tolist(),
        }
        errors.extend(agreement.tolist())
        progress.extend(metric["progress_rows"])
    return {
        "mean_agreement_error": float(np.mean(errors)),
        "median_agreement_error": float(np.median(errors)),
        "mean_target_progress": float(np.mean(progress)),
        "positive_fraction": float(np.mean([
            value > 0.0 for value in progress])),
        "per_transition": per_transition,
    }


def _capture_training_states(model, alignments, heartbeat, family):
    captured = {}
    for panel, alignment in alignments.items():
        captured[panel] = []
        for history, frame_batches in enumerate(alignment["batches"]):
            row = {}
            for frame in FRAME_NAMES:
                row[frame] = _capture_positions(
                    model, frame_batches[frame],
                    alignment["answer_positions"])
                heartbeat.step(
                    extra=f"{family}/{panel}/h{history}/{frame}/fit_capture")
            captured[panel].append(row)
    return captured


def _stack_captured(captured, panels, frame, row_slice):
    values = []
    for panel in panels:
        for history in range(2):
            values.append(
                captured[panel][history][frame][row_slice])
    return _flatten(torch.cat(values, dim=0))


def _capture_source_test(model, alignment, heartbeat, family):
    captured = {frame: [] for frame in FRAME_NAMES}
    for history, frame_batches in enumerate(alignment["batches"]):
        for frame in FRAME_NAMES:
            captured[frame].append(_capture_positions(
                model, frame_batches[frame],
                alignment["answer_positions"]))
            heartbeat.step(
                extra=f"{family}/{TEST_PANEL}/h{history}/{frame}/source_only")
    return {
        frame: _flatten(torch.cat(captured[frame], dim=0))
        for frame in FRAME_NAMES
    }


def _random_permutations(family_index, transition_index):
    result = {}
    for random_index in range(N_RANDOM):
        history_indices = []
        for history in range(2):
            seed = (
                RANDOM_SEED
                + family_index * 1000003
                + transition_index * 10007
                + random_index * 101
                + history)
            generator = torch.Generator().manual_seed(seed)
            permutation = torch.randperm(
                TEST_N, generator=generator)
            if torch.equal(permutation, torch.arange(TEST_N)):
                permutation = torch.roll(permutation, shifts=1)
            history_indices.append(permutation)
        result[random_index] = history_indices
    return result


def _shuffle_delta(delta, permutations):
    if delta.shape[0] != 2 * TEST_N:
        raise ValueError("random control expects paired test histories")
    return torch.cat([
        delta[
            history * TEST_N
            + permutations[history]]
        for history in range(2)
    ], dim=0)


def _build_predictions(
        atlas, pairwise, test_states, family_index):
    predictions = {}
    permutation_metadata = {}
    for transition_index, (source, target) in enumerate(TRANSITIONS):
        key = _transition_key(source, target)
        middle = _middle_frame(source, target)
        origin = test_states[source]
        atlas_prediction = _atlas_transport(
            atlas, source, target, origin)
        composed = _atlas_transport(
            atlas, middle, target,
            _atlas_transport(atlas, source, middle, origin))
        wrong = _atlas_transport(
            atlas, source, middle, origin)
        predictions[key] = {
            "atlas": atlas_prediction,
            "composed": composed,
            "pairwise": _pairwise_transport(
                pairwise, source, target, origin),
            "mean": _mean_transport(
                atlas, source, target, origin),
            "wrong_target": wrong,
        }
        permutations = _random_permutations(
            family_index, transition_index)
        permutation_metadata[key] = {
            str(index): [
                value.tolist() for value in permutations[index]
            ]
            for index in range(N_RANDOM)
        }
    return predictions, permutation_metadata


def _prediction_public(predictions):
    return {
        transition: {
            arm: {
                "shape": list(value.shape),
                "sha256": _tensor_sha256(value),
            }
            for arm, value in arms.items()
        }
        for transition, arms in predictions.items()
    }


def _write_prediction_freeze(
        out_dir, models_public, predictions_by_family,
        permutation_metadata):
    metadata = {
        "protocol_sha256": PROTOCOL_SHA256,
        "models": models_public,
        "predictions": {
            family: _prediction_public(
                predictions_by_family[family])
            for family in FAMILY_ORDER
        },
        "random_permutations": permutation_metadata,
    }
    json_path = os.path.join(
        out_dir, "causal_atlas_prediction_freeze.json")
    with open(json_path, "w") as handle:
        json.dump(metadata, handle, indent=2)
    arrays = {
        f"{family}__{transition}__{arm}": (
            predictions_by_family[family][transition][
                arm].numpy())
        for family in FAMILY_ORDER
        for transition in predictions_by_family[family]
        for arm in ("atlas", "composed")
    }
    npz_path = os.path.join(
        out_dir, "causal_atlas_predictions.npz")
    np.savez_compressed(npz_path, **arrays)
    return {
        "json_path": json_path,
        "npz_path": npz_path,
        "json_sha256": _artifact_sha256(json_path),
        "npz_sha256": _artifact_sha256(npz_path),
        "metadata": metadata,
    }


def _reload_atlas_predictions(freeze, predictions_by_family):
    with np.load(freeze["npz_path"]) as archive:
        for family in FAMILY_ORDER:
            for transition in predictions_by_family[family]:
                for arm in ("atlas", "composed"):
                    key = f"{family}__{transition}__{arm}"
                    value = torch.from_numpy(
                        np.array(archive[key])).float()
                    expected = freeze["metadata"]["predictions"][
                        family][transition][arm]["sha256"]
                    if _tensor_sha256(value) != expected:
                        raise AssertionError(
                            f"reloaded prediction changed: {key}")
                    predictions_by_family[family][transition][
                        arm] = value


def _capture_evaluation_baselines(
        model, alignment, source_test, heartbeat, family):
    union = sorted(set(
        alignment["answer_positions"]
        + alignment["instruction_positions"]
        + alignment["identical_positions"]))
    index = {position: offset for offset, position in enumerate(union)}
    answer_offsets = [
        index[position] for position in alignment["answer_positions"]]
    instruction_offsets = [
        index[position] for position in alignment[
            "instruction_positions"]]
    identical_offsets = [
        index[position] for position in alignment[
            "identical_positions"]]
    baselines = []
    for history, frame_batches in enumerate(alignment["batches"]):
        row = {}
        for frame in FRAME_NAMES:
            cache = _capture_baseline(
                model, frame_batches[frame], SOURCE_LAYER, union)
            cache["answer_source"] = cache["source"][:, answer_offsets, :]
            cache["instruction_source"] = (
                cache["source"][:, instruction_offsets, :])
            cache["identical_source"] = (
                cache["source"][:, identical_offsets, :])
            predicted_source = source_test[frame][
                history * TEST_N:(history + 1) * TEST_N]
            if not torch.allclose(
                    _flatten(cache["answer_source"]),
                    predicted_source, atol=1e-5, rtol=1e-5):
                raise AssertionError(
                    "source-only and evaluation recaptures differ")
            row[frame] = cache
            heartbeat.step(
                extra=f"{family}/{TEST_PANEL}/h{history}/{frame}/eval_capture")
        baselines.append(row)
    return baselines


def _expected_values(rows, history):
    key = "source" if int(history) == 0 else "target"
    return [row[key] for row in rows]


def _processed(checkpoint, direct):
    return checkpoint.float() - direct.float()


def _append_causal(
        accumulator, origin, target, patched, patched_direct,
        answer_accuracy):
    accumulator["accuracies"].append(float(answer_accuracy))
    for checkpoint in CHECKPOINT_LAYERS:
        origin_processed = _processed(
            origin[f"checkpoint_{checkpoint}"],
            origin["answer_source"][:, -1, :])
        target_processed = _processed(
            target[f"checkpoint_{checkpoint}"],
            target["answer_source"][:, -1, :])
        patched_processed = _processed(
            patched[f"checkpoint_{checkpoint}"],
            patched_direct)
        progress, distance = _row_transport(
            origin_processed, target_processed, patched_processed)
        accumulator[checkpoint]["progress"].extend(progress)
        accumulator[checkpoint]["distance"].extend(distance)


def _empty_causal():
    return {
        "accuracies": [],
        **{
            checkpoint: {"progress": [], "distance": []}
            for checkpoint in CHECKPOINT_LAYERS
        },
    }


def _causal_summary(accumulator):
    return {
        str(checkpoint): _direction_summary(
            accumulator[checkpoint]["progress"],
            accumulator[checkpoint]["distance"],
            accumulator["accuracies"])
        for checkpoint in CHECKPOINT_LAYERS
    }


def _prediction_history(value, history):
    return value[
        history * TEST_N:(history + 1) * TEST_N]


def _random_prediction(
        origin_flat, atlas_flat, permutation_metadata,
        random_index):
    permutations = [
        torch.tensor(value, dtype=torch.long)
        for value in permutation_metadata[str(random_index)]
    ]
    delta = atlas_flat - origin_flat
    return origin_flat + _shuffle_delta(delta, permutations)


def _run_transition_causal(
        model, alignment, baselines, rows, source, target,
        predictions, permutation_metadata, heartbeat, family):
    key = _transition_key(source, target)
    arm_names = [
        "atlas",
        "composed",
        "pairwise",
        "mean",
        "wrong_target",
        "exact_state_oracle",
        "instruction",
        "identical",
        *[
            f"random_{index:02d}"
            for index in CAUSAL_RANDOM_INDICES
        ],
    ]
    accumulators = {
        arm: _empty_causal() for arm in arm_names
    }
    origin_all = torch.cat([
        baselines[history][source]["answer_source"]
        for history in range(2)
    ], dim=0)
    origin_flat = _flatten(origin_all)
    random_predictions = {
        f"random_{index:02d}": _random_prediction(
            origin_flat, predictions["atlas"],
            permutation_metadata, index)
        for index in CAUSAL_RANDOM_INDICES
    }
    for history in range(2):
        origin = baselines[history][source]
        target_baseline = baselines[history][target]
        batch = alignment["batches"][history][source]
        expected = _expected_values(rows, history)
        for arm in arm_names:
            if arm == "exact_state_oracle":
                positions = alignment["answer_positions"]
                target_values = target_baseline["answer_source"]
                patched_direct = target_values[:, -1, :]
            elif arm in ("instruction", "identical"):
                positions = alignment[
                    f"{arm}_positions"]
                base = origin[f"{arm}_source"]
                atlas_values = _prediction_history(
                    predictions["atlas"], history).reshape_as(
                        origin["answer_source"])
                delta = atlas_values - origin["answer_source"]
                target_values = base + delta
                patched_direct = origin["answer_source"][:, -1, :]
            else:
                positions = alignment["answer_positions"]
                if arm.startswith("random_"):
                    flat = random_predictions[arm]
                else:
                    flat = predictions[arm]
                target_values = _prediction_history(
                    flat, history).reshape_as(
                        origin["answer_source"])
                patched_direct = target_values[:, -1, :]
            patched = _run_exact_patch(
                model, batch, SOURCE_LAYER,
                positions, target_values)
            accuracy = _generic_accuracy(
                patched["logits"], batch, expected, VALUES)
            _append_causal(
                accumulators[arm], origin, target_baseline,
                patched, patched_direct, accuracy)
            heartbeat.step(
                extra=f"{family}/{key}/h{history}/{arm}")
    return {
        arm: _causal_summary(accumulators[arm])
        for arm in arm_names
    }


def _offline_transition(
        atlas, source, target, test_states, predictions,
        permutation_metadata):
    key = _transition_key(source, target)
    origin = test_states[source]
    exact = test_states[target]
    metrics = {
        arm: _transport_metrics(origin, exact, value)
        for arm, value in predictions.items()
    }
    random_metrics = {}
    for index in range(N_RANDOM):
        random_value = _random_prediction(
            origin, predictions["atlas"],
            permutation_metadata, index)
        random_metrics[f"random_{index:02d}"] = (
            _transport_metrics(origin, exact, random_value))
    content = _content_equivariance(
        origin, exact, predictions["atlas"])
    return {
        "transition": key,
        "arms": metrics,
        "random": random_metrics,
        "content_equivariance": content,
    }


def _mean_causal(transitions, arm):
    values = [
        transition["causal"][arm][str(PRIMARY_CHECKPOINT)][
            "mean_progress"]
        for transition in transitions.values()
    ]
    return float(np.mean(values))


def _positive_causal(transitions, arm):
    return int(sum(
        transition["causal"][arm][str(PRIMARY_CHECKPOINT)][
            "mean_progress"] > 0.0
        for transition in transitions.values()
    ))


def _minimum_accuracy(transitions, arm):
    return float(min(
        transition["causal"][arm][str(PRIMARY_CHECKPOINT)][
            "minimum_answer_accuracy"]
        for transition in transitions.values()
    ))


def _family_adjudication(transitions, inverse, composition):
    exact = _mean_causal(transitions, "exact_state_oracle")
    atlas = _mean_causal(transitions, "atlas")
    composed = _mean_causal(transitions, "composed")
    pairwise = _mean_causal(transitions, "pairwise")
    mean = _mean_causal(transitions, "mean")
    wrong = _mean_causal(transitions, "wrong_target")
    instruction = _mean_causal(transitions, "instruction")
    identical = _mean_causal(transitions, "identical")
    causal_random = max(
        _mean_causal(transitions, f"random_{index:02d}")
        for index in CAUSAL_RANDOM_INDICES)
    recovery = atlas / exact if exact > 1e-8 else None
    content_cosine = float(np.mean([
        transition["offline"]["content_equivariance"][
            "mean_cosine"]
        for transition in transitions.values()
    ]))
    content_error = float(np.median([
        row
        for transition in transitions.values()
        for row in transition["offline"][
            "content_equivariance"]["relative_error_rows"]
    ]))
    exact_pass = bool(
        exact >= MINIMUM_EXACT_MEAN_PROGRESS - 1e-9
        and _positive_causal(
            transitions, "exact_state_oracle")
        >= MINIMUM_EXACT_POSITIVE_TRANSITIONS
        and _minimum_accuracy(
            transitions, "exact_state_oracle")
        >= MINIMUM_VALUE_ACCURACY - 1e-9)
    atlas_pass = bool(
        exact_pass
        and atlas >= MINIMUM_ATLAS_MEAN_PROGRESS - 1e-9
        and _positive_causal(transitions, "atlas")
        >= MINIMUM_ATLAS_POSITIVE_TRANSITIONS
        and _minimum_accuracy(transitions, "atlas")
        >= MINIMUM_VALUE_ACCURACY - 1e-9
        and recovery is not None
        and recovery >= MINIMUM_RECOVERY_OF_EXACT - 1e-9
        and atlas - max(pairwise, mean)
        >= MINIMUM_BASELINE_MARGIN - 1e-9
        and atlas - max(wrong, causal_random)
        >= MINIMUM_SPECIFICITY_MARGIN - 1e-9
        and instruction < max(
            0.10,
            MAXIMUM_POSITION_CONTROL_FRACTION * atlas)
        and identical < max(
            0.10,
            MAXIMUM_POSITION_CONTROL_FRACTION * atlas))
    algebra_pass = bool(
        composed >= MINIMUM_COMPOSED_PROGRESS - 1e-9
        and _positive_causal(transitions, "composed")
        >= MINIMUM_COMPOSED_POSITIVE_TRANSITIONS
        and _minimum_accuracy(transitions, "composed")
        >= MINIMUM_VALUE_ACCURACY - 1e-9
        and composition["median_agreement_error"]
        <= MAXIMUM_COMPOSITION_AGREEMENT_ERROR + 1e-9
        and inverse["median_relative_error"]
        <= MAXIMUM_INVERSE_LOOP_ERROR + 1e-9
        and content_cosine
        >= MINIMUM_CONTENT_EQUIVARIANCE_COSINE - 1e-9
        and content_error
        <= MAXIMUM_CONTENT_EQUIVARIANCE_ERROR + 1e-9)
    return {
        "exact_mean_progress": exact,
        "atlas_mean_progress": atlas,
        "composed_mean_progress": composed,
        "pairwise_mean_progress": pairwise,
        "mean_translation_progress": mean,
        "wrong_target_progress": wrong,
        "best_causal_random_progress": causal_random,
        "instruction_progress": instruction,
        "identical_progress": identical,
        "recovery_of_exact": recovery,
        "content_equivariance_mean_cosine": content_cosine,
        "content_equivariance_median_error": content_error,
        "exact_pass": exact_pass,
        "atlas_pass": atlas_pass,
        "algebra_pass": algebra_pass,
        "pass": bool(atlas_pass and algebra_pass),
    }


def _overall_adjudication(families):
    exact_families = [
        family for family, value in families.items()
        if value["adjudication"]["exact_pass"]
    ]
    atlas_families = [
        family for family, value in families.items()
        if value["adjudication"]["atlas_pass"]
    ]
    algebra_families = [
        family for family, value in families.items()
        if value["adjudication"]["algebra_pass"]
    ]
    full_families = [
        family for family, value in families.items()
        if value["adjudication"]["pass"]
    ]
    selected_score = min(
        float(np.mean([
            transition["offline"]["arms"]["atlas"][
                "mean_progress"]
            for transition in value["transitions"].values()
        ]))
        for value in families.values()
    )
    random_scores = []
    for index in range(N_RANDOM):
        arm = f"random_{index:02d}"
        random_scores.append(min(
            float(np.mean([
                transition["offline"]["random"][arm][
                    "mean_progress"]
                for transition in value["transitions"].values()
            ]))
            for value in families.values()
        ))
    random_p, random_exceed = _tail_probability(
        selected_score, random_scores)
    exact_eligible = (
        len(exact_families) >= MINIMUM_PASSING_FAMILIES)
    atlas_broad = (
        len(atlas_families) >= MINIMUM_PASSING_FAMILIES
        and random_p <= MAXIMUM_RANDOM_P + 1e-12)
    algebra_broad = (
        len(algebra_families) >= MINIMUM_PASSING_FAMILIES)
    full = (
        len(full_families) >= MINIMUM_PASSING_FAMILIES
        and random_p <= MAXIMUM_RANDOM_P + 1e-12)
    verdict = _verdict_from_flags(
        exact_eligible, atlas_broad, algebra_broad, full)
    return {
        "exact_families": exact_families,
        "atlas_families": atlas_families,
        "algebra_families": algebra_families,
        "full_families": full_families,
        "selected_offline_breadth_score": selected_score,
        "random_offline_breadth_scores": random_scores,
        "random_empirical_p": random_p,
        "random_exceed_count": random_exceed,
        "exact_eligible": exact_eligible,
        "atlas_broad": atlas_broad,
        "algebra_broad": algebra_broad,
        "pass": full,
        "verdict": verdict,
    }


def _verdict_from_flags(
        exact_eligible, atlas_broad, algebra_broad, full):
    """Keep the preregistered failure taxonomy explicit and testable."""
    if not exact_eligible:
        return "ASSAY_INELIGIBLE"
    if full:
        return "CAUSAL_ATLAS_WITH_COMPOSITION"
    if atlas_broad:
        return "CAUSAL_TRANSPORT_WITHOUT_ALGEBRA"
    if algebra_broad:
        return "ALGEBRAIC_FIT_WITHOUT_CAUSAL_CONTROL"
    return "NO_LOW_COMPLEXITY_CAUSAL_ATLAS"


def _self_check():
    generator = torch.Generator().manual_seed(12345)
    latent = torch.randn(40, 2, generator=generator)
    frame_states = {}
    for index, frame in enumerate(FRAME_NAMES):
        raw = torch.randn(2, 9, generator=generator)
        left, _singular, right_t = torch.linalg.svd(
            raw, full_matrices=False)
        basis = left @ right_t
        mean = torch.zeros(9)
        mean[index:index + 3] = float(index + 1)
        frame_states[frame] = mean + latent @ basis
    train = {
        frame: values[:30] for frame, values in frame_states.items()
    }
    test = {
        frame: values[30:] for frame, values in frame_states.items()
    }
    atlas = _fit_atlas(train, rank=2)
    progress = []
    for source, target in TRANSITIONS:
        predicted = _atlas_transport(
            atlas, source, target, test[source])
        progress.append(_transport_metrics(
            test[source], test[target], predicted)["mean_progress"])
    inverse = _inverse_metrics(atlas, test)
    composition = _composition_metrics(atlas, test)
    boundary = _PredictionBoundary()
    blocked = False
    try:
        boundary.require_evaluation()
    except RuntimeError:
        blocked = True
    boundary.freeze("A", "B")
    boundary.require_evaluation()
    probability, exceed = _tail_probability(
        1.0, [0.0] * N_RANDOM)
    rows = _row_splits()
    verdicts = {
        _verdict_from_flags(False, False, False, False),
        _verdict_from_flags(True, True, True, True),
        _verdict_from_flags(True, True, False, False),
        _verdict_from_flags(True, False, True, False),
        _verdict_from_flags(True, False, False, False),
    }
    expected_verdicts = set(PROTOCOL["verdicts"])
    passed = bool(
        min(progress) > 0.99
        and inverse["median_relative_error"] < 1e-4
        and composition["median_agreement_error"] < 1e-4
        and blocked
        and probability == 0.05
        and exceed == 0
        and len(rows["test"]) == TEST_N
        and verdicts == expected_verdicts)
    if not passed:
        raise AssertionError("causal-atlas self-check failed")
    return {
        "synthetic_transport_progress": progress,
        "synthetic_inverse_error":
            inverse["median_relative_error"],
        "synthetic_composition_error":
            composition["median_agreement_error"],
        "prediction_boundary_check": blocked,
        "random_tail_check": True,
        "row_split_check": True,
        "verdict_taxonomy_check": True,
        "pass": True,
    }


@torch.no_grad()
def run_delta_causal_atlas(
        model_path, out_dir,
        model_key="qwen7b_causal_atlas",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=TEST_N, self_test_only=False):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_world) != TEST_N:
        raise ValueError("causal-atlas v1 is frozen to twelve test histories")
    self_check = _self_check()
    if self_test_only:
        result = {
            "stage": "delta_causal_atlas",
            "protocol": PROTOCOL,
            "protocol_sha256": PROTOCOL_SHA256,
            "self_check": self_check,
            "verdict": "SELF_CHECK_PASS",
        }
        path = os.path.join(
            out_dir, "delta_causal_atlas_self_check.json")
        with open(path, "w") as handle:
            json.dump(result, handle, indent=2)
        log(
            "CAUSAL-ATLAS self-check pass "
            f"protocol={PROTOCOL_SHA256}")
        return result

    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    if PRIMARY_CHECKPOINT >= model_num_hidden_layers(model):
        raise ValueError("causal-atlas checkpoint is absent")
    splits = _row_splits()
    fit_rows = splits["train"] + splits["validation"]
    commands, padding_plan, tokenization_tables = _commands(tok)
    total = (
        len(FAMILY_ORDER) * (
            len(TRAIN_PANELS) * 2 * len(FRAME_NAMES)
            + 2 * len(FRAME_NAMES)
            + 2 * len(FRAME_NAMES)
            + len(TRANSITIONS) * 2 * 11))
    heartbeat = Heartbeat(
        total, "causal_atlas", every_sec=30, out_dir=out_dir)

    prepared = {}
    predictions_by_family = {}
    source_test_by_family = {}
    atlases = {}
    models_public = {}
    permutation_metadata = {}

    for family_index, family in enumerate(FAMILY_ORDER):
        spec = FAMILY_SPECS[family]
        fit_alignments = {
            panel: _frame_alignment(
                tok, dev, fit_rows, spec, commands[panel])
            for panel in TRAIN_PANELS
        }
        captured = _capture_training_states(
            model, fit_alignments, heartbeat, family)
        train_states = {
            frame: _stack_captured(
                captured, TRAIN_PANELS, frame,
                slice(0, TRAIN_N))
            for frame in FRAME_NAMES
        }
        validation_states = {
            frame: _stack_captured(
                captured, TRAIN_PANELS, frame,
                slice(TRAIN_N, TRAIN_N + VALIDATION_N))
            for frame in FRAME_NAMES
        }
        _selected_atlas, atlas_selection = _select_atlas(
            train_states, validation_states)
        chosen_rank = atlas_selection["selected"][
            "requested_rank"]
        combined_states = {
            frame: torch.cat([
                train_states[frame],
                validation_states[frame],
            ], dim=0)
            for frame in FRAME_NAMES
        }
        atlas = _fit_atlas(combined_states, chosen_rank)
        pairwise, pairwise_metadata = _refit_pairwise(
            train_states, validation_states)

        test_alignment = _frame_alignment(
            tok, dev, splits["test"], spec,
            commands[TEST_PANEL])
        source_test = _capture_source_test(
            model, test_alignment, heartbeat, family)
        predictions, family_permutations = _build_predictions(
            atlas, pairwise, source_test, family_index)
        predictions_by_family[family] = predictions
        source_test_by_family[family] = source_test
        atlases[family] = atlas
        prepared[family] = test_alignment
        permutation_metadata[family] = family_permutations
        models_public[family] = {
            "atlas_selection": atlas_selection,
            "atlas": _atlas_public(atlas),
            "pairwise_selection": {
                transition: metadata["selected"]
                for transition, metadata
                in pairwise_metadata.items()
            },
        }
        del captured
        del fit_alignments
        del pairwise

    freeze = _write_prediction_freeze(
        out_dir, models_public, predictions_by_family,
        permutation_metadata)
    boundary = _PredictionBoundary()
    boundary.freeze(
        freeze["json_sha256"], freeze["npz_sha256"])
    _reload_atlas_predictions(
        freeze, predictions_by_family)
    log(
        "FROZEN causal-atlas predictions "
        f"json={freeze['json_sha256']} "
        f"npz={freeze['npz_sha256']}")

    boundary.require_evaluation()
    families = {}
    for family in FAMILY_ORDER:
        alignment = prepared[family]
        baselines = _capture_evaluation_baselines(
            model, alignment, source_test_by_family[family],
            heartbeat, family)
        inverse = _inverse_metrics(
            atlases[family], source_test_by_family[family])
        composition = _composition_metrics(
            atlases[family], source_test_by_family[family])
        transitions = {}
        for source, target in TRANSITIONS:
            key = _transition_key(source, target)
            offline = _offline_transition(
                atlases[family], source, target,
                source_test_by_family[family],
                predictions_by_family[family][key],
                permutation_metadata[family][key])
            causal = _run_transition_causal(
                model, alignment, baselines, splits["test"],
                source, target,
                predictions_by_family[family][key],
                permutation_metadata[family][key],
                heartbeat, family)
            transitions[key] = {
                "source": source,
                "target": target,
                "middle": _middle_frame(source, target),
                "offline": offline,
                "causal": causal,
            }
        adjudication = _family_adjudication(
            transitions, inverse, composition)
        families[family] = {
            "model": models_public[family],
            "inverse": inverse,
            "composition": composition,
            "transitions": transitions,
            "adjudication": adjudication,
        }
        log(
            f"CAUSAL-ATLAS family={family} "
            f"exact={adjudication['exact_pass']} "
            f"atlas={adjudication['atlas_pass']} "
            f"algebra={adjudication['algebra_pass']} "
            f"progress={adjudication['atlas_mean_progress']:+.3f}")
    heartbeat.done()

    overall = _overall_adjudication(families)
    result = {
        "stage": "delta_causal_atlas",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "self_check": self_check,
        "prediction_freeze": {
            "json_path": freeze["json_path"],
            "npz_path": freeze["npz_path"],
            "json_sha256": freeze["json_sha256"],
            "npz_sha256": freeze["npz_sha256"],
        },
        "commands": commands,
        "padding_plan": padding_plan,
        "tokenization_tables": tokenization_tables,
        "rows": splits,
        "alignment": {
            family: {
                key: value
                for key, value in prepared[family].items()
                if key != "batches"
            }
            for family in FAMILY_ORDER
        },
        "families": families,
        "overall": overall,
        "verdict": overall["verdict"],
    }
    path = os.path.join(
        out_dir,
        f"results_delta_causal_atlas_{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"CAUSAL-ATLAS verdict={result['verdict']} "
        f"full={overall['full_families']} "
        f"random_p={overall['random_empirical_p']:.3f} "
        f"artifact={path}")
    return result
