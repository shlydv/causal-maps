"""Final fork: can family-specific maps predict operation-state transport?"""
from __future__ import annotations

import hashlib
import json

import torch

from .delta_predictive_conditional_transport import (
    CHECKPOINT_LAYERS,
    FAMILY_ORDER,
    MINIMUM_CONDITIONAL_GAIN,
    MINIMUM_EXACT_FAMILIES,
    MINIMUM_POSITIVE_ROWS,
    MINIMUM_PREDICTED_FAMILIES,
    MINIMUM_PREDICTED_PROGRESS,
    MINIMUM_RECOVERY_OF_EXACT,
    MINIMUM_VALUE_ACCURACY,
    MAXIMUM_MEDIAN_DISTANCE,
    RANKS,
    RIDGES,
    TEST_N,
    TRAIN_N,
    VALIDATION_N,
    _run_predictive_transport,
)


PROTOCOL_VERSION = (
    "2026-07-27-p2-within-family-conditional-transport-v1")
PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "purpose": (
        "Final branch-closing test: distinguish computation-family-specific "
        "control laws from exact state copying with no learnable low-"
        "complexity controller."
    ),
    "hypothesis": (
        "Within a fixed computation family, the matched BELIEF/SEARCH "
        "operation displacement is a low-complexity function of the current "
        "L21 answer-prefix state and generalizes to unseen worlds."
    ),
    "model": "Qwen2.5-7B-Instruct, 8-bit",
    "families": list(FAMILY_ORDER),
    "training_scope": "within_family_only",
    "locus": {
        "layer": 21,
        "positions": "three answer-prefix command tokens",
        "checkpoints": list(CHECKPOINT_LAYERS),
        "primary_checkpoint": 27,
    },
    "rows_per_family": {
        "train_directed_pairs": TRAIN_N,
        "validation_directed_pairs": VALIDATION_N,
        "test_histories": TEST_N,
        "test_directed_pairs": 6,
        "test_distractor_variants_per_pair": 2,
        "all_three_pair_sets_disjoint": True,
    },
    "predictor": {
        "input": "origin-operation state only",
        "target": "opposite-minus-origin displacement",
        "counterpart_test_state_available_during_prediction": False,
        "ranks": list(RANKS),
        "ridge_values": list(RIDGES),
        "selection": (
            "minimum same-family validation normalized MSE; refit on "
            "same-family train plus validation rows after selection"
        ),
    },
    "controls": [
        "exact matched counterpart state",
        "same-family global mean displacement",
        "same-family target-state centroid",
        "same-family nearest neighbour",
        "row-shuffled conditional prediction",
        "instruction-position conditional displacement",
        "matched identical-token conditional displacement",
    ],
    "per_family_gate": {
        "exact_reference_both_directions": True,
        "minimum_progress_each_direction":
            MINIMUM_PREDICTED_PROGRESS,
        "minimum_recovery_of_exact_each_direction":
            MINIMUM_RECOVERY_OF_EXACT,
        "minimum_positive_rows_each_direction":
            MINIMUM_POSITIVE_ROWS,
        "maximum_median_target_distance_each_direction":
            MAXIMUM_MEDIAN_DISTANCE,
        "minimum_value_accuracy_each_direction":
            MINIMUM_VALUE_ACCURACY,
        "minimum_gain_over_each_global_template":
            MINIMUM_CONDITIONAL_GAIN,
        "minimum_gain_over_row_shuffle":
            MINIMUM_CONDITIONAL_GAIN,
        "position_controls": (
            "instruction and identical each below max(0.10, half the "
            "conditional bidirectional score)"
        ),
    },
    "overall_gate": {
        "minimum_exact_reference_families":
            MINIMUM_EXACT_FAMILIES,
        "minimum_family_specific_predictive_families":
            MINIMUM_PREDICTED_FAMILIES,
    },
    "verdicts": {
        "FAMILY_SPECIFIC_CONTROL_LAWS": (
            "At least six families pass; continue the research branch."),
        "PARTIAL_FAMILY_SPECIFIC_CONTROL": (
            "Three to five pass; insufficient for the proposed general "
            "mechanism."),
        "NO_LEARNABLE_WITHIN_FAMILY_CONTROL": (
            "Exact references pass but fewer than three conditional maps "
            "pass; close the control-law branch."),
    },
    "stopping_rule": (
        "No prompt, split, rank, ridge, threshold, locus, nonlinear-model, "
        "or architecture rescue follows a non-positive result. Only "
        "FAMILY_SPECIFIC_CONTROL_LAWS licenses further ICML/ICLR-directed "
        "operator-structure experiments."
    ),
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest().upper()

VERDICT_MAP = {
    "PREDICTABLE_STATE_CONDITIONED_TRANSPORT":
        "FAMILY_SPECIFIC_CONTROL_LAWS",
    "PARTIAL_STATE_CONDITIONED_TRANSPORT":
        "PARTIAL_FAMILY_SPECIFIC_CONTROL",
    "EXEMPLAR_CONDITIONAL_TRANSPORT":
        "EXEMPLAR_ONLY_WITHIN_FAMILY_CONTROL",
    "GLOBAL_TEMPLATE_TRANSPORT":
        "WITHIN_FAMILY_GLOBAL_TEMPLATE_ONLY",
    "ORACLE_ONLY_STATE_TRANSPORT":
        "NO_LEARNABLE_WITHIN_FAMILY_CONTROL",
    "EXACT_REFERENCE_NOT_GENERAL":
        "WITHIN_FAMILY_EXACT_REFERENCE_FAILURE",
    "NONSPECIFIC_PREDICTED_TRANSPORT":
        "WITHIN_FAMILY_POSITION_CONTROL_FAILURE",
}


@torch.no_grad()
def run_delta_within_family_conditional_transport(
        model_path, out_dir,
        model_key="qwen7b_within_family_conditional_transport",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=12, self_test_only=False):
    return _run_predictive_transport(
        model_path, out_dir,
        model_key=model_key,
        protocol=PROTOCOL,
        protocol_sha256=PROTOCOL_SHA256,
        experiment_stage="delta_within_family_conditional_transport",
        training_scope="within_family",
        verdict_map=VERDICT_MAP,
        quantization=quantization,
        device_map=device_map,
        max_memory=max_memory,
        n_world=n_world,
        self_test_only=self_test_only)
