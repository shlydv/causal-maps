"""Offline shared-plus-context audit for saved response-map tensors.

This module uses no model inference. It evaluates whether a family-specific
response-map correction estimated on four row IDs predicts the remaining four
row IDs better than shared-only and norm-matched wrong-family controls.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


FAMILIES = (
    "private_belief",
    "two_hop_pointer",
    "maximum_score",
    "constraint_elimination",
)
OPERATIONS = ("belief", "search")
CHECKPOINTS = (24, 27)

MINIMUM_RESIDUAL_COSINE = 0.30
MINIMUM_ERROR_REDUCTION = 0.10
MINIMUM_WRONG_CONTROL_ADVANTAGE = 0.10
MINIMUM_POSITIVE_CELLS = 6


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _relative_squared_error(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    denominator = float(np.dot(target, target))
    if denominator <= 1e-12:
        return float("inf")
    difference = prediction - target
    return float(np.dot(difference, difference) / denominator)


def _samples(array: np.ndarray, row_ids: tuple[int, ...]) -> np.ndarray:
    """Return [history*row, flattened_map] while keeping row groups intact."""
    selected = np.asarray(array[:, row_ids], dtype=np.float32)
    return selected.reshape(-1, int(np.prod(selected.shape[2:])))


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _cell_summary(records: list[dict[str, float]]) -> dict[str, float]:
    keys = records[0]
    return {
        f"median_{key}": _median([record[key] for record in records])
        for key in keys
    }


def _split_audit(
    arrays: dict[tuple[int, str, str], np.ndarray],
    checkpoint: int,
    train_rows: tuple[int, ...],
) -> dict[tuple[str, str], dict[str, float]]:
    test_rows = tuple(row for row in range(8) if row not in train_rows)
    train_means: dict[tuple[str, str], np.ndarray] = {}
    test_samples: dict[tuple[str, str], np.ndarray] = {}
    test_means: dict[tuple[str, str], np.ndarray] = {}
    for family in FAMILIES:
        for operation in OPERATIONS:
            array = arrays[(checkpoint, family, operation)]
            train = _samples(array, train_rows)
            test = _samples(array, test_rows)
            train_means[(family, operation)] = train.mean(axis=0)
            test_samples[(family, operation)] = test
            test_means[(family, operation)] = test.mean(axis=0)

    result = {}
    for family in FAMILIES:
        for operation in OPERATIONS:
            other_families = tuple(
                candidate for candidate in FAMILIES if candidate != family
            )
            shared_train = np.mean(
                [
                    train_means[(candidate, operation)]
                    for candidate in other_families
                ],
                axis=0,
            )
            shared_test = np.mean(
                [
                    test_means[(candidate, operation)]
                    for candidate in other_families
                ],
                axis=0,
            )
            family_train = train_means[(family, operation)]
            family_test = test_means[(family, operation)]
            correction_train = family_train - shared_train
            correction_test = family_test - shared_test
            correction_norm = float(np.linalg.norm(correction_train))
            shared_norm = float(np.linalg.norm(shared_train))

            shared_errors = []
            context_errors = []
            shared_cosines = []
            context_cosines = []
            for target in test_samples[(family, operation)]:
                shared_errors.append(
                    _relative_squared_error(shared_train, target)
                )
                context_errors.append(
                    _relative_squared_error(family_train, target)
                )
                shared_cosines.append(_cosine(shared_train, target))
                context_cosines.append(_cosine(family_train, target))

            wrong_errors = []
            wrong_cosines = []
            for wrong_family in other_families:
                wrong_others = tuple(
                    candidate
                    for candidate in FAMILIES
                    if candidate != wrong_family
                )
                wrong_shared = np.mean(
                    [
                        train_means[(candidate, operation)]
                        for candidate in wrong_others
                    ],
                    axis=0,
                )
                wrong_correction = (
                    train_means[(wrong_family, operation)] - wrong_shared
                )
                wrong_norm = float(np.linalg.norm(wrong_correction))
                if wrong_norm > 1e-12:
                    wrong_correction = (
                        wrong_correction * correction_norm / wrong_norm
                    )
                else:
                    wrong_correction = np.zeros_like(wrong_correction)
                wrong_prediction = shared_train + wrong_correction
                for target in test_samples[(family, operation)]:
                    wrong_errors.append(
                        _relative_squared_error(wrong_prediction, target)
                    )
                    wrong_cosines.append(_cosine(wrong_prediction, target))

            mean_shared_error = float(np.mean(shared_errors))
            mean_context_error = float(np.mean(context_errors))
            mean_wrong_error = float(np.mean(wrong_errors))
            result[(family, operation)] = {
                "residual_cosine": _cosine(
                    correction_train, correction_test
                ),
                "correction_to_shared_norm": (
                    correction_norm / max(shared_norm, 1e-12)
                ),
                "shared_relative_squared_error": mean_shared_error,
                "context_relative_squared_error": mean_context_error,
                "wrong_relative_squared_error": mean_wrong_error,
                "context_error_reduction_vs_shared": (
                    (mean_shared_error - mean_context_error)
                    / max(mean_shared_error, 1e-12)
                ),
                "context_error_reduction_vs_wrong": (
                    (mean_wrong_error - mean_context_error)
                    / max(mean_wrong_error, 1e-12)
                ),
                "shared_prediction_cosine": float(
                    np.mean(shared_cosines)
                ),
                "context_prediction_cosine": float(
                    np.mean(context_cosines)
                ),
                "wrong_prediction_cosine": float(
                    np.mean(wrong_cosines)
                ),
            }
    return result


def _variance_decomposition(
    arrays: dict[tuple[int, str, str], np.ndarray],
    checkpoint: int,
) -> dict:
    by_operation = {}
    deviation_rows = []
    for operation in OPERATIONS:
        family_samples = {
            family: np.asarray(
                arrays[(checkpoint, family, operation)],
                dtype=np.float32,
            ).reshape(16, -1)
            for family in FAMILIES
        }
        family_means = {
            family: samples.mean(axis=0)
            for family, samples in family_samples.items()
        }
        grand_mean = np.mean(list(family_means.values()), axis=0)
        between = 16.0 * sum(
            float(np.sum((mean - grand_mean) ** 2))
            for mean in family_means.values()
        )
        within = sum(
            float(np.sum((samples - family_means[family]) ** 2))
            for family, samples in family_samples.items()
        )
        mean_sample_energy = float(
            np.mean(
                [
                    np.sum(samples * samples, axis=1).mean()
                    for samples in family_samples.values()
                ]
            )
        )
        grand_energy = float(np.sum(grand_mean * grand_mean))
        centered_total = between + within
        by_operation[operation] = {
            "shared_grand_mean_energy": grand_energy,
            "shared_energy_fraction_of_mean_sample_energy": (
                grand_energy / max(mean_sample_energy, 1e-12)
            ),
            "between_family_sum_squares": between,
            "within_family_sum_squares": within,
            "between_family_fraction_of_centered_variation": (
                between / max(centered_total, 1e-12)
            ),
            "within_family_fraction_of_centered_variation": (
                within / max(centered_total, 1e-12)
            ),
        }
        deviation_rows.extend(
            family_means[family] - grand_mean for family in FAMILIES
        )

    deviations = np.stack(deviation_rows).astype(np.float64)
    gram = deviations @ deviations.T
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    singular_values = np.sqrt(eigenvalues)
    energy = eigenvalues / max(float(eigenvalues.sum()), 1e-12)
    return {
        "by_operation": by_operation,
        "family_operation_deviation_singular_values": (
            singular_values.tolist()
        ),
        "family_operation_deviation_energy_fractions": energy.tolist(),
        "rank1_deviation_energy": float(energy[:1].sum()),
        "rank2_deviation_energy": float(energy[:2].sum()),
        "rank4_deviation_energy": float(energy[:4].sum()),
    }


def analyze(npz_path: Path) -> dict:
    with np.load(npz_path) as archive:
        arrays = {
            (checkpoint, family, operation): np.array(
                archive[f"{family}_{operation}_L{checkpoint}"],
                dtype=np.float32,
            )
            for checkpoint in CHECKPOINTS
            for family in FAMILIES
            for operation in OPERATIONS
        }

    split_rows = tuple(itertools.combinations(range(8), 4))
    checkpoint_results = {}
    for checkpoint in CHECKPOINTS:
        cell_records = {
            (family, operation): []
            for family in FAMILIES
            for operation in OPERATIONS
        }
        for train_rows in split_rows:
            split = _split_audit(arrays, checkpoint, train_rows)
            for cell, metrics in split.items():
                cell_records[cell].append(metrics)
        cells = {
            f"{family}/{operation}": _cell_summary(records)
            for (family, operation), records in cell_records.items()
        }
        residual_cosines = [
            cell["median_residual_cosine"] for cell in cells.values()
        ]
        improvements = [
            cell["median_context_error_reduction_vs_shared"]
            for cell in cells.values()
        ]
        wrong_advantages = [
            cell["median_context_error_reduction_vs_wrong"]
            for cell in cells.values()
        ]
        checkpoint_results[str(checkpoint)] = {
            "cells": cells,
            "summary": {
                "median_residual_cosine": _median(residual_cosines),
                "positive_residual_cosine_cells": sum(
                    value > 0.0 for value in residual_cosines
                ),
                "median_context_error_reduction_vs_shared": _median(
                    improvements
                ),
                "positive_context_improvement_cells": sum(
                    value > 0.0 for value in improvements
                ),
                "median_context_error_reduction_vs_wrong": _median(
                    wrong_advantages
                ),
                "positive_wrong_control_advantage_cells": sum(
                    value > 0.0 for value in wrong_advantages
                ),
                "median_correction_to_shared_norm": _median(
                    [
                        cell["median_correction_to_shared_norm"]
                        for cell in cells.values()
                    ]
                ),
            },
            "variance_decomposition": _variance_decomposition(
                arrays, checkpoint
            ),
        }

    primary = checkpoint_results["27"]["summary"]
    replication = checkpoint_results["24"]["summary"]
    gates = {
        "residual_reproducibility": bool(
            primary["median_residual_cosine"]
            >= MINIMUM_RESIDUAL_COSINE
            and primary["positive_residual_cosine_cells"]
            >= MINIMUM_POSITIVE_CELLS
        ),
        "unseen_row_improvement": bool(
            primary["median_context_error_reduction_vs_shared"]
            >= MINIMUM_ERROR_REDUCTION
            and primary["positive_context_improvement_cells"]
            >= MINIMUM_POSITIVE_CELLS
        ),
        "wrong_context_control": bool(
            primary["median_context_error_reduction_vs_wrong"]
            >= MINIMUM_WRONG_CONTROL_ADVANTAGE
        ),
        "l24_directional_replication": bool(
            replication["median_residual_cosine"] > 0.0
            and replication[
                "median_context_error_reduction_vs_shared"
            ] > 0.0
        ),
    }
    verdict = (
        "PROSPECTIVE_INVERSE_CONTROL_DESIGN_LICENSED"
        if all(gates.values())
        else "SHARED_CONTEXT_GEOMETRY_BRANCH_CLOSED"
    )
    with npz_path.open("rb") as handle:
        npz_sha256 = hashlib.sha256(handle.read()).hexdigest().upper()
    return {
        "analysis": "paper2_shared_context_offline_audit",
        "status": "exploratory_zero_gpu",
        "source_npz": str(npz_path),
        "source_npz_sha256": npz_sha256,
        "split_policy": {
            "number_of_row_splits": len(split_rows),
            "training_row_ids_per_split": 4,
            "test_row_ids_per_split": 4,
            "histories_grouped_by_row": True,
        },
        "thresholds": {
            "minimum_median_residual_cosine":
                MINIMUM_RESIDUAL_COSINE,
            "minimum_median_error_reduction":
                MINIMUM_ERROR_REDUCTION,
            "minimum_median_wrong_control_advantage":
                MINIMUM_WRONG_CONTROL_ADVANTAGE,
            "minimum_positive_cells": MINIMUM_POSITIVE_CELLS,
        },
        "checkpoints": checkpoint_results,
        "gates": gates,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(
        {
            "verdict": result["verdict"],
            "gates": result["gates"],
            "L27": result["checkpoints"]["27"]["summary"],
            "L24": result["checkpoints"]["24"]["summary"],
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
