"""Prospective three-mode destination-specific latent route selector."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from .delta_anchor_write import _resolve
from .delta_causal_atlas import (
    FRAME_NAMES,
    PANEL_LABELS,
    _PredictionBoundary,
    _artifact_sha256,
    _commands,
    _frame_alignment,
    _row_splits,
)
from .delta_heterogeneous_family_screen import FAMILY_SPECS
from .delta_latent_instruction_compiler import (
    CHECKPOINTS,
    EARLY_LAYERS,
    PATCH_WIDTH,
    PROGRAM_WIDTH,
    _all_positions,
    _answer_state,
    _append,
    _baseline_accuracy,
    _capture,
    _context_match,
    _derangements,
    _empty_accumulator,
    _exact_early_pass,
    _late_oracle_pass,
    _late_patch_cache,
    _processed,
    _run_early,
    _summary,
)
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)


PROTOCOL_VERSION = "2026-07-27-p2-three-mode-selector-v1"
FRAMES = tuple(FRAME_NAMES)
TRAIN_PANELS = ("anchor", "synonym_a")
CALIBRATION_PANEL = "synonym_a"
TEST_PANEL = "synonym_b"
TRAIN_FAMILIES = (
    "private_belief",
    "latest_update",
    "key_value_lookup",
    "conditional_selection",
)
CALIBRATION_FAMILIES = (
    "constraint_elimination",
    "temporal_slot",
)
TEST_FAMILIES = (
    "maximum_score",
    "two_hop_pointer",
)
N_RANDOM = 19
RANDOM_SEED = 31415927

MINIMUM_CALIBRATION_SCORE = 0.10
MINIMUM_VALUE_ACCURACY = 0.80
MINIMUM_PORTABLE_PROGRESS = 0.20
MINIMUM_PORTABLE_RECOVERY = 0.35
MINIMUM_POSITIVE_ROWS = 20
MINIMUM_TARGET_VS_DECOY_WINS = 20
MINIMUM_DESTINATION_WINS = 18
MINIMUM_CODE_MARGIN = 0.10
MINIMUM_MEDIATION = 0.70
MAXIMUM_NULL_FRACTION = 0.50
MINIMUM_CONTEXT_ADVANTAGE = 0.10
MAXIMUM_RANDOM_P = 0.05
N_CELLS = len(TEST_FAMILIES) * len(FRAMES) * (len(FRAMES) - 1)


PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "status": "prospective heldout three-mode destination selector",
    "hypothesis": (
        "One training-derived early code per destination selects that "
        "destination from either alternative source after contextual "
        "processing by the frozen intervening layers."
    ),
    "interpretation_boundary": (
        "The modes are pretrained lexical readout routes. A pass establishes "
        "destination-addressed route selection, not replacement of arbitrary "
        "natural-language instructions or computations."
    ),
    "model": "Qwen2.5-7B-Instruct, 8-bit, Tesla T4",
    "frames": list(FRAMES),
    "labels": PANEL_LABELS,
    "early_layers": list(EARLY_LAYERS),
    "generated_layer": 21,
    "checkpoints": list(CHECKPOINTS),
    "family_split": {
        "train": list(TRAIN_FAMILIES),
        "calibration": list(CALIBRATION_FAMILIES),
        "test": list(TEST_FAMILIES),
    },
    "lexical_split": {
        "train": list(TRAIN_PANELS),
        "calibration": CALIBRATION_PANEL,
        "test": TEST_PANEL,
    },
    "world_split": {
        "train": 24,
        "calibration": 8,
        "test": 12,
        "directed_pairs_disjoint": True,
    },
    "program": (
        "For destination t, mean of h_t minus the mean of both non-target "
        "source states over training families, panels, rows and histories; "
        "one six-state code per destination and candidate layer; no scale fit."
    ),
    "selection": (
        "Maximize the minimum destination-specific calibration score over "
        "all family/source/target cells; earlier layer breaks ties."
    ),
    "calibration_gate": {
        "minimum_score": MINIMUM_CALIBRATION_SCORE,
        "minimum_value_accuracy": MINIMUM_VALUE_ACCURACY,
    },
    "primary_gates": {
        "minimum_progress_l21_l27": MINIMUM_PORTABLE_PROGRESS,
        "minimum_recovery_of_exact": MINIMUM_PORTABLE_RECOVERY,
        "minimum_positive_rows": MINIMUM_POSITIVE_ROWS,
        "minimum_target_vs_decoy_wins":
            MINIMUM_TARGET_VS_DECOY_WINS,
        "minimum_three_mode_destination_wins":
            MINIMUM_DESTINATION_WINS,
        "minimum_margin_over_competing_code": MINIMUM_CODE_MARGIN,
        "minimum_value_accuracy": MINIMUM_VALUE_ACCURACY,
        "minimum_block_and_rescue_fraction": MINIMUM_MEDIATION,
        "maximum_sign_or_identical_fraction": MAXIMUM_NULL_FRACTION,
        "minimum_context_error_reduction": MINIMUM_CONTEXT_ADVANTAGE,
        "maximum_add_one_random_p": MAXIMUM_RANDOM_P,
    },
    "controls": [
        "exact matched L21 target",
        "exact matched early target-minus-source",
        "competing destination code",
        "sign-reversed intended code",
        "intended code at six matched identical-token positions",
        "19 exactly norm-matched isotropic random codes",
        "L21 source-state restoration",
        "generated-L21-state rescue",
        "19 receiver-state row derangements",
    ],
    "verdicts": [
        "DESTINATION_SPECIFIC_CONTEXTUAL_SELECTOR",
        "SELECTIVE_STEERING_WITHOUT_NATIVE_MEDIATION",
        "GENERIC_EVICTION_NOT_SELECTION",
        "NO_PORTABLE_THREE_MODE_CONTROL",
        "NO_EARLY_THREE_MODE_CAPACITY",
        "CAUSAL_TARGET_UNRESOLVED",
        "CALIBRATION_SELECTOR_NULL",
    ],
    "stopping_rule": (
        "No prompt, mode, family, panel, split, layer, scale, site, threshold, "
        "random seed or verdict rescue follows the output."
    ),
    "random_seed": RANDOM_SEED,
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(
        PROTOCOL, sort_keys=True, separators=(",", ":")
    ).encode()
).hexdigest().upper()


def _transitions():
    return tuple(
        (source, target)
        for source in FRAMES
        for target in FRAMES
        if source != target
    )


def _decoy(source, target):
    values = [
        frame for frame in FRAMES
        if frame not in (source, target)
    ]
    if len(values) != 1:
        raise ValueError("three-mode transition has no unique decoy")
    return values[0]


def _expected(rows, history):
    key = "source" if int(history) == 0 else "target"
    return [row[key] for row in rows]


def _fit_destination_programs(training):
    programs = {}
    for layer in EARLY_LAYERS:
        programs[int(layer)] = {}
        for target in FRAMES:
            values = []
            sources = [frame for frame in FRAMES if frame != target]
            for family in TRAIN_FAMILIES:
                for panel in TRAIN_PANELS:
                    for history in range(2):
                        cells = training[family][panel][history]
                        source_mean = 0.5 * (
                            cells[sources[0]][f"early_{layer}"]
                            + cells[sources[1]][f"early_{layer}"]
                        )
                        values.append(
                            cells[target][f"early_{layer}"] - source_mean)
            programs[int(layer)][target] = (
                torch.cat(values, dim=0).mean(dim=0))
    return programs


def _random_programs(programs):
    rng = np.random.default_rng(RANDOM_SEED)
    result = {}
    for target in sorted(programs):
        reference = programs[target].float()
        norm = float(reference.norm())
        values = []
        for _index in range(N_RANDOM):
            draw = torch.tensor(
                rng.standard_normal(tuple(reference.shape)),
                dtype=torch.float32)
            draw *= norm / float(draw.norm().clamp_min(1e-8))
            values.append(draw)
        result[target] = values
    return result


def _identical_positions(alignment):
    batches = [
        alignment["batches"][history][frame]
        for history in range(2)
        for frame in FRAMES
    ]
    excluded = set(_all_positions(alignment))
    instruction_start = int(alignment["instruction_positions"][0])
    candidates = []
    for position in range(int(batches[0]["ids"].shape[1])):
        if position in excluded or position >= instruction_start:
            continue
        reference = batches[0]["ids"][:, position]
        if all(torch.equal(
                reference, batch["ids"][:, position])
               for batch in batches[1:]):
            candidates.append(int(position))
    if len(candidates) < PROGRAM_WIDTH:
        raise ValueError("fewer than six three-mode identical positions")
    return candidates[-PROGRAM_WIDTH:]


def _representation(cache, checkpoint):
    if checkpoint == "l21":
        return _answer_state(cache).float().flatten(1)
    return _processed(cache, int(checkpoint[1:])).float().flatten(1)


def _empty_selection():
    return {
        checkpoint: {
            "destination": [],
            "target_vs_decoy": [],
            "preference": [],
        }
        for checkpoint in ("l21", "l27")
    }


def _append_selection(
        accumulator, patched, baselines, source, target, decoy):
    frame_index = {frame: index for index, frame in enumerate(FRAMES)}
    for checkpoint in ("l21", "l27"):
        generated = _representation(patched, checkpoint)
        states = torch.stack([
            _representation(baselines[frame], checkpoint)
            for frame in FRAMES
        ], dim=1)
        distances = (states - generated[:, None, :]).norm(dim=-1)
        nearest = distances.argmin(dim=1)
        target_index = frame_index[target]
        decoy_index = frame_index[decoy]
        target_distance = distances[:, target_index]
        decoy_distance = distances[:, decoy_index]
        target_decoy_scale = (
            states[:, target_index, :] - states[:, decoy_index, :]
        ).norm(dim=-1).clamp_min(1e-8)
        accumulator[checkpoint]["destination"].extend(
            (nearest == target_index).tolist())
        accumulator[checkpoint]["target_vs_decoy"].extend(
            (target_distance < decoy_distance).tolist())
        accumulator[checkpoint]["preference"].extend(
            ((decoy_distance - target_distance)
             / target_decoy_scale).tolist())


def _selection_summary(accumulator):
    result = {}
    for checkpoint, values in accumulator.items():
        n_rows = len(values["destination"])
        result[checkpoint] = {
            "n_rows": n_rows,
            "destination_wins": int(sum(values["destination"])),
            "destination_accuracy": float(
                np.mean(values["destination"])),
            "target_vs_decoy_wins": int(
                sum(values["target_vs_decoy"])),
            "target_vs_decoy_accuracy": float(
                np.mean(values["target_vs_decoy"])),
            "mean_target_preference": float(
                np.mean(values["preference"])),
            "target_preference_rows": [
                float(value) for value in values["preference"]
            ],
        }
    return result


def _selector_score(primary, wrong, selection):
    margin = float(
        primary["l27"]["mean_progress"]
        - wrong["l27"]["mean_progress"])
    return float(min(
        primary["l21"]["mean_progress"],
        primary["l27"]["mean_progress"],
        margin,
        selection["l21"]["target_vs_decoy_accuracy"] - 0.5,
        selection["l27"]["target_vs_decoy_accuracy"] - 0.5,
    ))


def _select_layer(calibration):
    rows = []
    for layer in EARLY_LAYERS:
        cells = calibration[str(layer)]
        rows.append({
            "layer": int(layer),
            "score": float(min(cell["selector_score"] for cell in cells)),
            "minimum_value_accuracy": float(min(
                min(
                    cell["primary"]["minimum_value_accuracy"],
                    cell["baseline_minimum_value_accuracy"],
                )
                for cell in cells)),
        })
    eligible = [
        row for row in rows
        if row["minimum_value_accuracy"]
        >= MINIMUM_VALUE_ACCURACY - 1e-9
    ]
    pool = eligible if eligible else rows
    selected = sorted(
        pool, key=lambda row: (-row["score"], row["layer"]))[0]
    return rows, selected


def _portable_pass(primary, exact, wrong, selection):
    exact_effect = max(
        float(exact["l27"]["mean_progress"]), 1e-8)
    recovery = float(
        primary["l27"]["mean_progress"] / exact_effect)
    margin = float(
        primary["l27"]["mean_progress"]
        - wrong["l27"]["mean_progress"])
    passed = bool(
        primary["l21"]["mean_progress"]
        >= MINIMUM_PORTABLE_PROGRESS - 1e-9
        and primary["l27"]["mean_progress"]
        >= MINIMUM_PORTABLE_PROGRESS - 1e-9
        and recovery >= MINIMUM_PORTABLE_RECOVERY - 1e-9
        and primary["l21"]["positive_rows"] >= MINIMUM_POSITIVE_ROWS
        and primary["l27"]["positive_rows"] >= MINIMUM_POSITIVE_ROWS
        and selection["l21"]["target_vs_decoy_wins"]
        >= MINIMUM_TARGET_VS_DECOY_WINS
        and selection["l27"]["target_vs_decoy_wins"]
        >= MINIMUM_TARGET_VS_DECOY_WINS
        and selection["l27"]["destination_wins"]
        >= MINIMUM_DESTINATION_WINS
        and margin >= MINIMUM_CODE_MARGIN - 1e-9
        and primary["minimum_value_accuracy"]
        >= MINIMUM_VALUE_ACCURACY - 1e-9
    )
    return passed, recovery, margin


def _verdict(
        late_count, exact_count, selector_count, movement_count,
        mediation_count, context_count, specificity):
    if int(late_count) < N_CELLS:
        return "CAUSAL_TARGET_UNRESOLVED"
    if int(exact_count) < N_CELLS:
        return "NO_EARLY_THREE_MODE_CAPACITY"
    if (
        int(selector_count) == N_CELLS
        and int(mediation_count) == N_CELLS
        and int(context_count) == N_CELLS
        and bool(specificity)
    ):
        return "DESTINATION_SPECIFIC_CONTEXTUAL_SELECTOR"
    if int(selector_count) == N_CELLS:
        return "SELECTIVE_STEERING_WITHOUT_NATIVE_MEDIATION"
    if int(movement_count) >= N_CELLS - 2:
        return "GENERIC_EVICTION_NOT_SELECTION"
    return "NO_PORTABLE_THREE_MODE_CONTROL"


def _self_check():
    torch.manual_seed(11)
    programs = {
        frame: torch.randn(PROGRAM_WIDTH, 13)
        for frame in FRAMES
    }
    randoms = _random_programs(programs)
    norm_error = max(
        abs(float(value.norm()) - float(programs[target].norm()))
        for target in FRAMES
        for value in randoms[target]
    )

    batch = 24
    hidden = 7
    native = {
        frame: torch.randn(batch, PATCH_WIDTH, hidden)
        for frame in FRAMES
    }
    routes = {
        frame: torch.randn(batch, hidden)
        for frame in FRAMES
    }
    cache = {
        frame: {
            "l21_program": value,
            "checkpoint_27": value[:, -1, :] + routes[frame],
        }
        for frame, value in native.items()
    }
    selection = _empty_selection()
    _append_selection(
        selection, cache["communication"], cache,
        "epistemic", "communication", "search")
    selection = _selection_summary(selection)

    calibration = {
        str(layer): [
            {
                "selector_score": 0.05,
                "primary": {"minimum_value_accuracy": 1.0},
                "baseline_minimum_value_accuracy": 1.0,
            }
            for _ in range(12)
        ]
        for layer in EARLY_LAYERS
    }
    for cell in calibration[str(EARLY_LAYERS[2])]:
        cell["selector_score"] = 0.30
    _rows, selected = _select_layer(calibration)

    observed = {
        _verdict(11, 12, 12, 12, 12, 12, True),
        _verdict(12, 11, 12, 12, 12, 12, True),
        _verdict(12, 12, 12, 12, 12, 12, True),
        _verdict(12, 12, 12, 12, 11, 12, False),
        _verdict(12, 12, 8, 10, 8, 12, False),
        _verdict(12, 12, 5, 5, 5, 12, False),
        "CALIBRATION_SELECTOR_NULL",
    }
    return {
        "frame_count": len(FRAMES),
        "transition_count": len(_transitions()),
        "unique_decoys": all(
            _decoy(source, target) not in (source, target)
            for source, target in _transitions()),
        "random_count": min(len(values) for values in randoms.values()),
        "random_norm_error": float(norm_error),
        "exact_target_selection": bool(
            selection["l21"]["destination_wins"] == batch
            and selection["l27"]["target_vs_decoy_wins"] == batch),
        "selected_layer": int(selected["layer"]),
        "verdicts_reachable": observed == set(PROTOCOL["verdicts"]),
        "pass": bool(
            len(FRAMES) == 3
            and len(_transitions()) == 6
            and norm_error <= 1e-4
            and selection["l27"]["destination_wins"] == batch
            and selected["layer"] == EARLY_LAYERS[2]
            and observed == set(PROTOCOL["verdicts"])
        ),
    }


@torch.no_grad()
def run_delta_three_mode_selector(
        model_path, out_dir,
        model_key="qwen7b_three_mode_selector",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=12, self_test_only=False):
    os.makedirs(out_dir, exist_ok=True)
    check = _self_check()
    if not check["pass"]:
        raise AssertionError(f"three-mode mathematical guard failed: {check}")
    if self_test_only:
        result = {
            "stage": "delta_three_mode_selector",
            "protocol": PROTOCOL,
            "protocol_sha256": PROTOCOL_SHA256,
            "self_check": check,
            "verdict": "SELF_TEST_ONLY",
        }
        path = os.path.join(
            out_dir,
            f"results_delta_three_mode_selector_{model_key}.json")
        with open(path, "w") as handle:
            json.dump(result, handle, indent=2)
        return result
    if int(n_world) != 12:
        raise ValueError("v1 is frozen to twelve held-out test rows")

    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    if max(CHECKPOINTS) >= model_num_hidden_layers(model):
        raise ValueError("frozen selector layers are absent")
    commands, padding_plan, tokenization_tables = _commands(tok)
    splits = _row_splits()
    if len(splits["test"]) != int(n_world):
        raise AssertionError("held-out row count changed")

    training_total = (
        len(TRAIN_FAMILIES) * len(TRAIN_PANELS) * 2 * len(FRAMES)
    )
    heartbeat = Heartbeat(
        training_total, "three_mode_training_capture",
        every_sec=30, out_dir=out_dir)
    training = {}
    alignment_metadata = {}
    for family in TRAIN_FAMILIES:
        training[family] = {}
        for panel in TRAIN_PANELS:
            alignment = _frame_alignment(
                tok, dev, splits["train"], FAMILY_SPECS[family],
                commands[panel])
            positions = _all_positions(alignment)
            if len(positions) != PROGRAM_WIDTH:
                raise ValueError("training program width changed")
            training[family][panel] = []
            for history in range(2):
                row = {}
                for frame in FRAMES:
                    row[frame] = _capture(
                        model, alignment["batches"][history][frame],
                        EARLY_LAYERS, positions, capture_route=False)
                    heartbeat.step(
                        extra=f"{family}/{panel}/h{history}/{frame}")
                training[family][panel].append(row)
            alignment_metadata[f"{family}/{panel}"] = {
                "instruction_positions":
                    list(alignment["instruction_positions"]),
                "answer_positions": list(alignment["answer_positions"]),
            }
    programs_by_layer = _fit_destination_programs(training)

    calibration_total = (
        len(CALIBRATION_FAMILIES) * 2 * len(FRAMES)
        + len(CALIBRATION_FAMILIES) * len(EARLY_LAYERS)
        * 2 * len(FRAMES) * (len(FRAMES) - 1)
    )
    heartbeat = Heartbeat(
        calibration_total, "three_mode_calibration",
        every_sec=30, out_dir=out_dir)
    calibration = {str(layer): [] for layer in EARLY_LAYERS}
    calibration_public = {}
    for family in CALIBRATION_FAMILIES:
        alignment = _frame_alignment(
            tok, dev, splits["validation"], FAMILY_SPECS[family],
            commands[CALIBRATION_PANEL])
        positions = _all_positions(alignment)
        baselines = []
        baseline_accuracies = []
        for history in range(2):
            row = {}
            for frame in FRAMES:
                batch = alignment["batches"][history][frame]
                row[frame] = _capture(
                    model, batch, EARLY_LAYERS, positions)
                baseline_accuracies.append(_baseline_accuracy(
                    row[frame], batch,
                    _expected(splits["validation"], history)))
                heartbeat.step(
                    extra=f"{family}/baseline/h{history}/{frame}")
            baselines.append(row)
        calibration_public[family] = {}
        for layer in EARLY_LAYERS:
            generated = [{frame: {} for frame in FRAMES}
                         for _ in range(2)]
            for history in range(2):
                for source in FRAMES:
                    batch = alignment["batches"][history][source]
                    for target in FRAMES:
                        if target == source:
                            continue
                        generated[history][source][target] = _run_early(
                            model, batch, layer, positions,
                            programs_by_layer[layer][target], positions)
                        heartbeat.step(
                            extra=(
                                f"{family}/L{layer}/h{history}/"
                                f"{source}_to_{target}"))
            calibration_public[family][str(layer)] = {}
            for source, target in _transitions():
                decoy = _decoy(source, target)
                primary_acc = _empty_accumulator()
                wrong_acc = _empty_accumulator()
                selection_acc = _empty_selection()
                for history in range(2):
                    batch = alignment["batches"][history][source]
                    expected = _expected(
                        splits["validation"], history)
                    primary = generated[history][source][target]
                    wrong = generated[history][source][decoy]
                    accuracy = _baseline_accuracy(
                        primary, batch, expected)
                    _append(
                        primary_acc, baselines[history][source],
                        baselines[history][target], primary, accuracy)
                    _append(
                        wrong_acc, baselines[history][source],
                        baselines[history][target], wrong,
                        _baseline_accuracy(wrong, batch, expected))
                    _append_selection(
                        selection_acc, primary, baselines[history],
                        source, target, decoy)
                primary_summary = _summary(primary_acc)
                wrong_summary = _summary(wrong_acc)
                selection_summary = _selection_summary(selection_acc)
                cell = {
                    "source": source,
                    "target": target,
                    "decoy": decoy,
                    "primary": primary_summary,
                    "wrong": wrong_summary,
                    "selection": selection_summary,
                    "selector_score": _selector_score(
                        primary_summary, wrong_summary,
                        selection_summary),
                    "baseline_minimum_value_accuracy": float(
                        min(baseline_accuracies)),
                }
                calibration[str(layer)].append(cell)
                calibration_public[family][str(layer)][
                    f"{source}_to_{target}"
                ] = cell
    layer_rows, selected = _select_layer(calibration)
    selected_layer = int(selected["layer"])
    calibration_pass = bool(
        selected["score"] >= MINIMUM_CALIBRATION_SCORE - 1e-9
        and selected["minimum_value_accuracy"]
        >= MINIMUM_VALUE_ACCURACY - 1e-9
    )
    log(
        "THREE-MODE-SELECTOR calibration "
        f"selected=L{selected_layer} score={selected['score']:+.4f} "
        f"accuracy={selected['minimum_value_accuracy']:.0%} "
        f"pass={calibration_pass}")

    selected_programs = {
        target: value.detach().float().cpu()
        for target, value in programs_by_layer[selected_layer].items()
    }
    random_programs = _random_programs(selected_programs)
    program_norms = {
        target: float(value.norm())
        for target, value in selected_programs.items()
    }
    random_norm_error = float(max(
        abs(float(value.norm()) - program_norms[target])
        for target, values in random_programs.items()
        for value in values
    ))
    permutations = _derangements(len(splits["test"]) * 2)
    test_alignments = {}
    test_layouts = {}
    for family in TEST_FAMILIES:
        alignment = _frame_alignment(
            tok, dev, splits["test"], FAMILY_SPECS[family],
            commands[TEST_PANEL])
        test_alignments[family] = alignment
        test_layouts[family] = {
            "instruction_positions":
                list(alignment["instruction_positions"]),
            "answer_positions": list(alignment["answer_positions"]),
            "program_positions": _all_positions(alignment),
            "identical_positions": _identical_positions(alignment),
        }

    freeze_json_path = os.path.join(
        out_dir, "three_mode_selector_freeze.json")
    freeze_npz_path = os.path.join(
        out_dir, "three_mode_selector_programs.npz")
    freeze = {
        "protocol_sha256": PROTOCOL_SHA256,
        "selected_layer": selected_layer,
        "selection_rows": layer_rows,
        "calibration_pass": calibration_pass,
        "program_shapes": {
            target: list(value.shape)
            for target, value in selected_programs.items()
        },
        "program_sha256": {
            target: hashlib.sha256(
                value.contiguous().numpy().tobytes()
            ).hexdigest().upper()
            for target, value in selected_programs.items()
        },
        "program_norms": program_norms,
        "maximum_random_norm_error": random_norm_error,
        "random_seed": RANDOM_SEED,
        "n_random": N_RANDOM,
        "derangements": [
            permutation.tolist() for permutation in permutations
        ],
        "test_families": list(TEST_FAMILIES),
        "test_panel": TEST_PANEL,
        "test_layouts": test_layouts,
    }
    with open(freeze_json_path, "w") as handle:
        json.dump(freeze, handle, indent=2)
    arrays = {
        f"program_{target}": value.numpy()
        for target, value in selected_programs.items()
    }
    for target, values in random_programs.items():
        for index, value in enumerate(values):
            arrays[f"random_{target}_{index:02d}"] = value.numpy()
    np.savez_compressed(freeze_npz_path, **arrays)
    boundary = _PredictionBoundary()
    boundary.freeze(
        _artifact_sha256(freeze_json_path),
        _artifact_sha256(freeze_npz_path))
    log(
        "FROZEN three-mode selector "
        f"json={boundary.json_sha256} npz={boundary.npz_sha256}")

    base_result = {
        "stage": "delta_three_mode_selector",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "self_check": check,
        "commands": commands,
        "padding_plan": padding_plan,
        "tokenization_tables": tokenization_tables,
        "family_split": PROTOCOL["family_split"],
        "row_split": PROTOCOL["world_split"],
        "alignment_metadata": alignment_metadata,
        "calibration": calibration_public,
        "layer_selection": layer_rows,
        "selected_layer": selected_layer,
        "prediction_freeze": {
            "json_path": freeze_json_path,
            "npz_path": freeze_npz_path,
            "json_sha256": boundary.json_sha256,
            "npz_sha256": boundary.npz_sha256,
        },
        "random_control_audit": {
            "program_norms": program_norms,
            "maximum_random_norm_error": random_norm_error,
            "within_destination_norm_variance": 0.0,
            "norm_effect_correlation_identifiable": False,
            "reason": (
                "All nineteen random codes for a destination have the exact "
                "same norm as its learned code."
            ),
        },
    }
    if not calibration_pass:
        result = {
            **base_result,
            "verdict": "CALIBRATION_SELECTOR_NULL",
        }
        path = os.path.join(
            out_dir,
            f"results_delta_three_mode_selector_{model_key}.json")
        with open(path, "w") as handle:
            json.dump(result, handle, indent=2)
        log("THREE-MODE-SELECTOR verdict=CALIBRATION_SELECTOR_NULL")
        return result

    boundary.require_evaluation()
    test_total = (
        len(TEST_FAMILIES) * 2 * len(FRAMES)
        + len(TEST_FAMILIES) * 2 * len(FRAMES) * (len(FRAMES) - 1)
        + N_CELLS * 2 * (6 + N_RANDOM)
    )
    heartbeat = Heartbeat(
        test_total, "three_mode_heldout_test",
        every_sec=30, out_dir=out_dir)
    families = {}
    late_count = exact_count = selector_count = movement_count = 0
    mediation_count = context_count = 0
    primary_scores = []
    reverse_ratios = []
    identical_ratios = []
    random_scores_by_index = [[] for _ in range(N_RANDOM)]

    for family in TEST_FAMILIES:
        alignment = test_alignments[family]
        layout = test_layouts[family]
        positions = list(layout["program_positions"])
        answer_positions = list(layout["answer_positions"])
        identical_positions = list(layout["identical_positions"])
        baselines = []
        baseline_accuracies = {}
        for history in range(2):
            row = {}
            for frame in FRAMES:
                batch = alignment["batches"][history][frame]
                row[frame] = _capture(
                    model, batch, (selected_layer,), positions)
                baseline_accuracies[
                    f"h{history}/{frame}"
                ] = _baseline_accuracy(
                    row[frame], batch,
                    _expected(splits["test"], history))
                heartbeat.step(
                    extra=f"{family}/baseline/h{history}/{frame}")
            baselines.append(row)
        family_baseline_pass = bool(
            min(baseline_accuracies.values())
            >= MINIMUM_VALUE_ACCURACY - 1e-9
        )

        portable_cache = [{frame: {} for frame in FRAMES}
                          for _ in range(2)]
        for history in range(2):
            for source in FRAMES:
                batch = alignment["batches"][history][source]
                for target in FRAMES:
                    if target == source:
                        continue
                    portable_cache[history][source][target] = _run_early(
                        model, batch, selected_layer, positions,
                        selected_programs[target], positions)
                    heartbeat.step(
                        extra=(
                            f"{family}/portable/h{history}/"
                            f"{source}_to_{target}"))

        family_result = {
            "positions": layout,
            "baseline_accuracies": baseline_accuracies,
            "directions": {},
        }
        for source, target in _transitions():
            decoy = _decoy(source, target)
            arm_names = (
                "late_oracle", "exact_early", "portable",
                "wrong_destination", "reverse", "identical",
                "block", "rescue",
            )
            arm_accumulators = {
                name: _empty_accumulator() for name in arm_names
            }
            selection_accumulators = {
                name: _empty_selection()
                for name in (
                    "portable", "wrong_destination", "reverse",
                )
            }
            random_accumulators = [
                _empty_accumulator() for _ in range(N_RANDOM)
            ]
            random_selection = [
                _empty_selection() for _ in range(N_RANDOM)
            ]
            generated_states = []
            target_states = []
            source_states = []

            for history in range(2):
                batch = alignment["batches"][history][source]
                expected = _expected(splits["test"], history)
                origin = baselines[history][source]
                target_cache = baselines[history][target]
                portable = portable_cache[history][source][target]
                wrong = portable_cache[history][source][decoy]

                late = _late_patch_cache(
                    model, batch, answer_positions,
                    _answer_state(target_cache))
                heartbeat.step(
                    extra=f"{family}/{source}_to_{target}/h{history}/late")
                _append(
                    arm_accumulators["late_oracle"],
                    origin, target_cache, late,
                    _baseline_accuracy(late, batch, expected))

                exact_delta = (
                    target_cache[f"early_{selected_layer}"]
                    - origin[f"early_{selected_layer}"])
                exact = _run_early(
                    model, batch, selected_layer, positions,
                    exact_delta, positions)
                heartbeat.step(
                    extra=f"{family}/{source}_to_{target}/h{history}/exact")
                _append(
                    arm_accumulators["exact_early"],
                    origin, target_cache, exact,
                    _baseline_accuracy(exact, batch, expected))

                _append(
                    arm_accumulators["portable"],
                    origin, target_cache, portable,
                    _baseline_accuracy(portable, batch, expected))
                _append(
                    arm_accumulators["wrong_destination"],
                    origin, target_cache, wrong,
                    _baseline_accuracy(wrong, batch, expected))
                _append_selection(
                    selection_accumulators["portable"],
                    portable, baselines[history],
                    source, target, decoy)
                _append_selection(
                    selection_accumulators["wrong_destination"],
                    wrong, baselines[history],
                    source, target, decoy)
                generated_states.append(_answer_state(portable))
                target_states.append(_answer_state(target_cache))
                source_states.append(_answer_state(origin))

                reverse = _run_early(
                    model, batch, selected_layer, positions,
                    -selected_programs[target], positions)
                heartbeat.step(
                    extra=(
                        f"{family}/{source}_to_{target}/"
                        f"h{history}/reverse"))
                _append(
                    arm_accumulators["reverse"],
                    origin, target_cache, reverse,
                    _baseline_accuracy(reverse, batch, expected))
                _append_selection(
                    selection_accumulators["reverse"],
                    reverse, baselines[history],
                    source, target, decoy)

                identical = _run_early(
                    model, batch, selected_layer, identical_positions,
                    selected_programs[target], positions)
                heartbeat.step(
                    extra=(
                        f"{family}/{source}_to_{target}/"
                        f"h{history}/identical"))
                _append(
                    arm_accumulators["identical"],
                    origin, target_cache, identical,
                    _baseline_accuracy(identical, batch, expected))

                block = _run_early(
                    model, batch, selected_layer, positions,
                    selected_programs[target], positions,
                    block_answer=_answer_state(origin))
                heartbeat.step(
                    extra=f"{family}/{source}_to_{target}/h{history}/block")
                _append(
                    arm_accumulators["block"],
                    origin, target_cache, block,
                    _baseline_accuracy(block, batch, expected))

                rescue = _late_patch_cache(
                    model, batch, answer_positions,
                    _answer_state(portable))
                heartbeat.step(
                    extra=f"{family}/{source}_to_{target}/h{history}/rescue")
                _append(
                    arm_accumulators["rescue"],
                    origin, target_cache, rescue,
                    _baseline_accuracy(rescue, batch, expected))

                for index, random_code in enumerate(
                        random_programs[target]):
                    random_cache = _run_early(
                        model, batch, selected_layer, positions,
                        random_code, positions)
                    heartbeat.step(
                        extra=(
                            f"{family}/{source}_to_{target}/"
                            f"h{history}/random{index:02d}"))
                    _append(
                        random_accumulators[index],
                        origin, target_cache, random_cache,
                        _baseline_accuracy(
                            random_cache, batch, expected))
                    _append_selection(
                        random_selection[index],
                        random_cache, baselines[history],
                        source, target, decoy)

            summaries = {
                name: _summary(accumulator)
                for name, accumulator in arm_accumulators.items()
            }
            selections = {
                name: _selection_summary(accumulator)
                for name, accumulator in selection_accumulators.items()
            }
            random_summaries = [
                _summary(accumulator)
                for accumulator in random_accumulators
            ]
            random_selections = [
                _selection_summary(accumulator)
                for accumulator in random_selection
            ]

            late_pass = bool(
                family_baseline_pass
                and _late_oracle_pass(summaries["late_oracle"]))
            exact_pass = _exact_early_pass(summaries["exact_early"])
            portable_pass, recovery, code_margin = _portable_pass(
                summaries["portable"], summaries["exact_early"],
                summaries["wrong_destination"],
                selections["portable"])
            cell_random_values = [
                float(item["l27"]["mean_progress"])
                for item in random_summaries
            ]
            cell_random_exceed = int(sum(
                value
                >= summaries["portable"]["l27"]["mean_progress"]
                for value in cell_random_values
            ))
            cell_random_percentile = float(
                100.0 * sum(
                    value
                    < summaries["portable"]["l27"]["mean_progress"]
                    for value in cell_random_values
                ) / N_RANDOM
            )
            movement_pass = bool(
                summaries["portable"]["l27"]["mean_progress"]
                >= MINIMUM_PORTABLE_PROGRESS - 1e-9
                and summaries["portable"]["l27"]["positive_rows"]
                >= MINIMUM_POSITIVE_ROWS
                and summaries["portable"]["minimum_value_accuracy"]
                >= MINIMUM_VALUE_ACCURACY - 1e-9
            )
            primary_effect = float(
                summaries["portable"]["l27"]["mean_progress"])
            block_effect = float(
                summaries["block"]["l27"]["mean_progress"])
            rescue_effect = float(
                summaries["rescue"]["l27"]["mean_progress"])
            block_fraction = float(
                (primary_effect - block_effect)
                / max(abs(primary_effect), 1e-8))
            rescue_fraction = float(
                rescue_effect / max(abs(primary_effect), 1e-8))
            mediation_pass = bool(
                primary_effect > 0.0
                and block_fraction >= MINIMUM_MEDIATION - 1e-9
                and rescue_fraction >= MINIMUM_MEDIATION - 1e-9
            )
            context = _context_match(
                torch.cat(generated_states, dim=0),
                torch.cat(target_states, dim=0),
                torch.cat(source_states, dim=0),
                permutations)

            late_count += int(late_pass)
            exact_count += int(exact_pass)
            selector_count += int(portable_pass)
            movement_count += int(movement_pass)
            mediation_count += int(mediation_pass)
            context_count += int(context["pass"])
            primary_scores.append(primary_effect)
            reverse_ratios.append(float(
                summaries["reverse"]["l27"]["mean_progress"]
                / max(primary_effect, 1e-8)))
            identical_ratios.append(float(
                summaries["identical"]["l27"]["mean_progress"]
                / max(primary_effect, 1e-8)))
            for index, random_summary in enumerate(random_summaries):
                random_scores_by_index[index].append(
                    random_summary["l27"]["mean_progress"])

            family_result["directions"][
                f"{source}_to_{target}"
            ] = {
                "source": source,
                "target": target,
                "decoy": decoy,
                "arms": summaries,
                "selection": selections,
                "random_arms": random_summaries,
                "random_selection": random_selections,
                "context_match": context,
                "mediation": {
                    "primary_l27_progress": primary_effect,
                    "blocked_l27_progress": block_effect,
                    "rescue_l27_progress": rescue_effect,
                    "block_fraction": block_fraction,
                    "rescue_fraction": rescue_fraction,
                    "pass": mediation_pass,
                },
                "adjudication": {
                    "late_oracle_pass": late_pass,
                    "exact_early_pass": exact_pass,
                    "selector_pass": portable_pass,
                    "movement_pass": movement_pass,
                    "portable_recovery_of_exact": recovery,
                    "code_identity_margin": code_margin,
                    "random_codes_equal_or_stronger":
                        cell_random_exceed,
                    "learned_code_random_percentile":
                        cell_random_percentile,
                    "cell_add_one_random_p": float(
                        (1.0 + cell_random_exceed)
                        / (1.0 + N_RANDOM)),
                },
            }
        families[family] = family_result

    primary_breadth = float(min(primary_scores))
    random_breadth = [
        float(min(values)) for values in random_scores_by_index
    ]
    random_exceed = int(sum(
        value >= primary_breadth for value in random_breadth))
    random_p = float((1.0 + random_exceed) / (1.0 + N_RANDOM))
    maximum_reverse_fraction = float(max(reverse_ratios))
    maximum_identical_fraction = float(max(identical_ratios))
    specificity = bool(
        selector_count == N_CELLS
        and maximum_reverse_fraction
        <= MAXIMUM_NULL_FRACTION + 1e-9
        and maximum_identical_fraction
        <= MAXIMUM_NULL_FRACTION + 1e-9
        and random_p <= MAXIMUM_RANDOM_P + 1e-12
    )
    verdict = _verdict(
        late_count, exact_count, selector_count, movement_count,
        mediation_count, context_count, specificity)
    overall = {
        "cell_count": N_CELLS,
        "late_oracle_count": late_count,
        "exact_early_count": exact_count,
        "selector_count": selector_count,
        "movement_count": movement_count,
        "mediation_count": mediation_count,
        "context_match_count": context_count,
        "primary_breadth_score": primary_breadth,
        "maximum_reverse_fraction": maximum_reverse_fraction,
        "maximum_identical_fraction": maximum_identical_fraction,
        "random_breadth_scores": random_breadth,
        "random_exceed_count": random_exceed,
        "random_add_one_p": random_p,
        "specificity_pass": specificity,
        "verdict": verdict,
    }
    result = {
        **base_result,
        "families": families,
        "overall": overall,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_three_mode_selector_{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2)
    log(
        "THREE-MODE-SELECTOR "
        f"verdict={verdict} exact={exact_count}/{N_CELLS} "
        f"selector={selector_count}/{N_CELLS} "
        f"movement={movement_count}/{N_CELLS} "
        f"mediation={mediation_count}/{N_CELLS} "
        f"context={context_count}/{N_CELLS} "
        f"random_p={random_p:.3f} artifact={path}")
    return result
