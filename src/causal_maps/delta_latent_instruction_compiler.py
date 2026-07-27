"""Prospective latent-instruction compiler assay.

A training-derived early operation code is applied before the established
L21 answer-state interface.  The primary question is whether the frozen model
then constructs the receiver's own held-out target state and downstream route,
rather than merely carrying a shared displacement or taking an alternate path.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from .delta_anchor_write import _resolve
from .delta_causal_atlas import (
    PANEL_LABELS,
    _PredictionBoundary,
    _artifact_sha256,
    _commands,
    _frame_alignment,
    _row_splits,
)
from .delta_cross_domain_controller import _generic_accuracy
from .delta_exact_transplant_locus_diagnostic import _run_exact_patch
from .delta_heterogeneous_family_screen import FAMILY_SPECS, VALUES
from .logutil import Heartbeat, log
from .model_utils import (
    get_decoder_layers,
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)
from .patching import _split_output


PROTOCOL_VERSION = "2026-07-27-p2-latent-instruction-compiler-v1"
EARLY_LAYERS = (2, 6, 10, 14, 18)
GENERATED_LAYER = 21
CHECKPOINTS = (24, 27)
PRIMARY_CHECKPOINT = 27
FRAMES = ("epistemic", "search")
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
PATCH_WIDTH = 3
PROGRAM_WIDTH = 6
N_RANDOM = 19
RANDOM_SEED = 8675309

MINIMUM_CALIBRATION_SCORE = 0.20
MINIMUM_VALUE_ACCURACY = 0.80
MINIMUM_LATE_ORACLE_PROGRESS = 0.40
MINIMUM_EXACT_EARLY_L21_PROGRESS = 0.60
MINIMUM_EXACT_EARLY_L27_PROGRESS = 0.40
MINIMUM_PORTABLE_L21_PROGRESS = 0.25
MINIMUM_PORTABLE_L27_PROGRESS = 0.25
MINIMUM_PORTABLE_RECOVERY = 0.50
MINIMUM_POSITIVE_ROWS = 20
MINIMUM_MEDIATION = 0.70
MAXIMUM_NULL_FRACTION = 0.50
MINIMUM_CONTEXT_ADVANTAGE = 0.10
MAXIMUM_RANDOM_P = 0.05


PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "status": "prospective heldout causal-state generation test",
    "hypothesis": (
        "A portable early operation code is transformed by frozen intervening "
        "layers into receiver-specific L21 operation states and downstream "
        "routes."
    ),
    "interpretation_boundary": (
        "Exact early target states are a compiler-capacity oracle. Only the "
        "training-derived code is a prospective portable intervention."
    ),
    "model": "Qwen2.5-7B-Instruct, 8-bit, Tesla T4",
    "early_layers": list(EARLY_LAYERS),
    "generated_layer": GENERATED_LAYER,
    "checkpoints": list(CHECKPOINTS),
    "primary_checkpoint": PRIMARY_CHECKPOINT,
    "frames": list(FRAMES),
    "positions": {
        "program": (
            "two position-matched three-token command occurrences"
        ),
        "generated_state": "three-token answer occurrence at L21",
        "processed_route": (
            "checkpoint final state minus generated L21 final answer state"
        ),
    },
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
        "Per direction and candidate layer, mean paired target-minus-source "
        "six-state residual over training families, panels, rows and both "
        "clean/natural histories; no fitted scale."
    ),
    "layer_selection": (
        "Maximize the minimum of L21 generated-state progress and processed-"
        "L27 route progress over every calibration cell; earlier layer is "
        "the deterministic tie-breaker."
    ),
    "calibration_gate": {
        "minimum_score": MINIMUM_CALIBRATION_SCORE,
        "minimum_value_accuracy": MINIMUM_VALUE_ACCURACY,
    },
    "test_arms": [
        "exact same-row target state at selected early layer/all six positions",
        "training-derived early program/all six positions",
        "answer-occurrence program ablation",
        "instruction-occurrence program ablation",
        "program at six matched identical-token positions",
        "sign-reversed program",
        "19 same-norm isotropic random programs",
        "exact matched L21 target answer state",
        "early program plus L21 source-state restoration",
        "generated L21 state transplanted without early program",
    ],
    "primary_gates": {
        "minimum_late_oracle_progress": MINIMUM_LATE_ORACLE_PROGRESS,
        "minimum_exact_early_l21_progress":
            MINIMUM_EXACT_EARLY_L21_PROGRESS,
        "minimum_exact_early_l27_progress":
            MINIMUM_EXACT_EARLY_L27_PROGRESS,
        "minimum_portable_l21_progress": MINIMUM_PORTABLE_L21_PROGRESS,
        "minimum_portable_l27_progress": MINIMUM_PORTABLE_L27_PROGRESS,
        "minimum_portable_recovery": MINIMUM_PORTABLE_RECOVERY,
        "minimum_positive_rows": MINIMUM_POSITIVE_ROWS,
        "minimum_value_accuracy": MINIMUM_VALUE_ACCURACY,
        "minimum_block_and_rescue_fraction": MINIMUM_MEDIATION,
        "occurrence_ablation_interpretation": (
            "answer-only and instruction-only are localization arms, not "
            "nulls; matching the six-position primary implies compression"
        ),
        "maximum_identical_or_reverse_fraction": MAXIMUM_NULL_FRACTION,
        "minimum_receiver_context_error_reduction":
            MINIMUM_CONTEXT_ADVANTAGE,
        "maximum_add_one_random_p": MAXIMUM_RANDOM_P,
    },
    "verdicts": [
        "CONTEXT_COMPILED_LATENT_INSTRUCTION",
        "CONTEXTUAL_COMPILER_WITHOUT_PORTABLE_CODE",
        "PORTABLE_STEERING_WITHOUT_NATIVE_COMPILATION",
        "NO_EARLY_COMPILATION",
        "CALIBRATION_CODE_NULL",
        "CAUSAL_TARGET_UNRESOLVED",
    ],
    "stopping_rule": (
        "No prompt, family, lexical panel, split, layer, scale, site, "
        "threshold, random seed or verdict rescue follows the output."
    ),
    "random_seed": RANDOM_SEED,
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(
        PROTOCOL, sort_keys=True, separators=(",", ":")
    ).encode()
).hexdigest().upper()


def _directions():
    return (
        ("epistemic", "search"),
        ("search", "epistemic"),
    )


def _transition_key(source, target):
    return f"{source}_to_{target}"


def _expected(rows, history):
    key = "source" if int(history) == 0 else "target"
    return [row[key] for row in rows]


def _all_positions(alignment):
    return (
        list(alignment["instruction_positions"])
        + list(alignment["answer_positions"])
    )


def _six_identical_positions(alignment):
    """Choose a deterministic six-token null before the instruction command."""
    batches = [
        alignment["batches"][history][frame]
        for history in range(2)
        for frame in FRAMES
    ]
    length = int(batches[0]["ids"].shape[1])
    excluded = set(_all_positions(alignment))
    instruction_start = int(alignment["instruction_positions"][0])
    candidates = []
    for position in range(length):
        if position in excluded or position >= instruction_start:
            continue
        reference = batches[0]["ids"][:, position]
        if all(torch.equal(
                reference, batch["ids"][:, position])
               for batch in batches[1:]):
            candidates.append(int(position))
    if len(candidates) < PROGRAM_WIDTH:
        raise ValueError("fewer than six fully matched control positions")
    return candidates[-PROGRAM_WIDTH:]


@torch.no_grad()
def _capture(
        model, batch, early_layers, program_positions,
        capture_route=True):
    blocks = get_decoder_layers(model)
    cache = {}
    handles = []

    def positions_hook(name, positions):
        def hook(_module, _args, output):
            states, _rebuild = _split_output(output)
            cache[name] = (
                states[:, positions, :].detach().float().cpu())
        return hook

    def final_hook(name):
        def hook(_module, _args, output):
            states, _rebuild = _split_output(output)
            cache[name] = states[:, -1, :].detach().float().cpu()
        return hook

    for layer in early_layers:
        handles.append(blocks[int(layer)].register_forward_hook(
            positions_hook(f"early_{int(layer)}", program_positions)))
    if capture_route:
        handles.append(blocks[GENERATED_LAYER].register_forward_hook(
            positions_hook("l21_program", program_positions)))
        for checkpoint in CHECKPOINTS:
            handles.append(blocks[int(checkpoint)].register_forward_hook(
                final_hook(f"checkpoint_{int(checkpoint)}")))
    try:
        output = model(
            input_ids=batch["ids"], attention_mask=batch["am"],
            use_cache=False)
        if capture_route:
            cache["logits"] = (
                output.logits[:, -1, :].detach().float().cpu())
    finally:
        for handle in handles:
            handle.remove()
    return cache


@torch.no_grad()
def _run_early(
        model, batch, early_layer, patch_positions, delta,
        program_positions, block_answer=None):
    """Apply an early additive/exact delta and capture the generated route."""
    blocks = get_decoder_layers(model)
    cache = {}
    handles = []
    answer_positions = program_positions[-PATCH_WIDTH:]

    def early_hook(_module, _args, output):
        states, rebuild = _split_output(output)
        states = states.clone()
        value = delta.to(device=states.device, dtype=states.dtype)
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.shape[0] == 1:
            value = value.expand(states.shape[0], -1, -1)
        if value.shape != states[:, patch_positions, :].shape:
            raise ValueError(
                "early program shape does not match patch positions: "
                f"{tuple(value.shape)} versus "
                f"{tuple(states[:, patch_positions, :].shape)}")
        states[:, patch_positions, :] += value
        return rebuild(states)

    def generated_hook(_module, _args, output):
        states, rebuild = _split_output(output)
        if block_answer is not None:
            states = states.clone()
            value = block_answer.to(
                device=states.device, dtype=states.dtype)
            states[:, answer_positions, :] = value
            output = rebuild(states)
        cache["l21_program"] = (
            states[:, program_positions, :].detach().float().cpu())
        return output

    def final_hook(name):
        def hook(_module, _args, output):
            states, _rebuild = _split_output(output)
            cache[name] = states[:, -1, :].detach().float().cpu()
        return hook

    handles.append(blocks[int(early_layer)].register_forward_hook(early_hook))
    handles.append(blocks[GENERATED_LAYER].register_forward_hook(
        generated_hook))
    for checkpoint in CHECKPOINTS:
        handles.append(blocks[int(checkpoint)].register_forward_hook(
            final_hook(f"checkpoint_{int(checkpoint)}")))
    try:
        output = model(
            input_ids=batch["ids"], attention_mask=batch["am"],
            use_cache=False)
        cache["logits"] = output.logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return cache


def _answer_state(cache):
    return cache["l21_program"][:, -PATCH_WIDTH:, :]


def _processed(cache, checkpoint):
    return (
        cache[f"checkpoint_{int(checkpoint)}"].float()
        - _answer_state(cache)[:, -1, :].float()
    )


def _transport(origin, target, patched):
    origin = origin.float().flatten(1)
    target = target.float().flatten(1)
    patched = patched.float().flatten(1)
    direction = target - origin
    displacement = patched - origin
    norm2 = direction.square().sum(dim=-1).clamp_min(1e-8)
    progress = (displacement * direction).sum(dim=-1) / norm2
    distance = (
        (patched - target).norm(dim=-1)
        / direction.norm(dim=-1).clamp_min(1e-8)
    )
    return progress, distance


def _empty_accumulator():
    return {
        "accuracies": [],
        "l21": {"progress": [], "distance": []},
        **{
            f"l{checkpoint}": {"progress": [], "distance": []}
            for checkpoint in CHECKPOINTS
        },
    }


def _append(
        accumulator, origin, target, patched, accuracy):
    accumulator["accuracies"].append(float(accuracy))
    progress, distance = _transport(
        _answer_state(origin), _answer_state(target),
        _answer_state(patched))
    accumulator["l21"]["progress"].extend(progress.tolist())
    accumulator["l21"]["distance"].extend(distance.tolist())
    for checkpoint in CHECKPOINTS:
        progress, distance = _transport(
            _processed(origin, checkpoint),
            _processed(target, checkpoint),
            _processed(patched, checkpoint))
        accumulator[f"l{checkpoint}"]["progress"].extend(progress.tolist())
        accumulator[f"l{checkpoint}"]["distance"].extend(distance.tolist())


def _summary(accumulator):
    result = {
        "minimum_value_accuracy": float(min(accumulator["accuracies"])),
        "value_accuracy_by_history": [
            float(value) for value in accumulator["accuracies"]
        ],
    }
    for name in ("l21", "l24", "l27"):
        progress = accumulator[name]["progress"]
        distance = accumulator[name]["distance"]
        result[name] = {
            "n_rows": len(progress),
            "mean_progress": float(np.mean(progress)),
            "median_progress": float(np.median(progress)),
            "positive_rows": int(sum(value > 0.0 for value in progress)),
            "progress_rows": [float(value) for value in progress],
            "mean_distance_ratio": float(np.mean(distance)),
            "median_distance_ratio": float(np.median(distance)),
            "distance_ratio_rows": [float(value) for value in distance],
        }
    return result


def _baseline_accuracy(cache, batch, expected):
    return _generic_accuracy(
        cache["logits"], batch, expected, VALUES)


def _late_patch_cache(model, batch, positions, target_values):
    result = _run_exact_patch(
        model, batch, GENERATED_LAYER, positions, target_values)
    return {
        "l21_program": target_values,
        **{
            f"checkpoint_{checkpoint}":
                result[f"checkpoint_{checkpoint}"]
            for checkpoint in CHECKPOINTS
        },
        "logits": result["logits"],
    }


def _fit_programs(training):
    programs = {}
    for layer in EARLY_LAYERS:
        programs[int(layer)] = {}
        for source, target in _directions():
            values = []
            for family in TRAIN_FAMILIES:
                for panel in TRAIN_PANELS:
                    for history in range(2):
                        cells = training[family][panel][history]
                        values.append(
                            cells[target][f"early_{layer}"]
                            - cells[source][f"early_{layer}"]
                        )
            programs[int(layer)][_transition_key(source, target)] = (
                torch.cat(values, dim=0).mean(dim=0))
    return programs


def _calibration_score(cell_summaries):
    return float(min(
        min(
            cell["l21"]["mean_progress"],
            cell["l27"]["mean_progress"],
        )
        for cell in cell_summaries
    ))


def _select_layer(calibration):
    rows = []
    for layer in EARLY_LAYERS:
        cells = calibration[str(layer)]
        rows.append({
            "layer": int(layer),
            "score": _calibration_score(cells),
            "minimum_value_accuracy": float(min(
                min(
                    cell["minimum_value_accuracy"],
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


def _random_programs(programs):
    rng = np.random.default_rng(RANDOM_SEED)
    result = {}
    for direction in sorted(programs):
        reference = programs[direction].float()
        norm = float(reference.norm())
        values = []
        for _index in range(N_RANDOM):
            draw = torch.tensor(
                rng.standard_normal(tuple(reference.shape)),
                dtype=torch.float32)
            draw *= norm / float(draw.norm().clamp_min(1e-8))
            values.append(draw)
        result[direction] = values
    return result


def _derangements(n_rows):
    if int(n_rows) <= N_RANDOM:
        raise ValueError(
            "at least N_RANDOM + 1 rows are required for distinct "
            "cyclic derangements")
    rng = np.random.default_rng(RANDOM_SEED + 1)
    identity = np.arange(int(n_rows))
    result = []
    while len(result) < N_RANDOM:
        candidate = rng.permutation(int(n_rows))
        if np.any(candidate == identity):
            continue
        if any(np.array_equal(candidate, old) for old in result):
            continue
        result.append(candidate)
    return result


def _context_match(generated, target, source, permutations):
    """Test receiver-specific operation structure with a same-origin null.

    Every permuted target is differenced from the *receiver's* source state,
    not its donor's source state.  Otherwise the common ``-source`` term in
    two ordinary operation deltas would manufacture a matching advantage.
    """
    generated = generated.float().flatten(1)
    target = target.float().flatten(1)
    source = source.float().flatten(1)
    generated_delta = generated - source
    target_delta = target - source
    generated_delta = (
        generated_delta
        - generated_delta.mean(dim=0, keepdim=True)
    )
    target_delta = (
        target_delta - target_delta.mean(dim=0, keepdim=True)
    )
    scale = float(
        target_delta.square().sum(dim=-1).mean().clamp_min(1e-8))
    matched = float(
        (generated_delta - target_delta)
        .square().sum(dim=-1).mean() / scale)
    random_errors = []
    for permutation in permutations:
        permuted_delta = (
            target[torch.tensor(permutation)] - source)
        permuted_delta = (
            permuted_delta
            - permuted_delta.mean(dim=0, keepdim=True)
        )
        random_errors.append(float(
            (generated_delta - permuted_delta)
            .square().sum(dim=-1).mean() / scale
        ))
    median_random = float(np.median(random_errors))
    advantage = float(
        (median_random - matched) / max(median_random, 1e-8))
    exceed = sum(value <= matched for value in random_errors)
    p_value = (1.0 + exceed) / (1.0 + len(random_errors))
    return {
        "matched_normalized_error": matched,
        "random_normalized_errors": random_errors,
        "median_random_normalized_error": median_random,
        "relative_error_reduction": advantage,
        "random_exceed_count": int(exceed),
        "add_one_p": float(p_value),
        "pass": bool(
            advantage >= MINIMUM_CONTEXT_ADVANTAGE - 1e-9
            and p_value <= MAXIMUM_RANDOM_P + 1e-12
        ),
    }


def _late_oracle_pass(summary):
    return bool(
        summary["l27"]["mean_progress"]
        >= MINIMUM_LATE_ORACLE_PROGRESS - 1e-9
        and summary["l27"]["positive_rows"] >= MINIMUM_POSITIVE_ROWS
        and summary["minimum_value_accuracy"]
        >= MINIMUM_VALUE_ACCURACY - 1e-9
    )


def _exact_early_pass(summary):
    return bool(
        summary["l21"]["mean_progress"]
        >= MINIMUM_EXACT_EARLY_L21_PROGRESS - 1e-9
        and summary["l27"]["mean_progress"]
        >= MINIMUM_EXACT_EARLY_L27_PROGRESS - 1e-9
        and summary["l21"]["positive_rows"] >= MINIMUM_POSITIVE_ROWS
        and summary["l27"]["positive_rows"] >= MINIMUM_POSITIVE_ROWS
        and summary["minimum_value_accuracy"]
        >= MINIMUM_VALUE_ACCURACY - 1e-9
    )


def _portable_pass(summary, exact_summary):
    exact_effect = max(
        float(exact_summary["l27"]["mean_progress"]), 1e-8)
    recovery = float(
        summary["l27"]["mean_progress"] / exact_effect)
    return bool(
        summary["l21"]["mean_progress"]
        >= MINIMUM_PORTABLE_L21_PROGRESS - 1e-9
        and summary["l27"]["mean_progress"]
        >= MINIMUM_PORTABLE_L27_PROGRESS - 1e-9
        and recovery >= MINIMUM_PORTABLE_RECOVERY - 1e-9
        and summary["l21"]["positive_rows"] >= MINIMUM_POSITIVE_ROWS
        and summary["l27"]["positive_rows"] >= MINIMUM_POSITIVE_ROWS
        and summary["minimum_value_accuracy"]
        >= MINIMUM_VALUE_ACCURACY - 1e-9
    ), recovery


def _verdict(
        late_count, exact_count, portable_count, route_count,
        mediation_count, context_count, specificity):
    if int(late_count) < 4:
        return "CAUSAL_TARGET_UNRESOLVED"
    if int(exact_count) < 4:
        if int(route_count) >= 3:
            return "PORTABLE_STEERING_WITHOUT_NATIVE_COMPILATION"
        return "NO_EARLY_COMPILATION"
    if (
        int(portable_count) == 4
        and int(mediation_count) == 4
        and int(context_count) == 4
        and bool(specificity)
    ):
        return "CONTEXT_COMPILED_LATENT_INSTRUCTION"
    if int(portable_count) >= 3 and int(route_count) >= 3:
        return "PORTABLE_STEERING_WITHOUT_NATIVE_COMPILATION"
    return "CONTEXTUAL_COMPILER_WITHOUT_PORTABLE_CODE"


def _self_check():
    # A shared input code passes through a context-dependent nonlinear map.
    code = torch.tensor([[1.0, -0.5], [0.2, 0.7]])
    contexts = torch.linspace(
        -1.5, 1.5, steps=24 * 2 * 2
    ).reshape(24, 2, 2)
    target = torch.tanh(contexts + code)
    generated = target.clone()
    permutations = _derangements(len(contexts))
    context = _context_match(
        generated, target, torch.tanh(contexts), permutations)
    randoms = _random_programs({"a_to_b": code})["a_to_b"]
    random_norms = [float(value.norm()) for value in randoms]
    synthetic_cells = lambda value: [  # noqa: E731
        {
            "l21": {"mean_progress": value},
            "l27": {"mean_progress": value + 0.05},
            "minimum_value_accuracy": 1.0,
            "baseline_minimum_value_accuracy": 1.0,
        }
        for _index in range(4)
    ]
    synthetic_calibration = {
        str(layer): synthetic_cells(0.1)
        for layer in EARLY_LAYERS
    }
    synthetic_calibration[str(EARLY_LAYERS[2])] = synthetic_cells(0.4)
    _rows, selected = _select_layer(synthetic_calibration)
    freeze_guard = _PredictionBoundary()
    unfrozen_rejected = False
    try:
        freeze_guard.require_evaluation()
    except RuntimeError:
        unfrozen_rejected = True
    freeze_guard.freeze("JSON", "NPZ")
    freeze_guard.require_evaluation()
    verdicts = {
        _verdict(3, 4, 4, 4, 4, 4, True),
        _verdict(4, 3, 2, 2, 2, 2, False),
        _verdict(4, 3, 3, 3, 2, 2, False),
        _verdict(4, 4, 2, 2, 2, 2, False),
        _verdict(4, 4, 4, 4, 4, 4, True),
    }
    expected = set(PROTOCOL["verdicts"]) - {"CALIBRATION_CODE_NULL"}
    return {
        "context_match_pass": bool(context["pass"]),
        "random_count": len(randoms),
        "random_norm_error": float(max(
            abs(value - float(code.norm()))
            for value in random_norms
        )),
        "verdicts_reachable": verdicts == expected,
        "selected_layer": int(selected["layer"]),
        "unfrozen_evaluation_rejected": unfrozen_rejected,
        "pass": bool(
            context["pass"]
            and len(randoms) == N_RANDOM
            and max(
                abs(value - float(code.norm()))
                for value in random_norms
            ) < 1e-5
            and int(selected["layer"]) == EARLY_LAYERS[2]
            and unfrozen_rejected
            and verdicts == expected
        ),
    }


def _public_cache(cache):
    return {
        "minimum": float(min(
            cache["baseline_accuracies"].values())),
        "by_cell": {
            key: float(value)
            for key, value in cache["baseline_accuracies"].items()
        },
    }


@torch.no_grad()
def run_delta_latent_instruction_compiler(
        model_path, out_dir,
        model_key="qwen7b_latent_instruction_compiler",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=12, self_test_only=False):
    os.makedirs(out_dir, exist_ok=True)
    check = _self_check()
    if not check["pass"]:
        raise AssertionError(f"compiler mathematical guard failed: {check}")
    if self_test_only:
        result = {
            "stage": "delta_latent_instruction_compiler",
            "protocol": PROTOCOL,
            "protocol_sha256": PROTOCOL_SHA256,
            "self_check": check,
            "verdict": "SELF_TEST_ONLY",
        }
        path = os.path.join(
            out_dir,
            "results_delta_latent_instruction_compiler_"
            f"{model_key}.json")
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
        raise ValueError("frozen compiler layers are absent")
    commands, padding_plan, tokenization_tables = _commands(tok)
    splits = _row_splits()
    if len(splits["test"]) != int(n_world):
        raise AssertionError("held-out row count changed")

    training_total = (
        len(TRAIN_FAMILIES) * len(TRAIN_PANELS) * 2 * len(FRAMES)
    )
    heartbeat = Heartbeat(
        training_total,
        "latent_compiler_training_capture",
        every_sec=30, out_dir=out_dir)
    training = {}
    alignment_metadata = {}
    for family in TRAIN_FAMILIES:
        training[family] = {}
        for panel in TRAIN_PANELS:
            alignment = _frame_alignment(
                tok, dev, splits["train"], FAMILY_SPECS[family],
                commands[panel])
            program_positions = _all_positions(alignment)
            if len(program_positions) != PROGRAM_WIDTH:
                raise ValueError("training program width changed")
            training[family][panel] = []
            for history in range(2):
                row = {}
                for frame in FRAMES:
                    row[frame] = _capture(
                        model, alignment["batches"][history][frame],
                        EARLY_LAYERS, program_positions,
                        capture_route=False)
                    heartbeat.step(
                        extra=f"{family}/{panel}/h{history}/{frame}")
                training[family][panel].append(row)
            alignment_metadata[f"{family}/{panel}"] = {
                "instruction_positions":
                    list(alignment["instruction_positions"]),
                "answer_positions": list(alignment["answer_positions"]),
            }
    programs_by_layer = _fit_programs(training)

    calibration_total = (
        len(CALIBRATION_FAMILIES) * len(EARLY_LAYERS)
        * len(_directions()) * 2
        + len(CALIBRATION_FAMILIES) * len(FRAMES) * 2
    )
    heartbeat = Heartbeat(
        calibration_total,
        "latent_compiler_calibration",
        every_sec=30, out_dir=out_dir)
    calibration = {str(layer): [] for layer in EARLY_LAYERS}
    calibration_public = {}
    for family in CALIBRATION_FAMILIES:
        alignment = _frame_alignment(
            tok, dev, splits["validation"], FAMILY_SPECS[family],
            commands[CALIBRATION_PANEL])
        program_positions = _all_positions(alignment)
        baselines = []
        calibration_baseline_accuracies = []
        for history in range(2):
            row = {}
            for frame in FRAMES:
                batch = alignment["batches"][history][frame]
                row[frame] = _capture(
                    model, batch,
                    EARLY_LAYERS, program_positions)
                calibration_baseline_accuracies.append(
                    _baseline_accuracy(
                        row[frame], batch,
                        _expected(splits["validation"], history)))
                heartbeat.step(
                    extra=f"{family}/baseline/h{history}/{frame}")
            baselines.append(row)
        calibration_public[family] = {}
        for layer in EARLY_LAYERS:
            calibration_public[family][str(layer)] = {}
            for source, target in _directions():
                key = _transition_key(source, target)
                accumulator = _empty_accumulator()
                code = programs_by_layer[layer][key]
                for history in range(2):
                    batch = alignment["batches"][history][source]
                    patched = _run_early(
                        model, batch, layer, program_positions, code,
                        program_positions)
                    heartbeat.step(
                        extra=f"{family}/L{layer}/{key}/h{history}")
                    accuracy = _baseline_accuracy(
                        patched, batch,
                        _expected(splits["validation"], history))
                    _append(
                        accumulator, baselines[history][source],
                        baselines[history][target], patched, accuracy)
                cell = _summary(accumulator)
                cell["baseline_minimum_value_accuracy"] = float(
                    min(calibration_baseline_accuracies))
                calibration[str(layer)].append(cell)
                calibration_public[family][str(layer)][key] = cell
    layer_rows, selected = _select_layer(calibration)
    selected_layer = int(selected["layer"])
    calibration_pass = bool(
        selected["score"] >= MINIMUM_CALIBRATION_SCORE - 1e-9
        and selected["minimum_value_accuracy"]
        >= MINIMUM_VALUE_ACCURACY - 1e-9
    )
    log(
        "LATENT-COMPILER calibration "
        f"selected=L{selected_layer} score={selected['score']:+.4f} "
        f"accuracy={selected['minimum_value_accuracy']:.0%} "
        f"pass={calibration_pass}")

    selected_programs = {
        key: value.detach().float().cpu()
        for key, value in programs_by_layer[selected_layer].items()
    }
    random_programs = _random_programs(selected_programs)
    permutations = _derangements(len(splits["test"]) * 2)
    # Token layouts are input metadata, not target activations. Resolve and
    # freeze them now so no test position can be selected after the boundary.
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
            "identical_positions": _six_identical_positions(alignment),
        }
    freeze_json_path = os.path.join(
        out_dir, "latent_instruction_compiler_freeze.json")
    freeze_npz_path = os.path.join(
        out_dir, "latent_instruction_compiler_programs.npz")
    freeze = {
        "protocol_sha256": PROTOCOL_SHA256,
        "selected_layer": selected_layer,
        "selection_rows": layer_rows,
        "calibration_pass": calibration_pass,
        "program_shapes": {
            key: list(value.shape)
            for key, value in selected_programs.items()
        },
        "program_sha256": {
            key: hashlib.sha256(
                value.contiguous().numpy().tobytes()
            ).hexdigest().upper()
            for key, value in selected_programs.items()
        },
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
        f"program_{key}": value.numpy()
        for key, value in selected_programs.items()
    }
    for key, values in random_programs.items():
        for index, value in enumerate(values):
            arrays[f"random_{key}_{index:02d}"] = value.numpy()
    np.savez_compressed(freeze_npz_path, **arrays)
    boundary = _PredictionBoundary()
    boundary.freeze(
        _artifact_sha256(freeze_json_path),
        _artifact_sha256(freeze_npz_path))
    log(
        "FROZEN latent compiler predictions "
        f"json={boundary.json_sha256} npz={boundary.npz_sha256}")

    if not calibration_pass:
        result = {
            "stage": "delta_latent_instruction_compiler",
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
            "verdict": "CALIBRATION_CODE_NULL",
        }
        path = os.path.join(
            out_dir,
            "results_delta_latent_instruction_compiler_"
            f"{model_key}.json")
        with open(path, "w") as handle:
            json.dump(result, handle, indent=2)
        log(
            "LATENT-INSTRUCTION-COMPILER "
            "verdict=CALIBRATION_CODE_NULL")
        return result

    boundary.require_evaluation()
    test_total = (
        len(TEST_FAMILIES) * len(FRAMES) * 2
        + len(TEST_FAMILIES) * len(_directions()) * 2
        * (9 + N_RANDOM)
    )
    heartbeat = Heartbeat(
        test_total,
        "latent_compiler_heldout_test",
        every_sec=30, out_dir=out_dir)
    families = {}
    all_primary_scores = []
    all_answer_scores = []
    all_instruction_scores = []
    all_identical_scores = []
    all_reverse_scores = []
    random_scores_by_index = [[] for _ in range(N_RANDOM)]
    late_count = exact_count = portable_count = route_count = 0
    mediation_count = context_count = 0

    for family in TEST_FAMILIES:
        alignment = test_alignments[family]
        layout = test_layouts[family]
        program_positions = list(layout["program_positions"])
        instruction_positions = list(layout["instruction_positions"])
        answer_positions = list(layout["answer_positions"])
        identical_positions = list(layout["identical_positions"])
        baselines = []
        baseline_accuracies = {}
        for history in range(2):
            row = {}
            for frame in FRAMES:
                batch = alignment["batches"][history][frame]
                row[frame] = _capture(
                    model, batch, (selected_layer,), program_positions)
                baseline_accuracies[
                    f"h{history}/{frame}"
                ] = _baseline_accuracy(
                    row[frame], batch,
                    _expected(splits["test"], history))
                heartbeat.step(
                    extra=f"{family}/baseline/h{history}/{frame}")
            baselines.append(row)
        family_result = {
            "positions": {
                "instruction": instruction_positions,
                "answer": answer_positions,
                "program": program_positions,
                "identical": identical_positions,
            },
            "baseline_accuracies": baseline_accuracies,
            "directions": {},
        }
        family_baseline_pass = bool(
            min(baseline_accuracies.values())
            >= MINIMUM_VALUE_ACCURACY - 1e-9
        )
        for source, target in _directions():
            key = _transition_key(source, target)
            code = selected_programs[key]
            arm_accumulators = {
                name: _empty_accumulator()
                for name in (
                    "late_oracle",
                    "exact_early",
                    "portable",
                    "answer_only",
                    "instruction_only",
                    "identical",
                    "reverse",
                    "block",
                    "rescue",
                )
            }
            random_accumulators = [
                _empty_accumulator() for _ in range(N_RANDOM)
            ]
            generated_states = []
            target_states = []
            source_states = []
            for history in range(2):
                batch = alignment["batches"][history][source]
                origin = baselines[history][source]
                target_cache = baselines[history][target]
                expected = _expected(splits["test"], history)

                late = _late_patch_cache(
                    model, batch, answer_positions,
                    _answer_state(target_cache))
                heartbeat.step(extra=f"{family}/{key}/h{history}/late")
                _append(
                    arm_accumulators["late_oracle"],
                    origin, target_cache, late,
                    _baseline_accuracy(late, batch, expected))

                exact_delta = (
                    target_cache[f"early_{selected_layer}"]
                    - origin[f"early_{selected_layer}"])
                exact = _run_early(
                    model, batch, selected_layer,
                    program_positions, exact_delta, program_positions)
                heartbeat.step(extra=f"{family}/{key}/h{history}/exact")
                _append(
                    arm_accumulators["exact_early"],
                    origin, target_cache, exact,
                    _baseline_accuracy(exact, batch, expected))

                portable = _run_early(
                    model, batch, selected_layer,
                    program_positions, code, program_positions)
                heartbeat.step(
                    extra=f"{family}/{key}/h{history}/portable")
                _append(
                    arm_accumulators["portable"],
                    origin, target_cache, portable,
                    _baseline_accuracy(portable, batch, expected))
                generated_states.append(_answer_state(portable))
                target_states.append(_answer_state(target_cache))
                source_states.append(_answer_state(origin))

                answer_only = _run_early(
                    model, batch, selected_layer, answer_positions,
                    code[-PATCH_WIDTH:], program_positions)
                heartbeat.step(
                    extra=f"{family}/{key}/h{history}/answer")
                _append(
                    arm_accumulators["answer_only"],
                    origin, target_cache, answer_only,
                    _baseline_accuracy(answer_only, batch, expected))

                instruction_only = _run_early(
                    model, batch, selected_layer, instruction_positions,
                    code[:PATCH_WIDTH], program_positions)
                heartbeat.step(
                    extra=f"{family}/{key}/h{history}/instruction")
                _append(
                    arm_accumulators["instruction_only"],
                    origin, target_cache, instruction_only,
                    _baseline_accuracy(instruction_only, batch, expected))

                identical = _run_early(
                    model, batch, selected_layer, identical_positions,
                    code, program_positions)
                heartbeat.step(
                    extra=f"{family}/{key}/h{history}/identical")
                _append(
                    arm_accumulators["identical"],
                    origin, target_cache, identical,
                    _baseline_accuracy(identical, batch, expected))

                reverse = _run_early(
                    model, batch, selected_layer, program_positions,
                    -code, program_positions)
                heartbeat.step(
                    extra=f"{family}/{key}/h{history}/reverse")
                _append(
                    arm_accumulators["reverse"],
                    origin, target_cache, reverse,
                    _baseline_accuracy(reverse, batch, expected))

                block = _run_early(
                    model, batch, selected_layer, program_positions,
                    code, program_positions,
                    block_answer=_answer_state(origin))
                heartbeat.step(
                    extra=f"{family}/{key}/h{history}/block")
                _append(
                    arm_accumulators["block"],
                    origin, target_cache, block,
                    _baseline_accuracy(block, batch, expected))

                rescue = _late_patch_cache(
                    model, batch, answer_positions,
                    _answer_state(portable))
                heartbeat.step(
                    extra=f"{family}/{key}/h{history}/rescue")
                _append(
                    arm_accumulators["rescue"],
                    origin, target_cache, rescue,
                    _baseline_accuracy(rescue, batch, expected))

                for index, random_code in enumerate(
                        random_programs[key]):
                    random_cache = _run_early(
                        model, batch, selected_layer, program_positions,
                        random_code, program_positions)
                    heartbeat.step(
                        extra=(
                            f"{family}/{key}/h{history}/"
                            f"random{index:02d}"))
                    _append(
                        random_accumulators[index],
                        origin, target_cache, random_cache,
                        _baseline_accuracy(
                            random_cache, batch, expected))

            summaries = {
                name: _summary(accumulator)
                for name, accumulator in arm_accumulators.items()
            }
            random_summaries = [
                _summary(accumulator)
                for accumulator in random_accumulators
            ]
            late_pass = bool(
                family_baseline_pass
                and _late_oracle_pass(summaries["late_oracle"])
            )
            exact_pass = _exact_early_pass(summaries["exact_early"])
            portable_pass, recovery = _portable_pass(
                summaries["portable"], summaries["exact_early"])
            route_pass = bool(
                summaries["portable"]["l27"]["mean_progress"]
                >= MINIMUM_PORTABLE_L27_PROGRESS - 1e-9
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
            portable_count += int(portable_pass)
            route_count += int(route_pass)
            mediation_count += int(mediation_pass)
            context_count += int(context["pass"])
            all_primary_scores.append(primary_effect)
            all_answer_scores.append(
                summaries["answer_only"]["l27"]["mean_progress"])
            all_instruction_scores.append(
                summaries["instruction_only"]["l27"]["mean_progress"])
            all_identical_scores.append(
                summaries["identical"]["l27"]["mean_progress"])
            all_reverse_scores.append(
                summaries["reverse"]["l27"]["mean_progress"])
            for index, random_summary in enumerate(random_summaries):
                random_scores_by_index[index].append(
                    random_summary["l27"]["mean_progress"])
            family_result["directions"][key] = {
                "arms": summaries,
                "random_arms": random_summaries,
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
                    "portable_pass": portable_pass,
                    "portable_recovery_of_exact": recovery,
                    "route_pass": route_pass,
                },
            }
        families[family] = family_result

    primary_breadth = float(min(all_primary_scores))
    answer_breadth = float(min(all_answer_scores))
    instruction_breadth = float(min(all_instruction_scores))
    identical_breadth = float(min(all_identical_scores))
    reverse_breadth = float(min(all_reverse_scores))
    random_breadth = [
        float(min(values)) for values in random_scores_by_index
    ]
    random_exceed = sum(
        value >= primary_breadth for value in random_breadth)
    random_p = (1.0 + random_exceed) / (1.0 + N_RANDOM)
    occurrence_margin = float(
        primary_breadth - max(answer_breadth, instruction_breadth))
    null_limit = MAXIMUM_NULL_FRACTION * max(primary_breadth, 0.0)
    specificity = bool(
        identical_breadth <= null_limit + 1e-9
        and reverse_breadth <= null_limit + 1e-9
        and random_p <= MAXIMUM_RANDOM_P + 1e-12
    )
    verdict = _verdict(
        late_count, exact_count, portable_count, route_count,
        mediation_count, context_count, specificity)
    overall = {
        "cell_count": 4,
        "late_oracle_count": late_count,
        "exact_early_count": exact_count,
        "portable_count": portable_count,
        "portable_route_count": route_count,
        "mediation_count": mediation_count,
        "context_match_count": context_count,
        "primary_breadth_score": primary_breadth,
        "answer_only_breadth_score": answer_breadth,
        "instruction_only_breadth_score": instruction_breadth,
        "occurrence_ablation_margin": occurrence_margin,
        "identical_breadth_score": identical_breadth,
        "reverse_breadth_score": reverse_breadth,
        "random_breadth_scores": random_breadth,
        "random_exceed_count": int(random_exceed),
        "random_add_one_p": float(random_p),
        "specificity_pass": specificity,
        "verdict": verdict,
    }
    result = {
        "stage": "delta_latent_instruction_compiler",
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
        "families": families,
        "overall": overall,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        "results_delta_latent_instruction_compiler_"
        f"{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2)
    log(
        "LATENT-INSTRUCTION-COMPILER "
        f"verdict={verdict} exact={exact_count}/4 "
        f"portable={portable_count}/4 mediation={mediation_count}/4 "
        f"context={context_count}/4 random_p={random_p:.3f} "
        f"artifact={path}")
    return result
