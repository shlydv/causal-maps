"""Held-out test of a donor-averaged answer-prefix route displacement."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from .delta_anchor_write import _resolve
from .delta_depth_replication import _first_passing
from .delta_distributed_label_transplant import (
    _aligned_batches,
    _capture_full_l21,
    _tail_probability,
)
from .delta_operation_handoff_depth import (
    CAPTURE_LAYERS,
    LAYERS,
    _evaluate_sites,
    _full_sites,
)
from .delta_preprint_battery import _compatible_world_rows
from .delta_source_head_mediation import _public_task, _task_context
from .delta_sparse_transport import _attention_geometry
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

PROTOCOL_VERSION = "2026-07-25-p2-content-cancelled-controller-v1"
PROTOCOL_SHA256 = (
    "2CA54C50FBAAF94B2A4F63AB81CD81F28637E6347896F0726BEF9F7FC3CCDDFB")
DONOR_N = 15
TEST_N = 15
PATCH_WIDTH = 3
N_RANDOM = 19
DIRECTION_SEED = 48131
POSITION_SEED = 48133
EPS = 1e-8


def _curve(model, context, n_heads, head_dim):
    return {
        str(layer): _evaluate_sites(
            model, context,
            _full_sites(tuple(x for x in LAYERS if x <= layer), n_heads),
            head_dim)
        for layer in LAYERS
    }


def _summary(curve):
    return {
        "l24_minimum_mediation": float(
            curve["24"]["mediation"]["minimum_fraction"]),
        "first_passing_prefix": _first_passing(curve),
    }


def _movement(original, belief_to_search, search_to_belief):
    belief = float(original["belief"]["l24_minimum_mediation"])
    search = float(original["search"]["l24_minimum_mediation"])
    gap = belief - search
    down = belief - float(belief_to_search)
    up = float(search_to_belief) - search
    return {
        "original_gap": gap,
        "belief_to_search_movement": down,
        "search_to_belief_movement": up,
        "bidirectional_score": min(down, up),
        "belief_to_search_pass": bool(
            down >= 0.05 - 1e-9 and down + 1e-9 >= 0.5 * gap),
        "search_to_belief_pass": bool(
            up >= 0.05 - 1e-9 and up + 1e-9 >= 0.5 * gap),
    }


def _world_mediation(task, cell):
    """Per-row analogue of the frozen aggregate mediation fraction."""
    source = task["source_intervention"]
    blocked = cell["blocked_intervention"]
    result = []
    for sf, sr, bf, br in zip(
            source["forward_effect_rows"],
            source["reverse_effect_rows"],
            blocked["forward_effect_rows"],
            blocked["reverse_effect_rows"]):
        if abs(float(sf)) <= EPS or abs(float(sr)) <= EPS:
            result.append(None)
            continue
        forward = 1.0 - abs(float(bf)) / abs(float(sf))
        reverse = 1.0 - abs(float(br)) / abs(float(sr))
        result.append(min(forward, reverse))
    return result


def _world_movements(original_belief, original_search,
                     patched_belief, patched_search):
    rows = []
    for index, (ob, os_, pb, ps) in enumerate(zip(
            original_belief, original_search,
            patched_belief, patched_search)):
        if any(value is None for value in (ob, os_, pb, ps)):
            rows.append({
                "world_offset": index,
                "belief_to_search_movement": None,
                "search_to_belief_movement": None,
                "predicted_sign": False,
            })
            continue
        down = float(ob) - float(pb)
        up = float(ps) - float(os_)
        rows.append({
            "world_offset": index,
            "belief_to_search_movement": down,
            "search_to_belief_movement": up,
            "predicted_sign": bool(down > 0.0 and up > 0.0),
        })
    valid_down = [
        row["belief_to_search_movement"] for row in rows
        if row["belief_to_search_movement"] is not None
    ]
    valid_up = [
        row["search_to_belief_movement"] for row in rows
        if row["search_to_belief_movement"] is not None
    ]
    return {
        "rows": rows,
        "all_predicted_sign": bool(
            len(rows) == TEST_N
            and all(row["predicted_sign"] for row in rows)),
        "belief_to_search_range": (
            [min(valid_down), max(valid_down)] if valid_down else None),
        "search_to_belief_range": (
            [min(valid_up), max(valid_up)] if valid_up else None),
    }


def _fixed_patch(states, positions, displacement, sign):
    patches = []
    for arm_states in states:
        values = (
            arm_states[:, positions, :]
            + float(sign) * displacement.unsqueeze(0))
        patches.append((positions, values))
    return tuple(patches)


def _norm_matched_directions(displacement, n_random=N_RANDOM,
                             seed=DIRECTION_SEED):
    rng = np.random.default_rng(int(seed))
    target_norms = displacement.norm(dim=-1)
    if bool((target_norms <= EPS).any()):
        raise ValueError("content-cancelled displacement has a zero-norm row")
    controls = []
    shape = tuple(int(value) for value in displacement.shape)
    for _index in range(int(n_random)):
        value = torch.from_numpy(
            rng.standard_normal(shape).astype(np.float32))
        value = value / value.norm(dim=-1, keepdim=True).clamp_min(EPS)
        value = value * target_norms.unsqueeze(-1)
        controls.append(value)
    return controls


def _random_position_sets(candidates, n_random=N_RANDOM,
                          seed=POSITION_SEED):
    if len(candidates) < PATCH_WIDTH:
        raise ValueError("too few identical-token random-position candidates")
    rng = np.random.default_rng(int(seed))
    controls = []
    seen = set()
    while len(controls) < int(n_random):
        choice = tuple(sorted(
            int(value) for value in rng.choice(
                candidates, size=PATCH_WIDTH, replace=False)))
        if choice not in seen:
            seen.add(choice)
            controls.append(list(choice))
    return controls


def _functional_score(movement, tasks):
    functional = all(
        task["eligible"] and task["source_intervention"]["sufficient"]
        for task in tasks)
    return {
        **movement,
        "functional": bool(functional),
        "functional_bidirectional_score": (
            float(movement["bidirectional_score"])
            if functional else -1e9),
    }


def _donor_alignment(samples, displacement):
    center = displacement.flatten()
    center_norm = float(center.norm())
    cosines = []
    for sample in samples:
        flat = sample.flatten()
        cosine = float(
            torch.dot(flat, center)
            / max(float(flat.norm()) * center_norm, EPS))
        cosines.append(cosine)
    return {
        "n_samples": len(cosines),
        "cosines": cosines,
        "mean_cosine": float(np.mean(cosines)),
        "minimum_cosine": float(np.min(cosines)),
        "positive_fraction": float(np.mean(
            [value > 0.0 for value in cosines])),
    }


def _verdict(tasks, summaries, movement, world_movement,
             instruction_control, direction_control, position_control):
    selected_names = (
        "belief_original", "search_original",
        "belief_to_search", "search_to_belief",
    )
    if not all(tasks[name]["eligible"] for name in selected_names):
        return "BEHAVIORALLY_INELIGIBLE"
    if not all(tasks[name]["source_intervention"]["sufficient"]
               for name in selected_names):
        return "SOURCE_SITE_INELIGIBLE"
    if any(summaries[name]["first_passing_prefix"] is None
           for name in selected_names):
        return "DEPTH_UNRESOLVED"
    if movement["original_gap"] < 0.05 - 1e-9:
        return "ORIGINAL_GAP_ABSENT"

    down = movement["belief_to_search_pass"]
    up = movement["search_to_belief_pass"]
    if down != up:
        return "ASYMMETRIC_CONTENT_CANCELLED_EFFECT"
    if not (down and up):
        return "NO_CONTENT_CANCELLED_CONTROLLER"
    if not world_movement["all_predicted_sign"]:
        return "NONUNIFORM_CONTENT_CANCELLED_EFFECT"

    if direction_control["empirical_p"] > 0.05 + 1e-12:
        return "NONSPECIFIC_RANDOM_DIRECTION"
    if position_control["empirical_p"] > 0.05 + 1e-12:
        return "NONSPECIFIC_POSITION_EFFECT"
    if (instruction_control["functional_bidirectional_score"]
            >= 0.5 * movement["bidirectional_score"] - 1e-12):
        return "INSTRUCTION_LOCUS_EFFECT"

    categorical = bool(
        summaries["belief_to_search"]["first_passing_prefix"]
        > summaries["belief_original"]["first_passing_prefix"]
        or summaries["search_to_belief"]["first_passing_prefix"]
        < summaries["search_original"]["first_passing_prefix"])
    return (
        "CONTENT_CANCELLED_PREFIX_CONTROLLER"
        if categorical
        else "CONTINUOUS_CONTENT_CANCELLED_PREFIX_CONTROLLER")


@torch.no_grad()
def run_delta_content_cancelled_controller(
        model_path, out_dir,
        model_key="qwen7b_content_cancelled_controller",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=30):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_world) != DONOR_N + TEST_N:
        raise ValueError("v1 is frozen to exactly 30 worlds")
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    rows, indices = _compatible_world_rows(
        tok, torch.device("cpu"), int(n_world))
    if len(rows) != DONOR_N + TEST_N:
        raise ValueError("v1 requires all 30 compatible worlds")
    if max(LAYERS) >= model_num_hidden_layers(model):
        raise ValueError("content-cancelled controller layer is absent")
    n_heads, head_dim = _attention_geometry(model)
    donor_rows = rows[:DONOR_N]
    test_rows = rows[DONOR_N:]

    try:
        (donor_pairs, donor_differing, donor_groups, _donor_candidates,
         donor_marker, donor_readout) = _aligned_batches(
             tok, dev, donor_rows)
        (test_pairs, test_differing, test_groups, candidates,
         marker, readout) = _aligned_batches(tok, dev, test_rows)
        if donor_differing != test_differing:
            raise ValueError("donor/evaluation differing positions diverge")
        if donor_groups != test_groups:
            raise ValueError("donor/evaluation label groups diverge")
        if (donor_marker, donor_readout) != (marker, readout):
            raise ValueError("donor/evaluation marker or readout diverges")
        answer_positions = list(test_groups[1])
        instruction_positions = list(test_groups[0])
        if len(answer_positions) != PATCH_WIDTH:
            raise ValueError("answer-prefix width is not three")
    except ValueError as exc:
        result = {
            "stage": "delta_content_cancelled_controller",
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "model_key": model_key,
            "alignment_error": str(exc),
            "verdict": "TOKEN_ALIGNMENT_INVALID",
        }
        path = os.path.join(
            out_dir,
            f"results_delta_content_cancelled_controller_{model_key}.json")
        with open(path, "w") as handle:
            json.dump(result, handle, indent=2)
        log(f"CONTENT-CANCELLED CONTROLLER "
            f"verdict=TOKEN_ALIGNMENT_INVALID artifact={path}")
        return result

    donor_samples = []
    for belief_batch, search_batch in donor_pairs:
        belief = _capture_full_l21(model, belief_batch)
        search = _capture_full_l21(model, search_batch)
        donor_samples.extend(
            belief[:, answer_positions, :]
            - search[:, answer_positions, :])
    donor_sample_tensor = torch.stack(donor_samples)
    displacement = donor_sample_tensor.mean(dim=0)

    evaluation_states = {"belief": [], "search": []}
    for belief_batch, search_batch in test_pairs:
        evaluation_states["belief"].append(
            _capture_full_l21(model, belief_batch))
        evaluation_states["search"].append(
            _capture_full_l21(model, search_batch))

    random_directions = _norm_matched_directions(displacement)
    random_positions = _random_position_sets(candidates)
    total_steps = (
        4 * (3 + len(LAYERS))
        + 2 * 4
        + N_RANDOM * 2 * 4
        + N_RANDOM * 2 * 4)
    hb = Heartbeat(
        total_steps, "content_cancelled_controller",
        every_sec=30, out_dir=out_dir)

    selected_specs = {
        "belief_original": ("distributed_belief", None),
        "search_original": ("distributed_search", None),
        "belief_to_search": (
            "distributed_belief",
            _fixed_patch(
                evaluation_states["belief"], answer_positions,
                displacement, -1.0)),
        "search_to_belief": (
            "distributed_search",
            _fixed_patch(
                evaluation_states["search"], answer_positions,
                displacement, +1.0)),
    }
    contexts = {}
    tasks = {}
    curves = {}
    summaries = {}
    for name, (query, patch) in selected_specs.items():
        context = _task_context(
            model, tok, dev, test_rows, query,
            CAPTURE_LAYERS, head_dim, sequence_patch=patch)
        contexts[name] = context
        tasks[name] = _public_task(context)
        hb.step(extra=f"{name}/baseline")
        hb.step(extra=f"{name}/source")
        hb.step(extra=f"{name}/base_path")
        curves[name] = {}
        for layer, cell in _curve(
                model, context, n_heads, head_dim).items():
            curves[name][layer] = cell
            hb.step(extra=f"{name}/prefixL{layer}")
        summaries[name] = _summary(curves[name])

    original = {
        "belief": summaries["belief_original"],
        "search": summaries["search_original"],
    }
    movement = _movement(
        original,
        summaries["belief_to_search"]["l24_minimum_mediation"],
        summaries["search_to_belief"]["l24_minimum_mediation"])
    original_belief_rows = _world_mediation(
        tasks["belief_original"], curves["belief_original"]["24"])
    original_search_rows = _world_mediation(
        tasks["search_original"], curves["search_original"]["24"])
    patched_belief_rows = _world_mediation(
        tasks["belief_to_search"], curves["belief_to_search"]["24"])
    patched_search_rows = _world_mediation(
        tasks["search_to_belief"], curves["search_to_belief"]["24"])
    world_movement = _world_movements(
        original_belief_rows, original_search_rows,
        patched_belief_rows, patched_search_rows)

    control_layers = (21, 22, 23, 24)
    l24_sites = _full_sites((22, 23, 24), n_heads)

    def evaluate_control(name, belief_patch, search_patch):
        values = {}
        public = []
        for direction, query, patch in (
                ("belief_to_search", "distributed_belief", belief_patch),
                ("search_to_belief", "distributed_search", search_patch)):
            context = _task_context(
                model, tok, dev, test_rows, query,
                control_layers, head_dim, sequence_patch=patch)
            task = _public_task(context)
            public.append(task)
            hb.step(extra=f"{name}/{direction}/baseline")
            hb.step(extra=f"{name}/{direction}/source")
            hb.step(extra=f"{name}/{direction}/base_path")
            cell = _evaluate_sites(
                model, context, l24_sites, head_dim)
            values[direction] = float(
                cell["mediation"]["minimum_fraction"])
            hb.step(extra=f"{name}/{direction}/L24")
        return _functional_score(
            _movement(
                original, values["belief_to_search"],
                values["search_to_belief"]),
            public)

    instruction_control = evaluate_control(
        "instruction",
        _fixed_patch(
            evaluation_states["belief"], instruction_positions,
            displacement, -1.0),
        _fixed_patch(
            evaluation_states["search"], instruction_positions,
            displacement, +1.0))

    direction_cells = []
    for random_index, random_direction in enumerate(random_directions):
        cell = evaluate_control(
            f"random_direction{random_index}",
            _fixed_patch(
                evaluation_states["belief"], answer_positions,
                random_direction, -1.0),
            _fixed_patch(
                evaluation_states["search"], answer_positions,
                random_direction, +1.0))
        direction_cells.append({
            "random_index": random_index,
            "per_position_norms": random_direction.norm(
                dim=-1).tolist(),
            **cell,
        })

    position_cells = []
    for random_index, positions in enumerate(random_positions):
        cell = evaluate_control(
            f"random_position{random_index}",
            _fixed_patch(
                evaluation_states["belief"], positions,
                displacement, -1.0),
            _fixed_patch(
                evaluation_states["search"], positions,
                displacement, +1.0))
        position_cells.append({
            "random_index": random_index,
            "positions": positions,
            **cell,
        })
    hb.done()

    selected_score = float(movement["bidirectional_score"])
    direction_scores = [
        cell["functional_bidirectional_score"]
        for cell in direction_cells
    ]
    direction_p, direction_exceed = _tail_probability(
        selected_score, direction_scores)
    position_scores = [
        cell["functional_bidirectional_score"]
        for cell in position_cells
    ]
    position_p, position_exceed = _tail_probability(
        selected_score, position_scores)
    direction_control = {
        "n_random": N_RANDOM,
        "seed": DIRECTION_SEED,
        "empirical_p": direction_p,
        "exceed_count": direction_exceed,
        "cells": direction_cells,
    }
    position_control = {
        "n_random": N_RANDOM,
        "seed": POSITION_SEED,
        "empirical_p": position_p,
        "exceed_count": position_exceed,
        "cells": position_cells,
    }
    verdict = _verdict(
        tasks, summaries, movement, world_movement,
        instruction_control, direction_control, position_control)

    controller_path = os.path.join(
        out_dir, f"content_cancelled_controller_{model_key}.npy")
    np.save(controller_path, displacement.numpy())
    with open(controller_path, "rb") as handle:
        controller_sha256 = hashlib.sha256(handle.read()).hexdigest().upper()

    result = {
        "stage": "delta_content_cancelled_controller",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "claim": (
            "A donor-averaged, content-cancelled answer-prefix "
            "displacement reconfigures an unchanged state's causal route "
            "on held-out worlds."),
        "world_selection": {
            "requested": int(n_world),
            "selected": len(rows),
            "indices_from_30": indices,
            "donor_indices": indices[:DONOR_N],
            "evaluation_indices": indices[DONOR_N:],
        },
        "alignment": {
            "marker_position": marker,
            "readout_position": readout,
            "differing_positions": test_differing,
            "instruction_positions": instruction_positions,
            "answer_prefix_positions": answer_positions,
            "identical_post_marker_candidates": candidates,
        },
        "controller": {
            "construction": (
                "mean(BELIEF-SEARCH) over donor worlds 0-14 and "
                "clean/natural arms"),
            "shape": list(displacement.shape),
            "frobenius_norm": float(displacement.norm()),
            "per_position_norms": displacement.norm(dim=-1).tolist(),
            "artifact": os.path.basename(controller_path),
            "artifact_sha256": controller_sha256,
            "donor_alignment": _donor_alignment(
                donor_sample_tensor, displacement),
        },
        "layers": list(LAYERS),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "tasks": tasks,
        "cumulative_prefix": curves,
        "selected_summaries": summaries,
        "primary_movement": movement,
        "per_world_movement": world_movement,
        "instruction_control": instruction_control,
        "random_direction_control": direction_control,
        "random_position_control": position_control,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_content_cancelled_controller_{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"CONTENT-CANCELLED CONTROLLER verdict={verdict} "
        f"movement={movement} worlds="
        f"{world_movement['all_predicted_sign']} "
        f"p_direction={direction_p} p_position={position_p} "
        f"artifact={path}")
    return result
