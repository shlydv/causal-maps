"""Prospective leave-color-out test of a shared low-rank route controller."""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import os

import numpy as np
import torch

from .delta_anchor_write import _resolve
from .delta_content_cancelled_controller import (
    DIRECTION_SEED,
    N_RANDOM,
    PATCH_WIDTH,
    POSITION_SEED,
    _fixed_patch,
    _functional_score,
    _movement,
    _norm_matched_directions,
    _random_position_sets,
    _summary,
    _world_mediation,
)
from .delta_controller_matrix import (
    _controller_from_alignment,
    _fresh_domain_rows,
    _location_alignment,
)
from .delta_cross_domain_controller import (
    DOMAIN_SPECS,
    _domain_alignment,
    _domain_rows,
    _generic_curve,
    _generic_evaluate_sites,
    _generic_task_context,
    _public_generic_task,
)
from .delta_distributed_label_transplant import _capture_full_l21, _tail_probability
from .delta_operation_handoff_depth import CAPTURE_LAYERS, LAYERS, _full_sites
from .delta_preprint_battery import _compatible_world_rows
from .delta_sparse_transport import _attention_geometry
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

PROTOCOL_VERSION = "2026-07-25-p2-leave-color-out-shared-v1"
PROTOCOL_SHA256 = (
    "ACE78DBD4FBB8BFBF9C070571D40B977AF59B7139E80ECB54B5AA1B2D819AA37")
DONOR_N = 15
TEST_N = 30
SIGN_MINIMUM = 27
SIGN_P_MAX = 0.01
PASS_VERDICTS = ("PASS",)


def _fresh_color_rows():
    """Thirty exact histories absent from both earlier color experiments."""
    values = DOMAIN_SPECS["color_state"]["values"]
    old_rows = _domain_rows(values) + _fresh_domain_rows(values)
    used_prompt_histories = {
        (state, row["d1"], row["d2"])
        for row in old_rows
        for state in (row["source"], row["target"])
    }
    rows = []
    for index in range(TEST_N):
        source_index = index % len(values)
        shift = 1 + index // len(values)
        source = values[source_index]
        target = values[(source_index + shift) % len(values)]
        remaining = [
            values[(source_index + offset) % len(values)]
            for offset in range(1, len(values) + 1)
            if values[(source_index + offset) % len(values)]
            not in (source, target)
        ]
        for d1, d2 in itertools.permutations(reversed(remaining), 2):
            clean_signature = (source, d1, d2)
            natural_signature = (target, d1, d2)
            if (clean_signature not in used_prompt_histories
                    and natural_signature not in used_prompt_histories):
                used_prompt_histories.add(clean_signature)
                used_prompt_histories.add(natural_signature)
                break
        else:
            raise AssertionError(
                f"could not construct unseen color history {index}")
        rows.append({
            "row_index": index,
            "source": source,
            "target": target,
            "state": source,
            "d1": d1,
            "d2": d2,
        })
    signatures = {
        (row["source"], row["target"], row["state"], row["d1"], row["d2"])
        for row in rows
    }
    if len(signatures) != TEST_N:
        raise AssertionError("fresh color histories are not unique")
    old_signatures = {
        (row["source"], row["target"], row["state"], row["d1"], row["d2"])
        for row in old_rows
    }
    if signatures & old_signatures:
        raise AssertionError("fresh color histories overlap an earlier run")
    prompt_signatures = {
        (state, row["d1"], row["d2"])
        for row in rows
        for state in (row["source"], row["target"])
    }
    if len(prompt_signatures) != 2 * TEST_N:
        raise AssertionError("fresh rendered prompt histories are not unique")
    return rows


def _shared_controllers(location, ownership):
    flat = torch.stack([
        location.flatten().float(),
        ownership.flatten().float(),
    ])
    _u, singular, vh = torch.linalg.svd(flat, full_matrices=False)
    axis = vh[0]
    projections = flat @ axis
    if float(projections.mean()) < 0.0:
        axis = -axis
        projections = -projections
    pc1 = (axis * projections.mean()).reshape_as(location)
    mean = 0.5 * (location.float() + ownership.float())
    energy = singular.square()
    return pc1, mean, {
        "singular_values": singular.tolist(),
        "pc1_energy_fraction": float(energy[0] / energy.sum()),
        "oriented_projections": projections.tolist(),
        "pc1_mean_cosine": float(
            torch.dot(pc1.flatten(), mean.flatten())
            / (pc1.norm() * mean.norm()).clamp_min(1e-8)),
    }


def _cosine(left, right):
    a = left.flatten().float()
    b = right.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-8))


def _controller_metadata(controller):
    buffer = io.BytesIO()
    np.save(buffer, controller.numpy())
    return {
        "shape": list(controller.shape),
        "frobenius_norm": float(controller.norm()),
        "per_position_norms": controller.norm(dim=-1).tolist(),
        "npy_sha256": hashlib.sha256(
            buffer.getvalue()).hexdigest().upper(),
    }


def _sign_tail(successes, total):
    return float(sum(
        math.comb(total, value)
        for value in range(successes, total + 1)
    ) / (2 ** total))


def _statistical_world_movement(original_belief, original_search,
                                patched_belief, patched_search):
    rows = []
    down_values = []
    up_values = []
    for index, (ob, os_, pb, ps) in enumerate(zip(
            original_belief, original_search,
            patched_belief, patched_search)):
        if any(value is None for value in (ob, os_, pb, ps)):
            rows.append({
                "world_offset": index,
                "belief_to_search_movement": None,
                "search_to_belief_movement": None,
            })
            continue
        down = float(ob) - float(pb)
        up = float(ps) - float(os_)
        down_values.append(down)
        up_values.append(up)
        rows.append({
            "world_offset": index,
            "belief_to_search_movement": down,
            "search_to_belief_movement": up,
            "belief_predicted_sign": bool(down > 0.0),
            "search_predicted_sign": bool(up > 0.0),
        })
    down_success = sum(value > 0.0 for value in down_values)
    up_success = sum(value > 0.0 for value in up_values)
    valid = len(down_values) == TEST_N and len(up_values) == TEST_N
    return {
        "rows": rows,
        "valid_worlds": len(down_values),
        "belief_to_search": {
            "successes": down_success,
            "fraction": down_success / TEST_N if valid else None,
            "exact_one_sided_sign_p": (
                _sign_tail(down_success, TEST_N) if valid else None),
            "range": (
                [min(down_values), max(down_values)]
                if down_values else None),
            "mean": float(np.mean(down_values)) if down_values else None,
        },
        "search_to_belief": {
            "successes": up_success,
            "fraction": up_success / TEST_N if valid else None,
            "exact_one_sided_sign_p": (
                _sign_tail(up_success, TEST_N) if valid else None),
            "range": [min(up_values), max(up_values)] if up_values else None,
            "mean": float(np.mean(up_values)) if up_values else None,
        },
        "statistical_uniformity_pass": bool(
            valid
            and down_success >= SIGN_MINIMUM
            and up_success >= SIGN_MINIMUM
            and _sign_tail(down_success, TEST_N) <= SIGN_P_MAX
            and _sign_tail(up_success, TEST_N) <= SIGN_P_MAX),
    }


def _arm_verdict(tasks, summaries, movement, world):
    names = (
        "belief_original", "search_original",
        "belief_to_search", "search_to_belief",
    )
    if not all(tasks[name]["eligible"] for name in names):
        return "BEHAVIORALLY_INELIGIBLE"
    if not all(tasks[name]["source_intervention"]["sufficient"]
               for name in names):
        return "SOURCE_SITE_INELIGIBLE"
    if any(summaries[name]["first_passing_prefix"] is None
           for name in names):
        return "DEPTH_UNRESOLVED"
    if movement["original_gap"] < 0.03 - 1e-9:
        return "ORIGINAL_GAP_ABSENT"
    if not (movement["belief_to_search_pass"]
            and movement["search_to_belief_pass"]):
        return "AGGREGATE_MOVEMENT_FAILED"
    if not world["statistical_uniformity_pass"]:
        return "STATISTICAL_UNIFORMITY_FAILED"
    return "PASS"


def _original_gate(tasks, summaries, movement):
    names = ("belief_original", "search_original")
    if not all(tasks[name]["eligible"] for name in names):
        return "BEHAVIORALLY_INELIGIBLE"
    if not all(tasks[name]["source_intervention"]["sufficient"]
               for name in names):
        return "SOURCE_SITE_INELIGIBLE"
    if any(summaries[name]["first_passing_prefix"] is None
           for name in names):
        return "DEPTH_UNRESOLVED"
    if movement["original_gap"] < 0.03 - 1e-9:
        return "ORIGINAL_GAP_ABSENT"
    return "ELIGIBLE"


@torch.no_grad()
def run_delta_leave_color_out_shared(
        model_path, out_dir,
        model_key="qwen7b_leave_color_out_shared",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=30):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_world) != TEST_N:
        raise ValueError("v1 is frozen to exactly 30 evaluation histories")
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    if max(LAYERS) >= model_num_hidden_layers(model):
        raise ValueError("leave-color-out layers are absent")
    n_heads, head_dim = _attention_geometry(model)
    spec = DOMAIN_SPECS["color_state"]

    location_rows, location_indices = _compatible_world_rows(
        tok, torch.device("cpu"), 30)
    if len(location_rows) != 30:
        raise ValueError("all 30 compatible location worlds are required")
    ownership_rows = _fresh_domain_rows(
        DOMAIN_SPECS["ownership"]["values"])[:DONOR_N]
    oracle_color_rows = _fresh_domain_rows(spec["values"])[:DONOR_N]
    evaluation_rows = _fresh_color_rows()

    donor_alignments = {
        "location": _location_alignment(
            tok, dev, location_rows[:DONOR_N]),
        "ownership": _domain_alignment(
            tok, dev, ownership_rows, DOMAIN_SPECS["ownership"]),
        "color_oracle": _domain_alignment(
            tok, dev, oracle_color_rows, spec),
    }
    donor_controllers = {}
    for name, alignment in donor_alignments.items():
        controller, _samples = _controller_from_alignment(model, alignment)
        donor_controllers[name] = controller

    pc1, donor_mean, shared_geometry = _shared_controllers(
        donor_controllers["location"], donor_controllers["ownership"])
    controllers = {
        "location": donor_controllers["location"],
        "ownership": donor_controllers["ownership"],
        "donor_mean": donor_mean,
        "leave_color_out_pc1": pc1,
        "color_oracle": donor_controllers["color_oracle"],
    }
    shared_geometry.update({
        "pc1_color_oracle_cosine": _cosine(
            pc1, donor_controllers["color_oracle"]),
        "mean_color_oracle_cosine": _cosine(
            donor_mean, donor_controllers["color_oracle"]),
        "location_ownership_cosine": _cosine(
            donor_controllers["location"],
            donor_controllers["ownership"]),
        "color_excluded_from_primary_construction": True,
    })

    controller_path = os.path.join(
        out_dir, f"leave_color_out_controllers_{model_key}.npz")
    np.savez(
        controller_path,
        **{name: value.numpy() for name, value in controllers.items()})
    with open(controller_path, "rb") as handle:
        archive_sha = hashlib.sha256(handle.read()).hexdigest().upper()

    alignment = _domain_alignment(tok, dev, evaluation_rows, spec)
    if alignment["answer_positions"] != donor_alignments[
            "color_oracle"]["answer_positions"]:
        raise ValueError("color donor/evaluation answer positions diverge")
    states = {"belief": [], "search": []}
    for belief_batch, search_batch in alignment["batches"]:
        states["belief"].append(_capture_full_l21(model, belief_batch))
        states["search"].append(_capture_full_l21(model, search_batch))
    source = [row["source"] for row in evaluation_rows]
    target = [row["target"] for row in evaluation_rows]
    clean_belief, clean_search = alignment["batches"][0]
    natural_belief, natural_search = alignment["batches"][1]

    def build_context(operation, layers, sequence_patch=None):
        clean = clean_belief if operation == "belief" else clean_search
        natural = (
            natural_belief if operation == "belief" else natural_search)
        return _generic_task_context(
            model, clean, natural, list(spec["values"]),
            source, target, layers, head_dim,
            sequence_patch=sequence_patch)

    hb = Heartbeat(
        2 * (3 + len(LAYERS))
        + len(controllers) * 2 * (3 + len(LAYERS)),
        "leave_color_out_shared", every_sec=30, out_dir=out_dir)
    original_tasks = {}
    original_curves = {}
    original_summaries = {}
    for operation in ("belief", "search"):
        task_name = f"{operation}_original"
        context = build_context(operation, CAPTURE_LAYERS)
        original_tasks[task_name] = _public_generic_task(context)
        hb.step(extra=f"{task_name}/baseline")
        hb.step(extra=f"{task_name}/source")
        hb.step(extra=f"{task_name}/base_path")
        original_curves[task_name] = {}
        for layer, cell in _generic_curve(
                model, context, n_heads, head_dim).items():
            original_curves[task_name][layer] = cell
            hb.step(extra=f"{task_name}/prefixL{layer}")
        original_summaries[task_name] = _summary(
            original_curves[task_name])
    original = {
        "belief": original_summaries["belief_original"],
        "search": original_summaries["search_original"],
    }

    arms = {}
    for controller_name, controller in controllers.items():
        tasks = dict(original_tasks)
        curves = dict(original_curves)
        summaries = dict(original_summaries)
        for operation, sign, task_name in (
                ("belief", -1.0, "belief_to_search"),
                ("search", +1.0, "search_to_belief")):
            patch = _fixed_patch(
                states[operation], alignment["answer_positions"],
                controller, sign)
            context = build_context(
                operation, CAPTURE_LAYERS, sequence_patch=patch)
            tasks[task_name] = _public_generic_task(context)
            hb.step(extra=f"{controller_name}/{task_name}/baseline")
            hb.step(extra=f"{controller_name}/{task_name}/source")
            hb.step(extra=f"{controller_name}/{task_name}/base_path")
            curves[task_name] = {}
            for layer, cell in _generic_curve(
                    model, context, n_heads, head_dim).items():
                curves[task_name][layer] = cell
                hb.step(extra=(
                    f"{controller_name}/{task_name}/prefixL{layer}"))
            summaries[task_name] = _summary(curves[task_name])
        movement = _movement(
            original,
            summaries["belief_to_search"]["l24_minimum_mediation"],
            summaries["search_to_belief"]["l24_minimum_mediation"])
        world = _statistical_world_movement(
            _world_mediation(
                tasks["belief_original"],
                curves["belief_original"]["24"]),
            _world_mediation(
                tasks["search_original"],
                curves["search_original"]["24"]),
            _world_mediation(
                tasks["belief_to_search"],
                curves["belief_to_search"]["24"]),
            _world_mediation(
                tasks["search_to_belief"],
                curves["search_to_belief"]["24"]))
        verdict = _arm_verdict(tasks, summaries, movement, world)
        functional = _functional_score(movement, list(tasks.values()))
        arms[controller_name] = {
            "tasks": tasks,
            "cumulative_prefix": curves,
            "summaries": summaries,
            "movement": movement,
            "per_world_movement": world,
            "functional": functional["functional"],
            "functional_bidirectional_score": (
                functional["functional_bidirectional_score"]),
            "verdict": verdict,
        }
        log(
            f"LEAVE-COLOR-OUT {controller_name} verdict={verdict} "
            f"movement={movement} signs="
            f"{world['belief_to_search']['successes']}/"
            f"{world['search_to_belief']['successes']}")
    hb.done()

    control_layers = (21, 22, 23, 24)
    l24_sites = _full_sites((22, 23, 24), n_heads)
    control_hb = Heartbeat(
        2 * 4 + N_RANDOM * 2 * 4 + N_RANDOM * 2 * 4,
        "leave_color_out_controls", every_sec=30, out_dir=out_dir)

    def evaluate_control(label, displacement, positions):
        values = {}
        public = []
        for operation, sign, direction in (
                ("belief", -1.0, "belief_to_search"),
                ("search", +1.0, "search_to_belief")):
            patch = _fixed_patch(
                states[operation], positions, displacement, sign)
            context = build_context(
                operation, control_layers, sequence_patch=patch)
            public.append(_public_generic_task(context))
            control_hb.step(extra=f"{label}/{direction}/baseline")
            control_hb.step(extra=f"{label}/{direction}/source")
            control_hb.step(extra=f"{label}/{direction}/base_path")
            cell = _generic_evaluate_sites(
                model, context, l24_sites, head_dim)
            values[direction] = float(
                cell["mediation"]["minimum_fraction"])
            control_hb.step(extra=f"{label}/{direction}/L24")
        return _functional_score(
            _movement(
                original, values["belief_to_search"],
                values["search_to_belief"]),
            public)

    selected_score = arms["leave_color_out_pc1"][
        "functional_bidirectional_score"]
    instruction = evaluate_control(
        "instruction", pc1, alignment["instruction_positions"])
    random_directions = _norm_matched_directions(
        pc1, seed=DIRECTION_SEED + 4001)
    direction_cells = []
    for index, direction in enumerate(random_directions):
        direction_cells.append({
            "random_index": index,
            "per_position_norms": direction.norm(dim=-1).tolist(),
            **evaluate_control(
                f"random_direction{index}", direction,
                alignment["answer_positions"]),
        })
    direction_scores = [
        cell["functional_bidirectional_score"]
        for cell in direction_cells
    ]
    direction_p, direction_exceed = _tail_probability(
        selected_score, direction_scores)

    position_sets = _random_position_sets(
        alignment["random_candidates"], seed=POSITION_SEED + 4001)
    position_cells = []
    for index, positions in enumerate(position_sets):
        position_cells.append({
            "random_index": index,
            "positions": positions,
            **evaluate_control(
                f"random_position{index}", pc1, positions),
        })
    position_scores = [
        cell["functional_bidirectional_score"]
        for cell in position_cells
    ]
    position_p, position_exceed = _tail_probability(
        selected_score, position_scores)
    control_hb.done()

    specific = bool(
        arms["leave_color_out_pc1"]["verdict"] == "PASS"
        and direction_p <= 0.05 + 1e-12
        and position_p <= 0.05 + 1e-12
        and instruction["functional_bidirectional_score"]
        < 0.5 * selected_score - 1e-12)
    controls = {
        "selected_score": selected_score,
        "instruction": instruction,
        "random_direction": {
            "n_random": N_RANDOM,
            "seed": DIRECTION_SEED + 4001,
            "empirical_p": direction_p,
            "exceed_count": direction_exceed,
            "cells": direction_cells,
        },
        "random_position": {
            "n_random": N_RANDOM,
            "seed": POSITION_SEED + 4001,
            "empirical_p": position_p,
            "exceed_count": position_exceed,
            "cells": position_cells,
        },
        "specific": specific,
    }

    original_movement = arms["color_oracle"]["movement"]
    target_gate = _original_gate(
        original_tasks, original_summaries, original_movement)
    oracle_pass = arms["color_oracle"]["verdict"] == "PASS"
    pc1_pass = arms["leave_color_out_pc1"]["verdict"] == "PASS"
    mean_pass = arms["donor_mean"]["verdict"] == "PASS"
    individual_pass = any(
        arms[name]["verdict"] == "PASS"
        for name in ("location", "ownership"))
    if target_gate != "ELIGIBLE" or not oracle_pass:
        verdict = "COLOR_TARGET_UNRESOLVED"
    elif pc1_pass and specific and not individual_pass:
        verdict = "LEAVE_COLOR_OUT_SHARED_COMPONENT"
    elif pc1_pass and specific:
        verdict = "TARGET_EXCLUDED_TRANSFER"
    elif mean_pass:
        verdict = "MEAN_ONLY_TARGET_EXCLUDED_TRANSFER"
    else:
        verdict = "NO_LEAVE_COLOR_OUT_TRANSFER"

    result = {
        "stage": "delta_leave_color_out_shared",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "claim": (
            "A rank-one answer-prefix component learned from location and "
            "ownership only is prospectively tested on fresh color histories."),
        "splits": {
            "location_donor_indices": location_indices[:DONOR_N],
            "ownership_donor_rows": ownership_rows,
            "color_oracle_donor_rows": oracle_color_rows,
            "fresh_color_evaluation_rows": evaluation_rows,
            "fresh_histories_excluded_from_fitting": True,
            "color_excluded_from_primary_construction": True,
        },
        "controller_construction": {
            "primary": "leave_color_out_pc1",
            "method": (
                "uncentered PC1 of flattened donor-mean location and "
                "ownership controllers, oriented and mean-projection scaled"),
            "shared_geometry": shared_geometry,
        },
        "controllers": {
            name: _controller_metadata(value)
            for name, value in controllers.items()
        },
        "controller_archive": {
            "artifact": os.path.basename(controller_path),
            "sha256": archive_sha,
        },
        "prospective_gate": {
            "n_world": TEST_N,
            "minimum_positive_per_direction": SIGN_MINIMUM,
            "minimum_positive_fraction": SIGN_MINIMUM / TEST_N,
            "maximum_exact_one_sided_sign_p": SIGN_P_MAX,
            "minimum_aggregate_movement": 0.05,
            "minimum_gap_fraction": 0.5,
        },
        "target_original_gate": target_gate,
        "arms": arms,
        "controls": controls,
        "layers": list(LAYERS),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir, f"results_delta_leave_color_out_shared_{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"LEAVE-COLOR-OUT verdict={verdict} target={target_gate} "
        f"oracle={arms['color_oracle']['verdict']} "
        f"pc1={arms['leave_color_out_pc1']['verdict']} "
        f"specific={specific} artifact={path}")
    return result
