"""Prospective endogenous-necessity test for the color residual controller."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from .delta_anchor_write import _resolve
from .delta_content_cancelled_controller import (
    DIRECTION_SEED,
    EPS,
    N_RANDOM,
    POSITION_SEED,
    _fixed_patch,
    _movement,
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
    _generic_curve,
    _generic_evaluate_sites,
    _generic_task_context,
    _public_generic_task,
)
from .delta_distributed_label_transplant import (
    _capture_full_l21,
    _tail_probability,
)
from .delta_leave_color_out_shared import (
    SIGN_MINIMUM,
    SIGN_P_MAX,
    TEST_N,
    _arm_verdict,
    _controller_metadata,
    _original_gate,
    _shared_controllers,
    _sign_tail,
    _statistical_world_movement,
)
from .delta_operation_handoff_depth import CAPTURE_LAYERS, LAYERS, _full_sites
from .delta_preprint_battery import _compatible_world_rows
from .delta_residual_only_confirmation import _fresh_color_rows_v3
from .delta_shared_adapter_decomposition import (
    _color_decomposition,
    _orthogonal_norm_matched_residuals,
)
from .delta_sparse_transport import _attention_geometry
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

PROTOCOL_VERSION = "2026-07-25-p2-endogenous-residual-necessity-v1"
PROTOCOL_SHA256 = (
    "DC81755570515423FABB3E346C714811B26AACBBB1A93CAB3EF6238935B3B3DF")
DONOR_N = 15
N_DIRECTION = 19
N_POSITION = N_RANDOM
DIRECTION_NULL_SEED = DIRECTION_SEED + 10001
POSITION_NULL_SEED = POSITION_SEED + 10001
CARRIER_SENTENCE = "A small wall clock in the room showed noon throughout."
NUMERICAL_TOLERANCE = 1e-5
MINIMUM_ORIGINAL_GAP = 0.03
MINIMUM_GAP_REDUCTION = 0.04
MAXIMUM_REMAINING_GAP = 0.02


def _carrier_spec():
    spec = dict(DOMAIN_SPECS["color_state"])
    spec["rule"] = f"{spec['rule']} {CARRIER_SENTENCE}"
    return spec


def _nested_max(rows):
    """Maximum of a non-empty batch-by-position diagnostic matrix."""
    return float(max(max(float(value) for value in row) for row in rows))


def _equalization_patch_pair(
        belief_states, search_states, positions, direction):
    """Remove half of each endogenous B-S direction component from each side."""
    if len(belief_states) != len(search_states):
        raise ValueError("BELIEF/SEARCH state batches differ")
    axis = direction.detach().double().cpu()
    if tuple(axis.shape) != (len(positions), belief_states[0].shape[-1]):
        raise ValueError("equalization direction has the wrong shape")
    axis_norm2 = axis.square().sum(dim=-1)
    if bool((axis_norm2 <= EPS).any()):
        raise ValueError("equalization direction contains a zero-norm row")

    belief_patches = []
    search_patches = []
    midpoint_errors = []
    coordinate_errors = []
    orthogonal_errors = []
    coefficient_rows = []
    for belief, search in zip(belief_states, search_states):
        b = belief[:, positions, :].double().cpu()
        s = search[:, positions, :].double().cpu()
        difference = b - s
        coefficient = (
            (difference * axis.unsqueeze(0)).sum(dim=-1)
            / axis_norm2.unsqueeze(0))
        component = coefficient.unsqueeze(-1) * axis.unsqueeze(0)
        new_b = (b - 0.5 * component).float()
        new_s = (s + 0.5 * component).float()

        actual_b = new_b.double()
        actual_s = new_s.double()
        actual_difference = actual_b - actual_s
        coordinate_error = (
            (actual_difference * axis.unsqueeze(0)).sum(dim=-1)
            / axis_norm2.unsqueeze(0))
        expected_orthogonal = difference - component
        orthogonal_error = (
            (actual_difference - expected_orthogonal).norm(dim=-1)
            / difference.norm(dim=-1).clamp_min(EPS))
        midpoint_error = (
            0.5 * (actual_b + actual_s) - 0.5 * (b + s)
        ).abs().amax(dim=-1)

        belief_patches.append((list(positions), new_b))
        search_patches.append((list(positions), new_s))
        coefficient_rows.extend(coefficient.tolist())
        midpoint_errors.extend(midpoint_error.tolist())
        coordinate_errors.extend(coordinate_error.abs().tolist())
        orthogonal_errors.extend(orthogonal_error.tolist())

    invariants = {
        "maximum_midpoint_absolute_error": _nested_max(midpoint_errors),
        "maximum_residual_coordinate_difference": _nested_max(
            coordinate_errors),
        "maximum_relative_orthogonal_component_error": _nested_max(
            orthogonal_errors),
        "coefficient_range": [
            float(min(min(row) for row in coefficient_rows)),
            float(max(max(row) for row in coefficient_rows)),
        ],
    }
    invariants["pass"] = bool(
        invariants["maximum_midpoint_absolute_error"]
        <= NUMERICAL_TOLERANCE
        and invariants["maximum_residual_coordinate_difference"]
        <= NUMERICAL_TOLERANCE
        and invariants["maximum_relative_orthogonal_component_error"]
        <= NUMERICAL_TOLERANCE)
    return tuple(belief_patches), tuple(search_patches), invariants


def _gap_reduction(
        original_belief, original_search,
        equalized_belief, equalized_search,
        original_tasks, original_curves,
        equalized_tasks, equalized_curves):
    original_gap = float(original_belief - original_search)
    remaining_gap = float(equalized_belief - equalized_search)
    reduction = float(original_gap - remaining_gap)

    original_belief_rows = _world_mediation(
        original_tasks["belief_original"],
        original_curves["belief_original"]["24"])
    original_search_rows = _world_mediation(
        original_tasks["search_original"],
        original_curves["search_original"]["24"])
    equalized_belief_rows = _world_mediation(
        equalized_tasks["belief_equalized"],
        equalized_curves["belief_equalized"]["24"])
    equalized_search_rows = _world_mediation(
        equalized_tasks["search_equalized"],
        equalized_curves["search_equalized"]["24"])

    rows = []
    reductions = []
    for index, (ob, os_, eb, es) in enumerate(zip(
            original_belief_rows, original_search_rows,
            equalized_belief_rows, equalized_search_rows)):
        if any(value is None for value in (ob, os_, eb, es)):
            rows.append({
                "world_offset": index,
                "original_gap": None,
                "equalized_gap": None,
                "gap_reduction": None,
                "predicted_sign": False,
            })
            continue
        row_original = float(ob) - float(os_)
        row_equalized = float(eb) - float(es)
        row_reduction = row_original - row_equalized
        reductions.append(row_reduction)
        rows.append({
            "world_offset": index,
            "original_gap": row_original,
            "equalized_gap": row_equalized,
            "gap_reduction": row_reduction,
            "predicted_sign": bool(row_reduction > 0.0),
        })
    successes = sum(value > 0.0 for value in reductions)
    valid = len(reductions) == TEST_N
    sign_p = _sign_tail(successes, TEST_N) if valid else None
    return {
        "original_gap": original_gap,
        "equalized_gap": remaining_gap,
        "gap_reduction": reduction,
        "absolute_remaining_gap": abs(remaining_gap),
        "rows": rows,
        "valid_worlds": len(reductions),
        "positive_reductions": successes,
        "exact_one_sided_sign_p": sign_p,
        "aggregate_pass": bool(
            original_gap >= MINIMUM_ORIGINAL_GAP - 1e-9
            and reduction >= MINIMUM_GAP_REDUCTION - 1e-9
            and abs(remaining_gap) <= MAXIMUM_REMAINING_GAP + 1e-9),
        "statistical_pass": bool(
            valid
            and successes >= SIGN_MINIMUM
            and sign_p is not None
            and sign_p <= SIGN_P_MAX + 1e-12),
    }


@torch.no_grad()
def run_delta_endogenous_residual_necessity(
        model_path, out_dir,
        model_key="qwen7b_endogenous_residual_necessity",
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
        raise ValueError("endogenous-necessity layers are absent")
    n_heads, head_dim = _attention_geometry(model)

    spec = _carrier_spec()
    location_rows, location_indices = _compatible_world_rows(
        tok, torch.device("cpu"), 30)
    if len(location_rows) != 30:
        raise ValueError("all 30 compatible location worlds are required")
    ownership_rows = _fresh_domain_rows(
        DOMAIN_SPECS["ownership"]["values"])[:DONOR_N]
    color_donor_rows = _fresh_domain_rows(
        DOMAIN_SPECS["color_state"]["values"])[:DONOR_N]
    donor_alignments = {
        "location": _location_alignment(
            tok, dev, location_rows[:DONOR_N]),
        "ownership": _domain_alignment(
            tok, dev, ownership_rows, DOMAIN_SPECS["ownership"]),
        "color": _domain_alignment(
            tok, dev, color_donor_rows, DOMAIN_SPECS["color_state"]),
    }
    donor_controllers = {
        name: _controller_from_alignment(model, alignment)[0]
        for name, alignment in donor_alignments.items()
    }
    shared_pc1, _donor_mean, shared_geometry = _shared_controllers(
        donor_controllers["location"], donor_controllers["ownership"])
    projection, residual, decomposition = _color_decomposition(
        shared_pc1, donor_controllers["color"])

    controller_path = os.path.join(
        out_dir, f"endogenous_necessity_controllers_{model_key}.npz")
    np.savez(
        controller_path,
        shared_pc1=shared_pc1.numpy(),
        color_projection=projection.numpy(),
        color_residual=residual.numpy())
    with open(controller_path, "rb") as handle:
        archive_sha = hashlib.sha256(handle.read()).hexdigest().upper()

    evaluation_rows = _fresh_color_rows_v3()
    alignment = _domain_alignment(tok, dev, evaluation_rows, spec)
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

    residual_belief_patch, residual_search_patch, residual_invariants = (
        _equalization_patch_pair(
            states["belief"], states["search"],
            alignment["answer_positions"], residual))

    full_hb = Heartbeat(
        6 * (3 + len(LAYERS)),
        "endogenous_necessity_full_curves", every_sec=30, out_dir=out_dir)

    def evaluate_full(label, operation, patch=None):
        context = build_context(operation, CAPTURE_LAYERS, patch)
        task = _public_generic_task(context)
        for phase in ("baseline", "source", "base_path"):
            full_hb.step(extra=f"{label}/{phase}")
        curve = {}
        for layer, cell in _generic_curve(
                model, context, n_heads, head_dim).items():
            curve[layer] = cell
            full_hb.step(extra=f"{label}/prefixL{layer}")
        return task, curve, _summary(curve)

    original_tasks = {}
    original_curves = {}
    original_summaries = {}
    for operation in ("belief", "search"):
        name = f"{operation}_original"
        (original_tasks[name],
         original_curves[name],
         original_summaries[name]) = evaluate_full(name, operation)

    calibration_tasks = dict(original_tasks)
    calibration_curves = dict(original_curves)
    calibration_summaries = dict(original_summaries)
    for operation, sign, name in (
            ("belief", -1.0, "belief_to_search"),
            ("search", +1.0, "search_to_belief")):
        patch = _fixed_patch(
            states[operation], alignment["answer_positions"],
            residual, sign)
        (calibration_tasks[name],
         calibration_curves[name],
         calibration_summaries[name]) = evaluate_full(
             f"calibration_{name}", operation, patch)

    equalized_tasks = {}
    equalized_curves = {}
    equalized_summaries = {}
    for operation, patch, name in (
            ("belief", residual_belief_patch, "belief_equalized"),
            ("search", residual_search_patch, "search_equalized")):
        (equalized_tasks[name],
         equalized_curves[name],
         equalized_summaries[name]) = evaluate_full(name, operation, patch)
    full_hb.done()

    original = {
        "belief": original_summaries["belief_original"],
        "search": original_summaries["search_original"],
    }
    calibration_movement = _movement(
        original,
        calibration_summaries["belief_to_search"][
            "l24_minimum_mediation"],
        calibration_summaries["search_to_belief"][
            "l24_minimum_mediation"])
    calibration_world = _statistical_world_movement(
        _world_mediation(
            original_tasks["belief_original"],
            original_curves["belief_original"]["24"]),
        _world_mediation(
            original_tasks["search_original"],
            original_curves["search_original"]["24"]),
        _world_mediation(
            calibration_tasks["belief_to_search"],
            calibration_curves["belief_to_search"]["24"]),
        _world_mediation(
            calibration_tasks["search_to_belief"],
            calibration_curves["search_to_belief"]["24"]))
    calibration_verdict = _arm_verdict(
        calibration_tasks, calibration_summaries,
        calibration_movement, calibration_world)
    target_gate = _original_gate(
        original_tasks, original_summaries, calibration_movement)

    necessity = _gap_reduction(
        original["belief"]["l24_minimum_mediation"],
        original["search"]["l24_minimum_mediation"],
        equalized_summaries["belief_equalized"]["l24_minimum_mediation"],
        equalized_summaries["search_equalized"]["l24_minimum_mediation"],
        original_tasks, original_curves,
        equalized_tasks, equalized_curves)
    selected_functional = bool(
        all(task["eligible"] and task["source_intervention"]["sufficient"]
            for task in [
                *original_tasks.values(), *equalized_tasks.values()])
        and all(summary["first_passing_prefix"] is not None
                for summary in [
                    *original_summaries.values(),
                    *equalized_summaries.values()]))
    necessity["functional"] = selected_functional
    necessity["functional_gap_reduction"] = (
        necessity["gap_reduction"] if selected_functional else -1e9)
    necessity_pass = bool(
        necessity["aggregate_pass"]
        and necessity["statistical_pass"]
        and selected_functional
        and residual_invariants["pass"])

    control_layers = (21, 22, 23, 24)
    l24_sites = _full_sites((22, 23, 24), n_heads)
    control_hb = Heartbeat(
        (2 + N_DIRECTION + N_POSITION) * 2 * 4,
        "endogenous_necessity_controls", every_sec=30, out_dir=out_dir)

    def evaluate_control(label, direction, positions):
        belief_patch, search_patch, invariants = _equalization_patch_pair(
            states["belief"], states["search"], positions, direction)
        values = {}
        public = []
        for operation, patch in (
                ("belief", belief_patch), ("search", search_patch)):
            context = build_context(
                operation, control_layers, sequence_patch=patch)
            public.append(_public_generic_task(context))
            for phase in ("baseline", "source", "base_path"):
                control_hb.step(extra=f"{label}/{operation}/{phase}")
            cell = _generic_evaluate_sites(
                model, context, l24_sites, head_dim)
            values[operation] = float(
                cell["mediation"]["minimum_fraction"])
            control_hb.step(extra=f"{label}/{operation}/L24")
        functional = all(
            task["eligible"] and task["source_intervention"]["sufficient"]
            for task in public)
        remaining_gap = values["belief"] - values["search"]
        reduction = (
            necessity["original_gap"] - remaining_gap)
        return {
            "belief_l24_minimum_mediation": values["belief"],
            "search_l24_minimum_mediation": values["search"],
            "equalized_gap": remaining_gap,
            "gap_reduction": reduction,
            "functional": bool(functional),
            "functional_gap_reduction": (
                float(reduction) if functional else -1e9),
            "invariants": invariants,
        }

    projection_control = evaluate_control(
        "projection_equalization", projection,
        alignment["answer_positions"])
    instruction_control = evaluate_control(
        "instruction_residual_equalization", residual,
        alignment["instruction_positions"])

    random_directions = _orthogonal_norm_matched_residuals(
        shared_pc1, residual, n_random=N_DIRECTION,
        seed=DIRECTION_NULL_SEED)
    direction_cells = []
    for index, direction in enumerate(random_directions):
        direction_cells.append({
            "random_index": index,
            "per_position_norms": direction.norm(dim=-1).tolist(),
            **evaluate_control(
                f"random_direction_{index}", direction,
                alignment["answer_positions"]),
        })

    position_sets = _random_position_sets(
        alignment["random_candidates"], n_random=N_POSITION,
        seed=POSITION_NULL_SEED)
    position_cells = []
    for index, positions in enumerate(position_sets):
        position_cells.append({
            "random_index": index,
            "positions": positions,
            **evaluate_control(
                f"random_position_{index}", residual, positions),
        })
    control_hb.done()

    selected_score = necessity["functional_gap_reduction"]
    direction_scores = [
        cell["functional_gap_reduction"] for cell in direction_cells]
    position_scores = [
        cell["functional_gap_reduction"] for cell in position_cells]
    direction_p, direction_exceed = _tail_probability(
        selected_score, direction_scores)
    position_p, position_exceed = _tail_probability(
        selected_score, position_scores)
    all_controls_functional = bool(
        projection_control["functional"]
        and instruction_control["functional"]
        and all(cell["functional"] for cell in direction_cells)
        and all(cell["functional"] for cell in position_cells))
    specific = bool(
        all_controls_functional
        and direction_p <= 0.05 + 1e-12
        and position_p <= 0.05 + 1e-12
        and projection_control["functional_gap_reduction"]
        < 0.5 * selected_score - 1e-12
        and instruction_control["functional_gap_reduction"]
        < 0.5 * selected_score - 1e-12)
    controls = {
        "selected_gap_reduction": selected_score,
        "projection_equalization": projection_control,
        "instruction_residual_equalization": instruction_control,
        "orthogonal_random_direction": {
            "n_random": N_DIRECTION,
            "seed": DIRECTION_NULL_SEED,
            "empirical_p": direction_p,
            "exceed_count": direction_exceed,
            "cells": direction_cells,
        },
        "random_position": {
            "n_random": N_POSITION,
            "seed": POSITION_NULL_SEED,
            "empirical_p": position_p,
            "exceed_count": position_exceed,
            "cells": position_cells,
        },
        "all_controls_functional": all_controls_functional,
        "specific": specific,
    }

    if target_gate != "ELIGIBLE" or calibration_verdict != "PASS":
        verdict = "CARRIER_TEMPLATE_UNRESOLVED"
    elif not necessity_pass:
        verdict = "RESIDUAL_STEERING_WITHOUT_NECESSITY"
    elif not specific:
        verdict = "NECESSITY_EFFECT_NONSPECIFIC"
    elif necessity_pass and specific:
        verdict = "ENDOGENOUS_RESIDUAL_NECESSITY"
    else:
        verdict = "ENDOGENOUS_NECESSITY_UNRESOLVED"

    result = {
        "stage": "delta_endogenous_residual_necessity",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "claim": (
            "Midpoint-preserving equalization tests whether the naturally "
            "occurring residual coordinate is necessary for the model's "
            "BELIEF/SEARCH route distinction."),
        "carrier_template": {
            "sentence": CARRIER_SENTENCE,
            "evaluation_rows": evaluation_rows,
            "rendered_histories_unique_within_set": True,
            "template_absent_from_prior_evaluations": True,
        },
        "splits": {
            "location_donor_indices": location_indices[:DONOR_N],
            "ownership_donor_rows": ownership_rows,
            "color_donor_rows": color_donor_rows,
            "evaluation_excluded_from_construction": True,
        },
        "construction": {
            "shared_geometry": shared_geometry,
            "decomposition": decomposition,
        },
        "controllers": {
            "shared_pc1": _controller_metadata(shared_pc1),
            "color_projection": _controller_metadata(projection),
            "color_residual": _controller_metadata(residual),
        },
        "controller_archive": {
            "artifact": os.path.basename(controller_path),
            "sha256": archive_sha,
        },
        "prospective_gate": {
            "n_world": TEST_N,
            "minimum_original_gap": MINIMUM_ORIGINAL_GAP,
            "minimum_gap_reduction": MINIMUM_GAP_REDUCTION,
            "maximum_absolute_remaining_gap": MAXIMUM_REMAINING_GAP,
            "minimum_positive_world_reductions": SIGN_MINIMUM,
            "maximum_exact_one_sided_sign_p": SIGN_P_MAX,
            "direction_null_maximum_p": 0.05,
            "position_null_maximum_p": 0.05,
            "numerical_tolerance": NUMERICAL_TOLERANCE,
        },
        "target_original_gate": target_gate,
        "original": {
            "tasks": original_tasks,
            "cumulative_prefix": original_curves,
            "summaries": original_summaries,
        },
        "additive_residual_calibration": {
            "tasks": calibration_tasks,
            "cumulative_prefix": calibration_curves,
            "summaries": calibration_summaries,
            "movement": calibration_movement,
            "per_world_movement": calibration_world,
            "verdict": calibration_verdict,
        },
        "endogenous_equalization": {
            "tasks": equalized_tasks,
            "cumulative_prefix": equalized_curves,
            "summaries": equalized_summaries,
            "numerical_invariants": residual_invariants,
            "necessity": necessity,
            "pass": necessity_pass,
        },
        "controls": controls,
        "layers": list(LAYERS),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_endogenous_residual_necessity_{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"ENDOGENOUS-NECESSITY verdict={verdict} "
        f"target={target_gate} calibration={calibration_verdict} "
        f"reduction={necessity['gap_reduction']:+.5f} "
        f"remaining={necessity['equalized_gap']:+.5f} "
        f"signs={necessity['positive_reductions']}/{TEST_N} "
        f"specific={specific} artifact={path}")
    return result
