"""Independent confirmation of the low-energy residual-only controller."""
from __future__ import annotations

import hashlib
import itertools
import json
import os

import numpy as np
import torch

from .delta_anchor_write import _resolve
from .delta_content_cancelled_controller import (
    DIRECTION_SEED,
    N_RANDOM,
    POSITION_SEED,
    _fixed_patch,
    _functional_score,
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
    _domain_rows,
    _generic_curve,
    _generic_evaluate_sites,
    _generic_task_context,
    _public_generic_task,
)
from .delta_distributed_label_transplant import _capture_full_l21, _tail_probability
from .delta_leave_color_out_shared import (
    SIGN_MINIMUM,
    SIGN_P_MAX,
    TEST_N,
    _arm_verdict,
    _controller_metadata,
    _fresh_color_rows,
    _original_gate,
    _shared_controllers,
    _statistical_world_movement,
)
from .delta_operation_handoff_depth import CAPTURE_LAYERS, LAYERS, _full_sites
from .delta_preprint_battery import _compatible_world_rows
from .delta_shared_adapter_decomposition import (
    _color_decomposition,
    _fresh_color_rows_v2,
    _orthogonal_norm_matched_residuals,
)
from .delta_sparse_transport import _attention_geometry
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

PROTOCOL_VERSION = "2026-07-25-p2-residual-only-confirmation-v1"
PROTOCOL_SHA256 = (
    "74708FBD05FF1A528C07999C7243CAD9AAC0A3FC05A8752BB163BC9578A36C71")
DONOR_N = 15
N_DIRECTION = 39
N_POSITION = N_RANDOM
DIRECTION_NULL_SEED = DIRECTION_SEED + 8001
POSITION_NULL_SEED = POSITION_SEED + 8001
MONOTONIC_TOLERANCE = 0.005
DOSES = (-1.0, -0.5, 0.25, 0.5, 0.75, 1.0, 1.25)
DOSE_NAMES = {
    -1.0: "residual_m100",
    -0.5: "residual_m050",
    0.25: "residual_p025",
    0.5: "residual_p050",
    0.75: "residual_p075",
    1.0: "residual_p100",
    1.25: "residual_p125",
}


def _fresh_color_rows_v3():
    values = DOMAIN_SPECS["color_state"]["values"]
    prior_rows = (
        _domain_rows(values)
        + _fresh_domain_rows(values)
        + _fresh_color_rows()
        + _fresh_color_rows_v2()
    )
    used = {
        (state, row["d1"], row["d2"])
        for row in prior_rows
        for state in (row["source"], row["target"])
    }
    shifts = (1, 3, 5, 7)
    rows = []
    for index in range(TEST_N):
        source_index = index % len(values)
        shift = shifts[index // len(values)]
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
            if clean_signature not in used and natural_signature not in used:
                used.add(clean_signature)
                used.add(natural_signature)
                break
        else:
            raise AssertionError(
                f"could not construct third fresh color history {index}")
        rows.append({
            "row_index": index,
            "source": source,
            "target": target,
            "state": source,
            "d1": d1,
            "d2": d2,
        })
    signatures = {
        (state, row["d1"], row["d2"])
        for row in rows
        for state in (row["source"], row["target"])
    }
    if len(signatures) != 2 * TEST_N:
        raise AssertionError("third fresh prompt histories are not unique")
    return rows


@torch.no_grad()
def run_delta_residual_only_confirmation(
        model_path, out_dir,
        model_key="qwen7b_residual_only_confirmation",
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
        raise ValueError("residual-only layers are absent")
    n_heads, head_dim = _attention_geometry(model)
    spec = DOMAIN_SPECS["color_state"]

    location_rows, location_indices = _compatible_world_rows(
        tok, torch.device("cpu"), 30)
    if len(location_rows) != 30:
        raise ValueError("all 30 compatible location worlds are required")
    ownership_rows = _fresh_domain_rows(
        DOMAIN_SPECS["ownership"]["values"])[:DONOR_N]
    color_donor_rows = _fresh_domain_rows(spec["values"])[:DONOR_N]
    donor_alignments = {
        "location": _location_alignment(
            tok, dev, location_rows[:DONOR_N]),
        "ownership": _domain_alignment(
            tok, dev, ownership_rows, DOMAIN_SPECS["ownership"]),
        "color": _domain_alignment(tok, dev, color_donor_rows, spec),
    }
    donor_controllers = {
        name: _controller_from_alignment(model, alignment)[0]
        for name, alignment in donor_alignments.items()
    }
    shared_pc1, _donor_mean, shared_geometry = _shared_controllers(
        donor_controllers["location"], donor_controllers["ownership"])
    projection, residual, decomposition = _color_decomposition(
        shared_pc1, donor_controllers["color"])

    dose_controllers = {
        DOSE_NAMES[dose]: float(dose) * residual
        for dose in DOSES
    }
    controller_path = os.path.join(
        out_dir, f"residual_only_controllers_{model_key}.npz")
    np.savez(
        controller_path,
        shared_pc1=shared_pc1.numpy(),
        color_projection=projection.numpy(),
        color_residual=residual.numpy(),
        **{name: value.numpy()
           for name, value in dose_controllers.items()})
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

    hb = Heartbeat(
        2 * (3 + len(LAYERS))
        + len(dose_controllers) * 2 * (3 + len(LAYERS)),
        "residual_only_full_curves", every_sec=30, out_dir=out_dir)
    original_tasks = {}
    original_curves = {}
    original_summaries = {}
    for operation in ("belief", "search"):
        task_name = f"{operation}_original"
        context = build_context(operation, CAPTURE_LAYERS)
        original_tasks[task_name] = _public_generic_task(context)
        for phase in ("baseline", "source", "base_path"):
            hb.step(extra=f"{task_name}/{phase}")
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

    arm_results = {}
    for arm_name, controller in dose_controllers.items():
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
            for phase in ("baseline", "source", "base_path"):
                hb.step(extra=f"{arm_name}/{task_name}/{phase}")
            curves[task_name] = {}
            for layer, cell in _generic_curve(
                    model, context, n_heads, head_dim).items():
                curves[task_name][layer] = cell
                hb.step(extra=f"{arm_name}/{task_name}/prefixL{layer}")
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
        arm_results[arm_name] = {
            "dose": next(
                dose for dose, name in DOSE_NAMES.items()
                if name == arm_name),
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
            f"RESIDUAL-ONLY {arm_name} verdict={verdict} "
            f"movement={movement} signs="
            f"{world['belief_to_search']['successes']}/"
            f"{world['search_to_belief']['successes']}")
    hb.done()

    control_layers = (21, 22, 23, 24)
    l24_sites = _full_sites((22, 23, 24), n_heads)
    control_hb = Heartbeat(
        2 * 4 + N_DIRECTION * 2 * 4 + N_POSITION * 2 * 4,
        "residual_only_controls", every_sec=30, out_dir=out_dir)

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
            for phase in ("baseline", "source", "base_path"):
                control_hb.step(extra=f"{label}/{direction}/{phase}")
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

    instruction = evaluate_control(
        "instruction_residual", residual,
        alignment["instruction_positions"])
    random_residuals = _orthogonal_norm_matched_residuals(
        shared_pc1, residual, n_random=N_DIRECTION,
        seed=DIRECTION_NULL_SEED)
    direction_cells = []
    for index, random_residual in enumerate(random_residuals):
        direction_cells.append({
            "random_index": index,
            "per_position_norms": random_residual.norm(dim=-1).tolist(),
            **evaluate_control(
                f"random_residual_{index}", random_residual,
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
                f"random_residual_position_{index}",
                residual, positions),
        })
    control_hb.done()

    selected = arm_results["residual_p100"]
    selected_score = selected["functional_bidirectional_score"]
    direction_scores = [
        cell["functional_bidirectional_score"]
        for cell in direction_cells
    ]
    position_scores = [
        cell["functional_bidirectional_score"]
        for cell in position_cells
    ]
    direction_p, direction_exceed = _tail_probability(
        selected_score, direction_scores)
    position_p, position_exceed = _tail_probability(
        selected_score, position_scores)
    specific = bool(
        direction_p <= 0.025 + 1e-12
        and position_p <= 0.05 + 1e-12
        and instruction["functional_bidirectional_score"]
        < 0.5 * selected_score - 1e-12)
    controls = {
        "selected_score": selected_score,
        "instruction": instruction,
        "orthogonal_random_residual": {
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
        "specific": specific,
    }

    positive_names = (
        "residual_p025", "residual_p050",
        "residual_p075", "residual_p100",
    )
    positive_scores = [
        arm_results[name]["functional_bidirectional_score"]
        for name in positive_names
    ]
    monotonic = all(
        right + MONOTONIC_TOLERANCE >= left
        for left, right in zip(positive_scores, positive_scores[1:]))
    polarity_rows = {}
    for name in ("residual_m100", "residual_m050"):
        movement = arm_results[name]["movement"]
        polarity_rows[name] = {
            "belief_to_search_movement": (
                movement["belief_to_search_movement"]),
            "search_to_belief_movement": (
                movement["search_to_belief_movement"]),
            "both_negative": bool(
                movement["belief_to_search_movement"] < 0.0
                and movement["search_to_belief_movement"] < 0.0),
        }
    polarity_pass = all(
        row["both_negative"] for row in polarity_rows.values())
    target_gate = _original_gate(
        original_tasks, original_summaries, selected["movement"])
    selected_pass = selected["verdict"] == "PASS"
    if target_gate != "ELIGIBLE":
        verdict = "COLOR_TARGET_UNRESOLVED"
    elif selected_pass and not specific:
        verdict = "RESIDUAL_ONLY_EFFECT_NONSPECIFIC"
    elif selected_pass and specific and not (monotonic and polarity_pass):
        verdict = "RESIDUAL_DOSE_OR_POLARITY_FAILED"
    elif selected_pass and specific:
        verdict = "RESIDUAL_ONLY_CAUSAL_CONTROLLER"
    else:
        verdict = "RESIDUAL_ONLY_NOT_REPLICATED"

    result = {
        "stage": "delta_residual_only_confirmation",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "claim": (
            "The low-energy component orthogonal to the dominant "
            "cross-domain controller is independently tested as a "
            "sign- and position-specific causal controller."),
        "splits": {
            "location_donor_indices": location_indices[:DONOR_N],
            "ownership_donor_rows": ownership_rows,
            "color_donor_rows": color_donor_rows,
            "third_fresh_color_evaluation_rows": evaluation_rows,
            "evaluation_excluded_from_construction": True,
        },
        "construction": {
            "shared_geometry": shared_geometry,
            "decomposition": decomposition,
            "projection_excluded_from_selected_arm_and_nulls": True,
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
            "minimum_positive_per_direction": SIGN_MINIMUM,
            "maximum_exact_one_sided_sign_p": SIGN_P_MAX,
            "direction_null_maximum_p": 0.025,
            "position_null_maximum_p": 0.05,
            "monotonic_tolerance": MONOTONIC_TOLERANCE,
            "doses": list(DOSES),
        },
        "target_original_gate": target_gate,
        "arms": arm_results,
        "signed_dose_response": {
            "positive_arm_order": list(positive_names),
            "positive_bidirectional_scores": positive_scores,
            "positive_monotonic_with_tolerance": monotonic,
            "negative_polarity": polarity_rows,
            "negative_polarity_pass": polarity_pass,
        },
        "controls": controls,
        "layers": list(LAYERS),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_residual_only_confirmation_{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"RESIDUAL-ONLY verdict={verdict} target={target_gate} "
        f"selected={selected['verdict']} specific={specific} "
        f"monotonic={monotonic} polarity={polarity_pass} "
        f"artifact={path}")
    return result
