"""Donor-only within/cross-domain matrix of answer-prefix controllers."""
from __future__ import annotations

import hashlib
import io
import json
import os

import numpy as np
import torch

from .delta_anchor_write import _resolve
from .delta_content_cancelled_controller import (
    DIRECTION_SEED,
    N_RANDOM,
    PATCH_WIDTH,
    POSITION_SEED,
    _curve as _location_curve,
    _donor_alignment,
    _fixed_patch,
    _functional_score,
    _movement,
    _norm_matched_directions,
    _random_position_sets,
    _summary,
    _world_mediation,
    _world_movements,
)
from .delta_cross_domain_controller import (
    DOMAIN_SPECS,
    _domain_alignment,
    _domain_verdict,
    _generic_curve,
    _generic_evaluate_sites,
    _generic_task_context,
    _natural_rows,
    _original_domain_eligible,
    _public_generic_task,
)
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

PROTOCOL_VERSION = "2026-07-25-p2-controller-matrix-v1"
PROTOCOL_SHA256 = (
    "CE32CBE7FBCBEA73570A34892523267FBD82FBE7C6C04CE708C4269B0CD6B3E2")
DOMAINS = ("location", "ownership", "color_state")
DONOR_N = 15
TEST_N = 15
PASS_VERDICTS = (
    "CROSS_DOMAIN_ROUTE_SWITCH",
    "CONTINUOUS_CROSS_DOMAIN_TRANSFER",
)


def _fresh_domain_rows(values, n_rows=30):
    if len(values) != 8 or int(n_rows) != 30:
        raise ValueError("v1 requires eight values and exactly 30 rows")
    rows = []
    for index in range(int(n_rows)):
        source_index = index % len(values)
        shift = 3 + index // len(values)
        source = values[source_index]
        target = values[(source_index + shift) % len(values)]
        remaining = [
            values[(source_index + offset) % len(values)]
            for offset in range(1, len(values) + 1)
            if values[(source_index + offset) % len(values)]
            not in (source, target)
        ]
        rows.append({
            "row_index": index,
            "source": source,
            "target": target,
            "state": source,
            "d1": remaining[0],
            "d2": remaining[1],
        })
    pairs = {(row["source"], row["target"]) for row in rows}
    if len(pairs) != 30:
        raise AssertionError("fresh source-target pairs are not unique")
    return rows


def _location_alignment(tok, dev, rows):
    (batches, differing, groups, candidates,
     marker, readout) = _aligned_batches(tok, dev, rows)
    if len(groups[1]) != PATCH_WIDTH:
        raise ValueError("location answer-prefix width changed")
    return {
        "batches": batches,
        "marker": marker,
        "readout": readout,
        "differing_positions": differing,
        "instruction_positions": list(groups[0]),
        "answer_positions": list(groups[1]),
        "random_candidates": candidates,
    }


def _controller_from_alignment(model, alignment):
    samples = []
    for belief_batch, search_batch in alignment["batches"]:
        belief = _capture_full_l21(model, belief_batch)
        search = _capture_full_l21(model, search_batch)
        samples.extend(
            belief[:, alignment["answer_positions"], :]
            - search[:, alignment["answer_positions"], :])
    sample_tensor = torch.stack(samples)
    return sample_tensor.mean(dim=0), sample_tensor


def _evaluation_states(model, alignment):
    states = {"belief": [], "search": []}
    for belief_batch, search_batch in alignment["batches"]:
        states["belief"].append(_capture_full_l21(model, belief_batch))
        states["search"].append(_capture_full_l21(model, search_batch))
    return states


def _controller_geometry(controllers):
    names = list(controllers)
    flat = torch.stack([
        controllers[name].flatten().float()
        for name in names
    ])
    norms = flat.norm(dim=1)
    cosine = (
        flat @ flat.T
        / (norms[:, None] * norms[None, :]).clamp_min(1e-8))
    per_position = {}
    for left in names:
        per_position[left] = {}
        for right in names:
            values = []
            for position in range(PATCH_WIDTH):
                a = controllers[left][position].float()
                b = controllers[right][position].float()
                values.append(float(
                    torch.dot(a, b)
                    / (a.norm() * b.norm()).clamp_min(1e-8)))
            per_position[left][right] = values
    singular = torch.linalg.svdvals(flat)
    energy = singular.square()
    cumulative = torch.cumsum(energy, dim=0) / energy.sum()
    return {
        "domain_order": names,
        "flattened_cosine": {
            left: {
                right: float(cosine[i, j])
                for j, right in enumerate(names)
            }
            for i, left in enumerate(names)
        },
        "per_position_cosine": per_position,
        "singular_values": singular.tolist(),
        "cumulative_energy": cumulative.tolist(),
    }


def _matrix_cell_pass(cell):
    return cell["verdict"] in PASS_VERDICTS


def _original_gate_reason(tasks, summaries, movement):
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


def _adjudicate(matrix, controls, original_gates):
    for target in ("location", "color_state"):
        if original_gates[target] != "ELIGIBLE":
            return f"{target.upper()}_{original_gates[target]}"
    location_within = _matrix_cell_pass(matrix["location"]["location"])
    color_within = _matrix_cell_pass(
        matrix["color_state"]["color_state"])
    if location_within and not controls["location"]["specific"]:
        return "LOCATION_WITHIN_DOMAIN_CONTROL_FAILED"
    if color_within and not controls["color_state"]["specific"]:
        return "COLOR_WITHIN_DOMAIN_CONTROL_FAILED"
    if location_within and not color_within:
        return "LOCATION_SPECIFIC_CONTROLLER"
    if not location_within:
        return "LOCATION_CONTROLLER_NOT_REPLICATED"
    if not color_within:
        return "COLOR_CONTROLLER_NOT_ESTABLISHED"

    location_to_color = _matrix_cell_pass(
        matrix["color_state"]["location"])
    color_to_location = _matrix_cell_pass(
        matrix["location"]["color_state"])
    if location_to_color and color_to_location:
        return "SHARED_RAW_CONTROLLER_DIRECTION"
    if location_to_color != color_to_location:
        return "ASYMMETRIC_CONTROLLER_COORDINATES"
    return "DOMAIN_SPECIFIC_CONTROLLER_COORDINATES"


@torch.no_grad()
def run_delta_controller_matrix(
        model_path, out_dir,
        model_key="qwen7b_controller_matrix",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=30):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_world) != DONOR_N + TEST_N:
        raise ValueError("v1 is frozen to exactly 30 worlds")
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    if max(LAYERS) >= model_num_hidden_layers(model):
        raise ValueError("controller-matrix layers are absent")
    n_heads, head_dim = _attention_geometry(model)

    location_rows, location_indices = _compatible_world_rows(
        tok, torch.device("cpu"), int(n_world))
    if len(location_rows) != 30:
        raise ValueError("all 30 compatible location worlds are required")
    domain_rows = {
        "location": location_rows,
        "ownership": _fresh_domain_rows(
            DOMAIN_SPECS["ownership"]["values"]),
        "color_state": _fresh_domain_rows(
            DOMAIN_SPECS["color_state"]["values"]),
    }

    donor_alignments = {}
    evaluation_alignments = {}
    for name in DOMAINS:
        donor_rows = domain_rows[name][:DONOR_N]
        evaluation_rows = domain_rows[name][DONOR_N:]
        if name == "location":
            donor_alignment = _location_alignment(tok, dev, donor_rows)
            evaluation_alignment = _location_alignment(
                tok, dev, evaluation_rows)
        else:
            spec = DOMAIN_SPECS[name]
            donor_alignment = _domain_alignment(
                tok, dev, donor_rows, spec)
            evaluation_alignment = _domain_alignment(
                tok, dev, evaluation_rows, spec)
        if (donor_alignment["answer_positions"]
                != evaluation_alignment["answer_positions"]):
            raise ValueError(
                f"{name}: donor/evaluation answer positions diverge")
        donor_alignments[name] = donor_alignment
        evaluation_alignments[name] = evaluation_alignment

    controllers = {}
    donor_samples = {}
    controller_metadata = {}
    for name in DOMAINS:
        controller, samples = _controller_from_alignment(
            model, donor_alignments[name])
        controllers[name] = controller
        donor_samples[name] = samples
        buffer = io.BytesIO()
        np.save(buffer, controller.numpy())
        controller_metadata[name] = {
            "shape": list(controller.shape),
            "frobenius_norm": float(controller.norm()),
            "per_position_norms": controller.norm(dim=-1).tolist(),
            "npy_sha256": hashlib.sha256(
                buffer.getvalue()).hexdigest().upper(),
            "donor_alignment": _donor_alignment(samples, controller),
        }

    controller_path = os.path.join(
        out_dir, f"controller_matrix_{model_key}.npz")
    np.savez(
        controller_path,
        **{name: controllers[name].numpy() for name in DOMAINS})
    with open(controller_path, "rb") as handle:
        controller_archive_sha = hashlib.sha256(
            handle.read()).hexdigest().upper()

    targets = {}
    for name in DOMAINS:
        rows = domain_rows[name][DONOR_N:]
        alignment = evaluation_alignments[name]
        target = {
            "name": name,
            "kind": "location" if name == "location" else "generic",
            "rows": rows,
            "alignment": alignment,
            "states": _evaluation_states(model, alignment),
            "source": (
                ["Paris"] * TEST_N if name == "location"
                else [row["source"] for row in rows]),
            "target": (
                ["Rome"] * TEST_N if name == "location"
                else [row["target"] for row in rows]),
        }
        if name != "location":
            target["spec"] = DOMAIN_SPECS[name]
        targets[name] = target

    def build_context(target, operation, layers, sequence_patch=None):
        if target["kind"] == "location":
            query = (
                "distributed_belief"
                if operation == "belief" else "distributed_search")
            return _task_context(
                model, tok, dev, target["rows"], query,
                layers, head_dim, sequence_patch=sequence_patch)
        batch_index = 0 if operation == "belief" else 1
        clean = target["alignment"]["batches"][0][batch_index]
        natural = target["alignment"]["batches"][1][batch_index]
        return _generic_task_context(
            model, clean, natural,
            list(target["spec"]["values"]),
            target["source"], target["target"],
            layers, head_dim, sequence_patch=sequence_patch)

    def public_task(target, context):
        return (
            _public_task(context)
            if target["kind"] == "location"
            else _public_generic_task(context))

    def full_curve(target, context):
        return (
            _location_curve(model, context, n_heads, head_dim)
            if target["kind"] == "location"
            else _generic_curve(model, context, n_heads, head_dim))

    def l24_cell(target, context, sites):
        return (
            _evaluate_sites(model, context, sites, head_dim)
            if target["kind"] == "location"
            else _generic_evaluate_sites(
                model, context, sites, head_dim))

    matrix_hb = Heartbeat(
        len(DOMAINS) * 2 * (3 + len(LAYERS))
        + len(DOMAINS) * len(DOMAINS) * 2 * (3 + len(LAYERS)),
        "controller_matrix", every_sec=30, out_dir=out_dir)
    originals = {}
    for target_name, target in targets.items():
        tasks = {}
        curves = {}
        summaries = {}
        for operation in ("belief", "search"):
            task_name = f"{operation}_original"
            context = build_context(
                target, operation, CAPTURE_LAYERS)
            tasks[task_name] = public_task(target, context)
            matrix_hb.step(extra=f"{target_name}/{task_name}/baseline")
            matrix_hb.step(extra=f"{target_name}/{task_name}/source")
            matrix_hb.step(extra=f"{target_name}/{task_name}/base_path")
            curves[task_name] = {}
            for layer, cell in full_curve(target, context).items():
                curves[task_name][layer] = cell
                matrix_hb.step(
                    extra=f"{target_name}/{task_name}/prefixL{layer}")
            summaries[task_name] = _summary(curves[task_name])
        originals[target_name] = {
            "tasks": tasks,
            "curves": curves,
            "summaries": summaries,
            "original": {
                "belief": summaries["belief_original"],
                "search": summaries["search_original"],
            },
        }

    matrix = {}
    for target_name, target in targets.items():
        matrix[target_name] = {}
        for source_name, controller in controllers.items():
            tasks = {
                **originals[target_name]["tasks"],
            }
            curves = {
                **originals[target_name]["curves"],
            }
            summaries = {
                **originals[target_name]["summaries"],
            }
            for operation, sign, task_name in (
                    ("belief", -1.0, "belief_to_search"),
                    ("search", +1.0, "search_to_belief")):
                patch = _fixed_patch(
                    target["states"][operation],
                    target["alignment"]["answer_positions"],
                    controller, sign)
                context = build_context(
                    target, operation, CAPTURE_LAYERS,
                    sequence_patch=patch)
                tasks[task_name] = public_task(target, context)
                matrix_hb.step(extra=(
                    f"{target_name}<-{source_name}/"
                    f"{task_name}/baseline"))
                matrix_hb.step(extra=(
                    f"{target_name}<-{source_name}/"
                    f"{task_name}/source"))
                matrix_hb.step(extra=(
                    f"{target_name}<-{source_name}/"
                    f"{task_name}/base_path"))
                curves[task_name] = {}
                for layer, cell in full_curve(target, context).items():
                    curves[task_name][layer] = cell
                    matrix_hb.step(extra=(
                        f"{target_name}<-{source_name}/"
                        f"{task_name}/prefixL{layer}"))
                summaries[task_name] = _summary(curves[task_name])

            movement = _movement(
                originals[target_name]["original"],
                summaries["belief_to_search"][
                    "l24_minimum_mediation"],
                summaries["search_to_belief"][
                    "l24_minimum_mediation"])
            world_movement = _world_movements(
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
            verdict = _domain_verdict(
                tasks, summaries, movement, world_movement)
            functional = _functional_score(
                movement, list(tasks.values()))
            matrix[target_name][source_name] = {
                "source_controller": source_name,
                "target_domain": target_name,
                "tasks": tasks,
                "cumulative_prefix": curves,
                "summaries": summaries,
                "movement": movement,
                "per_world_movement": world_movement,
                "functional": functional["functional"],
                "functional_bidirectional_score": (
                    functional["functional_bidirectional_score"]),
                "verdict": verdict,
            }
            log(
                f"CONTROLLER MATRIX {target_name}<-{source_name} "
                f"verdict={verdict} movement={movement}")
    matrix_hb.done()

    original_gates = {}
    for target_name in DOMAINS:
        within = matrix[target_name][target_name]
        original_gates[target_name] = _original_gate_reason(
            within["tasks"], within["summaries"], within["movement"])
    eligible_targets = [
        name for name in DOMAINS
        if original_gates[name] == "ELIGIBLE"
    ]

    control_hb = Heartbeat(
        len(eligible_targets) * (
            2 * 4 + N_RANDOM * 2 * 4 + N_RANDOM * 2 * 4),
        "controller_matrix_controls",
        every_sec=30, out_dir=out_dir)
    l24_sites = _full_sites((22, 23, 24), n_heads)
    control_layers = (21, 22, 23, 24)

    def evaluate_control(target_name, label, displacement, positions):
        target = targets[target_name]
        values = {}
        public = []
        for operation, sign, direction in (
                ("belief", -1.0, "belief_to_search"),
                ("search", +1.0, "search_to_belief")):
            patch = _fixed_patch(
                target["states"][operation],
                positions, displacement, sign)
            context = build_context(
                target, operation, control_layers,
                sequence_patch=patch)
            task = public_task(target, context)
            public.append(task)
            control_hb.step(extra=(
                f"{target_name}/{label}/{direction}/baseline"))
            control_hb.step(extra=(
                f"{target_name}/{label}/{direction}/source"))
            control_hb.step(extra=(
                f"{target_name}/{label}/{direction}/base_path"))
            cell = l24_cell(target, context, l24_sites)
            values[direction] = float(
                cell["mediation"]["minimum_fraction"])
            control_hb.step(extra=(
                f"{target_name}/{label}/{direction}/L24"))
        return _functional_score(
            _movement(
                originals[target_name]["original"],
                values["belief_to_search"],
                values["search_to_belief"]),
            public)

    controls = {}
    for target_index, target_name in enumerate(eligible_targets):
        target = targets[target_name]
        controller = controllers[target_name]
        selected_score = matrix[target_name][target_name][
            "functional_bidirectional_score"]
        instruction = evaluate_control(
            target_name, "instruction", controller,
            target["alignment"]["instruction_positions"])

        random_directions = _norm_matched_directions(
            controller, seed=DIRECTION_SEED + 2003 * target_index)
        direction_cells = []
        for random_index, direction in enumerate(random_directions):
            cell = evaluate_control(
                target_name, f"random_direction{random_index}",
                direction, target["alignment"]["answer_positions"])
            direction_cells.append({
                "random_index": random_index,
                "per_position_norms": direction.norm(dim=-1).tolist(),
                **cell,
            })
        direction_scores = [
            cell["functional_bidirectional_score"]
            for cell in direction_cells
        ]
        direction_p, direction_exceed = _tail_probability(
            selected_score, direction_scores)

        position_sets = _random_position_sets(
            target["alignment"]["random_candidates"],
            seed=POSITION_SEED + 2011 * target_index)
        position_cells = []
        for random_index, positions in enumerate(position_sets):
            cell = evaluate_control(
                target_name, f"random_position{random_index}",
                controller, positions)
            position_cells.append({
                "random_index": random_index,
                "positions": positions,
                **cell,
            })
        position_scores = [
            cell["functional_bidirectional_score"]
            for cell in position_cells
        ]
        position_p, position_exceed = _tail_probability(
            selected_score, position_scores)
        specific = bool(
            _matrix_cell_pass(matrix[target_name][target_name])
            and direction_p <= 0.05 + 1e-12
            and position_p <= 0.05 + 1e-12
            and instruction["functional_bidirectional_score"]
            < 0.5 * selected_score - 1e-12)
        controls[target_name] = {
            "selected_score": selected_score,
            "instruction": instruction,
            "random_direction": {
                "n_random": N_RANDOM,
                "seed": DIRECTION_SEED + 2003 * target_index,
                "empirical_p": direction_p,
                "exceed_count": direction_exceed,
                "cells": direction_cells,
            },
            "random_position": {
                "n_random": N_RANDOM,
                "seed": POSITION_SEED + 2011 * target_index,
                "empirical_p": position_p,
                "exceed_count": position_exceed,
                "cells": position_cells,
            },
            "specific": specific,
        }
    control_hb.done()

    verdict = _adjudicate(matrix, controls, original_gates)
    result = {
        "stage": "delta_controller_matrix",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "claim": (
            "Domain-specific donor-only answer-prefix controllers are "
            "tested within and across held-out state domains."),
        "splits": {
            "location": {
                "indices_from_30": location_indices,
                "donor_indices": location_indices[:DONOR_N],
                "evaluation_indices": location_indices[DONOR_N:],
            },
            "ownership": {
                "donor_rows": domain_rows["ownership"][:DONOR_N],
                "evaluation_rows": domain_rows["ownership"][DONOR_N:],
            },
            "color_state": {
                "donor_rows": domain_rows["color_state"][:DONOR_N],
                "evaluation_rows": domain_rows["color_state"][DONOR_N:],
            },
        },
        "controllers": controller_metadata,
        "controller_archive": {
            "artifact": os.path.basename(controller_path),
            "sha256": controller_archive_sha,
        },
        "geometry": _controller_geometry(controllers),
        "original_gates": original_gates,
        "matrix_orientation": "matrix[target_domain][source_controller]",
        "matrix": matrix,
        "controls": controls,
        "layers": list(LAYERS),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir, f"results_delta_controller_matrix_{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"CONTROLLER MATRIX verdict={verdict} "
        f"original_gates={original_gates} artifact={path}")
    return result
