"""Held-out causal decomposition into a shared backbone and color adapter."""
from __future__ import annotations

import hashlib
import io
import itertools
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
from .delta_sparse_transport import _attention_geometry
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

PROTOCOL_VERSION = "2026-07-25-p2-shared-adapter-decomposition-v1"
PROTOCOL_SHA256 = (
    "4FB9E6BAA956CADD68F11F6C05BC88A56ECD17CAED3905BA038C2D508AEB13CB")
DONOR_N = 15
DOSES = (0.0, 0.25, 0.50, 0.75, 1.0)
MONOTONIC_TOLERANCE = 0.005


def _fresh_color_rows_v2():
    values = DOMAIN_SPECS["color_state"]["values"]
    prior_rows = (
        _domain_rows(values)
        + _fresh_domain_rows(values)
        + _fresh_color_rows()
    )
    used = {
        (state, row["d1"], row["d2"])
        for row in prior_rows
        for state in (row["source"], row["target"])
    }
    rows = []
    for index in range(TEST_N):
        source_index = index % len(values)
        shift = 4 + index // len(values)
        source = values[source_index]
        target = values[(source_index + shift) % len(values)]
        remaining = [
            values[(source_index + offset) % len(values)]
            for offset in range(1, len(values) + 1)
            if values[(source_index + offset) % len(values)]
            not in (source, target)
        ]
        for d1, d2 in itertools.permutations(remaining, 2):
            clean_signature = (source, d1, d2)
            natural_signature = (target, d1, d2)
            if clean_signature not in used and natural_signature not in used:
                used.add(clean_signature)
                used.add(natural_signature)
                break
        else:
            raise AssertionError(
                f"could not construct second fresh color history {index}")
        rows.append({
            "row_index": index,
            "source": source,
            "target": target,
            "state": source,
            "d1": d1,
            "d2": d2,
        })
    prompt_signatures = {
        (state, row["d1"], row["d2"])
        for row in rows
        for state in (row["source"], row["target"])
    }
    if len(prompt_signatures) != 2 * TEST_N:
        raise AssertionError("second fresh prompt histories are not unique")
    return rows


def _color_decomposition(shared, color):
    shared_flat = shared.flatten().float()
    color_flat = color.flatten().float()
    alpha = torch.dot(color_flat, shared_flat) / torch.dot(
        shared_flat, shared_flat).clamp_min(EPS)
    projection = alpha * shared.float()
    residual = color.float() - projection
    orthogonality = float(
        torch.dot(projection.flatten(), residual.flatten())
        / (projection.norm() * residual.norm()).clamp_min(EPS))
    color_energy = color.float().square().sum().clamp_min(EPS)
    return projection, residual, {
        "projection_scale_on_shared_pc1": float(alpha),
        "color_norm": float(color.norm()),
        "projection_norm": float(projection.norm()),
        "residual_norm": float(residual.norm()),
        "projection_energy_fraction": float(
            projection.square().sum() / color_energy),
        "residual_energy_fraction": float(
            residual.square().sum() / color_energy),
        "projection_residual_cosine": orthogonality,
        "per_position_residual_norms": residual.norm(dim=-1).tolist(),
        "per_position_residual_fraction_of_color_norm": (
            residual.norm(dim=-1)
            / color.float().norm(dim=-1).clamp_min(EPS)).tolist(),
    }


def _orthogonal_norm_matched_residuals(shared, residual, n_random=N_RANDOM,
                                       seed=DIRECTION_SEED + 6001):
    rng = np.random.default_rng(int(seed))
    controls = []
    shape = tuple(int(value) for value in residual.shape)
    target_norms = residual.norm(dim=-1)
    for _index in range(int(n_random)):
        value = torch.from_numpy(
            rng.standard_normal(shape).astype(np.float32))
        for position in range(value.shape[0]):
            axis = shared[position].float()
            value[position] -= (
                torch.dot(value[position], axis)
                / torch.dot(axis, axis).clamp_min(EPS)
            ) * axis
            value[position] = (
                value[position]
                / value[position].norm().clamp_min(EPS)
                * target_norms[position])
        controls.append(value)
    return controls


def _two_locus_patch(states, primary_positions, primary,
                     secondary_positions, secondary, sign):
    patches = []
    for arm_states in states:
        positions = list(primary_positions) + list(secondary_positions)
        values = torch.cat([
            arm_states[:, primary_positions, :]
            + float(sign) * primary.unsqueeze(0),
            arm_states[:, secondary_positions, :]
            + float(sign) * secondary.unsqueeze(0),
        ], dim=1)
        patches.append((positions, values))
    return tuple(patches)


@torch.no_grad()
def run_delta_shared_adapter_decomposition(
        model_path, out_dir,
        model_key="qwen7b_shared_adapter_decomposition",
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
        raise ValueError("shared-adapter layers are absent")
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
    shared_pc1, donor_mean, shared_geometry = _shared_controllers(
        donor_controllers["location"], donor_controllers["ownership"])
    projection, residual, decomposition = _color_decomposition(
        shared_pc1, donor_controllers["color"])

    arms = {
        "shared_pc1": shared_pc1,
        "color_projection": projection,
        "color_residual": residual,
        "projection_plus_025_residual": projection + 0.25 * residual,
        "projection_plus_050_residual": projection + 0.50 * residual,
        "projection_plus_075_residual": projection + 0.75 * residual,
        "projection_plus_100_residual": projection + residual,
    }
    controller_path = os.path.join(
        out_dir, f"shared_adapter_controllers_{model_key}.npz")
    np.savez(
        controller_path,
        location=donor_controllers["location"].numpy(),
        ownership=donor_controllers["ownership"].numpy(),
        color=donor_controllers["color"].numpy(),
        donor_mean=donor_mean.numpy(),
        **{name: value.numpy() for name, value in arms.items()})
    with open(controller_path, "rb") as handle:
        archive_sha = hashlib.sha256(handle.read()).hexdigest().upper()

    evaluation_rows = _fresh_color_rows_v2()
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
        2 * (3 + len(LAYERS)) + len(arms) * 2 * (3 + len(LAYERS)),
        "shared_adapter_full_curves", every_sec=30, out_dir=out_dir)
    original_tasks = {}
    original_curves = {}
    original_summaries = {}
    for operation in ("belief", "search"):
        task_name = f"{operation}_original"
        context = build_context(operation, CAPTURE_LAYERS)
        original_tasks[task_name] = _public_generic_task(context)
        for label in ("baseline", "source", "base_path"):
            hb.step(extra=f"{task_name}/{label}")
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
    for arm_name, controller in arms.items():
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
            for label in ("baseline", "source", "base_path"):
                hb.step(extra=f"{arm_name}/{task_name}/{label}")
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
            f"SHARED-ADAPTER {arm_name} verdict={verdict} "
            f"movement={movement} signs="
            f"{world['belief_to_search']['successes']}/"
            f"{world['search_to_belief']['successes']}")
    hb.done()

    control_layers = (21, 22, 23, 24)
    l24_sites = _full_sites((22, 23, 24), n_heads)
    position_subsets = (
        (0,), (1,), (2,), (0, 1), (0, 2), (1, 2),
    )
    control_hb = Heartbeat(
        len(position_subsets) * 2 * 4
        + 2 * 4 + N_RANDOM * 2 * 4 + N_RANDOM * 2 * 4,
        "shared_adapter_controls", every_sec=30, out_dir=out_dir)

    def evaluate_patch(label, make_patch):
        values = {}
        public = []
        for operation, sign, direction in (
                ("belief", -1.0, "belief_to_search"),
                ("search", +1.0, "search_to_belief")):
            context = build_context(
                operation, control_layers,
                sequence_patch=make_patch(operation, sign))
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

    subset_cells = []
    for subset in position_subsets:
        masked_residual = torch.zeros_like(residual)
        masked_residual[list(subset)] = residual[list(subset)]
        controller = projection + masked_residual
        cell = evaluate_patch(
            f"residual_subset_{'_'.join(map(str, subset))}",
            lambda operation, sign, value=controller: _fixed_patch(
                states[operation], alignment["answer_positions"],
                value, sign))
        subset_cells.append({
            "residual_position_offsets": list(subset),
            **cell,
        })

    def two_locus(secondary_positions, secondary):
        return lambda operation, sign: _two_locus_patch(
            states[operation],
            alignment["answer_positions"], projection,
            secondary_positions, secondary, sign)

    instruction = evaluate_patch(
        "instruction_residual",
        two_locus(alignment["instruction_positions"], residual))

    random_residuals = _orthogonal_norm_matched_residuals(
        shared_pc1, residual)
    direction_cells = []
    for index, random_residual in enumerate(random_residuals):
        cell = evaluate_patch(
            f"random_residual_{index}",
            lambda operation, sign, value=(
                    projection + random_residual): _fixed_patch(
                states[operation], alignment["answer_positions"],
                value, sign))
        direction_cells.append({
            "random_index": index,
            "per_position_norms": random_residual.norm(dim=-1).tolist(),
            "per_position_shared_cosines": [
                float(torch.dot(random_residual[p], shared_pc1[p])
                      / (random_residual[p].norm()
                         * shared_pc1[p].norm()).clamp_min(EPS))
                for p in range(random_residual.shape[0])
            ],
            **cell,
        })

    position_sets = _random_position_sets(
        alignment["random_candidates"], seed=POSITION_SEED + 6001)
    position_cells = []
    for index, positions in enumerate(position_sets):
        cell = evaluate_patch(
            f"random_residual_position_{index}",
            two_locus(positions, residual))
        position_cells.append({
            "random_index": index,
            "positions": positions,
            **cell,
        })
    control_hb.done()

    selected_score = arm_results["projection_plus_100_residual"][
        "functional_bidirectional_score"]
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
        direction_p <= 0.05 + 1e-12
        and position_p <= 0.05 + 1e-12
        and instruction["functional_bidirectional_score"]
        < 0.5 * selected_score - 1e-12)
    controls = {
        "selected_score": selected_score,
        "residual_position_subsets": subset_cells,
        "instruction_residual_on_projection": instruction,
        "orthogonal_random_residual": {
            "n_random": N_RANDOM,
            "seed": DIRECTION_SEED + 6001,
            "empirical_p": direction_p,
            "exceed_count": direction_exceed,
            "cells": direction_cells,
        },
        "random_residual_position": {
            "n_random": N_RANDOM,
            "seed": POSITION_SEED + 6001,
            "empirical_p": position_p,
            "exceed_count": position_exceed,
            "cells": position_cells,
        },
        "specific": specific,
    }

    dose_names = [
        "color_projection",
        "projection_plus_025_residual",
        "projection_plus_050_residual",
        "projection_plus_075_residual",
        "projection_plus_100_residual",
    ]
    dose_scores = [
        arm_results[name]["functional_bidirectional_score"]
        for name in dose_names
    ]
    monotonic = all(
        right + MONOTONIC_TOLERANCE >= left
        for left, right in zip(dose_scores, dose_scores[1:]))
    projection_pass = arm_results["color_projection"]["verdict"] == "PASS"
    residual_pass = arm_results["color_residual"]["verdict"] == "PASS"
    full_pass = (
        arm_results["projection_plus_100_residual"]["verdict"] == "PASS")
    component_best = max(
        arm_results["color_projection"]["functional_bidirectional_score"],
        arm_results["color_residual"]["functional_bidirectional_score"])
    synergy = bool(selected_score >= component_best + 0.02 - 1e-12)
    target_gate = _original_gate(
        original_tasks, original_summaries,
        arm_results["projection_plus_100_residual"]["movement"])

    if target_gate != "ELIGIBLE" or not full_pass:
        verdict = "COLOR_TARGET_UNRESOLVED"
    elif residual_pass:
        verdict = "DOMAIN_ADAPTER_SUFFICIENT"
    elif projection_pass:
        verdict = "SHARED_PROJECTION_SUFFICIENT"
    elif specific and synergy and monotonic:
        verdict = "COMPOSITIONAL_SHARED_PLUS_DOMAIN_ADAPTER"
    elif specific and synergy:
        verdict = "NONMONOTONIC_COMPOSITIONAL_RESCUE"
    else:
        verdict = "NO_SHARED_ADAPTER_COMPOSITION"

    result = {
        "stage": "delta_shared_adapter_decomposition",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "claim": (
            "A color controller is causally decomposed into a shared "
            "cross-domain projection and an orthogonal target adapter."),
        "splits": {
            "location_donor_indices": location_indices[:DONOR_N],
            "ownership_donor_rows": ownership_rows,
            "color_donor_rows": color_donor_rows,
            "second_fresh_color_evaluation_rows": evaluation_rows,
            "evaluation_excluded_from_construction": True,
        },
        "construction": {
            "shared_geometry": shared_geometry,
            "decomposition": decomposition,
            "doses": list(DOSES),
        },
        "controllers": {
            **{
                name: _controller_metadata(value)
                for name, value in donor_controllers.items()
            },
            **{
                name: _controller_metadata(value)
                for name, value in arms.items()
            },
        },
        "controller_archive": {
            "artifact": os.path.basename(controller_path),
            "sha256": archive_sha,
        },
        "prospective_gate": {
            "n_world": TEST_N,
            "minimum_positive_per_direction": SIGN_MINIMUM,
            "maximum_exact_one_sided_sign_p": SIGN_P_MAX,
            "monotonic_tolerance": MONOTONIC_TOLERANCE,
            "minimum_synergy_margin": 0.02,
        },
        "target_original_gate": target_gate,
        "arms": arm_results,
        "dose_response": {
            "arm_order": dose_names,
            "bidirectional_scores": dose_scores,
            "monotonic_with_tolerance": monotonic,
            "synergy_over_best_component": (
                selected_score - component_best),
            "synergy_pass": synergy,
        },
        "controls": controls,
        "layers": list(LAYERS),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_shared_adapter_decomposition_{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"SHARED-ADAPTER verdict={verdict} target={target_gate} "
        f"projection={arm_results['color_projection']['verdict']} "
        f"residual={arm_results['color_residual']['verdict']} "
        f"full={arm_results['projection_plus_100_residual']['verdict']} "
        f"specific={specific} monotonic={monotonic} synergy={synergy} "
        f"artifact={path}")
    return result
