"""Prospective causal-rank census of natural answer-prefix controllers."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from .delta_anchor_write import _resolve
from .delta_content_cancelled_controller import (
    _fixed_patch,
    _functional_score,
    _movement,
    _norm_matched_directions,
    _world_mediation,
    _world_movements,
)
from .delta_controller_matrix import (
    _controller_from_alignment,
    _fresh_domain_rows,
)
from .delta_cross_domain_controller import (
    DOMAIN_SPECS,
    _domain_alignment,
    _generic_evaluate_sites,
    _generic_task_context,
    _public_generic_task,
)
from .delta_distributed_label_transplant import _capture_full_l21
from .delta_endogenous_controller_factorial import (
    _carrier_spec,
    _fresh_color_rows_v4,
)
from .delta_lexical_class import LABELS, TASK_GRID, _padding_plan
from .delta_operation_handoff_depth import _evaluate_sites, _full_sites
from .delta_preprint_battery import _compatible_world_rows
from .delta_source_head_mediation import _public_task, _task_context
from .delta_sparse_transport import _attention_geometry
from .delta_synonym_prefix_transfer import _alignment as _lexical_alignment
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)


PROTOCOL_VERSION = "2026-07-26-p2-causal-rank-spectrum-v1"
DONOR_N = 15
LOCATION_TEST_N = 15
COLOR_TEST_N = 30
PATCH_WIDTH = 3
CONTROL_LAYERS = (21, 22, 23, 24)
RECONSTRUCTION_RANKS = (1, 2, 3, 4, 6, 11)
INDIVIDUAL_AXES = (1, 2, 3, 4, 5, 6)
N_RANDOM = 3
RANDOM_SEED = 82103
MINIMUM_REFERENCE_SCORE = 0.03
MINIMUM_RECOVERY_FRACTION = 0.80
MINIMUM_AXIS_FRACTION = 0.20
MINIMUM_AXIS_ABSOLUTE_SCORE = 0.015
MINIMUM_POSITIVE_WORLD_FRACTION = 0.80

PROTOCOL = {
    "version": PROTOCOL_VERSION,
    "basis": (
        "uncentered SVD of 9 donor-only epistemic/search lexical-pair "
        "controllers plus ownership and color controllers"),
    "basis_size": 11,
    "donor_n": DONOR_N,
    "targets": {
        "location": LOCATION_TEST_N,
        "color_state": COLOR_TEST_N,
    },
    "reconstruction_ranks": list(RECONSTRUCTION_RANKS),
    "individual_axes": list(INDIVIDUAL_AXES),
    "tail_axis_group": [7, 8, 9, 10, 11],
    "random_controls_per_target": N_RANDOM,
    "decision": {
        "minimum_reference_score": MINIMUM_REFERENCE_SCORE,
        "minimum_recovery_fraction": MINIMUM_RECOVERY_FRACTION,
        "minimum_axis_fraction": MINIMUM_AXIS_FRACTION,
        "minimum_axis_absolute_score": MINIMUM_AXIS_ABSOLUTE_SCORE,
        "minimum_positive_world_fraction":
            MINIMUM_POSITIVE_WORLD_FRACTION,
    },
}
PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
).hexdigest().upper()


def _lexical_controllers(model, alignment):
    epistemic = LABELS["epistemic"]
    search = LABELS["search"]
    states = {}
    for label in (*epistemic, *search):
        query = (
            TASK_GRID["epistemic"][label]
            if label in epistemic
            else TASK_GRID["search"][label]
        )
        states[label] = [
            _capture_full_l21(model, batch)
            for batch in alignment["batches"][query]
        ]
    positions = alignment["answer_positions"]
    controllers = {}
    for e_label in epistemic:
        for s_label in search:
            samples = []
            for e_state, s_state in zip(
                    states[e_label], states[s_label]):
                samples.extend(
                    e_state[:, positions, :]
                    - s_state[:, positions, :])
            controllers[f"{e_label}__{s_label}"] = (
                torch.stack(samples).mean(dim=0))
    return controllers, states


def _controller_basis(controllers):
    names = list(controllers)
    matrix = torch.stack([
        controllers[name].flatten().float()
        for name in names
    ])
    _u, singular, vh = torch.linalg.svd(
        matrix, full_matrices=False)
    mean = matrix.mean(dim=0)
    for index in range(vh.shape[0]):
        if float(torch.dot(vh[index], mean)) < 0.0:
            vh[index] *= -1.0
    coefficients = matrix @ vh.T
    energy = singular.square()
    return names, singular, vh, coefficients, {
        "singular_values": singular.tolist(),
        "energy_fraction": (energy / energy.sum()).tolist(),
        "cumulative_energy": (
            torch.cumsum(energy, dim=0) / energy.sum()).tolist(),
        "controller_coefficients": {
            name: coefficients[index].tolist()
            for index, name in enumerate(names)
        },
    }


def _component(controller, basis, axis_indices):
    flat = controller.flatten().float()
    coefficients = flat @ basis.T
    selected = [int(index) - 1 for index in axis_indices]
    value = (
        coefficients[selected].unsqueeze(0)
        @ basis[selected]
    ).reshape_as(controller)
    return value, {
        "axis_indices_one_based": list(axis_indices),
        "coefficients": coefficients[selected].tolist(),
        "frobenius_norm": float(value.norm()),
        "fraction_of_controller_energy": float(
            value.square().sum()
            / controller.float().square().sum().clamp_min(1e-8)),
        "cosine_with_controller": float(
            torch.dot(value.flatten(), controller.flatten().float())
            / (value.norm() * controller.float().norm()).clamp_min(1e-8)),
    }


def _natural_patch_pair(states, positions):
    return {
        "belief": tuple(
            (list(positions), value[:, positions, :].clone())
            for value in states["search"]),
        "search": tuple(
            (list(positions), value[:, positions, :].clone())
            for value in states["belief"]),
    }


def _positive_worlds(world):
    return sum(
        bool(row.get("predicted_sign"))
        for row in world["rows"])


@torch.no_grad()
def run_delta_causal_rank_spectrum(
        model_path, out_dir,
        model_key="qwen7b_causal_rank_spectrum",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=30):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_world) != 30:
        raise ValueError("v1 is frozen to exactly 30 location worlds")
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    if max(CONTROL_LAYERS) >= model_num_hidden_layers(model):
        raise ValueError("causal-rank layers are absent")
    n_heads, head_dim = _attention_geometry(model)
    l24_sites = _full_sites((22, 23, 24), n_heads)

    location_rows, location_indices = _compatible_world_rows(
        tok, torch.device("cpu"), int(n_world))
    if len(location_rows) != 30:
        raise ValueError("all 30 compatible location worlds are required")
    target_position, padding_plan, tokenization_tables = _padding_plan(
        tok, location_rows[0])
    if padding_plan is None:
        raise ValueError("lexical padding plan is absent")
    location_donor_rows = location_rows[:DONOR_N]
    location_test_rows = location_rows[DONOR_N:]
    location_donor_alignment = _lexical_alignment(
        tok, dev, location_donor_rows)
    location_test_alignment = _lexical_alignment(
        tok, dev, location_test_rows)
    if target_position != location_donor_alignment["readout"]:
        raise ValueError("lexical donor readout position changed")
    if (location_donor_alignment["answer_positions"]
            != location_test_alignment["answer_positions"]):
        raise ValueError("location donor/test answer positions differ")

    lexical_controllers, _donor_lexical_states = _lexical_controllers(
        model, location_donor_alignment)
    ownership_rows = _fresh_domain_rows(
        DOMAIN_SPECS["ownership"]["values"])[:DONOR_N]
    color_donor_rows = _fresh_domain_rows(
        DOMAIN_SPECS["color_state"]["values"])[:DONOR_N]
    ownership_alignment = _domain_alignment(
        tok, dev, ownership_rows, DOMAIN_SPECS["ownership"])
    color_donor_alignment = _domain_alignment(
        tok, dev, color_donor_rows, DOMAIN_SPECS["color_state"])
    ownership_controller, _ownership_samples = (
        _controller_from_alignment(model, ownership_alignment))
    color_controller, _color_samples = _controller_from_alignment(
        model, color_donor_alignment)
    controllers = {
        **lexical_controllers,
        "ownership": ownership_controller,
        "color_state": color_controller,
    }
    if len(controllers) != 11:
        raise AssertionError("causal-rank basis must contain 11 controllers")
    basis_names, singular, basis, coefficients, geometry = (
        _controller_basis(controllers))

    basis_path = os.path.join(
        out_dir, f"causal_rank_basis_{model_key}.npz")
    np.savez(
        basis_path,
        basis=basis.numpy(),
        singular_values=singular.numpy(),
        coefficients=coefficients.numpy(),
        **{
            f"controller_{name}": value.numpy()
            for name, value in controllers.items()
        })
    with open(basis_path, "rb") as handle:
        basis_sha = hashlib.sha256(handle.read()).hexdigest().upper()

    # Evaluation targets.
    anchor_belief = TASK_GRID["epistemic"]["BELIEF"]
    anchor_search = TASK_GRID["search"]["SEARCH"]
    location_states = {
        "belief": [
            _capture_full_l21(model, batch)
            for batch in location_test_alignment["batches"][anchor_belief]
        ],
        "search": [
            _capture_full_l21(model, batch)
            for batch in location_test_alignment["batches"][anchor_search]
        ],
    }
    color_rows = _fresh_color_rows_v4()
    color_spec = _carrier_spec()
    color_alignment = _domain_alignment(
        tok, dev, color_rows, color_spec)
    color_states = {"belief": [], "search": []}
    for belief_batch, search_batch in color_alignment["batches"]:
        color_states["belief"].append(
            _capture_full_l21(model, belief_batch))
        color_states["search"].append(
            _capture_full_l21(model, search_batch))

    targets = {
        "location": {
            "kind": "location",
            "rows": location_test_rows,
            "positions": location_test_alignment["answer_positions"],
            "states": location_states,
            "controller": controllers["BELIEF__SEARCH"],
            "n_world": LOCATION_TEST_N,
        },
        "color_state": {
            "kind": "generic",
            "rows": color_rows,
            "positions": color_alignment["answer_positions"],
            "states": color_states,
            "controller": color_controller,
            "alignment": color_alignment,
            "spec": color_spec,
            "source": [row["source"] for row in color_rows],
            "target": [row["target"] for row in color_rows],
            "n_world": COLOR_TEST_N,
        },
    }

    def build_context(target, operation, patch=None):
        if target["kind"] == "location":
            query = anchor_belief if operation == "belief" else anchor_search
            return _task_context(
                model, tok, dev, target["rows"], query,
                CONTROL_LAYERS, head_dim, sequence_patch=patch)
        batch_index = 0 if operation == "belief" else 1
        clean = target["alignment"]["batches"][0][batch_index]
        natural = target["alignment"]["batches"][1][batch_index]
        return _generic_task_context(
            model, clean, natural, list(target["spec"]["values"]),
            target["source"], target["target"],
            CONTROL_LAYERS, head_dim, sequence_patch=patch)

    def public_task(target, context):
        return (
            _public_task(context)
            if target["kind"] == "location"
            else _public_generic_task(context))

    def evaluate_cell(target, context):
        return (
            _evaluate_sites(model, context, l24_sites, head_dim)
            if target["kind"] == "location"
            else _generic_evaluate_sites(
                model, context, l24_sites, head_dim))

    # Precompute all frozen arms before evaluating any causal outcome.
    target_arms = {}
    component_metadata = {}
    for target_name, target in targets.items():
        arms = {"original": {"belief": None, "search": None}}
        natural = _natural_patch_pair(
            target["states"], target["positions"])
        arms["natural_prefix_interchange"] = natural
        component_metadata[target_name] = {}
        for rank in RECONSTRUCTION_RANKS:
            component, metadata = _component(
                target["controller"], basis, range(1, rank + 1))
            arm_name = f"rank_{rank:02d}_reconstruction"
            arms[arm_name] = {
                "belief": _fixed_patch(
                    target["states"]["belief"], target["positions"],
                    component, -1.0),
                "search": _fixed_patch(
                    target["states"]["search"], target["positions"],
                    component, +1.0),
            }
            component_metadata[target_name][arm_name] = metadata

        if target_name == "color_state":
            for axis in INDIVIDUAL_AXES:
                component, metadata = _component(
                    target["controller"], basis, [axis])
                arm_name = f"axis_{axis:02d}_alone"
                arms[arm_name] = {
                    "belief": _fixed_patch(
                        target["states"]["belief"], target["positions"],
                        component, -1.0),
                    "search": _fixed_patch(
                        target["states"]["search"], target["positions"],
                        component, +1.0),
                }
                component_metadata[target_name][arm_name] = metadata
            tail_component, tail_metadata = _component(
                target["controller"], basis, range(7, 12))
            arms["axes_07_11_tail"] = {
                "belief": _fixed_patch(
                    target["states"]["belief"], target["positions"],
                    tail_component, -1.0),
                "search": _fixed_patch(
                    target["states"]["search"], target["positions"],
                    tail_component, +1.0),
            }
            component_metadata[target_name][
                "axes_07_11_tail"] = tail_metadata

        random_directions = _norm_matched_directions(
            target["controller"], n_random=N_RANDOM,
            seed=RANDOM_SEED + (0 if target_name == "location" else 1009))
        for index, direction in enumerate(random_directions):
            arm_name = f"random_direction_{index:02d}"
            arms[arm_name] = {
                "belief": _fixed_patch(
                    target["states"]["belief"], target["positions"],
                    direction, -1.0),
                "search": _fixed_patch(
                    target["states"]["search"], target["positions"],
                    direction, +1.0),
            }
            component_metadata[target_name][arm_name] = {
                "frobenius_norm": float(direction.norm()),
                "per_position_norms": direction.norm(dim=-1).tolist(),
                "seed": (
                    RANDOM_SEED
                    + (0 if target_name == "location" else 1009)),
            }
        target_arms[target_name] = arms

    total_steps = sum(
        len(arms) * 2 * 4
        for arms in target_arms.values())
    hb = Heartbeat(
        total_steps, "causal_rank_spectrum",
        every_sec=30, out_dir=out_dir)
    results = {}
    for target_name, target in targets.items():
        results[target_name] = {}
        for arm_name, patches in target_arms[target_name].items():
            tasks = {}
            cells = {}
            values = {}
            for operation in ("belief", "search"):
                context = build_context(
                    target, operation, patches[operation])
                tasks[operation] = public_task(target, context)
                for phase in ("baseline", "source", "base_path"):
                    hb.step(extra=(
                        f"{target_name}/{arm_name}/{operation}/{phase}"))
                cells[operation] = evaluate_cell(target, context)
                values[operation] = float(
                    cells[operation]["mediation"]["minimum_fraction"])
                hb.step(extra=(
                    f"{target_name}/{arm_name}/{operation}/L24"))
            results[target_name][arm_name] = {
                "tasks": tasks,
                "l24_cells": cells,
                "values": values,
            }
    hb.done()

    # Score every intervention against untouched operation-specific routes.
    adjudication = {}
    for target_name, target in targets.items():
        original = results[target_name]["original"]
        original_summary = {
            operation: {
                "l24_minimum_mediation": original["values"][operation]}
            for operation in ("belief", "search")
        }
        all_functional = True
        for arm_name, arm in results[target_name].items():
            if arm_name == "original":
                continue
            movement = _movement(
                original_summary,
                arm["values"]["belief"], arm["values"]["search"])
            world = _world_movements(
                _world_mediation(
                    original["tasks"]["belief"],
                    original["l24_cells"]["belief"]),
                _world_mediation(
                    original["tasks"]["search"],
                    original["l24_cells"]["search"]),
                _world_mediation(
                    arm["tasks"]["belief"],
                    arm["l24_cells"]["belief"]),
                _world_mediation(
                    arm["tasks"]["search"],
                    arm["l24_cells"]["search"]))
            functional = _functional_score(
                movement,
                [
                    *original["tasks"].values(),
                    *arm["tasks"].values(),
                ])
            arm["movement"] = movement
            arm["per_world_movement"] = world
            arm["positive_worlds"] = _positive_worlds(world)
            arm["functional"] = functional["functional"]
            arm["functional_bidirectional_score"] = (
                functional["functional_bidirectional_score"])
            all_functional = bool(
                all_functional and functional["functional"])

        full_name = "rank_11_reconstruction"
        full_score = results[target_name][full_name][
            "functional_bidirectional_score"]
        minimum_positive_worlds = int(np.ceil(
            MINIMUM_POSITIVE_WORLD_FRACTION * target["n_world"]))
        cumulative = {}
        minimum_rank = None
        for rank in RECONSTRUCTION_RANKS:
            arm_name = f"rank_{rank:02d}_reconstruction"
            arm = results[target_name][arm_name]
            recovery = (
                arm["functional_bidirectional_score"] / full_score
                if full_score > 1e-8 else None)
            cumulative[str(rank)] = {
                "functional_bidirectional_score":
                    arm["functional_bidirectional_score"],
                "recovery_fraction_of_full": recovery,
                "positive_worlds": arm["positive_worlds"],
            }
            if (minimum_rank is None
                    and recovery is not None
                    and recovery >= MINIMUM_RECOVERY_FRACTION
                    and arm["movement"]["belief_to_search_movement"] > 0.0
                    and arm["movement"]["search_to_belief_movement"] > 0.0
                    and arm["positive_worlds"]
                    >= minimum_positive_worlds):
                minimum_rank = rank

        individual = {}
        active_axes = []
        if target_name == "color_state":
            for axis in INDIVIDUAL_AXES:
                arm_name = f"axis_{axis:02d}_alone"
                arm = results[target_name][arm_name]
                fraction = (
                    arm["functional_bidirectional_score"] / full_score
                    if full_score > 1e-8 else None)
                active = bool(
                    arm["functional"]
                    and arm["functional_bidirectional_score"]
                    >= MINIMUM_AXIS_ABSOLUTE_SCORE
                    and fraction is not None
                    and fraction >= MINIMUM_AXIS_FRACTION
                    and arm["movement"]["belief_to_search_movement"] > 0.0
                    and arm["movement"]["search_to_belief_movement"] > 0.0
                    and arm["positive_worlds"]
                    >= minimum_positive_worlds)
                individual[str(axis)] = {
                    "functional_bidirectional_score":
                        arm["functional_bidirectional_score"],
                    "fraction_of_full": fraction,
                    "positive_worlds": arm["positive_worlds"],
                    "active": active,
                }
                if active:
                    active_axes.append(axis)
            tail = results[target_name]["axes_07_11_tail"]
            tail_fraction = (
                tail["functional_bidirectional_score"] / full_score
                if full_score > 1e-8 else None)
            tail_active = bool(
                tail["functional"]
                and tail["functional_bidirectional_score"]
                >= MINIMUM_AXIS_ABSOLUTE_SCORE
                and tail_fraction is not None
                and tail_fraction >= MINIMUM_AXIS_FRACTION
                and tail["movement"]["belief_to_search_movement"] > 0.0
                and tail["movement"]["search_to_belief_movement"] > 0.0
                and tail["positive_worlds"]
                >= minimum_positive_worlds)
            individual["7_11_tail"] = {
                "functional_bidirectional_score":
                    tail["functional_bidirectional_score"],
                "fraction_of_full": tail_fraction,
                "positive_worlds": tail["positive_worlds"],
                "active": tail_active,
            }

        natural_score = results[target_name][
            "natural_prefix_interchange"][
                "functional_bidirectional_score"]
        random_scores = [
            results[target_name][f"random_direction_{index:02d}"][
                "functional_bidirectional_score"]
            for index in range(N_RANDOM)
        ]
        adjudication[target_name] = {
            "all_arms_functional": all_functional,
            "full_reconstruction_score": full_score,
            "natural_transplant_score": natural_score,
            "minimum_rank_for_80pct_full": minimum_rank,
            "cumulative_rank": cumulative,
            "individual_axes": individual,
            "active_individual_axes": active_axes,
            "tail_active": bool(
                individual.get("7_11_tail", {}).get("active", False)),
            "random_scores": random_scores,
        }

    color = adjudication["color_state"]
    location = adjudication["location"]
    if not (
            color["all_arms_functional"]
            and location["all_arms_functional"]):
        verdict = "BEHAVIOR_OR_SOURCE_INELIGIBLE"
    elif min(
            color["full_reconstruction_score"],
            location["full_reconstruction_score"],
            color["natural_transplant_score"],
            location["natural_transplant_score"],
    ) < MINIMUM_REFERENCE_SCORE:
        verdict = "REFERENCE_ROUTE_SWITCH_WEAK"
    elif (
            location["minimum_rank_for_80pct_full"] == 1
            and color["minimum_rank_for_80pct_full"] == 1
            and len(color["active_individual_axes"]) <= 1
            and not color["tail_active"]):
        verdict = "SINGLE_SHARED_CAUSAL_AXIS"
    elif (
            color["minimum_rank_for_80pct_full"] is not None
            and color["minimum_rank_for_80pct_full"] <= 4
            and (
                len(color["active_individual_axes"]) >= 2
                or color["tail_active"])
            and location["minimum_rank_for_80pct_full"] is not None
            and location["minimum_rank_for_80pct_full"] <= 4):
        verdict = "LOW_RANK_STRUCTURED_CAUSAL_SUBSPACE"
    elif (
            color["minimum_rank_for_80pct_full"] is None
            or color["minimum_rank_for_80pct_full"] > 4):
        verdict = "HIGH_RANK_OR_DOMAIN_SPECIFIC_CONTROL"
    else:
        verdict = "CAUSAL_RANK_UNRESOLVED"

    result = {
        "stage": "delta_causal_rank_spectrum",
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "claim": (
            "Natural answer-prefix contrasts span a causally tested spectrum "
            "that distinguishes one switch, a small structured control "
            "subspace, and high-rank task-specific control."),
        "splits": {
            "location_indices": location_indices,
            "location_donor_indices": location_indices[:DONOR_N],
            "location_test_indices": location_indices[DONOR_N:],
            "ownership_donor_rows": ownership_rows,
            "color_donor_rows": color_donor_rows,
            "color_test_rows": color_rows,
            "causal_test_excluded_from_basis_construction": True,
        },
        "padding_plan": padding_plan,
        "tokenization_tables": tokenization_tables,
        "basis": {
            "controller_names": basis_names,
            "geometry": geometry,
            "artifact": os.path.basename(basis_path),
            "sha256": basis_sha,
        },
        "target_component_metadata": component_metadata,
        "results": results,
        "adjudication": adjudication,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_causal_rank_spectrum_{model_key}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2, default=float)
    log(
        f"CAUSAL-RANK verdict={verdict} "
        f"location_rank={location['minimum_rank_for_80pct_full']} "
        f"color_rank={color['minimum_rank_for_80pct_full']} "
        f"color_active={color['active_individual_axes']} "
        f"tail={color['tail_active']} artifact={path}")
    return result
