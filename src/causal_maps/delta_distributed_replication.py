"""Cross-scale replication of distributed answer-prefix route switching."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .delta_anchor_write import TARGET, _resolve
from .delta_distributed_label_transplant import (
    N_RANDOM,
    RANDOM_SEED,
    TEST_N,
    TEST_START,
    _aligned_batches,
    _capture_full_l21,
    _tail_probability,
)
from .delta_operation_handoff_depth import _evaluate_sites, _full_sites
from .delta_preprint_battery import _compatible_world_rows
from .delta_source_head_mediation import _public_task, _task_context
from .delta_sparse_transport import _attention_geometry
from .delta_structured_workspace import _counterfactual
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

QWEN14B_PROTOCOL_VERSION = (
    "2026-07-24-p2-qwen14b-distributed-replication-v1")
QWEN14B_PROTOCOL_SHA256 = (
    "4E5955AF7714348F09CEEBF15D9DA0E242FB85F4125480B1CD8771589F737AE3")
MISTRAL7B_PROTOCOL_VERSION = (
    "2026-07-24-p2-mistral7b-distributed-replication-v1")
MISTRAL7B_PROTOCOL_SHA256 = (
    "FE9FA90ECC98CDA9778ED1567903CA4A64B0DB23DEA3123C73A61911B47EED07")
DEFAULT_SOURCE_LAYER = 32
DEFAULT_ROUTE_START = 33
DEFAULT_CHECKPOINTS = (34, 36, 38, 40, 41, 42, 44, 46, 47)
DEFAULT_DECISION_LAYER = 41
REPLICATION_SPECS = {
    "qwen14b": {
        "protocol_version": QWEN14B_PROTOCOL_VERSION,
        "protocol_sha256": QWEN14B_PROTOCOL_SHA256,
        "source_layer": 32,
        "route_start": 33,
        "checkpoints": DEFAULT_CHECKPOINTS,
        "decision_layer": 41,
        "expected_compatible": 30,
        "test_start": TEST_START,
        "test_n": TEST_N,
        "positive_verdict": "CROSS_SCALE_DISTRIBUTED_ROUTE_REPLICATION",
        "continuous_verdict": "CONTINUOUS_CROSS_SCALE_ROUTE_REPLICATION",
    },
    "mistral7b": {
        "protocol_version": MISTRAL7B_PROTOCOL_VERSION,
        "protocol_sha256": MISTRAL7B_PROTOCOL_SHA256,
        "source_layer": 24,
        "route_start": 25,
        "checkpoints": (25, 26, 27, 29, 30, 31),
        "decision_layer": 27,
        "expected_compatible": 18,
        "test_start": 0,
        "test_n": 18,
        "positive_verdict": (
            "CROSS_ARCHITECTURE_DISTRIBUTED_ROUTE_REPLICATION"),
        "continuous_verdict": (
            "CONTINUOUS_CROSS_ARCHITECTURE_ROUTE_REPLICATION"),
    },
}


def _first_passing(curve, checkpoints):
    for layer in checkpoints:
        if curve[str(layer)]["mediation"]["pass"]:
            return int(layer)
    return None


def _curve(model, context, n_heads, head_dim, route_start, checkpoints):
    return {
        str(checkpoint): _evaluate_sites(
            model, context,
            _full_sites(
                tuple(range(int(route_start), int(checkpoint) + 1)),
                n_heads),
            head_dim)
        for checkpoint in checkpoints
    }


def _summary(curve, checkpoints, decision_layer):
    return {
        "decision_minimum_mediation": float(
            curve[str(decision_layer)]["mediation"]["minimum_fraction"]),
        "first_passing_checkpoint": _first_passing(curve, checkpoints),
    }


def _movement(original, belief_to_search, search_to_belief):
    belief = float(original["belief"]["decision_minimum_mediation"])
    search = float(original["search"]["decision_minimum_mediation"])
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


def _verdict(tasks, summaries, random_tasks, primary, random_controls,
             positive_verdict, continuous_verdict):
    if not all(task["eligible"] for task in tasks.values()):
        return "BEHAVIORALLY_INELIGIBLE"
    if not all(
            task["source_intervention"]["sufficient"]
            for task in tasks.values()):
        return "SOURCE_SITE_INELIGIBLE"
    if any(
            not task["eligible"]
            or not task["source_intervention"]["sufficient"]
            for task in random_tasks):
        return "RANDOM_CONTROL_INELIGIBLE"

    original = {
        "belief": summaries["belief_original"],
        "search": summaries["search_original"],
    }
    if (
            original["belief"]["decision_minimum_mediation"]
            - original["search"]["decision_minimum_mediation"]
            < 0.05 - 1e-9):
        return "ORIGINAL_GAP_ABSENT"
    if (
            original["belief"]["first_passing_checkpoint"] is None
            or original["search"]["first_passing_checkpoint"] is None):
        return "ORIGINAL_DEPTH_UNRESOLVED"
    if not (
            primary["belief_to_search_pass"]
            and primary["search_to_belief_pass"]):
        if (
                primary["belief_to_search_pass"]
                != primary["search_to_belief_pass"]):
            return "ASYMMETRIC_CROSS_SCALE_SWITCH"
        return "NO_CROSS_SCALE_SWITCH"

    p_value, _exceed = _tail_probability(
        primary["bidirectional_score"],
        [cell["bidirectional_score"] for cell in random_controls])
    if p_value > 0.05 + 1e-12:
        return "NONSPECIFIC_CROSS_SCALE_SWITCH"
    switched_belief = summaries["answer_prefix_belief_to_search"][
        "first_passing_checkpoint"]
    switched_search = summaries["answer_prefix_search_to_belief"][
        "first_passing_checkpoint"]
    categorical = (
        switched_belief is None
        or switched_belief
        > original["belief"]["first_passing_checkpoint"]
        or (
            switched_search is not None
            and switched_search
            < original["search"]["first_passing_checkpoint"])
    )
    return positive_verdict if categorical else continuous_verdict


@torch.no_grad()
def run_delta_distributed_replication(
        model_path, out_dir,
        model_key="qwen14b_distributed_replication",
        quantization="awq", device_map=None, max_memory=None,
        n_world=30, source_layer=DEFAULT_SOURCE_LAYER,
        route_start=DEFAULT_ROUTE_START,
        route_checkpoints=DEFAULT_CHECKPOINTS,
        decision_layer=DEFAULT_DECISION_LAYER,
        replication_spec="qwen14b"):
    os.makedirs(out_dir, exist_ok=True)
    if replication_spec not in REPLICATION_SPECS:
        raise ValueError(f"unknown replication spec: {replication_spec}")
    spec = REPLICATION_SPECS[replication_spec]
    protocol_version = spec["protocol_version"]
    protocol_sha256 = spec["protocol_sha256"]
    source_layer = int(source_layer)
    route_start = int(route_start)
    checkpoints = tuple(int(x) for x in route_checkpoints)
    decision_layer = int(decision_layer)
    if checkpoints != spec["checkpoints"]:
        raise ValueError("v1 requires the frozen route checkpoints")
    if source_layer != spec["source_layer"]:
        raise ValueError("v1 requires the frozen source layer")
    if route_start != spec["route_start"]:
        raise ValueError("v1 requires the frozen route start")
    if decision_layer != spec["decision_layer"]:
        raise ValueError("v1 requires the frozen decision layer")
    if decision_layer not in checkpoints:
        raise ValueError("decision layer must be a checkpoint")

    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    if max(checkpoints) >= model_num_hidden_layers(model):
        raise ValueError("frozen replication layer is absent")
    dev = input_device(model)
    rows, indices = _compatible_world_rows(
        tok, torch.device("cpu"), int(n_world))
    if len(rows) != spec["expected_compatible"]:
        raise ValueError(
            "v1 compatible-world count changed: "
            f"expected {spec['expected_compatible']}, got {len(rows)}")
    test_start = int(spec["test_start"])
    test_n = int(spec["test_n"])
    test_rows = rows[test_start:test_start + test_n]

    try:
        (pairs, differing, groups, candidates,
         marker, readout) = _aligned_batches(tok, dev, test_rows)
    except ValueError as exc:
        result = {
            "stage": "delta_distributed_replication",
            "protocol_version": protocol_version,
            "protocol_sha256": protocol_sha256,
            "replication_spec": replication_spec,
            "model_key": model_key,
            "alignment_error": str(exc),
            "verdict": "TOKEN_ALIGNMENT_INVALID",
        }
        path = os.path.join(
            out_dir,
            f"results_delta_distributed_replication_{model_key}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        log(f"DISTRIBUTED REPLICATION verdict=TOKEN_ALIGNMENT_INVALID "
            f"artifact={path}")
        return result

    answer_positions = groups[1]
    full_states = []
    for belief_batch, search_batch in pairs:
        full_states.append({
            "belief": _capture_full_l21(
                model, belief_batch, source_layer=source_layer),
            "search": _capture_full_l21(
                model, search_batch, source_layer=source_layer),
        })

    rng = np.random.default_rng(RANDOM_SEED)
    random_sets = []
    seen = set()
    while len(random_sets) < N_RANDOM:
        choice = tuple(sorted(
            int(x) for x in rng.choice(
                candidates, size=len(answer_positions), replace=False)))
        if choice not in seen:
            seen.add(choice)
            random_sets.append(list(choice))

    route_layers = tuple(range(route_start, max(checkpoints) + 1))
    capture_layers = (source_layer,) + route_layers
    context_kwargs = {
        "source_layer": source_layer,
        "blocked_sites": (),
    }
    original_contexts = {
        "belief_original": _task_context(
            model, tok, dev, test_rows, "distributed_belief",
            capture_layers, _attention_geometry(model)[1],
            **context_kwargs),
        "search_original": _task_context(
            model, tok, dev, test_rows, "distributed_search",
            capture_layers, _attention_geometry(model)[1],
            **context_kwargs),
    }
    n_heads, head_dim = _attention_geometry(model)

    total_steps = (
        2 * (3 + len(checkpoints))
        + 3 * 2 * (3 + len(checkpoints))
        + N_RANDOM * 2 * 4)
    hb = Heartbeat(
        total_steps, "distributed_replication",
        every_sec=30, out_dir=out_dir)
    tasks = {}
    curves = {}
    for name, context in original_contexts.items():
        tasks[name] = _public_task(context)
        hb.step(extra=f"{name}/baseline")
        hb.step(extra=f"{name}/source")
        hb.step(extra=f"{name}/base_path")
        curves[name] = {}
        for layer, cell in _curve(
                model, context, n_heads, head_dim,
                route_start, checkpoints).items():
            curves[name][layer] = cell
            hb.step(extra=f"{name}/prefixL{layer}")

    arm_positions = {
        "instruction": groups[0],
        "answer_prefix": answer_positions,
        "all": differing,
    }
    for arm, positions in arm_positions.items():
        patches = {
            "belief_to_search": (
                (positions, full_states[0]["search"][:, positions, :]),
                (positions, full_states[1]["search"][:, positions, :])),
            "search_to_belief": (
                (positions, full_states[0]["belief"][:, positions, :]),
                (positions, full_states[1]["belief"][:, positions, :])),
        }
        queries = {
            "belief_to_search": "distributed_belief",
            "search_to_belief": "distributed_search",
        }
        for direction in ("belief_to_search", "search_to_belief"):
            name = f"{arm}_{direction}"
            context = _task_context(
                model, tok, dev, test_rows, queries[direction],
                capture_layers, head_dim,
                sequence_patch=patches[direction],
                **context_kwargs)
            tasks[name] = _public_task(context)
            hb.step(extra=f"{name}/baseline")
            hb.step(extra=f"{name}/source")
            hb.step(extra=f"{name}/base_path")
            curves[name] = {}
            for layer, cell in _curve(
                    model, context, n_heads, head_dim,
                    route_start, checkpoints).items():
                curves[name][layer] = cell
                hb.step(extra=f"{name}/prefixL{layer}")

    summaries = {
        name: _summary(curve, checkpoints, decision_layer)
        for name, curve in curves.items()
    }
    original = {
        "belief": summaries["belief_original"],
        "search": summaries["search_original"],
    }
    arm_summaries = {}
    for arm in arm_positions:
        arm_summaries[arm] = _movement(
            original,
            summaries[f"{arm}_belief_to_search"][
                "decision_minimum_mediation"],
            summaries[f"{arm}_search_to_belief"][
                "decision_minimum_mediation"])

    random_tasks = []
    random_controls = []
    random_capture_layers = (
        source_layer,
        *range(route_start, decision_layer + 1),
    )
    decision_sites = _full_sites(
        tuple(range(route_start, decision_layer + 1)), n_heads)
    for random_index, positions in enumerate(random_sets):
        direction_values = {}
        for direction, query, donor in (
                ("belief_to_search", "distributed_belief", "search"),
                ("search_to_belief", "distributed_search", "belief")):
            patch = (
                (positions, full_states[0][donor][:, positions, :]),
                (positions, full_states[1][donor][:, positions, :]))
            context = _task_context(
                model, tok, dev, test_rows, query,
                random_capture_layers, head_dim,
                sequence_patch=patch, **context_kwargs)
            random_tasks.append(_public_task(context))
            hb.step(extra=f"random{random_index}/{direction}/baseline")
            hb.step(extra=f"random{random_index}/{direction}/source")
            hb.step(extra=f"random{random_index}/{direction}/base_path")
            cell = _evaluate_sites(
                model, context, decision_sites, head_dim)
            direction_values[direction] = float(
                cell["mediation"]["minimum_fraction"])
            hb.step(extra=f"random{random_index}/{direction}/L"
                    f"{decision_layer}")
        movement = _movement(
            original,
            direction_values["belief_to_search"],
            direction_values["search_to_belief"])
        random_controls.append({
            "random_index": random_index,
            "positions": positions,
            **movement,
        })
    hb.done()

    primary = arm_summaries["answer_prefix"]
    random_scores = [
        cell["bidirectional_score"] for cell in random_controls
    ]
    random_p, random_exceed = _tail_probability(
        primary["bidirectional_score"], random_scores)
    verdict = _verdict(
        tasks, summaries, random_tasks, primary, random_controls,
        spec["positive_verdict"], spec["continuous_verdict"])
    result = {
        "stage": "delta_distributed_replication",
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        "replication_spec": replication_spec,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "claim": (
            "The causal route from an unchanged stored state can be "
            "reconfigured by transplanting distributed answer-prefix "
            "representations."),
        "interpretation_status": "candidate mechanism",
        "world_selection": {
            "requested": int(n_world),
            "selected": len(rows),
            "indices_from_30": indices,
            "test_indices": indices[test_start:test_start + test_n],
        },
        "geometry": {
            "source_layer": source_layer,
            "route_start": route_start,
            "route_checkpoints": list(checkpoints),
            "decision_layer": decision_layer,
            "n_heads": n_heads,
            "head_dim": head_dim,
        },
        "alignment": {
            "marker_position": marker,
            "readout_position": readout,
            "differing_positions": differing,
            "instruction_positions": groups[0],
            "answer_prefix_positions": answer_positions,
            "identical_post_marker_candidates": candidates,
            "belief_token_ids": pairs[0][0]["ids"][
                0, differing].detach().cpu().tolist(),
            "search_token_ids": pairs[0][1]["ids"][
                0, differing].detach().cpu().tolist(),
        },
        "random_control": {
            "n_random": N_RANDOM,
            "seed": RANDOM_SEED,
            "cardinality": len(answer_positions),
            "selected_score": primary["bidirectional_score"],
            "empirical_p": random_p,
            "exceed_count": random_exceed,
            "cells": random_controls,
        },
        "tasks": tasks,
        "cumulative_prefix": curves,
        "summaries": summaries,
        "arm_summaries": arm_summaries,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_distributed_replication_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"DISTRIBUTED REPLICATION verdict={verdict} "
        f"arms={arm_summaries} p={random_p} artifact={path}")
    return result
