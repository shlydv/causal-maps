"""Distributed L21 label-position transplantation with matched nulls."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .delta_anchor_write import TARGET, _resolve
from .delta_depth_replication import _first_passing
from .delta_operation_handoff_depth import (
    CAPTURE_LAYERS,
    LAYERS,
    _evaluate_sites,
    _full_sites,
)
from .delta_preprint_battery import _compatible_world_rows
from .delta_semantic_command_factor import BELIEF_QUESTION
from .delta_source_head_mediation import _public_task, _task_context
from .delta_sparse_transport import _attention_geometry
from .delta_structured_workspace import QUERY, _batch, _counterfactual
from .logutil import Heartbeat, log
from .model_utils import (
    get_decoder_layers,
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)
from .patching import _split_output

PROTOCOL_VERSION = "2026-07-24-p2-distributed-label-transplant-v1"
PROTOCOL_SHA256 = (
    "286C3C377921A99BF38BF7582806E6D885392C50631F003724261C6CBBEBD260")
TEST_START = 15
TEST_N = 15
N_RANDOM = 19
RANDOM_SEED = 9143
QUERY.update({
    "distributed_belief": (BELIEF_QUESTION, "BELIEF", "ac"),
    "distributed_search": (BELIEF_QUESTION, "X X SEARCH", "ac"),
})


def _contiguous_groups(positions):
    groups = []
    for position in positions:
        if not groups or position != groups[-1][-1] + 1:
            groups.append([position])
        else:
            groups[-1].append(position)
    return groups


def _aligned_batches(tok, dev, rows):
    natural_rows = _counterfactual(rows, {"ac": TARGET})
    pairs = []
    reference_mask = None
    marker = None
    for arm_rows in (rows, natural_rows):
        belief = _batch(
            tok, arm_rows, "distributed_belief", "narrative", dev)
        search = _batch(
            tok, arm_rows, "distributed_search", "narrative", dev)
        if belief["ids"].shape != search["ids"].shape:
            raise ValueError("position-matched label batches differ in shape")
        if belief["marker"] != search["marker"]:
            raise ValueError("position-matched label markers differ")
        diff = belief["ids"] != search["ids"]
        if not bool((diff == diff[0:1]).all()):
            raise ValueError("label difference mask varies across worlds")
        mask = diff[0].detach().cpu()
        if reference_mask is not None and not torch.equal(
                mask, reference_mask):
            raise ValueError("clean/natural difference masks diverge")
        reference_mask = mask
        if marker is not None and belief["marker"] != marker:
            raise ValueError("clean/natural marker positions diverge")
        marker = int(belief["marker"])
        pairs.append((belief, search))
    positions = torch.nonzero(
        reference_mask, as_tuple=False).flatten().tolist()
    if not positions or marker in positions:
        raise ValueError("invalid differing-position set")
    groups = _contiguous_groups(positions)
    if len(groups) != 2:
        raise ValueError(
            f"expected two label occurrences, found {groups}")
    readout = int(pairs[0][0]["ids"].shape[1] - 1)
    candidates = [
        position for position in range(marker + 1, readout + 1)
        if position not in set(positions)
    ]
    if len(candidates) < len(positions):
        raise ValueError("too few matched identical-token null positions")
    return pairs, positions, groups, candidates, marker, readout


@torch.no_grad()
def _capture_full_l21(model, batch, source_layer=21):
    cache = {}
    block = get_decoder_layers(model)[int(source_layer)]

    def hook(_module, _args, output):
        states, _rebuild = _split_output(output)
        cache["states"] = states.detach().float().cpu()

    handle = block.register_forward_hook(hook)
    try:
        model(
            input_ids=batch["ids"], attention_mask=batch["am"],
            use_cache=False)
    finally:
        handle.remove()
    if "states" not in cache:
        raise RuntimeError(
            f"L{int(source_layer)} sequence states were not captured")
    return cache["states"]


def _curve(model, context, n_heads, head_dim):
    return {
        str(layer): _evaluate_sites(
            model, context,
            _full_sites(tuple(x for x in LAYERS if x <= layer), n_heads),
            head_dim)
        for layer in LAYERS
    }


def _cell_summary(curve):
    return {
        "l24_minimum_mediation": float(
            curve["24"]["mediation"]["minimum_fraction"]),
        "first_passing_prefix": _first_passing(curve),
    }


def _movement(original, belief_to_search, search_to_belief):
    belief = original["belief"]["l24_minimum_mediation"]
    search = original["search"]["l24_minimum_mediation"]
    gap = belief - search
    down = belief - belief_to_search
    up = search_to_belief - search
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


def _tail_probability(observed, random_values):
    exceed = sum(float(value) >= float(observed) for value in random_values)
    return (1.0 + exceed) / (1.0 + len(random_values)), exceed


def _verdict(tasks, curves, random_tasks, random_controls):
    originals = ("belief_original", "search_original")
    arms = ("instruction", "answer_prefix", "all")
    names = list(originals) + [
        f"{arm}_{direction}"
        for arm in arms
        for direction in ("belief_to_search", "search_to_belief")
    ]
    if not all(tasks[name]["eligible"] for name in names):
        return "BEHAVIORALLY_INELIGIBLE"
    if not all(tasks[name]["source_intervention"]["sufficient"]
               for name in names):
        return "SOURCE_SITE_INELIGIBLE"
    if any(
            not task["eligible"]
            or not task["source_intervention"]["sufficient"]
            for task in random_tasks):
        return "RANDOM_CONTROL_INELIGIBLE"
    summaries = {
        name: _cell_summary(curves[name])
        for name in names
    }
    if any(summaries[name]["first_passing_prefix"] is None
           for name in names):
        return "DEPTH_UNRESOLVED"
    original = {
        "belief": summaries["belief_original"],
        "search": summaries["search_original"],
    }
    gap = (
        original["belief"]["l24_minimum_mediation"]
        - original["search"]["l24_minimum_mediation"])
    if gap < 0.05 - 1e-9:
        return "ORIGINAL_GAP_ABSENT"
    all_move = _movement(
        original,
        summaries["all_belief_to_search"]["l24_minimum_mediation"],
        summaries["all_search_to_belief"]["l24_minimum_mediation"])
    down_pass = all_move["belief_to_search_pass"]
    up_pass = all_move["search_to_belief_pass"]
    if down_pass and up_pass:
        random_scores = [
            cell["bidirectional_score"] for cell in random_controls
        ]
        p_value, _exceed = _tail_probability(
            all_move["bidirectional_score"], random_scores)
        if p_value > 0.05 + 1e-12:
            return "NONSPECIFIC_DISTRIBUTED_SWITCH"
        categorical = (
            summaries["all_search_to_belief"]["first_passing_prefix"]
            < summaries["search_original"]["first_passing_prefix"]
            or summaries["all_belief_to_search"]["first_passing_prefix"]
            > summaries["belief_original"]["first_passing_prefix"]
        )
        return (
            "SPECIFIC_DISTRIBUTED_LABEL_SWITCH"
            if categorical
            else "CONTINUOUS_SPECIFIC_DISTRIBUTED_SWITCH")
    if down_pass != up_pass:
        return "ASYMMETRIC_DISTRIBUTED_LABEL_SWITCH"
    return "NO_DISTRIBUTED_LABEL_SWITCH"


@torch.no_grad()
def run_delta_distributed_label_transplant(
        model_path, out_dir,
        model_key="qwen7b_distributed_label_transplant",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=30):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    rows, indices = _compatible_world_rows(
        tok, torch.device("cpu"), int(n_world))
    if len(rows) != 30:
        raise ValueError("v1 requires exactly 30 compatible worlds")
    if max(LAYERS) >= model_num_hidden_layers(model):
        raise ValueError("distributed-transplant layer is absent")
    n_heads, head_dim = _attention_geometry(model)
    test_rows = rows[TEST_START:TEST_START + TEST_N]

    try:
        (pairs, differing, groups, candidates,
         marker, readout) = _aligned_batches(tok, dev, test_rows)
    except ValueError as exc:
        result = {
            "stage": "delta_distributed_label_transplant",
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "model_key": model_key,
            "alignment_error": str(exc),
            "verdict": "TOKEN_ALIGNMENT_INVALID",
        }
        path = os.path.join(
            out_dir,
            "results_delta_distributed_label_transplant_"
            f"{model_key}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        log(f"DISTRIBUTED LABEL TRANSPLANT "
            f"verdict=TOKEN_ALIGNMENT_INVALID artifact={path}")
        return result

    full_states = []
    for belief_batch, search_batch in pairs:
        full_states.append({
            "belief": _capture_full_l21(model, belief_batch),
            "search": _capture_full_l21(model, search_batch),
        })

    rng = np.random.default_rng(RANDOM_SEED)
    random_sets = []
    seen = set()
    while len(random_sets) < N_RANDOM:
        choice = tuple(sorted(
            int(x) for x in rng.choice(
                candidates, size=len(differing), replace=False)))
        if choice not in seen:
            seen.add(choice)
            random_sets.append(list(choice))

    original_contexts = {
        "belief_original": _task_context(
            model, tok, dev, test_rows, "distributed_belief",
            CAPTURE_LAYERS, head_dim),
        "search_original": _task_context(
            model, tok, dev, test_rows, "distributed_search",
            CAPTURE_LAYERS, head_dim),
    }
    total_steps = (
        2 * (3 + len(LAYERS))
        + 3 * 2 * (3 + len(LAYERS))
        + N_RANDOM * 2 * 4)
    hb = Heartbeat(
        total_steps, "distributed_label_transplant",
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
                model, context, n_heads, head_dim).items():
            curves[name][layer] = cell
            hb.step(extra=f"{name}/prefixL{layer}")

    arm_positions = {
        "instruction": groups[0],
        "answer_prefix": groups[1],
        "all": differing,
    }
    arm_summaries = {}
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
                CAPTURE_LAYERS, head_dim,
                sequence_patch=patches[direction])
            tasks[name] = _public_task(context)
            hb.step(extra=f"{name}/baseline")
            hb.step(extra=f"{name}/source")
            hb.step(extra=f"{name}/base_path")
            curves[name] = {}
            for layer, cell in _curve(
                    model, context, n_heads, head_dim).items():
                curves[name][layer] = cell
                hb.step(extra=f"{name}/prefixL{layer}")
        original = {
            "belief": _cell_summary(curves["belief_original"]),
            "search": _cell_summary(curves["search_original"]),
        }
        arm_summaries[arm] = _movement(
            original,
            _cell_summary(curves[f"{arm}_belief_to_search"])[
                "l24_minimum_mediation"],
            _cell_summary(curves[f"{arm}_search_to_belief"])[
                "l24_minimum_mediation"])

    random_tasks = []
    random_controls = []
    random_capture_layers = (21, 22, 23, 24)
    original = {
        "belief": _cell_summary(curves["belief_original"]),
        "search": _cell_summary(curves["search_original"]),
    }
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
                sequence_patch=patch)
            random_tasks.append(_public_task(context))
            hb.step(extra=f"random{random_index}/{direction}/baseline")
            hb.step(extra=f"random{random_index}/{direction}/source")
            hb.step(extra=f"random{random_index}/{direction}/base_path")
            cell = _evaluate_sites(
                model, context,
                _full_sites((22, 23, 24), n_heads), head_dim)
            direction_values[direction] = float(
                cell["mediation"]["minimum_fraction"])
            hb.step(extra=f"random{random_index}/{direction}/L24")
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

    selected_score = arm_summaries["all"]["bidirectional_score"]
    random_scores = [
        cell["bidirectional_score"] for cell in random_controls
    ]
    random_p, random_exceed = _tail_probability(
        selected_score, random_scores)
    verdict = _verdict(tasks, curves, random_tasks, random_controls)
    result = {
        "stage": "delta_distributed_label_transplant",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "world_selection": {
            "requested": int(n_world),
            "selected": len(rows),
            "indices_from_30": indices,
            "test_indices": indices[TEST_START:TEST_START + TEST_N],
        },
        "alignment": {
            "marker_position": marker,
            "readout_position": readout,
            "differing_positions": differing,
            "instruction_positions": groups[0],
            "answer_prefix_positions": groups[1],
            "identical_post_marker_candidates": candidates,
            "belief_token_ids": pairs[0][0]["ids"][
                0, differing].detach().cpu().tolist(),
            "search_token_ids": pairs[0][1]["ids"][
                0, differing].detach().cpu().tolist(),
        },
        "random_control": {
            "n_random": N_RANDOM,
            "seed": RANDOM_SEED,
            "selected_score": selected_score,
            "empirical_p": random_p,
            "exceed_count": random_exceed,
            "cells": random_controls,
        },
        "layers": list(LAYERS),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "tasks": tasks,
        "cumulative_prefix": curves,
        "arm_summaries": arm_summaries,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        "results_delta_distributed_label_transplant_"
        f"{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"DISTRIBUTED LABEL TRANSPLANT verdict={verdict} "
        f"arms={arm_summaries} p={random_p} artifact={path}")
    return result
