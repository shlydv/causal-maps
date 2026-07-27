"""Causal transfer of anchor answer-prefix states into unseen synonyms."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .delta_anchor_write import TARGET, _resolve
from .delta_depth_replication import _first_passing
from .delta_distributed_label_transplant import (
    _capture_full_l21,
    _contiguous_groups,
    _tail_probability,
)
from .delta_lexical_class import TASK_GRID, _padding_plan
from .delta_operation_handoff_depth import (
    CAPTURE_LAYERS,
    LAYERS,
    _evaluate_sites,
    _full_sites,
)
from .delta_preprint_battery import _compatible_world_rows
from .delta_source_head_mediation import _public_task, _task_context
from .delta_sparse_transport import _attention_geometry
from .delta_structured_workspace import _batch, _counterfactual
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

PROTOCOL_VERSION = "2026-07-25-p2-synonym-prefix-transfer-v1"
PROTOCOL_SHA256 = (
    "077C94073F835670606A2A31FD0CCC05EFC3CEE711772BC1F350F6B764E5CCB5")
TEST_START = 15
TEST_N = 15
PATCH_WIDTH = 3
N_RANDOM = 19
RANDOM_SEED = 27191
ANCHOR_QUERY = {
    "belief": TASK_GRID["epistemic"]["BELIEF"],
    "search": TASK_GRID["search"]["SEARCH"],
}
RECIPIENTS = {
    "think": {
        "query": TASK_GRID["epistemic"]["THINK"],
        "class": "epistemic", "cross_donor": "search",
        "same_donor": "belief",
    },
    "know": {
        "query": TASK_GRID["epistemic"]["KNOW"],
        "class": "epistemic", "cross_donor": "search",
        "same_donor": "belief",
    },
    "find": {
        "query": TASK_GRID["search"]["FIND"],
        "class": "search", "cross_donor": "belief",
        "same_donor": "search",
    },
    "look": {
        "query": TASK_GRID["search"]["LOOK"],
        "class": "search", "cross_donor": "belief",
        "same_donor": "search",
    },
}


def _paired_batches(tok, dev, rows, query):
    natural_rows = _counterfactual(rows, {"ac": TARGET})
    return (
        _batch(tok, rows, query, "narrative", dev),
        _batch(tok, natural_rows, query, "narrative", dev),
    )


def _alignment(tok, dev, rows):
    all_queries = list(ANCHOR_QUERY.values()) + [
        spec["query"] for spec in RECIPIENTS.values()
    ]
    batches = {
        query: _paired_batches(tok, dev, rows, query)
        for query in all_queries
    }
    reference_shape = batches[all_queries[0]][0]["ids"].shape
    marker = int(batches[all_queries[0]][0]["marker"])
    readout = int(reference_shape[1] - 1)
    answer_positions = list(
        range(readout - PATCH_WIDTH + 1, readout + 1))
    for query in all_queries:
        for batch in batches[query]:
            if batch["ids"].shape != reference_shape:
                raise ValueError("position-matched synonym shapes diverge")
            if int(batch["marker"]) != marker:
                raise ValueError("position-matched synonym markers diverge")

    instruction_windows = {}
    excluded = set(answer_positions)
    for name, spec in RECIPIENTS.items():
        donor_query = ANCHOR_QUERY[spec["cross_donor"]]
        donor_ids = batches[donor_query][0]["ids"][0]
        recipient_ids = batches[spec["query"]][0]["ids"][0]
        differing = torch.nonzero(
            donor_ids != recipient_ids, as_tuple=False).flatten().tolist()
        instruction_diffs = [
            position for position in differing
            if marker < position < answer_positions[0]
        ]
        if not instruction_diffs:
            raise ValueError(
                f"instruction difference absent for {name}")
        instruction_end = max(instruction_diffs)
        window = list(range(
            instruction_end - PATCH_WIDTH + 1, instruction_end + 1))
        if min(window) <= marker:
            raise ValueError(f"invalid instruction window for {name}")
        instruction_windows[name] = window
        excluded.update(window)

    candidates = []
    for position in range(marker + 1, answer_positions[0]):
        if position in excluded:
            continue
        values = []
        for query in all_queries:
            for batch in batches[query]:
                values.append(int(batch["ids"][0, position]))
        if len(set(values)) == 1:
            candidates.append(position)
    if len(candidates) < PATCH_WIDTH:
        raise ValueError("too few universal identical-token null positions")
    return {
        "batches": batches,
        "marker": marker,
        "readout": readout,
        "answer_positions": answer_positions,
        "instruction_windows": instruction_windows,
        "random_candidates": candidates,
    }


def _patch(donor_states, donor, positions):
    return (
        (positions, donor_states[donor][0][:, positions, :]),
        (positions, donor_states[donor][1][:, positions, :]),
    )


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


def _movement(original, patched):
    individual = {}
    for name, spec in RECIPIENTS.items():
        before = float(original[name])
        after = float(patched[name])
        individual[name] = (
            before - after
            if spec["class"] == "epistemic"
            else after - before
        )
    means = {
        lexical_class: float(np.mean([
            individual[name] for name, spec in RECIPIENTS.items()
            if spec["class"] == lexical_class
        ]))
        for lexical_class in ("epistemic", "search")
    }
    return {
        "individual_signed_movements": individual,
        "class_mean_movements": means,
        "primary_score": min(means.values()),
        "all_predicted_sign": bool(all(
            value > 0 for value in individual.values())),
    }


def _verdict(tasks, selected_summaries, control_tasks,
             primary, random_controls):
    if not all(task["eligible"] for task in tasks.values()):
        return "BEHAVIORALLY_INELIGIBLE"
    if not all(
            task["source_intervention"]["sufficient"]
            for task in tasks.values()):
        return "SOURCE_SITE_INELIGIBLE"
    if any(
            not task["eligible"]
            or not task["source_intervention"]["sufficient"]
            for task in control_tasks):
        return "CONTROL_INELIGIBLE"
    if any(
            selected_summaries[f"original_{name}"][
                "first_passing_prefix"] is None
            or selected_summaries[f"cross_{name}"][
                "first_passing_prefix"] is None
            for name in RECIPIENTS):
        return "DEPTH_UNRESOLVED"

    original_means = {
        lexical_class: float(np.mean([
            selected_summaries[f"original_{name}"][
                "l24_minimum_mediation"]
            for name, spec in RECIPIENTS.items()
            if spec["class"] == lexical_class
        ]))
        for lexical_class in ("epistemic", "search")
    }
    if original_means["epistemic"] - original_means["search"] < 0.03 - 1e-9:
        return "ORIGINAL_SYNONYM_GAP_ABSENT"

    means = primary["class_mean_movements"]
    epistemic_pass = means["epistemic"] >= 0.05 - 1e-9
    search_pass = means["search"] >= 0.05 - 1e-9
    if epistemic_pass != search_pass:
        return "PARTIAL_CROSS_SYNONYM_TRANSFER"
    if not (epistemic_pass and search_pass
            and primary["all_predicted_sign"]):
        return "NO_CROSS_SYNONYM_TRANSFER"

    random_scores = [
        cell["primary_score"] for cell in random_controls
    ]
    p_value, _exceed = _tail_probability(
        primary["primary_score"], random_scores)
    if p_value > 0.05 + 1e-12:
        return "NONSPECIFIC_SYNONYM_TRANSFER"

    epistemic_categorical = any(
        selected_summaries[f"cross_{name}"]["first_passing_prefix"]
        > selected_summaries[f"original_{name}"]["first_passing_prefix"]
        for name, spec in RECIPIENTS.items()
        if spec["class"] == "epistemic"
    )
    search_categorical = any(
        selected_summaries[f"cross_{name}"]["first_passing_prefix"]
        < selected_summaries[f"original_{name}"]["first_passing_prefix"]
        for name, spec in RECIPIENTS.items()
        if spec["class"] == "search"
    )
    return (
        "CROSS_SYNONYM_ROUTE_TRANSFER"
        if epistemic_categorical and search_categorical
        else "CONTINUOUS_CROSS_SYNONYM_TRANSFER")


@torch.no_grad()
def run_delta_synonym_prefix_transfer(
        model_path, out_dir,
        model_key="qwen7b_synonym_prefix_transfer",
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
        raise ValueError("v1 requires all 30 compatible worlds")
    if max(LAYERS) >= model_num_hidden_layers(model):
        raise ValueError("synonym-transfer layer is absent")
    test_rows = rows[TEST_START:TEST_START + TEST_N]
    n_heads, head_dim = _attention_geometry(model)

    target_position, padding_plan, tokenization_tables = _padding_plan(
        tok, rows[0])
    if padding_plan is None:
        alignment_error = "lexical padding plan is absent"
    else:
        try:
            alignment = _alignment(tok, dev, test_rows)
            if int(target_position) != alignment["readout"]:
                raise ValueError("frozen lexical readout position changed")
        except ValueError as exc:
            alignment_error = str(exc)
        else:
            alignment_error = None
    if alignment_error is not None:
        result = {
            "stage": "delta_synonym_prefix_transfer",
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "model_key": model_key,
            "alignment_error": alignment_error,
            "padding_plan": padding_plan,
            "tokenization_tables": tokenization_tables,
            "verdict": "POSITION_MATCH_INVALID",
        }
        path = os.path.join(
            out_dir,
            f"results_delta_synonym_prefix_transfer_{model_key}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=float)
        log(f"SYNONYM PREFIX TRANSFER verdict=POSITION_MATCH_INVALID "
            f"artifact={path}")
        return result

    donor_states = {}
    for donor, query in ANCHOR_QUERY.items():
        donor_states[donor] = [
            _capture_full_l21(model, batch)
            for batch in alignment["batches"][query]
        ]

    rng = np.random.default_rng(RANDOM_SEED)
    random_sets = []
    seen = set()
    while len(random_sets) < N_RANDOM:
        choice = tuple(sorted(
            int(x) for x in rng.choice(
                alignment["random_candidates"],
                size=PATCH_WIDTH, replace=False)))
        if choice not in seen:
            seen.add(choice)
            random_sets.append(list(choice))

    total_steps = (
        len(RECIPIENTS) * 2 * (3 + len(LAYERS))
        + len(RECIPIENTS) * 2 * 4
        + N_RANDOM * len(RECIPIENTS) * 4)
    hb = Heartbeat(
        total_steps, "synonym_prefix_transfer",
        every_sec=30, out_dir=out_dir)
    tasks = {}
    curves = {}
    for name, spec in RECIPIENTS.items():
        for arm, sequence_patch in (
                ("original", None),
                ("cross", _patch(
                    donor_states, spec["cross_donor"],
                    alignment["answer_positions"]))):
            key = f"{arm}_{name}"
            context = _task_context(
                model, tok, dev, test_rows, spec["query"],
                CAPTURE_LAYERS, head_dim,
                sequence_patch=sequence_patch)
            tasks[key] = _public_task(context)
            hb.step(extra=f"{key}/baseline")
            hb.step(extra=f"{key}/source")
            hb.step(extra=f"{key}/base_path")
            curves[key] = {}
            for layer, cell in _curve(
                    model, context, n_heads, head_dim).items():
                curves[key][layer] = cell
                hb.step(extra=f"{key}/prefixL{layer}")

    selected_summaries = {
        key: _summary(curve) for key, curve in curves.items()
    }
    original_values = {
        name: selected_summaries[f"original_{name}"][
            "l24_minimum_mediation"]
        for name in RECIPIENTS
    }
    cross_values = {
        name: selected_summaries[f"cross_{name}"][
            "l24_minimum_mediation"]
        for name in RECIPIENTS
    }
    primary = _movement(original_values, cross_values)

    control_tasks = []
    control_summaries = {"within_class": {}, "instruction": {}}
    control_layers = (21, 22, 23, 24)
    l24_sites = _full_sites((22, 23, 24), n_heads)
    for name, spec in RECIPIENTS.items():
        controls = {
            "within_class": _patch(
                donor_states, spec["same_donor"],
                alignment["answer_positions"]),
            "instruction": _patch(
                donor_states, spec["cross_donor"],
                alignment["instruction_windows"][name]),
        }
        for control_name, sequence_patch in controls.items():
            context = _task_context(
                model, tok, dev, test_rows, spec["query"],
                control_layers, head_dim,
                sequence_patch=sequence_patch)
            control_tasks.append(_public_task(context))
            hb.step(extra=f"{control_name}/{name}/baseline")
            hb.step(extra=f"{control_name}/{name}/source")
            hb.step(extra=f"{control_name}/{name}/base_path")
            cell = _evaluate_sites(
                model, context, l24_sites, head_dim)
            control_summaries[control_name][name] = float(
                cell["mediation"]["minimum_fraction"])
            hb.step(extra=f"{control_name}/{name}/L24")
    control_movements = {
        name: _movement(original_values, values)
        for name, values in control_summaries.items()
    }

    random_controls = []
    for random_index, positions in enumerate(random_sets):
        values = {}
        eligible = True
        for name, spec in RECIPIENTS.items():
            context = _task_context(
                model, tok, dev, test_rows, spec["query"],
                control_layers, head_dim,
                sequence_patch=_patch(
                    donor_states, spec["cross_donor"], positions))
            public = _public_task(context)
            control_tasks.append(public)
            eligible = bool(
                eligible and public["eligible"]
                and public["source_intervention"]["sufficient"])
            hb.step(extra=f"random{random_index}/{name}/baseline")
            hb.step(extra=f"random{random_index}/{name}/source")
            hb.step(extra=f"random{random_index}/{name}/base_path")
            cell = _evaluate_sites(
                model, context, l24_sites, head_dim)
            values[name] = float(
                cell["mediation"]["minimum_fraction"])
            hb.step(extra=f"random{random_index}/{name}/L24")
        random_controls.append({
            "random_index": random_index,
            "positions": positions,
            "eligible": eligible,
            **_movement(original_values, values),
        })
    hb.done()

    random_scores = [
        cell["primary_score"] for cell in random_controls
    ]
    random_p, random_exceed = _tail_probability(
        primary["primary_score"], random_scores)
    verdict = _verdict(
        tasks, selected_summaries, control_tasks,
        primary, random_controls)
    result = {
        "stage": "delta_synonym_prefix_transfer",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "claim": (
            "Anchor-derived answer-prefix states causally reconfigure "
            "unseen synonymous lexical contexts."),
        "world_selection": {
            "requested": int(n_world),
            "selected": len(rows),
            "indices_from_30": indices,
            "test_indices": indices[TEST_START:TEST_START + TEST_N],
        },
        "anchor_queries": ANCHOR_QUERY,
        "recipients": RECIPIENTS,
        "padding_plan": padding_plan,
        "alignment": {
            "marker_position": alignment["marker"],
            "readout_position": alignment["readout"],
            "answer_prefix_positions": alignment["answer_positions"],
            "instruction_windows": alignment["instruction_windows"],
            "random_candidates": alignment["random_candidates"],
        },
        "layers": list(LAYERS),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "tasks": tasks,
        "cumulative_prefix": curves,
        "selected_summaries": selected_summaries,
        "primary_transfer": primary,
        "control_l24": control_summaries,
        "control_movements": control_movements,
        "random_control": {
            "n_random": N_RANDOM,
            "seed": RANDOM_SEED,
            "selected_score": primary["primary_score"],
            "empirical_p": random_p,
            "exceed_count": random_exceed,
            "cells": random_controls,
        },
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_synonym_prefix_transfer_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"SYNONYM PREFIX TRANSFER verdict={verdict} "
        f"primary={primary} p={random_p} artifact={path}")
    return result
