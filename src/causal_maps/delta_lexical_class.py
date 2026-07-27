"""Position-matched generalization across pretrained lexical label classes."""
from __future__ import annotations

import json
import os

import torch

from .delta_anchor_write import _resolve
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
from .delta_structured_workspace import QUERY, _batch
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

PROTOCOL_VERSION = "2026-07-24-p2-lexical-class-v1"
PROTOCOL_SHA256 = (
    "958066C8DD2100E8DF035D8657AA83146E48C60C64FC0DAF4A92079D09B06AAE")
CLASSES = ("epistemic", "communication", "search")
LABELS = {
    "epistemic": ("BELIEF", "THINK", "KNOW"),
    "communication": ("TELL", "REPORT", "SAY"),
    "search": ("SEARCH", "FIND", "LOOK"),
}
ANCHORS = {
    "epistemic": "BELIEF",
    "communication": "TELL",
    "search": "SEARCH",
}
MAX_PADDING = 16
TASK_GRID = {
    lexical_class: {
        label: f"lex_{lexical_class}_{label.lower()}"
        for label in LABELS[lexical_class]
    }
    for lexical_class in CLASSES
}
ALL_TASKS = tuple(
    TASK_GRID[lexical_class][label]
    for lexical_class in CLASSES
    for label in LABELS[lexical_class]
)
QUERY.update({
    TASK_GRID[lexical_class][label]: (
        BELIEF_QUESTION, label, "ac")
    for lexical_class in CLASSES
    for label in LABELS[lexical_class]
})


def _command(label, padding):
    return " ".join(["X"] * int(padding) + [label])


def _padding_plan(tok, row):
    tables = {}
    common = None
    cpu = torch.device("cpu")
    for lexical_class in CLASSES:
        for label in LABELS[lexical_class]:
            query = TASK_GRID[lexical_class][label]
            table = {}
            for padding in range(MAX_PADDING + 1):
                command = _command(label, padding)
                QUERY[query] = (BELIEF_QUESTION, command, "ac")
                batch = _batch(
                    tok, [row], query, "narrative", cpu)
                position = int(batch["ids"].shape[1] - 1)
                table.setdefault(position, {
                    "padding": int(padding),
                    "command": command,
                })
            tables[query] = table
            positions = set(table)
            common = positions if common is None else common & positions
    if not common:
        return None, None, tables
    target = min(common)
    plan = {
        query: tables[query][target]
        for query in ALL_TASKS
    }
    for lexical_class in CLASSES:
        for label in LABELS[lexical_class]:
            query = TASK_GRID[lexical_class][label]
            QUERY[query] = (
                BELIEF_QUESTION, plan[query]["command"], "ac")
    return target, plan, tables


def _summary(tasks, prefix):
    cells = {
        lexical_class: {
            label: {
                "readout_position": int(
                    tasks[TASK_GRID[lexical_class][label]][
                        "readout_position"]),
                "l24_minimum_mediation": float(
                    prefix[TASK_GRID[lexical_class][label]]["24"][
                        "mediation"]["minimum_fraction"]),
                "first_passing_prefix": _first_passing(
                    prefix[TASK_GRID[lexical_class][label]]),
            }
            for label in LABELS[lexical_class]
        }
        for lexical_class in CLASSES
    }
    class_means = {
        lexical_class: sum(
            cells[lexical_class][label]["l24_minimum_mediation"]
            for label in LABELS[lexical_class]
        ) / len(LABELS[lexical_class])
        for lexical_class in CLASSES
    }
    heldout_means = {
        lexical_class: sum(
            cells[lexical_class][label]["l24_minimum_mediation"]
            for label in LABELS[lexical_class]
            if label != ANCHORS[lexical_class]
        ) / (len(LABELS[lexical_class]) - 1)
        for lexical_class in CLASSES
    }
    return {
        "cells": cells,
        "class_mean_l24": class_means,
        "heldout_synonym_mean_l24": heldout_means,
        "class_range": max(class_means.values())
                       - min(class_means.values()),
    }


def _verdict(tasks, prefix):
    if not all(tasks[name]["eligible"] for name in ALL_TASKS):
        return "BEHAVIORALLY_INELIGIBLE"
    if not all(tasks[name]["source_intervention"]["sufficient"]
               for name in ALL_TASKS):
        return "SOURCE_SITE_INELIGIBLE"
    summary = _summary(tasks, prefix)
    positions = {
        summary["cells"][lexical_class][label]["readout_position"]
        for lexical_class in CLASSES
        for label in LABELS[lexical_class]
    }
    if len(positions) != 1:
        return "POSITION_MATCH_INVALID"
    first = [
        summary["cells"][lexical_class][label]["first_passing_prefix"]
        for lexical_class in CLASSES
        for label in LABELS[lexical_class]
    ]
    if any(value is None for value in first):
        return "DEPTH_UNRESOLVED"
    means = summary["class_mean_l24"]
    heldout = summary["heldout_synonym_mean_l24"]
    full_difference = means["epistemic"] - means["search"]
    ordered = (
        means["epistemic"] > means["communication"]
        > means["search"])
    heldout_difference = heldout["epistemic"] - heldout["search"]
    if (ordered and full_difference >= 0.05 - 1e-9
            and heldout_difference >= 0.03 - 1e-9):
        return "LEXICAL_CLASS_GENERALIZATION"
    if full_difference >= 0.05 - 1e-9:
        return "ANCHOR_WORD_ONLY_EFFECT"
    if summary["class_range"] >= 0.05 - 1e-9:
        return "OTHER_LEXICAL_STRUCTURE"
    return "NO_LEXICAL_CLASS_EFFECT"


@torch.no_grad()
def run_delta_lexical_class(
        model_path, out_dir,
        model_key="qwen7b_lexical_class",
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
        raise ValueError("lexical-class layer is absent from model")
    n_heads, head_dim = _attention_geometry(model)

    target_position, padding_plan, tokenization_tables = _padding_plan(
        tok, rows[0])
    if padding_plan is None:
        result = {
            "stage": "delta_lexical_class",
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "model_key": model_key,
            "tokenization_tables": tokenization_tables,
            "verdict": "POSITION_MATCH_INVALID",
        }
        path = os.path.join(
            out_dir,
            f"results_delta_lexical_class_{model_key}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=float)
        log(f"LEXICAL CLASS verdict=POSITION_MATCH_INVALID "
            f"artifact={path}")
        return result

    hb = Heartbeat(
        len(ALL_TASKS) * (3 + len(LAYERS)),
        "lexical_class", every_sec=30, out_dir=out_dir)
    tasks = {}
    prefix = {}
    for query in ALL_TASKS:
        context = _task_context(
            model, tok, dev, rows, query,
            CAPTURE_LAYERS, head_dim, surface="narrative")
        tasks[query] = _public_task(context)
        hb.step(extra=f"{query}/baseline")
        hb.step(extra=f"{query}/source")
        hb.step(extra=f"{query}/base_path")
        prefix[query] = {}
        for layer in LAYERS:
            selected = tuple(x for x in LAYERS if x <= layer)
            cell = _evaluate_sites(
                model, context, _full_sites(selected, n_heads), head_dim)
            prefix[query][str(layer)] = cell
            hb.step(extra=(
                f"{query}/prefixL{layer}="
                f"{cell['mediation']['pass']}"))
    hb.done()

    summary = _summary(tasks, prefix)
    verdict = _verdict(tasks, prefix)
    result = {
        "stage": "delta_lexical_class",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "surface": "narrative",
        "question": BELIEF_QUESTION,
        "classes": list(CLASSES),
        "labels": LABELS,
        "anchors": ANCHORS,
        "task_grid": TASK_GRID,
        "target_readout_position": target_position,
        "padding_plan": padding_plan,
        "tokenization_tables": tokenization_tables,
        "world_selection": {
            "requested": int(n_world),
            "selected": len(rows),
            "indices_from_30": indices,
        },
        "layers": list(LAYERS),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "tasks": tasks,
        "cumulative_prefix": prefix,
        "lexical_summary": summary,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_lexical_class_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"LEXICAL CLASS verdict={verdict} "
        f"summary={summary} artifact={path}")
    return result

