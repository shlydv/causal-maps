"""Factor arbitrary output-label surface from explicitly defined meaning."""
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
from .delta_structured_workspace import (
    MARKER,
    QUERY,
    _batch,
    _user,
)
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

PROTOCOL_VERSION = "2026-07-24-p2-label-meaning-codebook-v1"
PROTOCOL_SHA256 = (
    "94115F0C01CF5713BD91A16996D67370CB43EF174249649BF0698F31236520B5")
SURFACES = ("ALPHA", "BETA", "GAMMA")
MEANINGS = ("BELIEF", "TELL", "SEARCH")
MAX_PADDING = 16
TASK_GRID = {
    surface: {
        meaning: f"code_{surface.lower()}_{meaning.lower()}"
        for meaning in MEANINGS
    }
    for surface in SURFACES
}
ALL_TASKS = tuple(
    TASK_GRID[surface][meaning]
    for surface in SURFACES
    for meaning in MEANINGS
)
QUERY.update({
    TASK_GRID[surface][meaning]: (
        BELIEF_QUESTION, surface, "ac")
    for surface in SURFACES
    for meaning in MEANINGS
})


def _renderer(tok, query, surface, meaning, padding):
    filler = ("X " * int(padding))
    insertion = (
        f"{MARKER}. {filler}In this response code, {surface} denotes a "
        f"{meaning} response.\n")

    def render(row):
        user = _user(row, query, "narrative")
        needle = f"{MARKER}.\n"
        if user.count(needle) != 1:
            raise ValueError("state marker insertion point is not unique")
        user = user.replace(needle, insertion)
        return tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)

    return render


def _padding_plan(tok, row):
    tables = {}
    common = None
    cpu = torch.device("cpu")
    for surface in SURFACES:
        for meaning in MEANINGS:
            query = TASK_GRID[surface][meaning]
            key = query
            table = {}
            for padding in range(MAX_PADDING + 1):
                render = _renderer(
                    tok, query, surface, meaning, padding)
                batch = _batch(
                    tok, [row], query, "narrative", cpu,
                    render_fn=render)
                position = int(batch["ids"].shape[1] - 1)
                table.setdefault(position, padding)
            tables[key] = table
            positions = set(table)
            common = positions if common is None else common & positions
    if not common:
        return None, None, tables
    target = min(common)
    plan = {
        query: int(tables[query][target])
        for query in ALL_TASKS
    }
    return target, plan, tables


def _summary(tasks, prefix):
    cells = {
        surface: {
            meaning: {
                "readout_position": int(
                    tasks[TASK_GRID[surface][meaning]][
                        "readout_position"]),
                "l24_minimum_mediation": float(
                    prefix[TASK_GRID[surface][meaning]]["24"][
                        "mediation"]["minimum_fraction"]),
                "first_passing_prefix": _first_passing(
                    prefix[TASK_GRID[surface][meaning]]),
            }
            for meaning in MEANINGS
        }
        for surface in SURFACES
    }
    meaning_means = {
        meaning: sum(
            cells[surface][meaning]["l24_minimum_mediation"]
            for surface in SURFACES
        ) / len(SURFACES)
        for meaning in MEANINGS
    }
    surface_means = {
        surface: sum(
            cells[surface][meaning]["l24_minimum_mediation"]
            for meaning in MEANINGS
        ) / len(MEANINGS)
        for surface in SURFACES
    }
    return {
        "cells": cells,
        "meaning_mean_l24": meaning_means,
        "surface_mean_l24": surface_means,
        "meaning_range": max(meaning_means.values())
                         - min(meaning_means.values()),
        "surface_range": max(surface_means.values())
                         - min(surface_means.values()),
    }


def _verdict(tasks, prefix):
    if not all(tasks[name]["eligible"] for name in ALL_TASKS):
        return "BEHAVIORALLY_INELIGIBLE"
    if not all(tasks[name]["source_intervention"]["sufficient"]
               for name in ALL_TASKS):
        return "SOURCE_SITE_INELIGIBLE"
    summary = _summary(tasks, prefix)
    positions = {
        summary["cells"][surface][meaning]["readout_position"]
        for surface in SURFACES
        for meaning in MEANINGS
    }
    if len(positions) != 1:
        return "POSITION_MATCH_INVALID"
    first = [
        summary["cells"][surface][meaning]["first_passing_prefix"]
        for surface in SURFACES
        for meaning in MEANINGS
    ]
    if any(value is None for value in first):
        return "DEPTH_UNRESOLVED"
    meaning_range = summary["meaning_range"]
    surface_range = summary["surface_range"]
    if (meaning_range >= 0.05 - 1e-9
            and meaning_range + 1e-9 >= 2.0 * surface_range):
        return "DEFINED_MEANING_DOMINANT"
    if (surface_range >= 0.05 - 1e-9
            and surface_range + 1e-9 >= 2.0 * meaning_range):
        return "SURFACE_TOKEN_DOMINANT"
    if max(meaning_range, surface_range) >= 0.05 - 1e-9:
        return "MIXED_CODE_FACTORS"
    return "NO_CODE_FACTOR_EFFECT"


@torch.no_grad()
def run_delta_label_meaning_codebook(
        model_path, out_dir,
        model_key="qwen7b_label_meaning_codebook",
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
        raise ValueError("codebook layer is absent from model")
    n_heads, head_dim = _attention_geometry(model)

    target_position, padding, tokenization_tables = _padding_plan(
        tok, rows[0])
    if padding is None:
        result = {
            "stage": "delta_label_meaning_codebook",
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "model_key": model_key,
            "tokenization_tables": tokenization_tables,
            "verdict": "POSITION_MATCH_INVALID",
        }
        path = os.path.join(
            out_dir,
            f"results_delta_label_meaning_codebook_{model_key}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=float)
        log(f"LABEL MEANING CODEBOOK verdict=POSITION_MATCH_INVALID "
            f"artifact={path}")
        return result

    hb = Heartbeat(
        len(ALL_TASKS) * (3 + len(LAYERS)),
        "label_meaning_codebook", every_sec=30, out_dir=out_dir)
    tasks = {}
    prefix = {}
    for surface in SURFACES:
        for meaning in MEANINGS:
            query = TASK_GRID[surface][meaning]
            render = _renderer(
                tok, query, surface, meaning, padding[query])
            context = _task_context(
                model, tok, dev, rows, query,
                CAPTURE_LAYERS, head_dim, surface="narrative",
                render_fn=render)
            tasks[query] = _public_task(context)
            hb.step(extra=f"{query}/baseline")
            hb.step(extra=f"{query}/source")
            hb.step(extra=f"{query}/base_path")
            prefix[query] = {}
            for layer in LAYERS:
                selected = tuple(x for x in LAYERS if x <= layer)
                cell = _evaluate_sites(
                    model, context, _full_sites(selected, n_heads),
                    head_dim)
                prefix[query][str(layer)] = cell
                hb.step(extra=(
                    f"{query}/prefixL{layer}="
                    f"{cell['mediation']['pass']}"))
    hb.done()

    summary = _summary(tasks, prefix)
    verdict = _verdict(tasks, prefix)
    result = {
        "stage": "delta_label_meaning_codebook",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "surface": "narrative",
        "question": BELIEF_QUESTION,
        "surfaces": list(SURFACES),
        "meanings": list(MEANINGS),
        "task_grid": TASK_GRID,
        "padding_plan": padding,
        "target_readout_position": target_position,
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
        "codebook_summary": summary,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_label_meaning_codebook_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"LABEL MEANING CODEBOOK verdict={verdict} "
        f"summary={summary} artifact={path}")
    return result

