"""Fully crossed question-by-command diagnostic for causal handoff depth."""
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
from .delta_semantic_command_factor import (
    BELIEF_QUESTION,
    SEARCH_QUESTION,
    TELL_QUESTION,
)
from .delta_source_head_mediation import _public_task, _task_context
from .delta_sparse_transport import _attention_geometry
from .delta_structured_workspace import QUERY
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

PROTOCOL_VERSION = "2026-07-24-p2-prompt-factorial-v1"
PROTOCOL_SHA256 = (
    "85A9E9A74A1886BB21B3242BAA731F73802498A999C6110BC5D86347192C90F6")
QUESTION_NAMES = ("belief", "tell", "search")
COMMANDS = ("BELIEF", "TELL", "SEARCH")
QUESTIONS = {
    "belief": BELIEF_QUESTION,
    "tell": TELL_QUESTION,
    "search": SEARCH_QUESTION,
}
TASK_GRID = {
    question: {
        command: f"factorial_{question}_{command.lower()}"
        for command in COMMANDS
    }
    for question in QUESTION_NAMES
}
ALL_TASKS = tuple(
    TASK_GRID[question][command]
    for question in QUESTION_NAMES
    for command in COMMANDS
)
QUERY.update({
    TASK_GRID[question][command]: (
        QUESTIONS[question], command, "ac")
    for question in QUESTION_NAMES
    for command in COMMANDS
})


def _first_matrix(prefix):
    return {
        question: {
            command: _first_passing(
                prefix[TASK_GRID[question][command]])
            for command in COMMANDS
        }
        for question in QUESTION_NAMES
    }


def _verdict(tasks, prefix):
    if not all(tasks[name]["eligible"] for name in ALL_TASKS):
        return "BEHAVIORALLY_INELIGIBLE"
    if not all(tasks[name]["source_intervention"]["sufficient"]
               for name in ALL_TASKS):
        return "SOURCE_SITE_INELIGIBLE"
    matrix = _first_matrix(prefix)
    values = [
        matrix[question][command]
        for question in QUESTION_NAMES
        for command in COMMANDS
    ]
    if any(value is None for value in values):
        return "DEPTH_UNRESOLVED"
    if len(set(values)) == 1:
        return "NO_DEPTH_VARIATION"
    columns_invariant = all(
        len({
            matrix[question][command]
            for question in QUESTION_NAMES
        }) == 1
        for command in COMMANDS
    )
    if columns_invariant:
        return "COMMAND_INVARIANT_ACROSS_QUESTIONS"
    rows_invariant = all(
        len(set(matrix[question].values())) == 1
        for question in QUESTION_NAMES
    )
    if rows_invariant:
        return "QUESTION_INVARIANT_ACROSS_COMMANDS"
    return "MIXED_OR_INTERACTION"


@torch.no_grad()
def run_delta_prompt_factorial(
        model_path, out_dir,
        model_key="qwen7b_prompt_factorial",
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
        raise ValueError("factorial layer is absent from model")
    n_heads, head_dim = _attention_geometry(model)

    hb = Heartbeat(
        len(ALL_TASKS) * (3 + len(LAYERS)),
        "prompt_factorial", every_sec=30, out_dir=out_dir)
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

    first = {
        query: _first_passing(prefix[query])
        for query in ALL_TASKS
    }
    matrix = _first_matrix(prefix)
    verdict = _verdict(tasks, prefix)
    result = {
        "stage": "delta_prompt_factorial",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "surface": "narrative",
        "questions": list(QUESTION_NAMES),
        "commands": list(COMMANDS),
        "task_grid": TASK_GRID,
        "query_contracts": {
            query: {
                "question": QUERY[query][0],
                "command": QUERY[query][1],
                "field": QUERY[query][2],
            }
            for query in ALL_TASKS
        },
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
        "first_passing_prefix": first,
        "first_passing_matrix": matrix,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_prompt_factorial_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"PROMPT FACTORIAL verdict={verdict} "
        f"matrix={matrix} artifact={path}")
    return result

