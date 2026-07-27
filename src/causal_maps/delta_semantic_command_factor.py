"""Crossed prompt diagnostic for semantic versus command-driven depth."""
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
from .delta_source_head_mediation import _public_task, _task_context
from .delta_sparse_transport import _attention_geometry
from .delta_structured_workspace import QUERY
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

PROTOCOL_VERSION = "2026-07-24-p2-semantic-command-factor-v1"
PROTOCOL_SHA256 = (
    "58CF68C4A6F1D63F8C973C789C2DD4E1BF51568C63B57C000D78C69F2EFB0794")
BELIEF_QUESTION = QUERY["belief_ac"][0]
TELL_QUESTION = QUERY["tell_ac"][0]
SEARCH_QUESTION = QUERY["search_ac"][0]
SEMANTIC_TASKS = (
    "factor_sem_belief",
    "factor_sem_tell",
    "factor_sem_search",
)
COMMAND_TASKS = (
    "factor_cmd_belief",
    "factor_cmd_tell",
    "factor_cmd_search",
)
ALL_TASKS = SEMANTIC_TASKS + COMMAND_TASKS
QUERY.update({
    "factor_sem_belief": (BELIEF_QUESTION, "ANSWER", "ac"),
    "factor_sem_tell": (TELL_QUESTION, "ANSWER", "ac"),
    "factor_sem_search": (SEARCH_QUESTION, "ANSWER", "ac"),
    "factor_cmd_belief": (BELIEF_QUESTION, "BELIEF", "ac"),
    "factor_cmd_tell": (BELIEF_QUESTION, "TELL", "ac"),
    "factor_cmd_search": (BELIEF_QUESTION, "SEARCH", "ac"),
})


def _verdict(tasks, prefix):
    if not all(tasks[name]["eligible"] for name in ALL_TASKS):
        return "BEHAVIORALLY_INELIGIBLE"
    if not all(tasks[name]["source_intervention"]["sufficient"]
               for name in ALL_TASKS):
        return "SOURCE_SITE_INELIGIBLE"
    first = {
        name: _first_passing(prefix[name])
        for name in ALL_TASKS
    }
    if any(value is None for value in first.values()):
        return "DEPTH_UNRESOLVED"
    values = list(first.values())
    if len(set(values)) == 1:
        return "NO_DEPTH_VARIATION"
    semantic = [first[name] for name in SEMANTIC_TASKS]
    command = [first[name] for name in COMMAND_TASKS]
    semantic_order = semantic[0] < semantic[1] < semantic[2]
    command_equal = len(set(command)) == 1
    if semantic_order and command_equal:
        return "SEMANTIC_OPERATION_EFFECT"
    if not semantic_order and not command_equal:
        return "COMMAND_PREFIX_EFFECT"
    return "MIXED_PROMPT_FACTORS"


@torch.no_grad()
def run_delta_semantic_command_factor(
        model_path, out_dir,
        model_key="qwen7b_semantic_command_factor",
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
        raise ValueError("factor layer is absent from model")
    n_heads, head_dim = _attention_geometry(model)

    hb = Heartbeat(
        len(ALL_TASKS) * (3 + len(LAYERS)),
        "semantic_command_factor", every_sec=30, out_dir=out_dir)
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
    verdict = _verdict(tasks, prefix)
    result = {
        "stage": "delta_semantic_command_factor",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "surface": "narrative",
        "panels": {
            "semantic": list(SEMANTIC_TASKS),
            "command": list(COMMAND_TASKS),
        },
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
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_semantic_command_factor_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"SEMANTIC COMMAND FACTOR verdict={verdict} "
        f"first={first} artifact={path}")
    return result
