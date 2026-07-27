"""Locked ledger/paraphrase replication of operation-dependent depth."""
from __future__ import annotations

import json
import os

import torch

from .delta_anchor_write import _resolve
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

PROTOCOL_VERSION = "2026-07-24-p2-depth-replication-v1"
PROTOCOL_SHA256 = (
    "8E43DB3ACAAD2E2A024DF12E26CE5B644613429C5382F264264C75C81A7A05E0")
REPLICATION_TASKS = (
    "rep_state_ac",
    "rep_report_ac",
    "rep_action_ac",
)
QUERY.update({
    "rep_state_ac": (
        "Consult Alice's private record for the cube. State its location.",
        "STATE", "ac"),
    "rep_report_ac": (
        "Alice must communicate her private cube record to a teammate. "
        "Which location does she communicate?",
        "REPORT", "ac"),
    "rep_action_ac": (
        "Alice must act to recover the cube using only her private record. "
        "Which location does she visit?",
        "GO", "ac"),
})


def _first_passing(curve):
    for layer in LAYERS:
        if curve[str(layer)]["mediation"]["pass"]:
            return layer
    return None


def _verdict(tasks, prefix):
    if not all(tasks[name]["eligible"] for name in REPLICATION_TASKS):
        return "BEHAVIORALLY_INELIGIBLE"
    if not all(tasks[name]["source_intervention"]["sufficient"]
               for name in REPLICATION_TASKS):
        return "SOURCE_SITE_INELIGIBLE"
    first = [_first_passing(prefix[name]) for name in REPLICATION_TASKS]
    if any(value is None for value in first):
        return "ORDERING_NOT_REPLICATED"
    if first[0] < first[1] < first[2]:
        return "DEPTH_ORDERING_REPLICATED"
    if first[0] <= first[1] <= first[2]:
        return "MONOTONE_WITH_TIE"
    return "ORDERING_NOT_REPLICATED"


@torch.no_grad()
def run_delta_depth_replication(
        model_path, out_dir,
        model_key="qwen7b_depth_replication",
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
        raise ValueError("replication layer is absent from model")
    n_heads, head_dim = _attention_geometry(model)

    hb = Heartbeat(
        len(REPLICATION_TASKS) * (3 + len(LAYERS)),
        "depth_replication", every_sec=30, out_dir=out_dir)
    tasks = {}
    prefix = {}
    for query in REPLICATION_TASKS:
        context = _task_context(
            model, tok, dev, rows, query,
            CAPTURE_LAYERS, head_dim, surface="ledger")
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
        for query in REPLICATION_TASKS
    }
    verdict = _verdict(tasks, prefix)
    result = {
        "stage": "delta_depth_replication",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "surface": "ledger",
        "query_contracts": {
            query: {
                "question": QUERY[query][0],
                "command": QUERY[query][1],
                "field": QUERY[query][2],
            }
            for query in REPLICATION_TASKS
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
        out_dir, f"results_delta_depth_replication_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"DEPTH REPLICATION verdict={verdict} "
        f"first={first} artifact={path}")
    return result
