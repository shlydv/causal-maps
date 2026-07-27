"""Full-readout-attention mediation curves across the remaining depth."""
from __future__ import annotations

import json
import os

import torch

from .delta_anchor_write import _resolve
from .delta_preprint_battery import _compatible_world_rows
from .delta_source_head_mediation import (
    TASKS,
    _mediation_pass,
    _public_task,
    _run_intervention,
    _task_context,
)
from .delta_sparse_transport import _attention_geometry, _cell, _site_value
from .delta_structured_workspace import _accuracy
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

PROTOCOL_VERSION = "2026-07-24-p2-operation-handoff-depth-v1"
PROTOCOL_SHA256 = (
    "2D6D4591EC38C1655824DC68DD451C3BBA140C27F8601FF50E648786256CE007")
LAYERS = (22, 23, 24, 25, 26, 27)
CAPTURE_LAYERS = (21,) + LAYERS


def _first_passing(curve):
    for layer in LAYERS:
        if curve[str(layer)]["mediation"]["pass"]:
            return layer
    return None


def _verdict(tasks, prefix):
    if not all(
            tasks[name]["eligible"]
            and tasks[name]["source_intervention"]["sufficient"]
            for name in TASKS):
        return "SOURCE_SITE_INELIGIBLE"
    first = {name: _first_passing(prefix[name]) for name in TASKS}
    if all(value is not None for value in first.values()):
        if (first["search_ac"] > first["belief_ac"]
                and first["search_ac"] > first["tell_ac"]):
            return "OPERATION_DEPENDENT_HANDOFF_DEPTH"
        if len(set(first.values())) == 1:
            return "SHARED_HANDOFF_DEPTH"
        return "PARTIAL_DEPTH_ORDERING"
    if (first["belief_ac"] is not None
            and first["tell_ac"] is not None
            and first["search_ac"] is None):
        return "SEARCH_ROUTE_OUTSIDE_LATE_READOUT_ATTENTION"
    return "PARTIAL_DEPTH_ORDERING"


def _full_sites(layers, n_heads):
    return tuple(
        (layer, head)
        for layer in layers
        for head in range(int(n_heads))
    )


@torch.no_grad()
def _evaluate_sites(model, context, sites, head_dim):
    forward = _run_intervention(
        model, context["clean"]["ids"], context["clean"]["am"],
        context["source_position"], context["natural_source"],
        context["readout_position"], sites,
        [_site_value(context["clean_heads"], site, head_dim)
         for site in sites], head_dim,
        readout_delta=context.get("readout_delta"),
        readout_patch=context.get("clean_readout_patch"),
        sequence_patch=context.get("clean_sequence_patch"),
        source_layer=context.get("source_layer", 21))
    reverse = _run_intervention(
        model, context["natural"]["ids"], context["natural"]["am"],
        context["source_position"], context["clean_source"],
        context["readout_position"], sites,
        [_site_value(context["natural_heads"], site, head_dim)
         for site in sites], head_dim,
        readout_delta=context.get("readout_delta"),
        readout_patch=context.get("natural_readout_patch"),
        sequence_patch=context.get("natural_sequence_patch"),
        source_layer=context.get("source_layer", 21))
    blocked = _cell(
        context["clean_logits"], context["natural_logits"],
        forward, reverse, context["clean"],
        context["source"], context["target"])
    mediation = _mediation_pass(
        context["source_intervention"], blocked,
        _accuracy(forward, context["clean"], context["source"]),
        _accuracy(reverse, context["natural"], context["target"]))
    return {
        "layers": sorted({layer for layer, _head in sites}),
        "n_sites": len(sites),
        "blocked_intervention": blocked,
        "mediation": mediation,
    }


@torch.no_grad()
def run_delta_operation_handoff_depth(
        model_path, out_dir,
        model_key="qwen7b_operation_handoff_depth",
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
        raise ValueError("handoff layer is absent from model")
    n_heads, head_dim = _attention_geometry(model)

    hb = Heartbeat(
        len(TASKS) * (3 + 3 * len(LAYERS)),
        "operation_handoff_depth", every_sec=30, out_dir=out_dir)
    public_tasks = {}
    individual = {}
    prefix = {}
    suffix = {}
    for query in TASKS:
        context = _task_context(
            model, tok, dev, rows, query,
            CAPTURE_LAYERS, head_dim)
        public_tasks[query] = _public_task(context)
        hb.step(extra=f"{query}/baseline")
        hb.step(extra=f"{query}/source")
        hb.step(extra=f"{query}/base_path")
        individual[query] = {}
        prefix[query] = {}
        suffix[query] = {}
        for layer in LAYERS:
            cell = _evaluate_sites(
                model, context, _full_sites((layer,), n_heads), head_dim)
            individual[query][str(layer)] = cell
            hb.step(extra=(
                f"{query}/individualL{layer}="
                f"{cell['mediation']['pass']}"))
        for layer in LAYERS:
            selected = tuple(x for x in LAYERS if x <= layer)
            cell = _evaluate_sites(
                model, context, _full_sites(selected, n_heads), head_dim)
            prefix[query][str(layer)] = cell
            hb.step(extra=(
                f"{query}/prefixL{layer}="
                f"{cell['mediation']['pass']}"))
        for layer in LAYERS:
            selected = tuple(x for x in LAYERS if x >= layer)
            cell = _evaluate_sites(
                model, context, _full_sites(selected, n_heads), head_dim)
            suffix[query][str(layer)] = cell
            hb.step(extra=(
                f"{query}/suffixL{layer}="
                f"{cell['mediation']['pass']}"))
    hb.done()

    first = {
        query: _first_passing(prefix[query])
        for query in TASKS
    }
    verdict = _verdict(public_tasks, prefix)
    result = {
        "stage": "delta_operation_handoff_depth",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "world_selection": {
            "requested": int(n_world),
            "selected": len(rows),
            "indices_from_30": indices,
        },
        "layers": list(LAYERS),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "tasks": public_tasks,
        "individual": individual,
        "cumulative_prefix": prefix,
        "cumulative_suffix": suffix,
        "first_passing_prefix": first,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_operation_handoff_depth_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"OPERATION HANDOFF DEPTH verdict={verdict} "
        f"first={first} artifact={path}")
    return result
