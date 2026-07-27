"""Exact paired transplantation of label-conditioned L21 readout states."""
from __future__ import annotations

import json
import os

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
from .delta_source_head_mediation import (
    _capture_source_heads,
    _public_task,
    _task_context,
)
from .delta_sparse_transport import _attention_geometry
from .delta_structured_workspace import QUERY, _batch, _counterfactual
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

PROTOCOL_VERSION = "2026-07-24-p2-paired-route-transplant-v1"
PROTOCOL_SHA256 = (
    "73B8A7484475F5A003953180D4747C1342A756C17570045D2E2B13D510884FE9")
TEST_START = 15
TEST_N = 15
QUERY.update({
    "paired_belief": (BELIEF_QUESTION, "BELIEF", "ac"),
    "paired_search": (BELIEF_QUESTION, "X X SEARCH", "ac"),
})


@torch.no_grad()
def _paired_states(model, tok, dev, rows, query):
    natural_rows = _counterfactual(rows, {"ac": TARGET})
    states = []
    positions = []
    for arm in (rows, natural_rows):
        batch = _batch(tok, arm, query, "narrative", dev)
        position = int(batch["ids"].shape[1] - 1)
        _logits, state, _heads = _capture_source_heads(
            model, batch["ids"], batch["am"],
            position, position, ())
        states.append(state)
        positions.append(position)
    if len(set(positions)) != 1:
        raise ValueError(f"paired readout positions differ for {query}")
    return (states[0], states[1]), positions[0]


def _summary(curves):
    return {
        name: {
            "l24_minimum_mediation": float(
                curves[name]["24"]["mediation"]["minimum_fraction"]),
            "first_passing_prefix": _first_passing(curves[name]),
        }
        for name in (
            "belief_original", "search_original",
            "belief_to_search", "search_to_belief")
    }


def _verdict(tasks, curves):
    names = (
        "belief_original", "search_original",
        "belief_to_search", "search_to_belief")
    if not all(tasks[name]["eligible"] for name in names):
        return "BEHAVIORALLY_INELIGIBLE"
    if not all(tasks[name]["source_intervention"]["sufficient"]
               for name in names):
        return "SOURCE_SITE_INELIGIBLE"
    summary = _summary(curves)
    if any(summary[name]["first_passing_prefix"] is None
           for name in names):
        return "DEPTH_UNRESOLVED"
    belief = summary["belief_original"]["l24_minimum_mediation"]
    search = summary["search_original"]["l24_minimum_mediation"]
    belief_minus = summary["belief_to_search"]["l24_minimum_mediation"]
    search_plus = summary["search_to_belief"]["l24_minimum_mediation"]
    gap = belief - search
    if gap < 0.05 - 1e-9:
        return "ORIGINAL_GAP_ABSENT"
    up = search_plus - search
    down = belief - belief_minus
    up_pass = up >= 0.05 - 1e-9 and up + 1e-9 >= 0.5 * gap
    down_pass = down >= 0.05 - 1e-9 and down + 1e-9 >= 0.5 * gap
    categorical = (
        summary["search_to_belief"]["first_passing_prefix"]
        < summary["search_original"]["first_passing_prefix"]
        or summary["belief_to_search"]["first_passing_prefix"]
        > summary["belief_original"]["first_passing_prefix"]
    )
    if up_pass and down_pass:
        return (
            "BIDIRECTIONAL_PAIRED_ROUTE_TRANSPLANT"
            if categorical
            else "CONTINUOUS_BIDIRECTIONAL_TRANSPLANT")
    if up_pass != down_pass:
        return "ASYMMETRIC_PAIRED_TRANSPLANT"
    return "NO_PAIRED_ROUTE_TRANSPLANT"


@torch.no_grad()
def run_delta_paired_route_transplant(
        model_path, out_dir,
        model_key="qwen7b_paired_route_transplant",
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
        raise ValueError("paired-transplant layer is absent from model")
    n_heads, head_dim = _attention_geometry(model)
    test_rows = rows[TEST_START:TEST_START + TEST_N]

    belief_states, belief_position = _paired_states(
        model, tok, dev, test_rows, "paired_belief")
    search_states, search_position = _paired_states(
        model, tok, dev, test_rows, "paired_search")
    if belief_position != search_position:
        raise ValueError("position-matched transplant contracts diverged")

    specs = {
        "belief_original": ("paired_belief", None),
        "search_original": ("paired_search", None),
        "belief_to_search": ("paired_belief", search_states),
        "search_to_belief": ("paired_search", belief_states),
    }
    hb = Heartbeat(
        len(specs) * (3 + len(LAYERS)),
        "paired_route_transplant", every_sec=30, out_dir=out_dir)
    contexts = {}
    public_tasks = {}
    curves = {}
    for name, (query, patch) in specs.items():
        context = _task_context(
            model, tok, dev, test_rows, query,
            CAPTURE_LAYERS, head_dim, surface="narrative",
            readout_patch=patch)
        contexts[name] = context
        public_tasks[name] = _public_task(context)
        hb.step(extra=f"{name}/baseline")
        hb.step(extra=f"{name}/source")
        hb.step(extra=f"{name}/base_path")
        curves[name] = {}
        for layer in LAYERS:
            selected = tuple(x for x in LAYERS if x <= layer)
            cell = _evaluate_sites(
                model, context, _full_sites(selected, n_heads), head_dim)
            curves[name][str(layer)] = cell
            hb.step(extra=(
                f"{name}/prefixL{layer}="
                f"{cell['mediation']['pass']}"))
    hb.done()

    summary = _summary(curves)
    verdict = _verdict(public_tasks, curves)
    result = {
        "stage": "delta_paired_route_transplant",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "surface": "narrative",
        "world_selection": {
            "requested": int(n_world),
            "selected": len(rows),
            "indices_from_30": indices,
            "test_indices": indices[TEST_START:TEST_START + TEST_N],
        },
        "transplant": {
            "source_layer": 21,
            "readout_position": belief_position,
            "paired_within_world": True,
            "clean_natural_arms_separate": True,
        },
        "layers": list(LAYERS),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "tasks": public_tasks,
        "cumulative_prefix": curves,
        "transplant_summary": summary,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_paired_route_transplant_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"PAIRED ROUTE TRANSPLANT verdict={verdict} "
        f"summary={summary} artifact={path}")
    return result

