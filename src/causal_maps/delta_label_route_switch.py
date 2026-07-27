"""Held-out bidirectional causal switching of lexical readout routes."""
from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F

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
from .delta_structured_workspace import (
    QUERY,
    _batch,
    _counterfactual,
)
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

PROTOCOL_VERSION = "2026-07-24-p2-label-route-switch-v1"
PROTOCOL_SHA256 = (
    "41682B62F951B4FC016FAB598F3E1837381E51E23C6EE4357F109577D18E956A")
DONOR_N = 15
TEST_N = 15
TASKS = ("switch_belief", "switch_search")
QUERY.update({
    "switch_belief": (BELIEF_QUESTION, "BELIEF", "ac"),
    "switch_search": (BELIEF_QUESTION, "X X SEARCH", "ac"),
})


@torch.no_grad()
def _readout_states(model, tok, dev, rows, query):
    natural_rows = _counterfactual(rows, {"ac": TARGET})
    out = []
    positions = []
    for arm in (rows, natural_rows):
        batch = _batch(tok, arm, query, "narrative", dev)
        position = int(batch["ids"].shape[1] - 1)
        _logits, states, _heads = _capture_source_heads(
            model, batch["ids"], batch["am"],
            position, position, ())
        out.append(states)
        positions.append(position)
    if len(set(positions)) != 1:
        raise ValueError(f"donor readout positions differ for {query}")
    return torch.cat(out, dim=0), positions[0]


def _curve(model, context, n_heads, head_dim):
    return {
        str(layer): _evaluate_sites(
            model, context,
            _full_sites(tuple(x for x in LAYERS if x <= layer), n_heads),
            head_dim)
        for layer in LAYERS
    }


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
            "BIDIRECTIONAL_ROUTE_SWITCH"
            if categorical else "CONTINUOUS_BIDIRECTIONAL_SWITCH")
    if up_pass != down_pass:
        return "ASYMMETRIC_ROUTE_SWITCH"
    return "NO_CAUSAL_ROUTE_SWITCH"


@torch.no_grad()
def run_delta_label_route_switch(
        model_path, out_dir,
        model_key="qwen7b_label_route_switch",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=30):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    rows, indices = _compatible_world_rows(
        tok, torch.device("cpu"), int(n_world))
    if len(rows) != DONOR_N + TEST_N:
        raise ValueError("v1 requires exactly 30 compatible worlds")
    if max(LAYERS) >= model_num_hidden_layers(model):
        raise ValueError("route-switch layer is absent from model")
    n_heads, head_dim = _attention_geometry(model)
    donor_rows = rows[:DONOR_N]
    test_rows = rows[DONOR_N:]

    belief_states, belief_position = _readout_states(
        model, tok, dev, donor_rows, "switch_belief")
    search_states, search_position = _readout_states(
        model, tok, dev, donor_rows, "switch_search")
    if belief_position != search_position:
        raise ValueError("position-matched donor contracts diverged")
    row_differences = belief_states - search_states
    control = row_differences.mean(dim=0)
    control_norm = float(control.norm())
    if control_norm <= 1e-8:
        raise ValueError("learned control vector has zero norm")
    cosine = F.cosine_similarity(
        row_differences,
        control.unsqueeze(0).expand_as(row_differences),
        dim=-1)

    hb = Heartbeat(
        4 * (3 + len(LAYERS)),
        "label_route_switch", every_sec=30, out_dir=out_dir)
    contexts = {}
    specs = {
        "belief_original": ("switch_belief", None),
        "search_original": ("switch_search", None),
        "belief_to_search": ("switch_belief", -control),
        "search_to_belief": ("switch_search", control),
    }
    curves = {}
    public_tasks = {}
    for name, (query, delta) in specs.items():
        context = _task_context(
            model, tok, dev, test_rows, query,
            CAPTURE_LAYERS, head_dim, surface="narrative",
            readout_delta=delta)
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
        "stage": "delta_label_route_switch",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "surface": "narrative",
        "query_contracts": {
            query: {
                "question": QUERY[query][0],
                "command": QUERY[query][1],
                "field": QUERY[query][2],
            }
            for query in TASKS
        },
        "world_selection": {
            "requested": int(n_world),
            "selected": len(rows),
            "indices_from_30": indices,
            "donor_indices": indices[:DONOR_N],
            "test_indices": indices[DONOR_N:],
        },
        "control": {
            "source_layer": 21,
            "readout_position": belief_position,
            "donor_state_count": int(row_differences.shape[0]),
            "l2_norm": control_norm,
            "row_difference_norm_mean": float(
                row_differences.norm(dim=-1).mean()),
            "row_to_mean_cosine_mean": float(cosine.mean()),
            "row_to_mean_cosine_min": float(cosine.min()),
        },
        "layers": list(LAYERS),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "tasks": public_tasks,
        "cumulative_prefix": curves,
        "switch_summary": summary,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_label_route_switch_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"LABEL ROUTE SWITCH verdict={verdict} "
        f"summary={summary} artifact={path}")
    return result

