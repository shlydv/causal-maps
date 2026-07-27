"""Causal mediation from the source anchor through frozen late heads."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .delta_anchor_write import SOURCE, TARGET, _anchor_position, _resolve
from .delta_preprint_battery import _compatible_world_rows
from .delta_sparse_transport import (
    G_ACC,
    _attention_geometry,
    _cell,
    _o_proj,
    _site_value,
)
from .delta_sparse_transport_confirmation import (
    CANDIDATE_LAYERS,
    FROZEN_TOP8,
)
from .delta_structured_workspace import (
    _accuracy,
    _batch,
    _counterfactual,
    _locations,
)
from .delta_trajectory import _ld
from .logutil import Heartbeat, log
from .model_utils import (
    get_decoder_layers,
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)
from .patching import _split_output

PROTOCOL_VERSION = "2026-07-24-p2-source-head-mediation-v1"
PROTOCOL_SHA256 = (
    "CD7F1794FA33D32C792C5217A2F5E88BA019A214F41D0A63DDCE70A0844EBAD6")
SOURCE_LAYER = 21
TASKS = ("belief_ac", "tell_ac", "search_ac")
N_RANDOM = 39
RANDOM_SEED = 6113
MEDIATION_GATE = 0.70
WRONG_DRIFT_GATE = 0.20
EPS = 1e-8


def _mediation_fraction(source_ratio, blocked_ratio):
    if source_ratio is None or abs(float(source_ratio)) <= EPS:
        return None
    return 1.0 - abs(float(blocked_ratio)) / abs(float(source_ratio))


def _mediation_pass(source_cell, blocked_cell, origin_forward_acc,
                    origin_reverse_acc):
    forward = _mediation_fraction(
        source_cell["forward_ratio"], blocked_cell["forward_ratio"])
    reverse = _mediation_fraction(
        source_cell["reverse_ratio"], blocked_cell["reverse_ratio"])
    passed = bool(
        source_cell["sufficient"]
        and forward is not None and reverse is not None
        and forward >= MEDIATION_GATE and reverse >= MEDIATION_GATE
        and float(origin_forward_acc) >= G_ACC
        and float(origin_reverse_acc) >= G_ACC
    )
    return {
        "forward_fraction": forward,
        "reverse_fraction": reverse,
        "minimum_fraction": (
            min(forward, reverse)
            if forward is not None and reverse is not None else None),
        "origin_forward_acc": float(origin_forward_acc),
        "origin_reverse_acc": float(origin_reverse_acc),
        "pass": passed,
    }


def _tail_probability(observed, random_values):
    exceed = sum(float(value) >= float(observed) for value in random_values)
    return (1.0 + exceed) / (1.0 + len(random_values)), exceed


def _verdict(tasks, random_cells, empirical_p, wrong_pass):
    if not all(tasks[name]["eligible"]
               and tasks[name]["source_intervention"]["sufficient"]
               for name in TASKS):
        return "SOURCE_SITE_INELIGIBLE"
    if (any(cell["mediation"]["pass"] for cell in random_cells)
            or float(empirical_p) > 0.025 + 1e-12):
        return "NONSPECIFIC_MEDIATION"
    passes = {
        name: bool(tasks[name]["mediation"]["pass"])
        for name in TASKS
    }
    if not wrong_pass:
        return "MIXED_MEDIATION"
    if all(passes.values()):
        return "SHARED_HEAD_MEDIATION"
    if passes == {
            "belief_ac": True, "tell_ac": True, "search_ac": False}:
        return "OPERATION_SPECIFIC_HEAD_MEDIATION"
    return "MIXED_MEDIATION"


@torch.no_grad()
def _capture_source_heads(model, ids, am, source_position,
                          readout_position, layers, readout_delta=None,
                          readout_patch=None, sequence_patch=None,
                          source_layer=SOURCE_LAYER):
    source_cache = {}
    head_cache = {}
    handles = []
    blocks = get_decoder_layers(model)

    def source_hook(_module, _args, output):
        states, rebuild = _split_output(output)
        if readout_delta is not None and readout_patch is not None:
            raise ValueError("readout delta and patch are mutually exclusive")
        if readout_patch is not None:
            states = states.clone()
            states[:, int(readout_position), :] = readout_patch.to(
                device=states.device, dtype=states.dtype)
        elif readout_delta is not None:
            states = states.clone()
            states[:, int(readout_position), :] += readout_delta.to(
                device=states.device, dtype=states.dtype)
        if sequence_patch is not None:
            positions, values = sequence_patch
            states = states.clone()
            states[:, positions, :] = values.to(
                device=states.device, dtype=states.dtype)
        source_cache["state"] = (
            states[:, int(source_position), :].detach().float().cpu())
        if (readout_delta is not None or readout_patch is not None
                or sequence_patch is not None):
            return rebuild(states)

    def head_hook(layer):
        def hook(_module, args):
            states = args[0]
            head_cache[int(layer)] = (
                states[:, int(readout_position), :].detach().float().cpu())
        return hook

    handles.append(
        blocks[int(source_layer)].register_forward_hook(source_hook))
    for layer in layers:
        handles.append(_o_proj(model, layer).register_forward_pre_hook(
            head_hook(layer)))
    try:
        output = model(input_ids=ids, attention_mask=am, use_cache=False)
        logits = output.logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    if "state" not in source_cache:
        raise RuntimeError("source state was not captured")
    missing = [layer for layer in layers if layer not in head_cache]
    if missing:
        raise RuntimeError(f"head layers were not captured: {missing}")
    return logits, source_cache["state"], head_cache


@torch.no_grad()
def _run_intervention(model, ids, am, source_position, source_value,
                      readout_position, clamp_sites, clamp_values, head_dim,
                      readout_delta=None, readout_patch=None,
                      sequence_patch=None, source_layer=SOURCE_LAYER):
    handles = []
    blocks = get_decoder_layers(model)

    def source_patch(_module, _args, output):
        states, rebuild = _split_output(output)
        states = states.clone()
        states[:, int(source_position), :] = source_value.to(
            device=states.device, dtype=states.dtype)
        if readout_delta is not None and readout_patch is not None:
            raise ValueError("readout delta and patch are mutually exclusive")
        if readout_patch is not None:
            states[:, int(readout_position), :] = readout_patch.to(
                device=states.device, dtype=states.dtype)
        elif readout_delta is not None:
            states[:, int(readout_position), :] += readout_delta.to(
                device=states.device, dtype=states.dtype)
        if sequence_patch is not None:
            positions, values = sequence_patch
            states[:, positions, :] = values.to(
                device=states.device, dtype=states.dtype)
        return rebuild(states)

    handles.append(
        blocks[int(source_layer)].register_forward_hook(source_patch))
    grouped = {}
    for site, value in zip(clamp_sites, clamp_values):
        grouped.setdefault(int(site[0]), []).append((int(site[1]), value))

    def clamp_hook(entries):
        def hook(_module, args):
            states = args[0].clone()
            for head, value in entries:
                start = int(head) * int(head_dim)
                stop = start + int(head_dim)
                states[:, int(readout_position), start:stop] = value.to(
                    device=states.device, dtype=states.dtype)
            return (states,) + tuple(args[1:])
        return hook

    for layer, entries in grouped.items():
        handles.append(_o_proj(model, layer).register_forward_pre_hook(
            clamp_hook(entries)))
    try:
        output = model(input_ids=ids, attention_mask=am, use_cache=False)
        return output.logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()


def _task_context(model, tok, dev, rows, query, layers, head_dim,
                  surface="narrative", render_fn=None,
                  readout_delta=None, readout_patch=None,
                  sequence_patch=None, source_layer=SOURCE_LAYER,
                  blocked_sites=FROZEN_TOP8):
    if readout_delta is not None and readout_patch is not None:
        raise ValueError("readout delta and patch are mutually exclusive")
    clean_patch, natural_patch = (
        readout_patch if readout_patch is not None else (None, None))
    clean_sequence, natural_sequence = (
        sequence_patch if sequence_patch is not None else (None, None))
    natural_rows = _counterfactual(rows, {"ac": TARGET})
    clean = _batch(
        tok, rows, query, surface, dev, render_fn=render_fn)
    natural = _batch(
        tok, natural_rows, query, surface, dev, render_fn=render_fn)
    if clean["ids"].shape != natural["ids"].shape:
        raise ValueError(f"unaligned clean/natural batch for {query}")
    source_position = _anchor_position(clean, natural)
    readout_position = int(clean["ids"].shape[1] - 1)
    clean_logits, clean_source, clean_heads = _capture_source_heads(
        model, clean["ids"], clean["am"], source_position,
        readout_position, layers, readout_delta=readout_delta,
        readout_patch=clean_patch, sequence_patch=clean_sequence,
        source_layer=source_layer)
    natural_logits, natural_source, natural_heads = _capture_source_heads(
        model, natural["ids"], natural["am"], source_position,
        readout_position, layers, readout_delta=readout_delta,
        readout_patch=natural_patch, sequence_patch=natural_sequence,
        source_layer=source_layer)
    source = _locations(rows, query)
    target = _locations(natural_rows, query)
    eligible = bool(min(
        _accuracy(clean_logits, clean, source),
        _accuracy(natural_logits, natural, target)) >= G_ACC)

    forward = _run_intervention(
        model, clean["ids"], clean["am"], source_position, natural_source,
        readout_position, (), (), head_dim,
        readout_delta=readout_delta, readout_patch=clean_patch,
        sequence_patch=clean_sequence, source_layer=source_layer)
    reverse = _run_intervention(
        model, natural["ids"], natural["am"], source_position, clean_source,
        readout_position, (), (), head_dim,
        readout_delta=readout_delta, readout_patch=natural_patch,
        sequence_patch=natural_sequence, source_layer=source_layer)
    source_cell = _cell(
        clean_logits, natural_logits, forward, reverse,
        clean, source, target)

    forward_blocked = _run_intervention(
        model, clean["ids"], clean["am"], source_position, natural_source,
        readout_position, blocked_sites,
        [_site_value(clean_heads, site, head_dim)
         for site in blocked_sites], head_dim,
        readout_delta=readout_delta, readout_patch=clean_patch,
        sequence_patch=clean_sequence, source_layer=source_layer)
    reverse_blocked = _run_intervention(
        model, natural["ids"], natural["am"], source_position, clean_source,
        readout_position, blocked_sites,
        [_site_value(natural_heads, site, head_dim)
         for site in blocked_sites], head_dim,
        readout_delta=readout_delta, readout_patch=natural_patch,
        sequence_patch=natural_sequence, source_layer=source_layer)
    blocked_cell = _cell(
        clean_logits, natural_logits, forward_blocked, reverse_blocked,
        clean, source, target)
    mediation = _mediation_pass(
        source_cell, blocked_cell,
        _accuracy(forward_blocked, clean, source),
        _accuracy(reverse_blocked, natural, target))
    return {
        "natural_rows": natural_rows,
        "clean": clean,
        "natural": natural,
        "source": source,
        "target": target,
        "source_position": source_position,
        "source_layer": int(source_layer),
        "readout_position": readout_position,
        "readout_delta": readout_delta,
        "clean_readout_patch": clean_patch,
        "natural_readout_patch": natural_patch,
        "clean_sequence_patch": clean_sequence,
        "natural_sequence_patch": natural_sequence,
        "clean_logits": clean_logits,
        "natural_logits": natural_logits,
        "clean_source": clean_source,
        "natural_source": natural_source,
        "clean_heads": clean_heads,
        "natural_heads": natural_heads,
        "eligible": eligible,
        "g0_clean": float(_accuracy(clean_logits, clean, source)),
        "g0_natural": float(_accuracy(natural_logits, natural, target)),
        "source_intervention": source_cell,
        "blocked_intervention": blocked_cell,
        "mediation": mediation,
    }


def _public_task(context):
    omitted = {
        "natural_rows", "clean", "natural", "source", "target",
        "clean_logits", "natural_logits", "clean_source", "natural_source",
        "clean_heads", "natural_heads", "readout_delta",
        "clean_readout_patch", "natural_readout_patch",
        "clean_sequence_patch", "natural_sequence_patch",
    }
    return {key: value for key, value in context.items()
            if key not in omitted}


@torch.no_grad()
def run_delta_source_head_mediation(
        model_path, out_dir,
        model_key="qwen7b_source_head_mediation",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=30, n_random=N_RANDOM, random_seed=RANDOM_SEED):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_random) != N_RANDOM:
        raise ValueError("v1 is frozen to 39 random clamps")
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    rows, indices = _compatible_world_rows(
        tok, torch.device("cpu"), int(n_world))
    if len(rows) != 30:
        raise ValueError("v1 requires all 30 compatible worlds")
    if max(CANDIDATE_LAYERS) >= model_num_hidden_layers(model):
        raise ValueError("candidate layer is absent from model")
    n_heads, head_dim = _attention_geometry(model)
    if any(head >= n_heads for _layer, head in FROZEN_TOP8):
        raise ValueError("frozen head is absent from model")

    hb = Heartbeat(
        len(TASKS) * 3 + int(n_random) + 2,
        "source_head_mediation", every_sec=30, out_dir=out_dir)
    contexts = {}
    for query in TASKS:
        context = _task_context(
            model, tok, dev, rows, query,
            CANDIDATE_LAYERS, head_dim)
        contexts[query] = context
        hb.step(extra=f"{query}/baseline")
        hb.step(extra=(
            f"{query}/source="
            f"{context['source_intervention']['sufficient']}"))
        hb.step(extra=(
            f"{query}/mediate={context['mediation']['pass']}"))

    primary = contexts["belief_ac"]
    all_sites = [
        (layer, head)
        for layer in CANDIDATE_LAYERS
        for head in range(n_heads)
    ]
    pool = [site for site in all_sites if site not in set(FROZEN_TOP8)]
    rng = np.random.default_rng(int(random_seed))
    random_cells = []
    for random_index in range(int(n_random)):
        choice = rng.choice(len(pool), size=len(FROZEN_TOP8), replace=False)
        sites = [pool[int(i)] for i in choice]
        forward_blocked = _run_intervention(
            model, primary["clean"]["ids"], primary["clean"]["am"],
            primary["source_position"], primary["natural_source"],
            primary["readout_position"], sites,
            [_site_value(primary["clean_heads"], site, head_dim)
             for site in sites], head_dim)
        reverse_blocked = _run_intervention(
            model, primary["natural"]["ids"], primary["natural"]["am"],
            primary["source_position"], primary["clean_source"],
            primary["readout_position"], sites,
            [_site_value(primary["natural_heads"], site, head_dim)
             for site in sites], head_dim)
        blocked = _cell(
            primary["clean_logits"], primary["natural_logits"],
            forward_blocked, reverse_blocked, primary["clean"],
            primary["source"], primary["target"])
        mediation = _mediation_pass(
            primary["source_intervention"], blocked,
            _accuracy(
                forward_blocked, primary["clean"], primary["source"]),
            _accuracy(
                reverse_blocked, primary["natural"], primary["target"]))
        random_cells.append({
            "random_index": random_index,
            "sites": [{"layer": a, "head": b} for a, b in sites],
            "blocked_intervention": blocked,
            "mediation": mediation,
        })
        hb.step(extra=f"random/{random_index}")

    selected_stat = primary["mediation"]["minimum_fraction"]
    random_stats = [
        cell["mediation"]["minimum_fraction"] for cell in random_cells
    ]
    empirical_p, exceed = _tail_probability(selected_stat, random_stats)

    wrong_rows = _counterfactual(rows, {"bc": TARGET})
    wrong_batch = _batch(
        tok, wrong_rows, "belief_ac", "narrative", dev)
    wrong_position = _anchor_position(primary["clean"], wrong_batch)
    _wrong_logits, wrong_state, _wrong_heads = _capture_source_heads(
        model, wrong_batch["ids"], wrong_batch["am"], wrong_position,
        primary["readout_position"], CANDIDATE_LAYERS)
    wrong_patch = _run_intervention(
        model, primary["clean"]["ids"], primary["clean"]["am"],
        wrong_position, wrong_state, primary["readout_position"],
        (), (), head_dim)
    source_ids = torch.tensor([
        primary["clean"]["amap"][x] for x in primary["source"]])
    target_ids = torch.tensor([
        primary["clean"]["amap"][x] for x in primary["target"]])
    clean_ld = _ld(primary["clean_logits"], target_ids, source_ids)
    wrong_ld = _ld(wrong_patch, target_ids, source_ids)
    natural_effect = primary["source_intervention"]["natural_effect"]
    wrong_ratio = (
        float((wrong_ld - clean_ld).mean()) / natural_effect
        if abs(natural_effect) > EPS else None)
    wrong_acc = float(_accuracy(
        wrong_patch, primary["clean"], primary["source"]))
    wrong_pass = bool(
        wrong_ratio is not None and abs(wrong_ratio) <= WRONG_DRIFT_GATE
        and wrong_acc >= G_ACC)
    wrong_control = {
        "query": "belief_ac",
        "patched_address": "bc",
        "source_layer": SOURCE_LAYER,
        "source_position": wrong_position,
        "target_drift_ratio": wrong_ratio,
        "alice_clean_accuracy": wrong_acc,
        "pass": wrong_pass,
    }
    hb.step(extra=f"wrong_address={wrong_pass}")
    hb.step(extra=f"null_p={empirical_p:.3f}")
    hb.done()

    public_tasks = {
        query: _public_task(context)
        for query, context in contexts.items()
    }
    verdict = _verdict(
        public_tasks, random_cells, empirical_p, wrong_pass)
    result = {
        "stage": "delta_source_head_mediation",
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
        "source_layer": SOURCE_LAYER,
        "candidate_layers": list(CANDIDATE_LAYERS),
        "n_heads": n_heads,
        "head_dim": head_dim,
        "frozen_top8": [
            {"layer": layer, "head": head}
            for layer, head in FROZEN_TOP8],
        "tasks": public_tasks,
        "random_null": {
            "n": len(random_cells),
            "observed_statistic": selected_stat,
            "random_exceedances": exceed,
            "empirical_p": empirical_p,
            "random_pass_count": sum(
                cell["mediation"]["pass"] for cell in random_cells),
            "max_random_statistic": max(random_stats),
            "cells": random_cells,
        },
        "wrong_address": wrong_control,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_source_head_mediation_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"SOURCE HEAD MEDIATION verdict={verdict} "
        f"p={empirical_p:.3f} artifact={path}")
    return result
