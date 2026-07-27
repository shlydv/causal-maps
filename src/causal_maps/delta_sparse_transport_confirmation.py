"""Locked cross-query confirmation of the sparse late transport set."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .delta_anchor_write import SOURCE, TARGET, _resolve
from .delta_preprint_battery import _compatible_world_rows
from .delta_sparse_transport import (
    G_ACC,
    _attention_geometry,
    _capture_heads,
    _cell,
    _forward_patch_heads,
    _site_value,
)
from .delta_structured_workspace import (
    _accuracy,
    _batch,
    _counterfactual,
    _locations,
)
from .logutil import Heartbeat, log
from .model_utils import (
    input_device,
    load_model_and_tokenizer,
    model_num_hidden_layers,
)

PROTOCOL_VERSION = "2026-07-24-p2-sparse-transport-confirm-v1"
PROTOCOL_SHA256 = (
    "A8C48BDC7CFD5D6317D3A9AD0037AFB2337C98376941BF34DB0ADE349282649E")
FROZEN_TOP4 = ((23, 11), (24, 21), (22, 1), (23, 6))
FROZEN_TOP8 = FROZEN_TOP4 + (
    (22, 25), (23, 4), (23, 13), (24, 27))
CONFIRM_TASKS = (
    ("tell_ac", "ac"),
    ("search_ac", "ac"),
)
EXPLORATORY_TASKS = (
    ("belief_as", "as"),
    ("belief_bc", "bc"),
)
RANDOM_SEED = 9407
CANDIDATE_LAYERS = (21, 22, 23, 24)


def _tail_probability(observed, random_values):
    exceed = sum(float(value) >= float(observed) for value in random_values)
    return (1.0 + exceed) / (1.0 + len(random_values)), exceed


def _bidirectional_stat(cell):
    values = [
        float(value)
        for value in (cell["forward_ratio"], cell["reverse_ratio"])
        if value is not None
    ]
    return min(values) if len(values) == 2 else float("-inf")


def _confirmation_verdict(tasks, random_cells, empirical_p):
    if not all(tasks[name]["eligible"] for name, _ in CONFIRM_TASKS):
        return "BEHAVIORALLY_INELIGIBLE"
    if (any(cell["sufficient"] for cell in random_cells)
            or float(empirical_p) > 0.01 + 1e-12):
        return "NONSPECIFIC_TRANSPORT"
    if all(tasks[name]["top4"]["sufficient"]
           for name, _ in CONFIRM_TASKS):
        return "LOCKED_SPARSE_TRANSPORT_CONFIRMED"
    return "QUERY_LIMITED_TRANSPORT"


def _task_batches(tok, rows, query, field, dev):
    natural_rows = _counterfactual(rows, {field: TARGET})
    clean = _batch(tok, rows, query, "narrative", dev)
    natural = _batch(tok, natural_rows, query, "narrative", dev)
    if clean["ids"].shape != natural["ids"].shape:
        raise ValueError(f"unaligned clean/natural batch for {query}")
    return natural_rows, clean, natural


@torch.no_grad()
def run_delta_sparse_transport_confirmation(
        model_path, out_dir,
        model_key="qwen7b_sparse_transport_confirmation",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=30, n_random=99, random_seed=RANDOM_SEED):
    os.makedirs(out_dir, exist_ok=True)
    if int(n_random) != 99:
        raise ValueError("confirmation v1 is frozen to 99 random sets")
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    rows, indices = _compatible_world_rows(
        tok, torch.device("cpu"), int(n_world))
    if len(rows) != 30:
        raise ValueError("confirmation v1 requires all 30 compatible worlds")

    # Random controls are sampled from the complete discovery search space,
    # including L21 even though no frozen top-eight site lies at that layer.
    layers = list(CANDIDATE_LAYERS)
    if max(layers) >= model_num_hidden_layers(model):
        raise ValueError("frozen layer is absent from model")
    n_heads, head_dim = _attention_geometry(model)
    if any(head >= n_heads for _layer, head in FROZEN_TOP8):
        raise ValueError("frozen head is absent from model")

    total = (len(CONFIRM_TASKS) + len(EXPLORATORY_TASKS)) * 3
    total += int(n_random) + 3
    hb = Heartbeat(
        total, "sparse_transport_confirmation",
        every_sec=30, out_dir=out_dir)

    caches = {}
    task_results = {}
    for query, field in CONFIRM_TASKS + EXPLORATORY_TASKS:
        natural_rows, clean, natural = _task_batches(
            tok, rows, query, field, dev)
        readout = int(clean["ids"].shape[1] - 1)
        clean_logits, clean_cache = _capture_heads(
            model, clean["ids"], clean["am"], layers, readout)
        natural_logits, natural_cache = _capture_heads(
            model, natural["ids"], natural["am"], layers, readout)
        hb.step(extra=f"{query}/baselines")
        source = _locations(rows, query)
        target = _locations(natural_rows, query)
        g0_clean = float(_accuracy(clean_logits, clean, source))
        g0_natural = float(_accuracy(natural_logits, natural, target))
        eligible = bool(min(g0_clean, g0_natural) >= G_ACC)
        cells = {}
        for label, sites in (("top4", FROZEN_TOP4),
                             ("top8", FROZEN_TOP8)):
            forward = _forward_patch_heads(
                model, clean["ids"], clean["am"], readout, sites,
                [_site_value(natural_cache, site, head_dim)
                 for site in sites], head_dim)
            reverse = _forward_patch_heads(
                model, natural["ids"], natural["am"], readout, sites,
                [_site_value(clean_cache, site, head_dim)
                 for site in sites], head_dim)
            cells[label] = _cell(
                clean_logits, natural_logits, forward, reverse,
                clean, source, target)
            hb.step(extra=f"{query}/{label}={cells[label]['sufficient']}")
        task_results[query] = {
            "field": field,
            "readout_position": readout,
            "g0_clean": g0_clean,
            "g0_natural": g0_natural,
            "eligible": eligible,
            **cells,
        }
        caches[query] = {
            "clean": clean,
            "natural": natural,
            "clean_logits": clean_logits,
            "natural_logits": natural_logits,
            "clean_cache": clean_cache,
            "natural_cache": natural_cache,
            "source": source,
            "target": target,
            "readout": readout,
        }

    primary = caches["tell_ac"]
    all_sites = [
        (layer, head) for layer in CANDIDATE_LAYERS
        for head in range(n_heads)
    ]
    if any(layer not in primary["clean_cache"]
           or layer not in primary["natural_cache"]
           for layer in CANDIDATE_LAYERS):
        raise RuntimeError("random-control candidate layer was not cached")
    pool = [site for site in all_sites if site not in set(FROZEN_TOP8)]
    rng = np.random.default_rng(int(random_seed))
    random_cells = []
    for random_index in range(int(n_random)):
        choice = rng.choice(len(pool), size=4, replace=False)
        sites = [pool[int(i)] for i in choice]
        forward = _forward_patch_heads(
            model, primary["clean"]["ids"], primary["clean"]["am"],
            primary["readout"], sites,
            [_site_value(primary["natural_cache"], site, head_dim)
             for site in sites], head_dim)
        reverse = _forward_patch_heads(
            model, primary["natural"]["ids"], primary["natural"]["am"],
            primary["readout"], sites,
            [_site_value(primary["clean_cache"], site, head_dim)
             for site in sites], head_dim)
        cell = _cell(
            primary["clean_logits"], primary["natural_logits"],
            forward, reverse, primary["clean"],
            primary["source"], primary["target"])
        random_cells.append({
            "random_index": random_index,
            "sites": [{"layer": a, "head": b} for a, b in sites],
            **cell,
        })
        hb.step(extra=f"random4/{random_index}")

    observed = _bidirectional_stat(task_results["tell_ac"]["top4"])
    random_statistics = [
        _bidirectional_stat(cell)
        for cell in random_cells
    ]
    empirical_p, exceed = _tail_probability(observed, random_statistics)

    layer_groups = {
        "L22": ((22, 1),),
        "L23": ((23, 11), (23, 6)),
        "L24": ((24, 21),),
    }
    grouped_controls = {}
    for label, sites in layer_groups.items():
        forward = _forward_patch_heads(
            model, primary["clean"]["ids"], primary["clean"]["am"],
            primary["readout"], sites,
            [_site_value(primary["natural_cache"], site, head_dim)
             for site in sites], head_dim)
        reverse = _forward_patch_heads(
            model, primary["natural"]["ids"], primary["natural"]["am"],
            primary["readout"], sites,
            [_site_value(primary["clean_cache"], site, head_dim)
             for site in sites], head_dim)
        grouped_controls[label] = {
            "sites": [{"layer": a, "head": b} for a, b in sites],
            **_cell(
                primary["clean_logits"], primary["natural_logits"],
                forward, reverse, primary["clean"],
                primary["source"], primary["target"]),
        }
        hb.step(extra=f"group/{label}")
    hb.done()

    verdict = _confirmation_verdict(
        task_results, random_cells, empirical_p)
    result = {
        "stage": "delta_sparse_transport_confirmation",
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
        "n_heads": n_heads,
        "head_dim": head_dim,
        "frozen_top4": [
            {"layer": layer, "head": head}
            for layer, head in FROZEN_TOP4],
        "frozen_top8": [
            {"layer": layer, "head": head}
            for layer, head in FROZEN_TOP8],
        "tasks": task_results,
        "random_null": {
            "query": "tell_ac",
            "n": len(random_cells),
            "observed_statistic": observed,
            "random_exceedances": exceed,
            "empirical_p": empirical_p,
            "random_pass_count": sum(
                cell["sufficient"] for cell in random_cells),
            "max_random_statistic": max(random_statistics),
            "cells": random_cells,
        },
        "same_layer_controls": grouped_controls,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir,
        f"results_delta_sparse_transport_confirmation_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"SPARSE TRANSPORT CONFIRM verdict={verdict} "
        f"p={empirical_p:.3f} artifact={path}")
    return result
