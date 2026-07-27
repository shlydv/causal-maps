"""Held-out head-level causal screen for the late query-readout handoff."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .delta_anchor_write import SOURCE, TARGET, _resolve
from .delta_preprint_battery import _compatible_world_rows
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
    model_hidden_size,
    model_num_hidden_layers,
    model_text_config,
)

PROTOCOL_VERSION = "2026-07-24-p2-sparse-transport-v1"
PROTOCOL_SHA256 = (
    "B1D8113AB720DFDAC303A74ADA130617CCA417A45AC3BFFDF63323266D041F0F")
DEFAULT_LAYERS = (21, 22, 23, 24)
DEFAULT_TOP_K = (1, 2, 4, 8)
G_ACC = 0.80
RATIO_GATE = (0.60, 1.40)
EPS = 1e-8
RANDOM_SEED = 4283


def _fixed_split(rows, n_donor=15):
    if not 1 <= int(n_donor) < len(rows):
        raise ValueError("n_donor must leave donor and evaluation rows")
    return list(rows[:int(n_donor)]), list(rows[int(n_donor):])


def _attention_geometry(model):
    config = model_text_config(model)
    n_heads = int(config.num_attention_heads)
    hidden = model_hidden_size(model)
    if hidden % n_heads:
        raise ValueError("hidden size is not divisible by attention heads")
    return n_heads, hidden // n_heads


def _o_proj(model, layer):
    block = get_decoder_layers(model)[int(layer)]
    attention = getattr(block, "self_attn", None)
    projection = getattr(attention, "o_proj", None)
    if projection is None:
        raise AttributeError(
            f"layer {layer} lacks self_attn.o_proj")
    return projection


@torch.no_grad()
def _capture_heads(model, ids, am, layers, readout):
    cache = {}
    handles = []

    def capture(layer):
        def hook(_module, args):
            states = args[0]
            cache[int(layer)] = (
                states[:, int(readout), :].detach().float().cpu())
        return hook

    for layer in layers:
        handles.append(_o_proj(model, layer).register_forward_pre_hook(
            capture(layer)))
    try:
        output = model(
            input_ids=ids, attention_mask=am, use_cache=False)
        logits = output.logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return logits, cache


@torch.no_grad()
def _forward_patch_heads(model, ids, am, readout, sites, values, head_dim):
    """Patch matched pre-o_proj head slices at the query readout position."""
    grouped = {}
    for (layer, head), value in zip(sites, values):
        grouped.setdefault(int(layer), []).append((int(head), value))
    handles = []

    def patch(entries):
        def hook(_module, args):
            states = args[0].clone()
            for head, value in entries:
                start = int(head) * int(head_dim)
                stop = start + int(head_dim)
                replacement = value.to(
                    device=states.device, dtype=states.dtype)
                states[:, int(readout), start:stop] = replacement
            return (states,) + tuple(args[1:])
        return hook

    for layer, entries in grouped.items():
        handles.append(_o_proj(model, layer).register_forward_pre_hook(
            patch(entries)))
    try:
        output = model(
            input_ids=ids, attention_mask=am, use_cache=False)
        return output.logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()


def _site_value(cache, site, head_dim):
    layer, head = site
    start = int(head) * int(head_dim)
    return cache[int(layer)][:, start:start + int(head_dim)]


def _cell(clean_logits, natural_logits, forward_logits, reverse_logits,
          batch, source, target):
    source_ids = torch.tensor([batch["amap"][x] for x in source])
    target_ids = torch.tensor([batch["amap"][x] for x in target])
    clean_ld = _ld(clean_logits, target_ids, source_ids)
    natural_ld = _ld(natural_logits, target_ids, source_ids)
    natural_rows = natural_ld - clean_ld
    natural_effect = float(natural_rows.mean())
    forward_ld = _ld(forward_logits, target_ids, source_ids)
    reverse_ld = _ld(reverse_logits, target_ids, source_ids)
    forward_rows = forward_ld - clean_ld
    reverse_rows = natural_ld - reverse_ld
    f_ratio = (
        float(forward_rows.mean()) / natural_effect
        if abs(natural_effect) > EPS else None)
    r_ratio = (
        float(reverse_rows.mean()) / natural_effect
        if abs(natural_effect) > EPS else None)
    lo, hi = RATIO_GATE
    sufficient = bool(
        f_ratio is not None and r_ratio is not None
        and lo <= f_ratio <= hi and lo <= r_ratio <= hi
        and _accuracy(forward_logits, batch, target) >= G_ACC
        and _accuracy(reverse_logits, batch, source) >= G_ACC)
    return {
        "natural_effect": natural_effect,
        "forward_ratio": f_ratio,
        "reverse_ratio": r_ratio,
        "forward_target_acc": float(
            _accuracy(forward_logits, batch, target)),
        "reverse_clean_acc": float(
            _accuracy(reverse_logits, batch, source)),
        "forward_effect_rows": forward_rows.tolist(),
        "reverse_effect_rows": reverse_rows.tolist(),
        "natural_effect_rows": natural_rows.tolist(),
        "sufficient": sufficient,
    }


def _discovery_score(clean_logits, patched_logits, batch, source, target,
                     natural_effect):
    source_ids = torch.tensor([batch["amap"][x] for x in source])
    target_ids = torch.tensor([batch["amap"][x] for x in target])
    clean_ld = _ld(clean_logits, target_ids, source_ids)
    patched_ld = _ld(patched_logits, target_ids, source_ids)
    effect_rows = patched_ld - clean_ld
    effect = float(effect_rows.mean())
    return {
        "effect": effect,
        "ratio": effect / natural_effect
        if abs(natural_effect) > EPS else None,
        "target_acc": float(_accuracy(
            patched_logits, batch, target)),
        "positive_fraction": float(
            (effect_rows > 0).float().mean()),
        "effect_rows": effect_rows.tolist(),
    }


def _verdict(eligible, top_cells, random_cells, full_layers):
    if not eligible:
        return "BEHAVIORALLY_INELIGIBLE"
    for k in sorted(top_cells, key=int):
        if top_cells[k]["sufficient"]:
            random_pass = any(
                cell["sufficient"] for cell in random_cells[k])
            return (
                "NONSPECIFIC_HEAD_SET"
                if random_pass else "SPARSE_TRANSPORT_PATH")
    if any(cell["sufficient"] for cell in full_layers.values()):
        return "DISTRIBUTED_ATTENTION_TRANSPORT"
    return "ATTENTION_OUTPUT_NOT_SUFFICIENT"


@torch.no_grad()
def run_delta_sparse_transport(
        model_path, out_dir, model_key="qwen7b_sparse_transport_d1",
        quantization="8bit", device_map=None, max_memory=None,
        n_world=30, n_donor=15, layers=DEFAULT_LAYERS,
        top_k=DEFAULT_TOP_K, n_random=5, random_seed=RANDOM_SEED):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    rows, indices = _compatible_world_rows(
        tok, torch.device("cpu"), int(n_world))
    donor_rows, eval_rows = _fixed_split(rows, n_donor)
    donor_natural_rows = _counterfactual(
        donor_rows, {"ac": TARGET})
    eval_natural_rows = _counterfactual(
        eval_rows, {"ac": TARGET})
    donor_clean = _batch(
        tok, donor_rows, "belief_ac", "narrative", dev)
    donor_natural = _batch(
        tok, donor_natural_rows, "belief_ac", "narrative", dev)
    eval_clean = _batch(
        tok, eval_rows, "belief_ac", "narrative", dev)
    eval_natural = _batch(
        tok, eval_natural_rows, "belief_ac", "narrative", dev)
    if (donor_clean["ids"].shape != donor_natural["ids"].shape
            or eval_clean["ids"].shape != eval_natural["ids"].shape):
        raise ValueError("clean/natural transport batches are not aligned")
    donor_readout = int(donor_clean["ids"].shape[1] - 1)
    eval_readout = int(eval_clean["ids"].shape[1] - 1)
    if donor_readout != eval_readout:
        raise ValueError("donor/evaluation readout positions differ")

    n_model_layers = model_num_hidden_layers(model)
    selected_layers = sorted({
        int(layer) for layer in layers
        if 0 <= int(layer) < n_model_layers
    })
    if not selected_layers:
        raise ValueError("no requested transport layer exists")
    selected_k = tuple(sorted({int(k) for k in top_k}))
    if selected_k != DEFAULT_TOP_K:
        raise ValueError("v1 is frozen to top-K [1,2,4,8]")
    n_heads, head_dim = _attention_geometry(model)
    all_sites = [
        (layer, head)
        for layer in selected_layers
        for head in range(n_heads)
    ]

    donor_clean_logits, donor_clean_cache = _capture_heads(
        model, donor_clean["ids"], donor_clean["am"],
        selected_layers, donor_readout)
    donor_natural_logits, donor_natural_cache = _capture_heads(
        model, donor_natural["ids"], donor_natural["am"],
        selected_layers, donor_readout)
    expected_width = n_heads * head_dim
    if any(cache.shape[1] != expected_width
           for cache in donor_clean_cache.values()):
        raise ValueError(
            "pre-o_proj width does not match attention-head geometry")
    donor_source = _locations(donor_rows, "belief_ac")
    donor_target = _locations(
        donor_natural_rows, "belief_ac")
    donor_source_ids = torch.tensor(
        [donor_clean["amap"][x] for x in donor_source])
    donor_target_ids = torch.tensor(
        [donor_clean["amap"][x] for x in donor_target])
    donor_clean_ld = _ld(
        donor_clean_logits, donor_target_ids, donor_source_ids)
    donor_natural_ld = _ld(
        donor_natural_logits, donor_target_ids, donor_source_ids)
    donor_natural_effect = float(
        (donor_natural_ld - donor_clean_ld).mean())

    total = len(all_sites) + len(selected_k) * (1 + int(n_random))
    total += len(selected_layers)
    hb = Heartbeat(
        total, "sparse_transport", every_sec=30, out_dir=out_dir)
    discovery = {}
    for site in all_sites:
        patched = _forward_patch_heads(
            model, donor_clean["ids"], donor_clean["am"],
            donor_readout, [site],
            [_site_value(donor_natural_cache, site, head_dim)],
            head_dim)
        score = _discovery_score(
            donor_clean_logits, patched, donor_clean,
            donor_source, donor_target, donor_natural_effect)
        discovery[f"L{site[0]}H{site[1]}"] = {
            "layer": site[0], "head": site[1], **score}
        hb.step(extra=(
            f"discover L{site[0]}H{site[1]} "
            f"ratio={score['ratio']:.3f}"))
    ranked = sorted(
        all_sites,
        key=lambda site: discovery[
            f"L{site[0]}H{site[1]}"]["effect"],
        reverse=True)

    eval_clean_logits, eval_clean_cache = _capture_heads(
        model, eval_clean["ids"], eval_clean["am"],
        selected_layers, eval_readout)
    eval_natural_logits, eval_natural_cache = _capture_heads(
        model, eval_natural["ids"], eval_natural["am"],
        selected_layers, eval_readout)
    eval_source = _locations(eval_rows, "belief_ac")
    eval_target = _locations(
        eval_natural_rows, "belief_ac")
    g0_clean = float(_accuracy(
        eval_clean_logits, eval_clean, eval_source))
    g0_natural = float(_accuracy(
        eval_natural_logits, eval_natural, eval_target))
    eligible = bool(min(g0_clean, g0_natural) >= G_ACC)

    top_cells = {}
    random_cells = {}
    rng = np.random.default_rng(int(random_seed))
    max_top = set(ranked[:max(selected_k)])
    random_pool = [site for site in all_sites if site not in max_top]
    for k in selected_k:
        sites = ranked[:k]
        fwd = _forward_patch_heads(
            model, eval_clean["ids"], eval_clean["am"],
            eval_readout, sites,
            [_site_value(eval_natural_cache, site, head_dim)
             for site in sites], head_dim)
        rev = _forward_patch_heads(
            model, eval_natural["ids"], eval_natural["am"],
            eval_readout, sites,
            [_site_value(eval_clean_cache, site, head_dim)
             for site in sites], head_dim)
        top_cells[str(k)] = {
            "sites": [{"layer": s[0], "head": s[1]} for s in sites],
            **_cell(
                eval_clean_logits, eval_natural_logits, fwd, rev,
                eval_clean, eval_source, eval_target),
        }
        hb.step(extra=f"top{k}={top_cells[str(k)]['sufficient']}")

        random_cells[str(k)] = []
        for random_index in range(int(n_random)):
            choice = rng.choice(
                len(random_pool), size=k, replace=False)
            random_sites = [random_pool[int(i)] for i in choice]
            rfwd = _forward_patch_heads(
                model, eval_clean["ids"], eval_clean["am"],
                eval_readout, random_sites,
                [_site_value(eval_natural_cache, site, head_dim)
                 for site in random_sites], head_dim)
            rrev = _forward_patch_heads(
                model, eval_natural["ids"], eval_natural["am"],
                eval_readout, random_sites,
                [_site_value(eval_clean_cache, site, head_dim)
                 for site in random_sites], head_dim)
            random_cells[str(k)].append({
                "random_index": random_index,
                "sites": [
                    {"layer": s[0], "head": s[1]}
                    for s in random_sites],
                **_cell(
                    eval_clean_logits, eval_natural_logits,
                    rfwd, rrev, eval_clean,
                    eval_source, eval_target),
            })
            hb.step(extra=f"random{k}/{random_index}")

    full_layers = {}
    for layer in selected_layers:
        sites = [(layer, head) for head in range(n_heads)]
        fwd = _forward_patch_heads(
            model, eval_clean["ids"], eval_clean["am"],
            eval_readout, sites,
            [_site_value(eval_natural_cache, site, head_dim)
             for site in sites], head_dim)
        rev = _forward_patch_heads(
            model, eval_natural["ids"], eval_natural["am"],
            eval_readout, sites,
            [_site_value(eval_clean_cache, site, head_dim)
             for site in sites], head_dim)
        full_layers[str(layer)] = _cell(
            eval_clean_logits, eval_natural_logits, fwd, rev,
            eval_clean, eval_source, eval_target)
        hb.step(extra=(
            f"fullL{layer}={full_layers[str(layer)]['sufficient']}"))
    hb.done()

    verdict = _verdict(
        eligible, top_cells, random_cells, full_layers)
    result = {
        "stage": "delta_sparse_transport",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "world_selection": {
            "requested": int(n_world),
            "selected": len(rows),
            "indices_from_30": indices,
            "n_donor": len(donor_rows),
            "n_evaluation": len(eval_rows),
            "donor_indices_from_selected": list(
                range(len(donor_rows))),
            "evaluation_indices_from_selected": list(
                range(len(donor_rows), len(rows))),
        },
        "layers": selected_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "readout_position": eval_readout,
        "g0_clean": g0_clean,
        "g0_natural": g0_natural,
        "eligible": eligible,
        "donor_natural_effect": donor_natural_effect,
        "discovery": discovery,
        "ranking": [
            {"layer": site[0], "head": site[1],
             "effect": discovery[
                 f"L{site[0]}H{site[1]}"]["effect"]}
            for site in ranked],
        "top_k": top_cells,
        "random": random_cells,
        "full_layer": full_layers,
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir, f"results_delta_sparse_transport_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"SPARSE TRANSPORT g0={g0_clean:.0%}/{g0_natural:.0%} "
        f"verdict={verdict} artifact={path}")
    return result
