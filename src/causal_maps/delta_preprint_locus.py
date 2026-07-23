"""Frozen multi-token causal-support curve for the Paper 1 headline world.

This is an additive robustness stage. It imports the validated structured-world
renderer and sparse capture path but owns its multi-position patch hook, so it
cannot change any completed battery's intervention behavior.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .delta_anchor_write import SOURCE, TARGET, _resolve
from .delta_preprint_battery import (_compatible_world_rows,
                                     _last_overlap_token)
from .delta_structured_workspace import (MARKER, _accuracy, _batch,
                                         _counterfactual, _locations)
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import (get_decoder_layers, input_device,
                          load_model_and_tokenizer, model_num_hidden_layers)
from .patching import _split_output

PROTOCOL_VERSION = "2026-07-21-v1"
LAYERS = (2, 4, 8, 12, 16, 20, 24, 26, 32, 36, 41, 46)
G0 = 0.80
RATIO_GATE = (0.60, 1.40)
N_RANDOM_LOCI = 3
RANDOM_SEED = 2718

SUMMARY_START = "Silently compute all six state variables now."
ANCHOR_NEEDLES = {
    "ac": "Alice watched a worker leave the cube in {ac}",
    "bc": "Later Bob, without Alice present, watched it go to {bc}",
    "as": "Alice separately watched the sphere go to {as}",
    "bs": "Bob later watched the sphere go to {bs}",
    "tc": "unseen final moves put the cube in {tc}",
    "ts": "and the sphere in {ts}",
}


def _overlap_span_tokens(tok, text, start_needle, end_needle):
    """All tokens overlapping the literal range, inclusive of both needles."""
    start = text.rfind(start_needle)
    finish_start = text.rfind(end_needle)
    if start < 0 or finish_start < start:
        raise ValueError("locus span literals absent or reversed")
    end = finish_start + len(end_needle)
    try:
        offsets = tok(text, add_special_tokens=False,
                      return_offsets_mapping=True)["offset_mapping"]
        hits = [i for i, (a, b) in enumerate(offsets)
                if a < end and b > start]
        if hits:
            return hits
    except (TypeError, KeyError, NotImplementedError):
        pass
    left = len(tok.encode(text[:start], add_special_tokens=False))
    right = len(tok.encode(text[:end], add_special_tokens=False))
    if right <= left:
        raise ValueError("could not tokenize locus span")
    return list(range(left, right))


def _uniform_locus_positions(tok, batch, rows):
    """Mechanically locate summary and six event-value anchors for all rows."""
    per_row_summary = [
        _overlap_span_tokens(tok, text, SUMMARY_START, MARKER)
        for text in batch["texts"]
    ]
    if len({tuple(x) for x in per_row_summary}) != 1:
        raise ValueError("summary-span positions vary across rows")
    summary = per_row_summary[0]
    per_field = {}
    for field, template in ANCHOR_NEEDLES.items():
        positions = [
            _last_overlap_token(tok, text, template.format(**row))
            for text, row in zip(batch["texts"], rows)
        ]
        if len(set(positions)) != 1:
            raise ValueError(f"{field} anchor positions vary: {positions}")
        per_field[field] = positions[0]
    if len(set(per_field.values())) != len(per_field):
        raise ValueError(f"anchor positions collide: {per_field}")
    if summary[-1] != batch["marker"]:
        raise ValueError("summary locus does not end at STATECHECK")
    return summary, per_field


def _build_loci(marker, summary, anchors, sequence_length=None,
                 n_random=N_RANDOM_LOCI, seed=RANDOM_SEED):
    """Pure frozen locus construction, separated for contract testing."""
    anchor_values = sorted(anchors.values())
    union = sorted(set(anchor_values) | set(summary))
    sequence_length = (int(marker) + 1 if sequence_length is None
                       else int(sequence_length))
    if sequence_length <= marker:
        raise ValueError("sequence must extend through marker")
    loci = {
        "marker_only": [int(marker)],
        "summary_span": list(summary),
        "source_anchors": anchor_values,
        "anchors_plus_summary": union,
        "full_prequery": list(range(int(marker) + 1)),
        "full_matched_prefix": list(range(sequence_length)),
    }
    for field in sorted(anchors):
        loci[f"anchors_without_{field}"] = [
            pos for key, pos in sorted(anchors.items()) if key != field
        ]
    candidates = sorted(set(range(int(marker) + 1)) - set(union))
    if n_random and len(candidates) < len(union):
        raise ValueError("not enough disjoint positions for size-matched loci")
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        loci[f"random_size_matched_{i}"] = sorted(
            rng.choice(candidates, size=len(union), replace=False).tolist())
    return loci


@torch.no_grad()
def _forward_multi_patch(model, ids, am, layer_idx, positions, value):
    """Patch a set of positions in one hook and return next-token logits."""
    layers = get_decoder_layers(model)
    positions = list(positions)

    def patch_hook(module, inp, out):
        hs, rebuild = _split_output(out)
        hs = hs.clone()
        index = torch.as_tensor(positions, dtype=torch.long, device=hs.device)
        replacement = value.to(device=hs.device, dtype=hs.dtype)
        if replacement.ndim != 3 or replacement.shape[1] != len(positions):
            raise ValueError("multi-position patch must have shape [B,P,D]")
        hs[:, index, :] = replacement
        return rebuild(hs)

    handle = layers[layer_idx].register_forward_hook(patch_hook)
    try:
        out = model(input_ids=ids, attention_mask=am, use_cache=False)
        return out.logits[:, -1, :].detach().float().cpu()
    finally:
        handle.remove()


def _cell_sufficient(cell):
    lo, hi = RATIO_GATE
    return (lo <= cell["forward_ratio"] <= hi
            and lo <= cell["reverse_ratio"] <= hi
            and cell["forward_target_acc"] >= G0
            and cell["reverse_clean_acc"] >= G0)


def _curve_verdict(per_layer):
    """Conservative descriptive category; full numeric curve remains primary."""
    def ever(locus):
        return any(_cell_sufficient(block[locus])
                   for block in per_layer.values())

    if not ever("full_matched_prefix"):
        return "UPPER_BOUND_FAILED"
    if ever("marker_only"):
        return "TOKEN_LOCAL_SUFFICIENT"
    if ever("source_anchors"):
        return "SOURCE_ANCHORS_SUFFICIENT"
    if ever("summary_span"):
        return "LOCAL_SUMMARY_SUFFICIENT"
    if ever("anchors_plus_summary"):
        return "LOCAL_UNION_SUFFICIENT"
    if ever("full_prequery"):
        return "DISTRIBUTED_PREQUERY_SUFFICIENT"
    return "QUERY_OR_READOUT_CONTEXT_REQUIRED"


def run_delta_preprint_locus_preflight(model_path, out_dir, n_world=30,
                                       n_random_loci=N_RANDOM_LOCI,
                                       random_seed=RANDOM_SEED):
    """Tokenizer-only contract check before allocating the 14B weights."""
    from transformers import AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)
    resolved = _resolve(model_path)
    tok = AutoTokenizer.from_pretrained(resolved)
    rows, indices = _compatible_world_rows(
        tok, torch.device("cpu"), n_world)
    natural_rows = _counterfactual(rows, {"ac": TARGET})
    cb = _batch(tok, rows, "belief_ac", "narrative", torch.device("cpu"))
    nb = _batch(tok, natural_rows, "belief_ac", "narrative",
                torch.device("cpu"))
    if cb["ids"].shape != nb["ids"].shape or cb["marker"] != nb["marker"]:
        raise ValueError("clean/natural preflight batches are not aligned")
    summary, anchors = _uniform_locus_positions(tok, cb, rows)
    natural_summary, natural_anchors = _uniform_locus_positions(
        tok, nb, natural_rows)
    if summary != natural_summary or anchors != natural_anchors:
        raise ValueError("clean/natural preflight loci differ")
    loci = _build_loci(
        cb["marker"], summary, anchors,
        sequence_length=int(cb["ids"].shape[1]),
        n_random=n_random_loci, seed=random_seed)
    result = {
        "stage": "delta_preprint_locus_preflight",
        "protocol_version": PROTOCOL_VERSION,
        "model_path": model_path,
        "resolved_model_path": resolved,
        "requested_worlds": n_world,
        "selected_worlds": len(rows),
        "indices_from_30": indices,
        "sequence_length": int(cb["ids"].shape[1]),
        "marker": cb["marker"],
        "summary": summary,
        "anchors": anchors,
        "locus_sizes": {name: len(pos) for name, pos in loci.items()},
        "verdict": "PREFLIGHT_PASS",
    }
    path = os.path.join(out_dir, "results_preprint_locus_preflight.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"LOCUS PREFLIGHT PASS n={len(rows)} seq={cb['ids'].shape[1]} "
        f"marker={cb['marker']} anchors={anchors}")
    return result


@torch.no_grad()
def run_delta_preprint_locus(model_path, out_dir, model_key="qwen14b_locus",
                             quantization="awq", device_map=None,
                             max_memory=None, n_world=30, layers=LAYERS,
                             n_random_loci=N_RANDOM_LOCI,
                             random_seed=RANDOM_SEED):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    rows, indices = _compatible_world_rows(tok, torch.device("cpu"), n_world)
    natural_rows = _counterfactual(rows, {"ac": TARGET})
    cb = _batch(tok, rows, "belief_ac", "narrative", dev)
    nb = _batch(tok, natural_rows, "belief_ac", "narrative", dev)
    if cb["ids"].shape != nb["ids"].shape or cb["marker"] != nb["marker"]:
        raise ValueError("clean/natural locus batches are not aligned")

    summary, anchors = _uniform_locus_positions(tok, cb, rows)
    natural_summary, natural_anchors = _uniform_locus_positions(
        tok, nb, natural_rows)
    if summary != natural_summary or anchors != natural_anchors:
        raise ValueError("clean/natural locus positions differ")
    loci = _build_loci(cb["marker"], summary, anchors,
                       sequence_length=int(cb["ids"].shape[1]),
                       n_random=n_random_loci, seed=random_seed)
    capture_positions = list(range(int(cb["ids"].shape[1])))

    n_layers = model_num_hidden_layers(model)
    layers = LAYERS if layers is None else layers
    layers = [int(x) for x in layers if 0 <= int(x) < n_layers]
    if not layers:
        raise ValueError("no requested locus layers exist in model")

    src = _locations(rows, "belief_ac")
    tgt = _locations(natural_rows, "belief_ac")
    sid = torch.tensor([cb["amap"][x] for x in src])
    tid = torch.tensor([cb["amap"][x] for x in tgt])

    clean_logits, _ = _forward(model, cb["ids"], cb["am"], ())
    natural_logits, _ = _forward(model, nb["ids"], nb["am"], ())
    clean_ld = _ld(clean_logits, tid, sid)
    natural_ld = _ld(natural_logits, tid, sid)
    natural_effect_rows = natural_ld - clean_ld
    natural_effect = float(natural_effect_rows.mean())
    g0_clean = float(_accuracy(clean_logits, cb, src))
    g0_natural = float(_accuracy(natural_logits, nb, tgt))

    result = {
        "stage": "delta_preprint_locus",
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "world_selection": {"requested": n_world, "selected": len(rows),
                            "indices_from_30": indices},
        "layers": layers,
        "positions": {"marker": cb["marker"], "summary": summary,
                      "anchors": anchors, "loci": loci},
        "g0_clean": g0_clean,
        "g0_natural": g0_natural,
        "clean_ld_rows": clean_ld.tolist(),
        "natural_ld_rows": natural_ld.tolist(),
        "natural_effect_rows": natural_effect_rows.tolist(),
        "natural_effect": natural_effect,
        "per_layer": {},
    }
    log(f"LOCUS G0 clean={g0_clean:.0%} natural={g0_natural:.0%} "
        f"n={len(rows)} marker={cb['marker']} loci={len(loci)}")
    if min(g0_clean, g0_natural) < G0 or abs(natural_effect) < 1e-8:
        result["verdict"] = "BEHAVIORALLY_INELIGIBLE"
    else:
        for layer in layers:
            log(f"=== locus layer L{layer} ===")
            _, clean_cache = _forward(
                model, cb["ids"], cb["am"], capture_positions, (layer,))
            _, natural_cache = _forward(
                model, nb["ids"], nb["am"], capture_positions, (layer,))
            cstate = clean_cache[layer]
            nstate = natural_cache[layer]
            layer_out = {}
            for name, positions in loci.items():
                # Capture order equals absolute positions 0..sequence_length-1.
                fwd = _forward_multi_patch(
                    model, cb["ids"], cb["am"], layer, positions,
                    nstate[:, positions, :])
                rev = _forward_multi_patch(
                    model, nb["ids"], nb["am"], layer, positions,
                    cstate[:, positions, :])
                fwd_ld = _ld(fwd, tid, sid)
                rev_ld = _ld(rev, tid, sid)
                fwd_rows = fwd_ld - clean_ld
                rev_rows = natural_ld - rev_ld
                cell = {
                    "n_positions": len(positions),
                    "forward_ratio": float(fwd_rows.mean()) / natural_effect,
                    "reverse_ratio": float(rev_rows.mean()) / natural_effect,
                    "forward_target_acc": float(_accuracy(fwd, cb, tgt)),
                    "reverse_clean_acc": float(_accuracy(rev, nb, src)),
                    "forward_ld_rows": fwd_ld.tolist(),
                    "reverse_ld_rows": rev_ld.tolist(),
                    "forward_effect_rows": fwd_rows.tolist(),
                    "reverse_effect_rows": rev_rows.tolist(),
                }
                cell["sufficient"] = _cell_sufficient(cell)
                layer_out[name] = cell
                log(f"  {name}: npos={len(positions)} "
                    f"fwd={cell['forward_ratio']:.3f}/{cell['forward_target_acc']:.0%} "
                    f"rev={cell['reverse_ratio']:.3f}/{cell['reverse_clean_acc']:.0%}")
            result["per_layer"][layer] = layer_out
            del clean_cache, natural_cache, cstate, nstate
        result["verdict"] = _curve_verdict(result["per_layer"])
        result["random_controls_sufficient"] = sum(
            cell["sufficient"]
            for block in result["per_layer"].values()
            for name, cell in block.items()
            if name.startswith("random_size_matched_"))

    path = os.path.join(out_dir,
                        f"results_delta_preprint_locus_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(f"LOCUS verdict={result['verdict']} artifact={path}")
    return result
