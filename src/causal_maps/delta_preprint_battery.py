"""Preprint battery — one model load, the full headline suite, all seeds.

PREPRINT_PLAN.md M1/M2 vehicle (pre-registered widening, 2026-07-15):
  per genuine row-generation seed in `seeds`:
    1. workspace matrix     (5 cells, n_rows=30, 50 nulls)
    2. entity matrix        (2 families x 3 cells, n_rows=30, 30 nulls)
  once per model (rows are seed-independent):
    3. anchor-write         (all 30 distinct structured worlds, 99 nulls)
    4. checkpoint trajectory (STATECHECK primary plus question/readout
                              controls, full-depth layer sweep)

All protocols identical to the frozen originals except the pre-registered
row widening and seed sweep; original runs stand as pilots. Sub-results are
written under out_dir/s{seed}/ and embedded in one battery JSON.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .delta_anchor_write import (SOURCE, TARGET, _anchor_position, _resolve,
                                 run_delta_anchor_write)
from .delta_entity_matrix import _run_model as run_entity_model
from .delta_structured_workspace import (QUERY, _accuracy, _batch,
                                         _counterfactual, _locations, _rows)
from .delta_trajectory import _forward, _ld
from .delta_workspace_matrix import run_delta_workspace_matrix
from .logutil import Heartbeat, log
from .model_utils import input_device, load_model_and_tokenizer

CHECKPOINT_LAYERS = (2, 4, 8, 12, 16, 20, 24, 26)


def _full_depth_layers(n_layers, requested=None):
    """Frozen depth grid plus late fractional-depth sites for larger models."""
    base = list(requested or CHECKPOINT_LAYERS)
    base += [round(.66 * n_layers), round(.75 * n_layers),
             round(.85 * n_layers), n_layers - 2]
    return sorted({int(x) for x in base if 0 <= int(x) < n_layers})


def _last_overlap_token(tok, text, needle):
    """Last token overlapping the final occurrence of a literal substring."""
    start = text.rfind(needle)
    if start < 0:
        raise ValueError(f"substring absent: {needle!r}")
    end = start + len(needle)
    try:
        offsets = tok(text, add_special_tokens=False,
                      return_offsets_mapping=True)["offset_mapping"]
        hits = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
        if hits:
            return hits[-1]
    except (TypeError, KeyError, NotImplementedError):
        pass
    prefix = tok.encode(text[:end], add_special_tokens=False)
    if not prefix:
        raise ValueError(f"could not locate token for: {needle!r}")
    return len(prefix) - 1


def _compatible_world_rows(tok, dev, requested):
    """Largest tokenizer-aligned bucket, mechanically selected from 30 worlds."""
    candidates = _rows(SOURCE, TARGET, "ac", "test", n_rows=30)
    ac_rows = _counterfactual(candidates, {"ac": TARGET})
    bc_rows = _counterfactual(candidates, {"bc": TARGET})
    queries = ("belief_ac", "tell_ac", "belief_as", "belief_bc",
               "truth_cube", "truth_sphere")
    buckets = {}
    for i, (row, ac_row, bc_row) in enumerate(zip(candidates, ac_rows, bc_rows)):
        signature = []
        for query in queries:
            b = _batch(tok, [row], query, "narrative", dev)
            signature.extend((int(b["ids"].shape[1]), int(b["marker"])))
        for query in ("belief_ac", "tell_ac"):
            b = _batch(tok, [ac_row], query, "narrative", dev)
            signature.extend((int(b["ids"].shape[1]), int(b["marker"])))
        b = _batch(tok, [bc_row], "belief_bc", "narrative", dev)
        signature.extend((int(b["ids"].shape[1]), int(b["marker"])))
        clean = _batch(tok, [row], "belief_ac", "narrative", dev)
        natural = _batch(tok, [ac_row], "belief_ac", "narrative", dev)
        signature.append(_anchor_position(clean, natural))
        buckets.setdefault(tuple(signature), []).append(i)
    selected = max(buckets.values(), key=lambda idx: (len(idx), -idx[0]))
    selected = selected[:min(requested, len(selected))]
    rows = [candidates[i] for i in selected]
    log(f"structured tokenizer bucket: requested={requested} selected={len(rows)} "
        f"from 30 candidates indices={selected}")
    return rows, selected


@torch.no_grad()
def _checkpoint_cell(model, tok, dev, layers, rows, query="belief_ac"):
    """Matched causal trajectory with a query-independent primary site.

    STATECHECK is the pre-registered checkpoint endpoint. The question and
    readout sites are positive/process controls showing where query-specific
    answer state becomes causally transportable.
    """
    nat = _counterfactual(rows, {"ac": TARGET})
    cb = _batch(tok, rows, query, "narrative", dev)
    nb = _batch(tok, nat, query, "narrative", dev)
    assert cb["marker"] == nb["marker"]
    marker = cb["marker"]
    question = QUERY[query][0]
    q_positions = [_last_overlap_token(tok, text, question)
                   for text in cb["texts"]]
    assert len(set(q_positions)) == 1, f"nonuniform question sites: {q_positions}"
    sites = {"checkpoint": marker, "question_end": q_positions[0],
             "readout": int(cb["ids"].shape[1] - 1)}
    layers = _full_depth_layers(int(model.config.num_hidden_layers), layers)
    positions = tuple(sites.values())
    cl, cc = _forward(model, cb["ids"], cb["am"], positions, tuple(layers))
    nl, nc = _forward(model, nb["ids"], nb["am"], positions, tuple(layers))
    src = _locations(rows, query)
    tgt = _locations(nat, query)
    sid = torch.tensor([cb["amap"][x] for x in src])
    tid = torch.tensor([cb["amap"][x] for x in tgt])

    def m(lg):
        return _ld(lg, tid, sid)

    g0c = _accuracy(cl, cb, src)
    g0n = _accuracy(nl, nb, tgt)
    nat_eff = float((m(nl) - m(cl)).mean())
    out = {"g0_clean": float(g0c), "g0_natural": float(g0n),
           "natural_effect": nat_eff, "sites": sites, "layers": layers,
           "per_site": {name: {} for name in sites}}
    if min(g0c, g0n) < 0.8:
        out["verdict"] = "INELICITABLE"
        return out
    for site_idx, (site_name, position) in enumerate(sites.items()):
        for L in layers:
            fwd, _ = _forward(model, cb["ids"], cb["am"], (position,),
                              patch=(L, position, nc[L][:, site_idx]))
            rev, _ = _forward(model, nb["ids"], nb["am"], (position,),
                              patch=(L, position, cc[L][:, site_idx]))
            lam = (float((m(fwd) - m(cl)).mean()) / nat_eff
                   if abs(nat_eff) > 1e-8 else float("nan"))
            out["per_site"][site_name][L] = {
                "lam": lam,
                "fwd_target_acc": float(_accuracy(fwd, cb, tgt)),
                "rev_clean_acc": float(_accuracy(rev, nb, src)),
            }
            log(f"  [{site_name} L{L}] lam={lam:.4f} "
                f"fwd_acc={out['per_site'][site_name][L]['fwd_target_acc']:.0%}")
    primary = [v["lam"] for v in out["per_site"]["checkpoint"].values()
               if np.isfinite(v["lam"])]
    if not primary:
        out["verdict"] = "CHECKPOINT_UNSCORABLE"
        return out
    out["max_abs_checkpoint_lam"] = float(max(abs(x) for x in primary))
    out["verdict"] = ("CHECKPOINT_INERT" if out["max_abs_checkpoint_lam"] < 0.3
                      else "CHECKPOINT_ACTIVE")
    return out


def run_delta_preprint_battery(model_path, out_dir, model_key="model",
                               quantization="8bit", device_map=None,
                               seeds=(0, 1, 2), layer=2,
                               layer_candidates=None, max_memory=None,
                               n_matrix=30, n_entity=30, n_world=30,
                               checkpoint_layers=CHECKPOINT_LAYERS,
                               matrix_null=50, entity_null=30,
                               anchor_null=99,
                               run_probe=False, probe_reps=6,
                               skip=()):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    results = {"stage": "delta_preprint_battery", "model_key": model_key,
               "model_path": model_path, "quantization": quantization,
               "layer": layer, "seeds": list(seeds),
               "n": {"matrix": n_matrix, "entity": n_entity,
                     "world": n_world},
               "n_null": {"matrix": matrix_null, "entity": entity_null,
                          "anchor": anchor_null},
               "per_seed": {}, "skipped": list(skip)}

    world_rows, world_indices = _compatible_world_rows(
        tok, torch.device("cpu"), n_world)
    results["structured_world_selection"] = {
        "requested": n_world, "selected": len(world_rows),
        "indices_from_30": world_indices,
        "rule": "largest tokenizer-aligned signature bucket",
    }

    for seed in seeds:
        sdir = os.path.join(out_dir, f"s{seed}")
        os.makedirs(sdir, exist_ok=True)
        block = {}
        if "matrix" not in skip:
            log(f"=== [{model_key}] seed {seed}: workspace matrix ===")
            block["matrix"] = run_delta_workspace_matrix(
                model_path, sdir, model_key=model_key,
                quantization=quantization, seed=seed, layer=layer,
                layer_candidates=layer_candidates, n_rows=n_matrix,
                n_null=matrix_null, model=model, tok=tok)
        if "entity" not in skip:
            log(f"=== [{model_key}] seed {seed}: entity matrix ===")
            hb = Heartbeat(2 * 2 * entity_null, f"entity_s{seed}", every_sec=30,
                           out_dir=sdir)
            block["entity"] = run_entity_model(
                model_key, model_path, quantization, layer, entity_null, seed, hb,
                n_rows=n_entity, model=model, tok=tok)
            hb.done()
        results["per_seed"][seed] = block

    if "anchor" not in skip:
        log(f"=== [{model_key}] exhaustive anchor write ===")
        adir = os.path.join(out_dir, "anchor_census")
        results["anchor"] = run_delta_anchor_write(
            model_path, adir, quantization=quantization, seed=0,
            n_null=anchor_null, n_rows=n_world, model=model, tok=tok,
            battery=True, clean_rows=world_rows)

    if "checkpoint" not in skip:
        log(f"=== [{model_key}] checkpoint cell ===")
        results["checkpoint"] = _checkpoint_cell(
            model, tok, dev, (list(checkpoint_layers)
                              if checkpoint_layers is not None else None),
            world_rows)

    if run_probe and "probe" not in skip:
        from .delta_preprint_probe import run_delta_preprint_probe
        log(f"=== [{model_key}] grouped checkpoint probe ===")
        results["probe"] = run_delta_preprint_probe(
            model_path, os.path.join(out_dir, "probe"),
            quantization=quantization, device_map=device_map,
            max_memory=max_memory, layers=checkpoint_layers,
            n_reps=probe_reps, model=model, tok=tok)

    # compact cross-seed summary for the paper tables
    summ = {}
    for name, getter in (
            ("matrix", lambda b: b.get("matrix", {}).get("verdict")),
            ("entity", lambda b: b.get("entity", {}).get("verdict"))):
        summ[name] = [getter(results["per_seed"][s]) for s in seeds]
    summ["anchor"] = results.get("anchor", {}).get("verdict")
    summ["checkpoint"] = results.get("checkpoint", {}).get("verdict")
    if run_probe:
        summ["probe_checkpoint"] = {
            surface: results.get("probe", {}).get("probe", {})
            .get(surface, {}).get("checkpoint", {}).get("accuracy")
            for surface in ("ledger", "narrative")
        }
    results["summary"] = summ
    with open(os.path.join(out_dir,
                           f"results_delta_preprint_battery_{model_key}.json"),
              "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"BATTERY [{model_key}] summary: {summ}")
    return results
