"""Capacity law: how many value bindings compose before interference.

k-variable prompt, all slots = a base value; install distinct target values at
all k slots via their directions; query the first variable. Sweep k and measure
retention(k) and cross-talk(k). Pre-registered (CAUSAL_MAPS_LOG.md 2026-07-13).
"""
import json
import os

import numpy as np
import torch

from .delta_crossskill import _variable_directions
from .delta_multislot import _selectivity, forward_add_multi
from .direction_transfer import PRIMARY_LAYER
from .logutil import Heartbeat, log
from .model_utils import (input_device, last_token_logits,
                          load_model_and_tokenizer, single_token_id)
from .nulls import permutation_pvalue
from .tensorize import _anchor_token_index

N_NULL = 100
K_LIST = [1, 2, 3, 4, 5, 6, 7, 8]
NAMES = list("ABCDEFGHIJKLMNOP")
N_TRIALS = 8


def _is_single(tok, w):
    try:
        single_token_id(tok, w)
        return True
    except ValueError:
        return False


def _prompt(tok, names, base, query):
    lets = " ".join(f"Let {n} = {base}." for n in names)
    user = f"{lets} What is the value of {query}?"
    templated = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    text = templated + f"{query} ="
    offs = [text.find(f"Let {n} = ") + len(f"Let {n} = ") for n in names]
    return text, offs


def _rand_like(d):  # [T,D] random, per-row norm matched to d
    r = torch.randn(d.shape)
    return r / r.norm(dim=1, keepdim=True).clamp(min=1e-8) * d.norm(dim=1, keepdim=True)


def run_delta_capacity(model_path, out_dir, quantization="8bit", device_map=None,
                       layer=PRIMARY_LAYER, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    Delta, values = _variable_directions(model, tok, layer, seed)
    val_ids = [single_token_id(tok, v) for v in values]
    vidx = {v: i for i, v in enumerate(values)}
    names = [n for n in NAMES if _is_single(tok, n)]
    dev = input_device(model)
    rng = np.random.default_rng(seed)
    ks = [k for k in K_LIST if k <= len(names) and k <= len(values) - 1]
    log(f"capacity: values={len(values)} names={len(names)} K={ks}")

    hb = Heartbeat(len(ks) * N_NULL, "delta_capacity", every_sec=15, out_dir=out_dir)
    base = values[-1]
    pool = [v for v in values if v != base]
    curve = []
    for k in ks:
        nm = names[:k]
        text, offs = _prompt(tok, nm, base, nm[0])
        slots = [_anchor_token_index(tok, text, o) for o in offs]
        assert all(s is not None for s in slots), f"slot find failed k={k}"
        ids1 = tok.encode(text, add_special_tokens=False)
        ids = torch.tensor([ids1] * N_TRIALS, dtype=torch.long, device=dev)
        am = torch.ones_like(ids)
        targets = [[str(x) for x in rng.choice(pool, size=k, replace=False)]
                   for _ in range(N_TRIALS)]
        t0_idx = np.array([vidx[targets[t][0]] for t in range(N_TRIALS)])
        base_lg = last_token_logits(model, ids, am)[:, val_ids].float().cpu().numpy()

        def dstack(col):
            return torch.stack([Delta[targets[t][col]] for t in range(N_TRIALS)])

        def add_dl(pos_deltas):
            lg = forward_add_multi(model, ids, am, layer, pos_deltas)[:, val_ids].float().cpu().numpy()
            return lg - base_lg

        dl_single = add_dl([(slots[0], dstack(0))])
        dl_all = add_dl([(slots[i], dstack(i)) for i in range(k)])
        sel_single = float(_selectivity(dl_single, t0_idx).mean())
        sel_all = float(_selectivity(dl_all, t0_idx).mean())
        # cross-talk: mean Δlogit of the distractor targets (cols 1..k-1) at V1
        xt = np.zeros(N_TRIALS)
        for c in range(1, k):
            cols = np.array([vidx[targets[t][c]] for t in range(N_TRIALS)])
            xt = xt + dl_all[np.arange(N_TRIALS), cols]
        crosstalk = float((xt / (k - 1)).mean()) if k > 1 else 0.0

        nulls = []
        for _ in range(N_NULL):
            pd = [(slots[i], _rand_like(dstack(i))) for i in range(k)]
            nulls.append(float(_selectivity(add_dl(pd), t0_idx).mean()))
            hb.step()
        p = permutation_pvalue(sel_all, np.array(nulls), "greater")
        retention = sel_all / sel_single if abs(sel_single) > 1e-6 else float("nan")
        row = {"k": k, "sel_single": sel_single, "sel_all": sel_all,
               "retention": retention, "crosstalk": crosstalk, "p_all": float(p)}
        curve.append(row)
        log(f"  k={k}: sel_single={sel_single:+.2f} sel_all={sel_all:+.2f} "
            f"retention={retention:.2f} crosstalk={crosstalk:+.2f} p={p:.3f}")
    hb.done()

    ok = [r["k"] for r in curve if np.isfinite(r["retention"])
          and r["retention"] >= 0.7 and r["p_all"] < 0.01]
    k_star = max(ok) if ok else 0
    results = {"stage": "delta_capacity", "model_path": model_path, "layer": int(layer),
               "n_trials": N_TRIALS, "n_null": N_NULL, "base_value": base,
               "K": [r["k"] for r in curve], "curve": curve, "k_star": k_star}
    with open(os.path.join(out_dir, "results_delta_capacity.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT capacity: k*={k_star} (largest k with retention>=0.7 and p<0.01) | "
        f"curve={[(r['k'], round(r['retention'], 2), round(r['crosstalk'], 2)) for r in curve]}")
    return results
