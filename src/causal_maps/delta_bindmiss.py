"""What native binding misses from span{Δ_u} — linear / positional / nonlinear.

BIND_MISS_PROTOCOL.md. Why-question after ASYM_INCOMPLETE_BASIS.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from . import variable_pairs
from .delta_crossskill import _variable_directions
from .direction_transfer import PRIMARY_LAYER
from .logutil import Heartbeat, log
from .model_utils import (get_decoder_layers, input_device, last_token_logits,
                          load_model_and_tokenizer, single_token_id)
from .nulls import permutation_pvalue
from .patching import (_split_output, cache_layer_outputs, forward_with_mean_knock,
                       forward_with_project)
from .tensorize import _anchor_token_index

N_NULL = 100
BEHAV_GATE = 0.80
LAYER = PRIMARY_LAYER
VERSION = 1
PROBE_ACC_GATE = 0.50
RECOVERY_GATE = 0.50


def _basis_span(dirs) -> torch.Tensor:
    M = torch.stack([d.float().reshape(-1) for d in dirs], dim=1)
    Q, _ = torch.linalg.qr(M, mode="reduced")
    return Q


def _project_rows(H: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """H [B,D], U [D,K] orthonormal cols → residual H - P_U H."""
    U = U.float()
    coeff = H.float() @ U
    return H.float() - coeff @ U.T


@torch.no_grad()
def _greedy_id(model, text, tok, device):
    e = tok.encode(text, add_special_tokens=False)
    ii = torch.tensor([e], dtype=torch.long, device=device)
    return int(last_token_logits(model, ii, torch.ones_like(ii)).argmax(-1).item())


@torch.no_grad()
def _slot_acts(model, ids, am, layer, pos):
    cache = cache_layer_outputs(model, ids, am, to_cpu=True)
    return cache[layer][:, pos, :].float()


@torch.no_grad()
def _pref_gold(logits, gold_idx, pool_ids):
    B = logits.shape[0]
    g = logits[torch.arange(B, device=logits.device), gold_idx]
    return (g - logits[:, pool_ids].mean(-1)).float().cpu().numpy()


def _centroid_probe_acc(R, labels, train_idx, test_idx, vals):
    """Nearest centroid in residual space. labels are value strings."""
    cents = {}
    for v in vals:
        ix = [i for i in train_idx if labels[i] == v]
        if not ix:
            return float("nan")
        cents[v] = R[ix].mean(0)
    correct = 0
    n = 0
    for i in test_idx:
        r = R[i]
        best, best_s = None, -1e30
        for v, c in cents.items():
            s = float(torch.nn.functional.cosine_similarity(
                r.unsqueeze(0), c.unsqueeze(0)).item())
            if s > best_s:
                best_s, best = s, v
        correct += int(best == labels[i])
        n += 1
    return correct / max(n, 1)


def run_delta_bindmiss(model_path, out_dir, quantization="8bit", device_map=None,
                       layer=LAYER, seed=0, n_null=N_NULL):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    device = input_device(model)
    rng = np.random.default_rng(seed)

    log(f"delta_bindmiss v{VERSION}: layer={layer} — what span{{Δ}} misses")

    Delta, values = _variable_directions(model, tok, layer, seed)
    vals = list(values)
    assert len(vals) >= 4
    val_ids = torch.tensor([single_token_id(tok, v) for v in vals],
                           dtype=torch.long, device=device)
    span = _basis_span([Delta[v] for v in vals]).to(device)
    log(f"  nV={len(vals)} span_rank={span.shape[1]}")

    # --- build native prompts from Variable pairs (cf side = Δ vocabulary) ---
    pairs = variable_pairs.make_variable_pairs(50, seed=seed, tok=tok)
    rows = []
    for p in pairs:
        v = p["meta"]["val_cf"]
        if v not in Delta:
            continue
        text = p["cf_text"]
        off = p["anchors"]["val_slot"]
        slot = _anchor_token_index(tok, text, off)
        if slot is None:
            continue
        gold = single_token_id(tok, v, leading_space=True)
        ok = _greedy_id(model, text, tok, device) == gold
        rows.append({"v": v, "text": text, "slot": slot, "gold": gold, "ok": ok,
                     "tid": p["template"]})

    behav = sum(r["ok"] for r in rows) / max(len(rows), 1)
    g0 = behav >= BEHAV_GATE
    log(f"  G0 bind={behav:.0%} n={len(rows)} pass={g0}")
    if not g0:
        results = {
            "stage": "delta_bindmiss", "version": VERSION, "model_path": model_path,
            "layer": layer, "verdict": "BIND_MISS_INELICITABLE",
            "G0": {"bind": behav, "pass": False},
        }
        with open(os.path.join(out_dir, "results_delta_bindmiss.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        log("VERDICT delta_bindmiss: BIND_MISS_INELICITABLE")
        return results

    ok_rows = [r for r in rows if r["ok"]] or rows
    # majority (S, slot)
    from collections import Counter
    key_counts = Counter(
        (len(tok.encode(r["text"], add_special_tokens=False)), r["slot"])
        for r in ok_rows)
    (maj, slot), _ = key_counts.most_common(1)[0]
    ok_rows = [r for r in ok_rows
               if len(tok.encode(r["text"], add_special_tokens=False)) == maj
               and r["slot"] == slot]
    assert len(ok_rows) >= 8, f"too few uniform prompts: {len(ok_rows)}"
    S = maj
    last = S - 1
    log(f"  n_prompts={len(ok_rows)} slot={slot} S={S} "
        f"per_v={ {v: sum(1 for r in ok_rows if r['v']==v) for v in vals} }")

    all_rows = ok_rows
    ids = torch.tensor([tok.encode(r["text"], add_special_tokens=False)
                        for r in all_rows], dtype=torch.long, device=device)
    am = torch.ones_like(ids)
    golds = torch.tensor([r["gold"] for r in all_rows], dtype=torch.long, device=device)
    labels = [r["v"] for r in all_rows]

    # activations + residuals
    H = _slot_acts(model, ids, am, layer, slot).to(device)  # [B,D]
    R = _project_rows(H, span)  # residual after removing extracted span

    # train/test split stratified
    by_v = {v: [i for i, lab in enumerate(labels) if lab == v] for v in vals}
    train_idx, test_idx = [], []
    for v, ix in by_v.items():
        rng.shuffle(ix)
        if len(ix) == 1:
            train_idx.append(ix[0]); test_idx.append(ix[0])
        else:
            n_te = max(1, len(ix) // 3)
            test_idx.extend(ix[:n_te])
            train_idx.extend(ix[n_te:])
    assert train_idx and test_idx

    probe_acc = _centroid_probe_acc(R.cpu(), labels, train_idx, test_idx, vals)
    # also probe on full H (ceiling) and on P_span H (should be high if Δ span carries value)
    P_H = (H.float() @ span) @ span.T
    probe_full = _centroid_probe_acc(H.cpu(), labels, train_idx, test_idx, vals)
    probe_proj = _centroid_probe_acc(P_H.cpu(), labels, train_idx, test_idx, vals)

    hb = Heartbeat(n_null, "bindmiss_probe_null", every_sec=15, out_dir=out_dir)
    null_accs = []
    lab_arr = np.array(labels)
    for _ in range(n_null):
        shuf = lab_arr.copy()
        rng.shuffle(shuf)
        null_accs.append(_centroid_probe_acc(
            R.cpu(), list(shuf), train_idx, test_idx, vals))
        hb.step()
    hb.done()
    p_probe = permutation_pvalue(probe_acc, np.array(null_accs), "greater")
    L1 = bool(probe_acc >= PROBE_ACC_GATE and p_probe < 0.01)
    log(f"  L1 probe_r={probe_acc:.0%}(p={p_probe:.3f}) "
        f"probe_full={probe_full:.0%} probe_span={probe_proj:.0%} pass={L1}")

    # --- causal residual ADD ---
    cents = {}
    for v in vals:
        ix = [i for i, lab in enumerate(labels) if lab == v]
        cents[v] = R[ix].mean(0)
    mean_c = torch.stack([cents[v] for v in vals]).mean(0)
    # held-out test rows only for causal
    te_ids = ids[test_idx]
    te_am = am[test_idx]
    te_gold = golds[test_idx]
    te_labels = [labels[i] for i in test_idx]

    lg_clean = last_token_logits(model, te_ids, te_am)
    pref_clean = float(_pref_gold(lg_clean, te_gold, val_ids).mean())
    lg_span = forward_with_project(model, te_ids, te_am, layer, slot, span, alpha=1.0)
    pref_span = float(_pref_gold(lg_span, te_gold, val_ids).mean())

    # per-row ADD d_v for that row's value
    prefs_add = []
    for j, i_global in enumerate(test_idx):
        v = labels[i_global]
        d = cents[v] - mean_c
        # norm-match to Δ_v
        ns = float(Delta[v].norm().clamp(min=1e-8))
        d = d / d.norm().clamp(min=1e-8) * ns
        # span-project then add d (compose: project hook then would need both —
        # approximate: add d on clean, and separately project; primary = project+add
        # Implement project+add in one hook
        ii = te_ids[j:j + 1]; aa = te_am[j:j + 1]
        lg = _forward_project_add(model, ii, aa, layer, slot, span, d.to(device))
        prefs_add.append(float(_pref_gold(lg, te_gold[j:j + 1], val_ids).mean()))
    pref_span_add = float(np.mean(prefs_add))
    denom = pref_clean - pref_span
    rec = (pref_span_add - pref_span) / denom if denom > 1e-6 else 0.0
    C1 = bool(denom > 1e-6 and rec >= RECOVERY_GATE)
    log(f"  C1 pref_clean={pref_clean:+.2f} span={pref_span:+.2f} "
        f"span+ADD={pref_span_add:+.2f} rec={rec:.2f} pass={C1}")

    # --- positional knocks ---
    positions = []
    for p in (slot - 1, slot, slot + 1, last):
        if 0 <= p < S and p not in positions:
            positions.append(p)

    lg0 = last_token_logits(model, ids, am)
    pref0 = float(_pref_gold(lg0, golds, val_ids).mean())
    drops = {}
    for p in positions:
        lgk = forward_with_mean_knock(model, ids, am, layer, p)
        pk = float(_pref_gold(lgk, golds, val_ids).mean())
        drops[p] = pref0 - pk
        log(f"  knock pos={p} drop={drops[p]:+.2f}")

    drop_val = drops[slot]
    others = [drops[p] for p in positions if p != slot]
    P1 = bool(others and max(others) >= 0.5 * max(drop_val, 1e-6))
    P1_local = bool(others and all(drop_val >= 2.0 * max(d, 0.0) for d in others))
    # if other drops negative, treat as 0 for local check
    P1_local = bool(others and all(drop_val >= 2.0 * max(d, 0.0) + 1e-6
                                   for d in others))
    log(f"  P1 distributed={P1} local={P1_local}")

    # verdict
    if L1 and C1:
        verdict = "BIND_MISS_LINEAR"
    elif L1 and not C1:
        verdict = "BIND_MISS_LINEAR_READOUT"
    elif (not L1) and P1:
        verdict = "BIND_MISS_DISTRIBUTED"
    elif (not L1) and (not P1) and P1_local:
        verdict = "BIND_MISS_NONLINEAR_LOCAL"
    else:
        verdict = "BIND_MISS_UNCLEAR"

    results = {
        "stage": "delta_bindmiss", "version": VERSION, "model_path": model_path,
        "layer": layer, "n_null": n_null, "n_prompts": len(all_rows),
        "G0": {"bind": behav, "pass": True},
        "span_rank": int(span.shape[1]),
        "probe": {
            "residual": probe_acc, "full_h": probe_full, "span_proj": probe_proj,
            "p_shuffle": float(p_probe), "L1": L1,
            "n_train": len(train_idx), "n_test": len(test_idx),
        },
        "causal": {
            "pref_clean": pref_clean, "pref_span": pref_span,
            "pref_span_add": pref_span_add, "recovery": rec, "C1": C1,
        },
        "position_drops": {str(p): drops[p] for p in positions},
        "P1_distributed": P1, "P1_local": P1_local,
        "gates": {"L1": L1, "C1": C1, "P1": P1, "P1_local": P1_local},
        "verdict": verdict,
        "framing": {
            "BIND_MISS_LINEAR": "missing content is linear+causal in residual at val_slot",
            "BIND_MISS_LINEAR_READOUT": "residual linearly decodes v but ADD does not recover",
            "BIND_MISS_DISTRIBUTED": "missing content not linear in-slot; knock hits other positions",
            "BIND_MISS_NONLINEAR_LOCAL": "in-slot, not linear in residual; site-local knock",
            "BIND_MISS_UNCLEAR": "no account cleared",
            "BIND_MISS_INELICITABLE": "G0 fail",
        }[verdict],
        "hard_stop": "no_position_menu_no_layer_sweep",
    }
    path = os.path.join(out_dir, "results_delta_bindmiss.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT delta_bindmiss: {verdict} | L1={L1} C1={C1} P1={P1}")
    return results


@torch.no_grad()
def _forward_project_add(model, ids, am, layer_idx, position, basis, delta):
    """h ← (h - P_S h) + delta at position."""
    layer = get_decoder_layers(model)[layer_idx]
    U = basis.float()
    if U.dim() == 1:
        U = (U / U.norm().clamp(min=1e-8)).unsqueeze(1)
    d = delta.float().reshape(-1)

    def hook(module, inp, out):
        hs, rebuild = _split_output(out)
        hs = hs.clone()
        h = hs[:, position, :].float()
        Udev = U.to(dtype=h.dtype, device=h.device)
        ddev = d.to(dtype=h.dtype, device=h.device)
        proj = (h @ Udev) @ Udev.T
        hs[:, position, :] = (h - proj + ddev).to(dtype=hs.dtype)
        return rebuild(hs)

    handle = layer.register_forward_hook(hook)
    try:
        out = model(input_ids=ids, attention_mask=am, use_cache=False)
    finally:
        handle.remove()
    return out.logits[:, -1, :]
