"""Consequence law — from write-consumption demo to quantitative law.

Pre-registered CAUSAL_MAPS_LOG.md 2026-07-14 (after GENERAL_PRINCIPLE).

Arm A (LAW): sweep the WHOLE digit codebook through the write on
  T1 = "X = {a}. Y = X + {b}. What is the value of Y?"
  Does output track f(v)=v+b for every written v, in every context?
Arm B (COMPOSITION): two written arguments consumed by ONE computation on
  T2 = "X = {a}. Z = {c}. Y = X + Z. What is the value of Y?"
  incl. MIXED cells (one injected argument + one textual argument).
Arm C (DISCOVERY): conflicting writes at one address — winner statistics only,
  no pass/fail gate (connects to overwrite-suppression findings).

Reuses the matrix interface: neutral donors ("Here is a token: v."), canonical-
continuation answer contract, per-arm G0 (a dead arm never kills the kernel).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .delta_multislot import forward_add_multi
from .delta_workspace_matrix import _chat, _common_prefix, _enc, JOINER, PRIMER
from .logutil import Heartbeat, log
from .model_utils import (input_device, last_token_logits,
                          load_model_and_tokenizer)
from .nulls import permutation_pvalue
from .patching import cache_layer_outputs, forward_with_add

N_NULL = 30                      # p-floor 1/31 ≈ .032 < .04 gate
G0 = 0.90
DIGITS = list(range(1, 10))
CONTEXTS_A = [(2, 3), (3, 2), (4, 5), (5, 1), (6, 2), (3, 4)]
CTX_C = (3, 2)
N_ROWS_B = 10
N_ROWS_C = 10
GATE_LAW_ACC = 0.90
GATE_PARTIAL_ACC = 0.70
GATE_RATIO = 0.70
GATE_B_ACC = 0.80
GATE_P = 0.04


def _t1(a, b):
    return f"X = {a}. Y = X + {b}. What is the value of Y?"


def _t2(a, c):
    return f"X = {a}. Z = {c}. Y = X + Z. What is the value of Y?"


def _digit_contract(tok, text_fn, probes):
    """Canonical-continuation contract for a template.
    probes: list of (args1, args2) pairs differing in exactly ONE template slot;
    returns positions per slot (diff position), plus digit answer ids/common."""
    positions = []
    for a1, a2 in probes:
        e1 = _enc(tok, _chat(tok, text_fn(*a1), PRIMER))
        e2 = _enc(tok, _chat(tok, text_fn(*a2), PRIMER))
        assert len(e1) == len(e2), f"probe lengths differ {len(e1)}/{len(e2)}"
        d = [i for i, (x, y) in enumerate(zip(e1, e2)) if x != y]
        assert len(d) == 1, f"probe differs at {len(d)} positions"
        positions.append(d[0])
    chat0 = _chat(tok, text_fn(*probes[0][0]), "")
    base = _enc(tok, chat0 + PRIMER)
    conts = {}
    for dgt in DIGITS:
        full = _enc(tok, chat0 + JOINER + str(dgt))
        assert full[:len(base)] == base, f"digit {dgt}: base not a prefix"
        conts[dgt] = full[len(base):]
    common = _common_prefix(list(conts.values()))
    amap = {}
    for dgt, c in conts.items():
        assert len(c) > len(common), f"digit {dgt}: no diverging token"
        amap[dgt] = c[len(common)]
    assert len(set(amap.values())) == 9, "digit ids collide"
    return {"positions": positions, "common": common, "amap": amap,
            "S": len(_enc(tok, _chat(tok, text_fn(*probes[0][0]), PRIMER)))
                 + len(common)}


def _rows_b(rng):
    rows, seen = [], set()
    guard = 0
    while len(rows) < N_ROWS_B and guard < 50000:
        guard += 1
        a, c, v, w = (int(rng.integers(1, 10)) for _ in range(4))
        if not (a + c <= 9 and v + w <= 9 and v + c <= 9 and a + w <= 9):
            continue
        if v == a or w == c or v + w == a + c:
            continue
        v2, w2 = int(rng.integers(1, 10)), int(rng.integers(1, 10))
        if not (v2 + w2 <= 9 and v2 != a and w2 != c
                and v2 + w2 not in (a + c, v + w)):
            continue
        key = (a, c, v, w)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"a": a, "c": c, "v": v, "w": w, "v2": v2, "w2": w2})
    assert len(rows) == N_ROWS_B, f"armB rows {len(rows)}"
    return rows


def _rows_c(rng):
    a, b = CTX_C
    pool = [v for v in DIGITS if v + b <= 9 and v != a]
    rows, seen = [], set()
    guard = 0
    while len(rows) < N_ROWS_C and guard < 5000:
        guard += 1
        v, w = (int(x) for x in rng.choice(pool, size=2, replace=False))
        if (v, w) in seen:
            continue
        seen.add((v, w))
        rows.append((v, w))
    assert len(rows) == N_ROWS_C, f"armC rows {len(rows)}"
    return rows


def run_delta_consequence_law(model_path, out_dir, model_key="model",
                              quantization="8bit", device_map=None, seed=0,
                              layer=2, n_null=N_NULL):
    os.makedirs(out_dir, exist_ok=True)
    if "*" in model_path:
        import glob as _glob
        hits = sorted(_glob.glob(model_path, recursive=True))
        assert hits, f"model_path glob matched nothing: {model_path}"
        mp = hits[0]
        if os.path.basename(mp) == "config.json":
            mp = os.path.dirname(mp)
        log(f"model_path glob -> {mp}")
        model_path = mp
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    dev = input_device(model)
    rng = np.random.default_rng(seed)

    c1 = _digit_contract(tok, _t1, [(((2, 3)), ((4, 3))), (((2, 3)), ((2, 4)))])
    c2 = _digit_contract(tok, _t2, [(((2, 3)), ((4, 3))), (((2, 3)), ((2, 5)))])
    pos_a1, pos_b1 = c1["positions"]
    pos_x2, pos_z2 = c2["positions"]
    assert pos_a1 != pos_b1 and pos_x2 != pos_z2
    log(f"consequence_law[{model_key}]: T1 pos(a,b)=({pos_a1},{pos_b1}) "
        f"common={c1['common']} | T2 pos(x,z)=({pos_x2},{pos_z2}) "
        f"common={c2['common']} | layer={layer}")

    # neutral donor states
    neut = [_chat(tok, f"Here is a token: {v}.", "") for v in DIGITS]
    enc_n = [_enc(tok, t) for t in neut]
    assert len({len(e) for e in enc_n}) == 1
    ids_n = torch.tensor(enc_n, dtype=torch.long, device=dev)
    cache_n = cache_layer_outputs(model, ids_n, torch.ones_like(ids_n),
                                  to_cpu=True)
    dn = [i for i, (x, y) in enumerate(zip(enc_n[0], enc_n[1])) if x != y]
    assert len(dn) == 1
    Z = {v: cache_n[layer][i, dn[0], :].float() for i, v in enumerate(DIGITS)}

    def batch(texts, common):
        e = [_enc(tok, _chat(tok, t, PRIMER)) + list(common) for t in texts]
        assert len({len(x) for x in e}) == 1
        t = torch.tensor(e, dtype=torch.long, device=dev)
        return t, torch.ones_like(t)

    @torch.no_grad()
    def logits_of(ids, am, mode=None, arg=None):
        if mode is None:
            lg = last_token_logits(model, ids, am)
        elif mode == "add":
            pos, delta = arg
            lg = forward_with_add(model, ids, am, layer, pos, delta)
        else:                                  # multi
            lg = forward_add_multi(model, ids, am, layer, arg)
        return lg.float()

    all_digit_ids1 = torch.tensor([c1["amap"][d] for d in DIGITS], device=dev)
    all_digit_ids2 = torch.tensor([c2["amap"][d] for d in DIGITS], device=dev)

    def picks(lg, digit_ids):
        return lg[:, digit_ids].argmax(-1).cpu().numpy()      # index into DIGITS

    hb = Heartbeat(2 * n_null, "consequence_law", every_sec=20, out_dir=out_dir)
    results = {"stage": "delta_consequence_law", "model_key": model_key,
               "model_path": model_path, "layer": int(layer), "seed": seed,
               "n_null": n_null}

    # ------------------------------- Arm A --------------------------------- #
    rowsA = [(a, b, v) for (a, b) in CONTEXTS_A
             for v in DIGITS if v != a and v + b <= 9]
    tA_clean = [_t1(a, b) for a, b, v in rowsA]
    tA_nat = [_t1(v, b) for a, b, v in rowsA]
    idsA, amA = batch(tA_clean, c1["common"])
    idsAn, amAn = batch(tA_nat, c1["common"])
    n = len(rowsA)
    ar = torch.arange(n, device=dev)
    fa = torch.tensor([c1["amap"][a + b] for a, b, v in rowsA], device=dev)
    fv = torch.tensor([c1["amap"][v + b] for a, b, v in rowsA], device=dev)
    fa_k = np.array([DIGITS.index(a + b) for a, b, v in rowsA])
    fv_k = np.array([DIGITS.index(v + b) for a, b, v in rowsA])

    def marginA(lg):
        return (lg[ar, fv] - lg[ar, fa]).cpu().numpy()

    lg_c = logits_of(idsA, amA)
    lg_n = logits_of(idsAn, amAn)
    g0_clean = float((picks(lg_c, all_digit_ids1) == fa_k).mean())
    g0_nat = float((picks(lg_n, all_digit_ids1) == fv_k).mean())
    armA = {"n_rows": n, "g0_clean": g0_clean, "g0_natural": g0_nat}
    if min(g0_clean, g0_nat) < G0:
        armA["verdict"] = "INELICITABLE"
        log(f"  [armA] G0 fail {g0_clean:.0%}/{g0_nat:.0%}")
    else:
        dA = torch.stack([Z[v] - Z[a] for a, b, v in rowsA])
        lg_add = logits_of(idsA, amA, "add", (pos_a1, dA))
        m_c, m_n, m_a = marginA(lg_c), marginA(lg_n), marginA(lg_add)
        acc = float((picks(lg_add, all_digit_ids1) == fv_k).mean())
        nat_eff, add_eff = m_n - m_c, m_a - m_c
        ok = np.abs(nat_eff) > 1.0
        ratios = add_eff[ok] / nat_eff[ok]
        med_ratio = float(np.median(ratios))
        norms = dA.norm(dim=1, keepdim=True)
        nulls = []
        for _ in range(n_null):
            r = torch.randn(dA.shape)
            r = r / r.norm(dim=1, keepdim=True).clamp(min=1e-8) * norms
            nulls.append(float((marginA(logits_of(idsA, amA, "add",
                                                  (pos_a1, r))) - m_c).mean()))
            hb.step()
        p = permutation_pvalue(float(add_eff.mean()), np.asarray(nulls),
                               "greater")
        per_cell = [{"a": a, "b": b, "v": v,
                     "hit": bool(picks(lg_add, all_digit_ids1)[i] == fv_k[i]),
                     "ratio": (float(add_eff[i] / nat_eff[i])
                               if abs(nat_eff[i]) > 1.0 else None)}
                    for i, (a, b, v) in enumerate(rowsA)]
        if acc >= GATE_LAW_ACC and med_ratio >= GATE_RATIO and p < GATE_P:
            vA = "CONSEQUENCE_LAW"
        elif acc >= GATE_PARTIAL_ACC:
            vA = "LAW_PARTIAL"
        else:
            vA = "LAW_FAIL"
        armA.update({"cell_accuracy": acc, "median_ratio": med_ratio,
                     "mean_add_effect": float(add_eff.mean()),
                     "mean_nat_effect": float(nat_eff.mean()),
                     "p": float(p), "per_cell": per_cell, "verdict": vA})
        log(f"  [armA] {vA} acc={acc:.0%} med_ratio={med_ratio:.2f} "
            f"p={p:.3f} n={n}")
    results["armA"] = armA

    # ------------------------------- Arm B --------------------------------- #
    rowsB = _rows_b(rng)
    tB_clean = [_t2(r["a"], r["c"]) for r in rowsB]
    tB_nat = [_t2(r["v"], r["w"]) for r in rowsB]
    idsB, amB = batch(tB_clean, c2["common"])
    idsBn, amBn = batch(tB_nat, c2["common"])
    nB = len(rowsB)
    arB = torch.arange(nB, device=dev)

    def idsum(key1, key2):
        return torch.tensor([c2["amap"][r[key1] + r[key2]] for r in rowsB],
                            device=dev)

    T_clean, T_vw = idsum("a", "c"), idsum("v", "w")
    T_vc, T_aw = idsum("v", "c"), idsum("a", "w")
    T_wr = torch.tensor([c2["amap"][r["v2"] + r["w2"]] for r in rowsB],
                        device=dev)
    k_clean = np.array([DIGITS.index(r["a"] + r["c"]) for r in rowsB])
    k_vw = np.array([DIGITS.index(r["v"] + r["w"]) for r in rowsB])
    k_vc = np.array([DIGITS.index(r["v"] + r["c"]) for r in rowsB])
    k_aw = np.array([DIGITS.index(r["a"] + r["w"]) for r in rowsB])
    k_wr = np.array([DIGITS.index(r["v2"] + r["w2"]) for r in rowsB])

    def marginB(lg):
        return (lg[arB, T_vw] - lg[arB, T_clean]).cpu().numpy()

    lgB_c = logits_of(idsB, amB)
    lgB_n = logits_of(idsBn, amBn)
    g0B_c = float((picks(lgB_c, all_digit_ids2) == k_clean).mean())
    g0B_n = float((picks(lgB_n, all_digit_ids2) == k_vw).mean())
    armB = {"n_rows": nB, "g0_clean": g0B_c, "g0_natural": g0B_n}
    if min(g0B_c, g0B_n) < G0:
        armB["verdict"] = "INELICITABLE"
        log(f"  [armB] G0 fail {g0B_c:.0%}/{g0B_n:.0%}")
    else:
        dX = torch.stack([Z[r["v"]] - Z[r["a"]] for r in rowsB])
        dZ = torch.stack([Z[r["w"]] - Z[r["c"]] for r in rowsB])
        dX2 = torch.stack([Z[r["v2"]] - Z[r["a"]] for r in rowsB])
        dZ2 = torch.stack([Z[r["w2"]] - Z[r["c"]] for r in rowsB])
        lg_both = logits_of(idsB, amB, "multi", [(pos_x2, dX), (pos_z2, dZ)])
        lg_x = logits_of(idsB, amB, "add", (pos_x2, dX))
        lg_z = logits_of(idsB, amB, "add", (pos_z2, dZ))
        lg_wr = logits_of(idsB, amB, "multi", [(pos_x2, dX2), (pos_z2, dZ2)])
        acc_both = float((picks(lg_both, all_digit_ids2) == k_vw).mean())
        acc_x = float((picks(lg_x, all_digit_ids2) == k_vc).mean())
        acc_z = float((picks(lg_z, all_digit_ids2) == k_aw).mean())
        acc_wr = float((picks(lg_wr, all_digit_ids2) == k_wr).mean())
        mB_c, mB_n, mB_b = marginB(lgB_c), marginB(lgB_n), marginB(lg_both)
        nat_eff = float((mB_n - mB_c).mean())
        add_eff = float((mB_b - mB_c).mean())
        ratio = add_eff / nat_eff if abs(nat_eff) > 1e-6 else float("nan")
        normsX = dX.norm(dim=1, keepdim=True)
        normsZ = dZ.norm(dim=1, keepdim=True)
        nulls = []
        for _ in range(n_null):
            r1 = torch.randn(dX.shape)
            r1 = r1 / r1.norm(dim=1, keepdim=True).clamp(min=1e-8) * normsX
            r2 = torch.randn(dZ.shape)
            r2 = r2 / r2.norm(dim=1, keepdim=True).clamp(min=1e-8) * normsZ
            lgn = logits_of(idsB, amB, "multi", [(pos_x2, r1), (pos_z2, r2)])
            nulls.append(float((marginB(lgn) - mB_c).mean()))
            hb.step()
        p = permutation_pvalue(add_eff, np.asarray(nulls), "greater")
        mixed_ok = acc_x >= GATE_B_ACC and acc_z >= GATE_B_ACC
        if (acc_both >= GATE_B_ACC and np.isfinite(ratio)
                and ratio >= GATE_RATIO and p < GATE_P):
            vB = "ARGUMENT_COMPOSITION"
        elif acc_both >= 0.5:
            vB = "COMPOSITION_PARTIAL"
        else:
            vB = "COMPOSITION_FAIL"
        armB.update({"acc_both": acc_both, "acc_mixed_x": acc_x,
                     "acc_mixed_z": acc_z, "acc_wrong_pair_own": acc_wr,
                     "ratio": float(ratio), "p": float(p),
                     "mixed_args_ok": bool(mixed_ok), "verdict": vB,
                     "rows": rowsB})
        log(f"  [armB] {vB} both={acc_both:.0%} mixedX={acc_x:.0%} "
            f"mixedZ={acc_z:.0%} wrong_own={acc_wr:.0%} ratio={ratio:.2f} "
            f"p={p:.3f} mixed_ok={mixed_ok}")
    results["armB"] = armB

    # ------------------------------- Arm C --------------------------------- #
    a, b = CTX_C
    rowsC = _rows_c(rng)
    tC = [_t1(a, b)] * len(rowsC)
    idsC, amC = batch(tC, c1["common"])
    nC = len(rowsC)
    fvC = np.array([DIGITS.index(v + b) for v, w in rowsC])
    fwC = np.array([DIGITS.index(w + b) for v, w in rowsC])
    faC = np.array([DIGITS.index(a + b)] * nC)
    dV = torch.stack([Z[v] - Z[a] for v, w in rowsC])
    dW = torch.stack([Z[w] - Z[a] for v, w in rowsC])
    conds = {"v_only": dV, "w_only": dW, "sum": dV + dW,
             "half_sum": 0.5 * (dV + dW)}
    armC = {"context": CTX_C, "rows": rowsC, "conditions": {}}
    for name, d in conds.items():
        lg = logits_of(idsC, amC, "add", (pos_a1, d))
        pk = picks(lg, all_digit_ids1)
        winner = {"f_v": float((pk == fvC).mean()),
                  "f_w": float((pk == fwC).mean()),
                  "f_a": float((pk == faC).mean())}
        winner["other"] = float(1.0 - winner["f_v"] - winner["f_w"]
                                - winner["f_a"])
        arC = torch.arange(nC, device=dev)
        iv = torch.tensor([c1["amap"][v + b] for v, w in rowsC], device=dev)
        iw = torch.tensor([c1["amap"][w + b] for v, w in rowsC], device=dev)
        mvw = float((lg[arC, iv] - lg[arC, iw]).mean())
        armC["conditions"][name] = {"winner": winner, "margin_fv_minus_fw": mvw}
        log(f"  [armC] {name}: winners {winner} margin_v-w={mvw:+.2f}")
    armC["verdict"] = "DISCOVERY_REPORTED"
    results["armC"] = armC
    hb.done()

    results["verdict"] = f"{armA.get('verdict')}|{armB.get('verdict')}"
    fp = os.path.join(out_dir,
                      f"results_delta_consequence_law_{model_key}.json")
    with open(fp, "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT consequence_law[{model_key}]: {results['verdict']}")
    return results
