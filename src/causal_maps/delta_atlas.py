"""Shared atlas — relational synthesis of UNSEEN value-states (predictive).

Pre-registered CAUSAL_MAPS_LOG.md 2026-07-14. Can a value-state we never
extracted be synthesized purely from its RELATIONS to other values (affine
barycentric coordinates over anchor states), and causally drive retrieval and
computation — within a model, and with coefficients fit in a DIFFERENT model?

  fit F = {1,2,4,5,7,8}   held-out H = {3,6,9}   (frozen split)
  c(h): z_src(h) ≈ Σ_f c_f z_src(f) + c0      (affine LSQ, source space)
  ẑ_tgt(h) = Σ_f c_f z_tgt(f) + c0·1?  — NO: c0 is a vector offset only in
  its own space; we use HOMOGENEOUS barycentric form instead: solve with the
  constraint-free design [anchors; 1] per dimension is ill-typed across spaces.
  Correct dimension-free form: affine combination c with Σc_f = 1 (barycentric),
  fit by LSQ on centered anchors; then ẑ_tgt(h) = Σ_f c_f z_tgt(f).

One kernel loads Qwen then Mistral sequentially (mounted paths). Arms:
  within-qwen, within-mistral (geometry control), cross (Qwen coeffs → Mistral
  anchors → tested in Mistral). Tasks: retrieve + consequence (T1 add).
Conditions: NATIVE / SYNTH / WRONG-SYNTH / 30 norm-matched randoms.
Gates + verdicts (SHARED_ATLAS / LOCAL_CHARTS / NO_AFFINE_CODEBOOK) frozen in
the log. Per-arm G0; a dead arm never kills the kernel.
"""
from __future__ import annotations

import gc
import json
import os

import numpy as np
import torch

from .delta_consequence_law import _digit_contract, _t1
from .delta_workspace_matrix import _chat, _enc
from .logutil import Heartbeat, log
from .model_utils import (input_device, last_token_logits,
                          load_model_and_tokenizer)
from .nulls import permutation_pvalue
from .patching import cache_layer_outputs, forward_with_add

N_NULL = 30
G0 = 0.90
DIGITS = list(range(1, 10))
FIT = [1, 2, 4, 5, 7, 8]
HELD = [3, 6, 9]
HELD_CONS = [3, 6]                       # 9 excluded: h+b would exceed 9
BASES_RET = [2, 5, 7, 8]                 # base value a per retrieve row (∈ FIT)
CTX_CONS = [(2, 3), (5, 1), (4, 2), (2, 2)]   # (a, b) contexts, a ∈ FIT
GATE_ACC = 0.80
GATE_RATIO = 0.70
GATE_P = 0.04


def _tr(a):
    return f"X = {a}. What is the value of X?"


def _resolve(model_path):
    if "*" in model_path:
        import glob as _glob
        hits = sorted(_glob.glob(model_path, recursive=True))
        assert hits, f"model_path glob matched nothing: {model_path}"
        mp = hits[0]
        if os.path.basename(mp) == "config.json":
            mp = os.path.dirname(mp)
        return mp
    return model_path


def barycentric_coeffs(z_anchor, z_target):
    """Affine (Σc=1) LSQ coefficients expressing z_target in the anchors.
    z_anchor [K, D] float64, z_target [D]. Returns c [K] with sum 1."""
    K = z_anchor.shape[0]
    mean = z_anchor.mean(0)
    A = (z_anchor - mean).T                      # [D, K]
    y = (z_target - mean)                        # [D]
    # least squares in the centered space, then renormalize to Σc = 1
    sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)   # [K]
    c = sol + (1.0 - sol.sum()) / K
    return c


def synthesize(c, z_anchor):
    """ẑ = Σ c_f z_f (barycentric ⇒ affine-equivariant across spaces)."""
    return (c.unsqueeze(1) * z_anchor).sum(0)


def _extract_phase(model_path, quantization, layer):
    """Load a model, extract anchor states + run its task phase later.
    Returns (model, tok, contract dicts, anchor Z dict)."""
    model, tok = load_model_and_tokenizer(_resolve(model_path),
                                          quantization=quantization)
    dev = input_device(model)
    cr = _digit_contract(tok, lambda a: _tr(a), [((2,), (4,))])
    cc = _digit_contract(tok, _t1, [(((2, 3)), ((4, 3))), (((2, 3)), ((2, 4)))])
    neut = [_chat(tok, f"Here is a token: {v}.", "") for v in DIGITS]
    enc_n = [_enc(tok, t) for t in neut]
    assert len({len(e) for e in enc_n}) == 1
    ids_n = torch.tensor(enc_n, dtype=torch.long, device=dev)
    cache = cache_layer_outputs(model, ids_n, torch.ones_like(ids_n),
                                to_cpu=True)
    dn = [i for i, (x, y) in enumerate(zip(enc_n[0], enc_n[1])) if x != y]
    assert len(dn) == 1
    Z = {v: cache[layer][i, dn[0], :].double() for i, v in enumerate(DIGITS)}
    return model, tok, dev, cr, cc, Z


def _run_arm(tag, model, tok, dev, cr, cc, Z_native, synth_states, layer,
             n_null, rng, hb):
    """synth_states: {h: ẑ(h) tensor} in THIS model's space (or None ⇒ skip)."""
    from .delta_workspace_matrix import PRIMER

    def batch(texts, common):
        e = [_enc(tok, _chat(tok, t, PRIMER)) + list(common) for t in texts]
        assert len({len(x) for x in e}) == 1
        t = torch.tensor(e, dtype=torch.long, device=dev)
        return t, torch.ones_like(t)

    @torch.no_grad()
    def lg_of(ids, am, delta=None, pos=None):
        if delta is None:
            return last_token_logits(model, ids, am).float()
        return forward_with_add(model, ids, am, layer, pos, delta).float()

    out = {}
    for task in ("retrieve", "consequence"):
        if task == "retrieve":
            con = cr
            pos_v = con["positions"][0]
            rows = [(a, h) for h in HELD for a in BASES_RET if a != h]
            texts_c = [_tr(a) for a, h in rows]
            texts_n = [_tr(h) for a, h in rows]
            f_clean = [a for a, h in rows]
            f_tgt = [h for a, h in rows]
            wrong_of = {3: 6, 6: 9, 9: 3}
            f_wr = [wrong_of[h] for a, h in rows]
            base_val = [a for a, h in rows]
        else:
            con = cc
            pos_v = con["positions"][0]
            rows = [(a, b, h) for h in HELD_CONS for (a, b) in CTX_CONS
                    if a != h and h + b <= 9]
            texts_c = [_t1(a, b) for a, b, h in rows]
            texts_n = [_t1(h, b) for a, b, h in rows]
            f_clean = [a + b for a, b, h in rows]
            f_tgt = [h + b for a, b, h in rows]
            wrong_of = {3: 6, 6: 3}
            f_wr = [wrong_of[h] + b for a, b, h in rows]
            base_val = [a for a, b, h in rows]
        held = ([h for a, h in rows] if task == "retrieve"
                else [h for a, b, h in rows])
        wrong_held = ([wrong_of[h] for h in held])

        ids_c, am_c = batch(texts_c, con["common"])
        ids_n2, am_n2 = batch(texts_n, con["common"])
        n = len(rows)
        ar = torch.arange(n, device=dev)
        amap = con["amap"]
        digit_ids = torch.tensor([amap[d] for d in DIGITS], device=dev)
        t_clean = torch.tensor([amap[x] for x in f_clean], device=dev)
        t_tgt = torch.tensor([amap[x] for x in f_tgt], device=dev)
        k_clean = np.array([DIGITS.index(x) for x in f_clean])
        k_tgt = np.array([DIGITS.index(x) for x in f_tgt])
        k_wr = np.array([DIGITS.index(x) for x in f_wr])

        def margin(lg):
            return (lg[ar, t_tgt] - lg[ar, t_clean]).cpu().numpy()

        def pick(lg):
            return lg[:, digit_ids].argmax(-1).cpu().numpy()

        lg_c = lg_of(ids_c, am_c)
        lg_n = lg_of(ids_n2, am_n2)
        g0c = float((pick(lg_c) == k_clean).mean())
        g0n = float((pick(lg_n) == k_tgt).mean())
        cell = {"n_rows": n, "g0_clean": g0c, "g0_natural": g0n}
        if min(g0c, g0n) < G0:
            cell["verdict"] = "INELICITABLE"
            out[task] = cell
            log(f"  [{tag}/{task}] G0 fail {g0c:.0%}/{g0n:.0%}")
            continue

        d_nat = torch.stack([(Z_native[h] - Z_native[a]).float()
                             for h, a in zip(held, base_val)])
        d_syn = torch.stack([(synth_states[h] - Z_native[a]).float()
                             for h, a in zip(held, base_val)])
        d_wr = torch.stack([(synth_states[w] - Z_native[a]).float()
                            for w, a in zip(wrong_held, base_val)])
        m_c = margin(lg_c)
        lg_nat = lg_of(ids_c, am_c, d_nat, pos_v)
        lg_syn = lg_of(ids_c, am_c, d_syn, pos_v)
        lg_wr = lg_of(ids_c, am_c, d_wr, pos_v)
        nat_eff = float((margin(lg_nat) - m_c).mean())
        syn_eff = float((margin(lg_syn) - m_c).mean())
        acc_nat = float((pick(lg_nat) == k_tgt).mean())
        acc_syn = float((pick(lg_syn) == k_tgt).mean())
        acc_wr = float((pick(lg_wr) == k_wr).mean())
        ratio = syn_eff / nat_eff if abs(nat_eff) > 1e-6 else float("nan")
        norms = d_syn.norm(dim=1, keepdim=True)
        nulls = []
        for _ in range(n_null):
            r = torch.randn(d_syn.shape)
            r = r / r.norm(dim=1, keepdim=True).clamp(min=1e-8) * norms
            nulls.append(float((margin(lg_of(ids_c, am_c, r, pos_v))
                                - m_c).mean()))
            hb.step()
        p = permutation_pvalue(syn_eff, np.asarray(nulls), "greater")
        ok = (acc_syn >= GATE_ACC and np.isfinite(ratio)
              and ratio >= GATE_RATIO and p < GATE_P and acc_wr >= GATE_ACC)
        cell.update({"native_acc": acc_nat, "synth_acc": acc_syn,
                     "wrong_synth_own_acc": acc_wr,
                     "native_effect": nat_eff, "synth_effect": syn_eff,
                     "ratio": float(ratio), "p": float(p),
                     "verdict": "SYNTH_PASS" if ok else "SYNTH_FAIL"})
        out[task] = cell
        log(f"  [{tag}/{task}] {cell['verdict']} native={acc_nat:.0%} "
            f"synth={acc_syn:.0%} wrong_own={acc_wr:.0%} ratio={ratio:.2f} "
            f"p={p:.3f}")
    out["pass"] = all(out[t].get("verdict") == "SYNTH_PASS"
                      for t in ("retrieve", "consequence"))
    return out


def run_delta_atlas(qwen_path, mistral_path, out_dir, quantization="8bit",
                    layer=2, seed=0, n_null=N_NULL):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    hb = Heartbeat(6 * n_null, "atlas", every_sec=20, out_dir=out_dir)
    results = {"stage": "delta_atlas", "layer": layer, "fit": FIT,
               "held": HELD, "n_null": n_null, "arms": {}}

    # ---------------- phase 1: Qwen ----------------------------------------
    log("=== atlas phase 1: Qwen ===")
    model, tok, dev, cr, cc, Zq = _extract_phase(qwen_path, quantization, layer)
    anch_q = torch.stack([Zq[f] for f in FIT])            # [6, Dq] float64
    coeffs = {h: barycentric_coeffs(anch_q, Zq[h]) for h in HELD}
    results["coeffs"] = {h: [float(x) for x in coeffs[h]] for h in HELD}
    log(f"  barycentric coeffs (fit in Qwen): "
        f"{ {h: [round(float(x), 3) for x in coeffs[h]] for h in HELD} }")
    synth_q = {h: synthesize(coeffs[h], anch_q) for h in HELD}
    for h in HELD:
        cos = torch.nn.functional.cosine_similarity(
            synth_q[h].unsqueeze(0), Zq[h].unsqueeze(0)).item()
        log(f"  within-Qwen ẑ({h}) vs z({h}): cos={cos:.4f}")
    results["arms"]["within_qwen"] = _run_arm(
        "within_qwen", model, tok, dev, cr, cc, Zq, synth_q, layer, n_null,
        rng, hb)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- phase 2: Mistral -------------------------------------
    log("=== atlas phase 2: Mistral ===")
    model, tok, dev, cr, cc, Zm = _extract_phase(mistral_path, quantization,
                                                 layer)
    anch_m = torch.stack([Zm[f] for f in FIT])
    coeffs_m = {h: barycentric_coeffs(anch_m, Zm[h]) for h in HELD}
    synth_m_within = {h: synthesize(coeffs_m[h], anch_m) for h in HELD}
    results["arms"]["within_mistral"] = _run_arm(
        "within_mistral", model, tok, dev, cr, cc, Zm, synth_m_within, layer,
        n_null, rng, hb)
    # CROSS: Qwen-fit coefficients applied to Mistral anchors
    synth_m_cross = {h: synthesize(coeffs[h], anch_m) for h in HELD}
    for h in HELD:
        cos = torch.nn.functional.cosine_similarity(
            synth_m_cross[h].unsqueeze(0), Zm[h].unsqueeze(0)).item()
        log(f"  cross ẑ_m({h}) [Qwen coeffs] vs z_m({h}): cos={cos:.4f}")
    results["arms"]["cross_qwen_to_mistral"] = _run_arm(
        "cross_q2m", model, tok, dev, cr, cc, Zm, synth_m_cross, layer,
        n_null, rng, hb)
    hb.done()

    wq = results["arms"]["within_qwen"]["pass"]
    wm = results["arms"]["within_mistral"]["pass"]
    cx = results["arms"]["cross_qwen_to_mistral"]["pass"]
    if wq and wm and cx:
        verdict = "SHARED_ATLAS"
    elif wq and wm:
        verdict = "LOCAL_CHARTS"
    elif not (wq or wm):
        verdict = "NO_AFFINE_CODEBOOK"
    else:
        verdict = "MIXED_WITHIN"
    results["verdict"] = verdict
    with open(os.path.join(out_dir, "results_delta_atlas.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT atlas: {verdict} | within_qwen={wq} within_mistral={wm} "
        f"cross={cx}")
    return results
