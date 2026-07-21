"""GENERIC_BOOST decomposition (Variable skill) — the reopened experiment.

Split the transferable Variable direction into a generic slot-update component
and a value-specific residual, then test whether the residual (a) transfers and
(b) is value-selective ("pushes cat, not grape").

Per-value direction (the sharpening that makes value-specificity recoverable):
    Δ_v = mean over the pairs with cf == v of (h_cf - h_clean)  at (L=2, val_slot)
Generic subspace = span{Δ_w : w != v}; g_v = proj(Δ_v), residual s_v = Δ_v - g_v.

For each value v, ADD a direction at (L=2, val_slot) to HELD-OUT clean prompts
(pairs with cf != v) and read Δlogit for every value token:
    transfer(d,v)    = mean_prompts Δlogit(v)
    selectivity(d,v) = mean_prompts [ Δlogit(v) - mean_{w!=v} Δlogit(w) ]

Gates (pre-registered in CAUSAL_MAPS_LOG.md, 2026-07-12 REOPEN):
  G0 residual non-trivial (median ||s||/||Δ|| >= 0.10)
  G1 residual transfers (mean transfer(s) > 0, p<0.01 vs same-norm random null)
  G2 residual value-selective (mean selectivity(s) > 0, p<0.01 vs null)
  G3 generic is the (non-selective) boost: norm-matched selectivity(s) > selectivity(g),
     p<0.01, and generic still boosts values (transfer(g) > 0)
  Verdict DECOMPOSED iff G0..G3 else PARTIAL; PURELY_GENERIC if G0 or G2 fails.
"""
import json
import os

import numpy as np
import torch

from . import variable_pairs
from .direction_transfer import PRIMARY_LAYER, _slot_acts
from .logutil import Heartbeat, log
from .model_utils import (input_device, last_token_logits,
                          load_model_and_tokenizer, single_token_id)
from .nulls import permutation_pvalue
from .patching import forward_with_add
from .tensorize import tensorize_pairs

N_NULL = 100


def proj_residual(delta_v, others):
    """Project delta_v onto span(others). Returns (generic g, residual s).
    others: list of 1-D tensors. Uses an orthonormal basis (QR)."""
    M = torch.stack(others, dim=1)              # [D, k]
    Q, _ = torch.linalg.qr(M, mode="reduced")   # [D, r] orthonormal columns
    g = Q @ (Q.t() @ delta_v)                   # projection onto span(others)
    return g, delta_v - g


@torch.no_grad()
def _val_logits(model, ids, am, val_ids, layer=None, pos=None, delta=None, scale=1.0):
    """Last-token logits restricted to val_ids [B, nV]. If delta given, ADD it."""
    if delta is None:
        logits = last_token_logits(model, ids, am)
    else:
        logits = forward_with_add(model, ids, am, layer, pos, delta, scale=scale)
    return logits[:, val_ids].float().cpu().numpy()


def run_delta_decompose(model_path, out_dir, quantization="8bit", device_map=None,
                        layer=PRIMARY_LAYER, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    pairs = variable_pairs.make_variable_pairs(50, seed=seed, tok=tok)
    batch = tensorize_pairs(tok, pairs, require_anchor_roles=("val_slot",))
    pos = batch["anchors"]["val_slot"]
    metas = batch["metas"]
    dev = input_device(model)

    # cf values that (a) are single-token and (b) have >=2 surviving pairs
    values = []
    for _, v1 in variable_pairs._VALUE_PAIRS:
        try:
            single_token_id(tok, v1)
        except ValueError:
            continue
        if sum(1 for m in metas if m["val_cf"] == v1) >= 2 and v1 not in values:
            values.append(v1)
    if len(values) < 3:
        raise RuntimeError(f"too few usable values: {values}")
    val_ids = [single_token_id(tok, v) for v in values]
    nV = len(values)
    log(f"delta_decompose: layer={layer} val_slot={pos} nV={nV} values={values}")

    hc, hf = _slot_acts(model, batch, layer, pos)      # [B, D] cpu float
    delta_per_pair = hf - hc
    idx_by_val = {v: [i for i, m in enumerate(metas) if m["val_cf"] == v] for v in values}
    Delta = {v: delta_per_pair[idx_by_val[v]].mean(0) for v in values}
    for v in values:
        log(f"  Δ_{v}: n={len(idx_by_val[v])} ||Δ||={float(Delta[v].norm()):.3f}")

    clean_ids_all = batch["clean"]["input_ids"]
    rng = np.random.default_rng(seed)
    per_value, agg = [], {k: {"transfer": [], "selectivity": []}
                          for k in ("matched", "generic", "residual",
                                    "generic_nm", "residual_nm")}
    null_transfer, null_selectivity = [], []

    # ETA heartbeat over the dominant cost (null sampling): prints timestamped
    # i/total + elapsed + eta every ~15s and writes progress.json (visible in
    # the Kaggle log).
    hb = Heartbeat(len(values) * N_NULL, "delta_decompose", every_sec=15,
                   out_dir=out_dir)

    for vi, v in enumerate(values):
        g, s = proj_residual(Delta[v], [Delta[w] for w in values if w != v])
        dv = Delta[v]
        norm_dv = float(dv.norm().clamp(min=1e-8))
        ns = float(s.norm().clamp(min=1e-8))
        ng = float(g.norm().clamp(min=1e-8))

        test_idx = [i for i, m in enumerate(metas) if m["val_cf"] != v]
        ids = clean_ids_all[test_idx].to(dev)
        am = torch.ones_like(ids)
        base = _val_logits(model, ids, am, val_ids)           # [B, nV]

        def eff(delta, scale):
            add = _val_logits(model, ids, am, val_ids, layer, pos, delta, scale)
            md = (add - base).mean(0)                          # [nV]
            transfer = float(md[vi])
            others_mean = float((md.sum() - md[vi]) / (nV - 1))
            return transfer, transfer - others_mean

        t_m, sel_m = eff(dv, 1.0)
        t_g, sel_g = eff(g, 1.0)
        t_s, sel_s = eff(s, 1.0)
        t_gnm, sel_gnm = eff(g, norm_dv / ng)                 # norm-matched to ||Δ||
        t_snm, sel_snm = eff(s, norm_dv / ns)

        nt, nsel = [], []
        for _ in range(N_NULL):
            r = torch.from_numpy(rng.normal(size=dv.numel()).astype(np.float32))
            r = r / r.norm().clamp(min=1e-8) * ns             # match residual norm
            tr, se = eff(r, 1.0)
            nt.append(tr); nsel.append(se)
            hb.step(extra=f"value {vi+1}/{nV}={v}")
        p_t = permutation_pvalue(t_s, np.asarray(nt), "greater")
        p_sel = permutation_pvalue(sel_s, np.asarray(nsel), "greater")

        cos_g = float(torch.nn.functional.cosine_similarity(
            dv.unsqueeze(0), g.unsqueeze(0)).item())
        per_value.append({
            "value": v, "n_pairs": len(idx_by_val[v]),
            "norm_delta": norm_dv, "norm_generic": ng, "norm_residual": ns,
            "residual_frac": ns / norm_dv, "cos_delta_generic": cos_g,
            "matched": {"transfer": t_m, "selectivity": sel_m},
            "generic": {"transfer": t_g, "selectivity": sel_g},
            "residual": {"transfer": t_s, "selectivity": sel_s,
                         "p_transfer": float(p_t), "p_selectivity": float(p_sel),
                         "null_transfer_mean": float(np.mean(nt)),
                         "null_selectivity_mean": float(np.mean(nsel))},
            "generic_nm": {"transfer": t_gnm, "selectivity": sel_gnm},
            "residual_nm": {"transfer": t_snm, "selectivity": sel_snm},
        })
        for k, tr, se in (("matched", t_m, sel_m), ("generic", t_g, sel_g),
                          ("residual", t_s, sel_s), ("generic_nm", t_gnm, sel_gnm),
                          ("residual_nm", t_snm, sel_snm)):
            agg[k]["transfer"].append(tr); agg[k]["selectivity"].append(se)
        null_transfer.append(nt); null_selectivity.append(nsel)
        log(f"  {v}: resfrac={ns/norm_dv:.2f} cosΔg={cos_g:.2f} | resid t={t_s:+.3f}"
            f"(p={p_t:.3f}) sel={sel_s:+.3f}(p={p_sel:.3f}) | gen_nm sel={sel_gnm:+.3f}")
    hb.done()

    def mean(a):
        return float(np.mean(a))

    median_resfrac = float(np.median([r["residual_frac"] for r in per_value]))
    mean_res_transfer = mean(agg["residual"]["transfer"])
    mean_res_sel = mean(agg["residual"]["selectivity"])
    null_t_agg = np.mean(np.asarray(null_transfer), axis=0)      # [N_NULL]
    null_sel_agg = np.mean(np.asarray(null_selectivity), axis=0)
    p_transfer = permutation_pvalue(mean_res_transfer, null_t_agg, "greater")
    p_selectivity = permutation_pvalue(mean_res_sel, null_sel_agg, "greater")

    # G3: residual more selective than generic (norm-matched), paired sign-flip null
    diff = np.asarray(agg["residual_nm"]["selectivity"]) - np.asarray(agg["generic_nm"]["selectivity"])
    rng2 = np.random.default_rng(seed + 1)
    nd = [float(np.mean(rng2.choice([-1, 1], size=diff.shape[0]) * diff))
          for _ in range(2000)]
    p_g3 = permutation_pvalue(float(np.mean(diff)), np.asarray(nd), "greater")
    generic_boosts = mean(agg["generic_nm"]["transfer"]) > 0

    G0 = median_resfrac >= 0.10
    G1 = mean_res_transfer > 0 and p_transfer < 0.01
    G2 = mean_res_sel > 0 and p_selectivity < 0.01
    G3 = float(np.mean(diff)) > 0 and p_g3 < 0.01 and generic_boosts
    verdict = ("PURELY_GENERIC" if (not G0 or not G2) else
               ("DECOMPOSED" if (G0 and G1 and G2 and G3) else "PARTIAL"))

    results = {
        "stage": "delta_decompose", "model_path": model_path,
        "layer": int(layer), "val_slot": int(pos), "values": values, "n_null": N_NULL,
        "median_residual_frac": median_resfrac,
        "mean_residual_transfer": mean_res_transfer, "p_transfer": float(p_transfer),
        "mean_residual_selectivity": mean_res_sel, "p_selectivity": float(p_selectivity),
        "mean_generic_selectivity_nm": mean(agg["generic_nm"]["selectivity"]),
        "mean_residual_selectivity_nm": mean(agg["residual_nm"]["selectivity"]),
        "sel_diff_resid_minus_generic_nm": float(np.mean(diff)), "p_g3": float(p_g3),
        "generic_transfer_nm": mean(agg["generic_nm"]["transfer"]),
        "mean_matched_transfer": mean(agg["matched"]["transfer"]),
        "mean_matched_selectivity": mean(agg["matched"]["selectivity"]),
        "mean_generic_transfer": mean(agg["generic"]["transfer"]),
        "mean_generic_selectivity": mean(agg["generic"]["selectivity"]),
        "gates": {"G0_residual_nontrivial": bool(G0), "G1_transfer": bool(G1),
                  "G2_selective": bool(G2), "G3_generic_is_boost": bool(G3)},
        "verdict": verdict, "per_value": per_value,
    }
    with open(os.path.join(out_dir, "results_delta_decompose.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    np.savez(os.path.join(out_dir, "delta_decompose_vectors.npz"),
             **{f"delta_{v}": Delta[v].numpy() for v in values})
    log(f"VERDICT delta_decompose: {verdict} | G0={G0} G1={G1} G2={G2} G3={G3} | "
        f"resfrac={median_resfrac:.2f} t={mean_res_transfer:+.3f}(p={p_transfer:.3f}) "
        f"sel={mean_res_sel:+.3f}(p={p_selectivity:.3f}) "
        f"seldiff={float(np.mean(diff)):+.3f}(p={p_g3:.3f})")
    return results
