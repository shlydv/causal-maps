"""Centroid-removal control (Variable skill) — decisive pre-extension test.

Remove the single empirical generic direction — the centroid g = mean of the
per-value directions Δ_v (the aggregate that produced GENERIC_BOOST) — from each
value direction and re-measure transfer + selectivity:

    d_v' = Δ_v − (Δ_v · ĝ) ĝ          ĝ = g / ‖g‖

If selectivity survives (norm-matched retention ≈ 1), the shared component
contributes nothing beyond centroid geometry. Pre-registered in
CAUSAL_MAPS_LOG.md (2026-07-12 CONTROL). Gates C1 transfer, C2 selective,
C3 selectivity preserved (KEY), C4 centroid non-selective -> CENTROID_IRRELEVANT.
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


def remove_direction(d, g):
    """Return d with its component along g removed (g need not be unit)."""
    ghat = g / g.norm().clamp(min=1e-8)
    return d - (d @ ghat) * ghat


@torch.no_grad()
def _val_logits(model, ids, am, val_ids, layer=None, pos=None, delta=None, scale=1.0):
    if delta is None:
        logits = last_token_logits(model, ids, am)
    else:
        logits = forward_with_add(model, ids, am, layer, pos, delta, scale=scale)
    return logits[:, val_ids].float().cpu().numpy()


def run_delta_centroid(model_path, out_dir, quantization="8bit", device_map=None,
                       layer=PRIMARY_LAYER, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    pairs = variable_pairs.make_variable_pairs(50, seed=seed, tok=tok)
    batch = tensorize_pairs(tok, pairs, require_anchor_roles=("val_slot",))
    pos = batch["anchors"]["val_slot"]
    metas = batch["metas"]
    dev = input_device(model)

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

    hc, hf = _slot_acts(model, batch, layer, pos)
    delta_per_pair = hf - hc
    idx_by_val = {v: [i for i, m in enumerate(metas) if m["val_cf"] == v] for v in values}
    Delta = {v: delta_per_pair[idx_by_val[v]].mean(0) for v in values}

    g = torch.stack([Delta[v] for v in values]).mean(0)   # centroid = empirical generic
    g_hat = g / g.norm().clamp(min=1e-8)
    log(f"delta_centroid: layer={layer} val_slot={pos} nV={nV} values={values} "
        f"||centroid||={float(g.norm()):.3f}")

    clean_ids_all = batch["clean"]["input_ids"]
    rng = np.random.default_rng(seed)
    per_value = []
    agg = {k: {"transfer": [], "selectivity": []}
           for k in ("matched", "dprime", "dprime_nm", "centroid_nm")}
    null_t, null_sel = [], []
    hb = Heartbeat(nV * N_NULL, "delta_centroid", every_sec=15, out_dir=out_dir)

    for vi, v in enumerate(values):
        dv = Delta[v]
        norm_dv = float(dv.norm().clamp(min=1e-8))
        dprime = remove_direction(dv, g)
        ndp = float(dprime.norm().clamp(min=1e-8))
        cos_dg = float(dv @ g_hat) / norm_dv
        removed_frac = 1.0 - ndp / norm_dv

        test_idx = [i for i, m in enumerate(metas) if m["val_cf"] != v]
        ids = clean_ids_all[test_idx].to(dev)
        am = torch.ones_like(ids)
        base = _val_logits(model, ids, am, val_ids)

        def eff(delta, scale):
            add = _val_logits(model, ids, am, val_ids, layer, pos, delta, scale)
            md = (add - base).mean(0)
            transfer = float(md[vi])
            others = float((md.sum() - md[vi]) / (nV - 1))
            return transfer, transfer - others

        t_m, sel_m = eff(dv, 1.0)                        # matched baseline
        t_dp, sel_dp = eff(dprime, 1.0)                  # d' at natural norm
        t_dpnm, sel_dpnm = eff(dprime, norm_dv / ndp)    # d' norm-matched to ||Δ||
        t_cnm, sel_cnm = eff(g_hat, norm_dv)             # centroid-only at ||Δ||

        nt, ns_ = [], []
        for _ in range(N_NULL):
            r = torch.from_numpy(rng.normal(size=dv.numel()).astype(np.float32))
            r = r / r.norm().clamp(min=1e-8) * ndp       # match d' norm
            tr, se = eff(r, 1.0)
            nt.append(tr); ns_.append(se)
            hb.step(extra=f"value {vi+1}/{nV}={v}")
        p_t = permutation_pvalue(t_dp, np.asarray(nt), "greater")
        p_sel = permutation_pvalue(sel_dp, np.asarray(ns_), "greater")

        ret_nat = sel_dp / sel_m if abs(sel_m) > 1e-6 else float("nan")
        ret_nm = sel_dpnm / sel_m if abs(sel_m) > 1e-6 else float("nan")
        per_value.append({
            "value": v, "cos_delta_centroid": cos_dg, "removed_frac": removed_frac,
            "norm_delta": norm_dv, "norm_dprime": ndp,
            "matched": {"transfer": t_m, "selectivity": sel_m},
            "dprime": {"transfer": t_dp, "selectivity": sel_dp,
                       "p_transfer": float(p_t), "p_selectivity": float(p_sel),
                       "null_sel_mean": float(np.mean(ns_))},
            "dprime_nm": {"transfer": t_dpnm, "selectivity": sel_dpnm},
            "centroid_nm": {"transfer": t_cnm, "selectivity": sel_cnm},
            "selectivity_retention_natural": ret_nat,
            "selectivity_retention_normmatched": ret_nm,
        })
        for k, tr, se in (("matched", t_m, sel_m), ("dprime", t_dp, sel_dp),
                          ("dprime_nm", t_dpnm, sel_dpnm), ("centroid_nm", t_cnm, sel_cnm)):
            agg[k]["transfer"].append(tr); agg[k]["selectivity"].append(se)
        null_t.append(nt); null_sel.append(ns_)
        log(f"  {v}: cosΔg={cos_dg:.2f} removed={removed_frac:.1%} | matched sel={sel_m:+.2f}"
            f" | d' sel={sel_dp:+.2f}(p={p_sel:.3f}) ret_nat={ret_nat:.2f} ret_nm={ret_nm:.2f}"
            f" | centroid_nm sel={sel_cnm:+.2f}")
    hb.done()

    def mean(a):
        return float(np.mean(a))

    def median(a):
        return float(np.median(a))

    mean_dp_transfer = mean(agg["dprime"]["transfer"])
    mean_dp_sel = mean(agg["dprime"]["selectivity"])
    null_t_agg = np.mean(np.asarray(null_t), axis=0)
    null_sel_agg = np.mean(np.asarray(null_sel), axis=0)
    p_transfer = permutation_pvalue(mean_dp_transfer, null_t_agg, "greater")
    p_selectivity = permutation_pvalue(mean_dp_sel, null_sel_agg, "greater")

    ret_nat_med = median([r["selectivity_retention_natural"] for r in per_value])
    ret_nm_med = median([r["selectivity_retention_normmatched"] for r in per_value])
    centroid_sel_nm = mean(agg["centroid_nm"]["selectivity"])

    C1 = mean_dp_transfer > 0 and p_transfer < 0.01
    C2 = mean_dp_sel > 0 and p_selectivity < 0.01
    C3 = ret_nm_med >= 0.90
    C4 = abs(centroid_sel_nm) < 0.5 * mean(agg["dprime_nm"]["selectivity"])
    verdict = "CENTROID_IRRELEVANT" if (C1 and C2 and C3) else "CENTROID_MATTERS"

    results = {
        "stage": "delta_centroid", "model_path": model_path,
        "layer": int(layer), "val_slot": int(pos), "values": values, "n_null": N_NULL,
        "centroid_norm": float(g.norm()),
        "mean_dprime_transfer": mean_dp_transfer, "p_transfer": float(p_transfer),
        "mean_dprime_selectivity": mean_dp_sel, "p_selectivity": float(p_selectivity),
        "selectivity_retention_natural_median": ret_nat_med,
        "selectivity_retention_normmatched_median": ret_nm_med,
        "mean_matched_selectivity": mean(agg["matched"]["selectivity"]),
        "mean_dprime_nm_selectivity": mean(agg["dprime_nm"]["selectivity"]),
        "mean_centroid_nm_selectivity": centroid_sel_nm,
        "mean_centroid_nm_transfer": mean(agg["centroid_nm"]["transfer"]),
        "mean_removed_frac": mean([r["removed_frac"] for r in per_value]),
        "gates": {"C1_transfer": bool(C1), "C2_selective": bool(C2),
                  "C3_selectivity_preserved": bool(C3), "C4_centroid_nonselective": bool(C4)},
        "verdict": verdict, "per_value": per_value,
    }
    with open(os.path.join(out_dir, "results_delta_centroid.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT delta_centroid: {verdict} | C1={C1} C2={C2} C3={C3} C4={C4} | "
        f"d' transfer={mean_dp_transfer:+.3f}(p={p_transfer:.3f}) "
        f"d' sel={mean_dp_sel:+.3f}(p={p_selectivity:.3f}) | "
        f"retention nm={ret_nm_med:.2f} nat={ret_nat_med:.2f} "
        f"centroid_nm_sel={centroid_sel_nm:+.3f} removed={results['mean_removed_frac']:.1%}")
    return results
