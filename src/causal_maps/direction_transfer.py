"""Direction-transfer test (Variable skill) — the natural last move.

Hypothesis (Sahil, 2026-07-12): intelligence lives in low-dimensional superposed
directions that are prompt-conditioned in *where* they apply but reusable in
*what* they compute. P1 failed because it scored site-maps (wrong coordinate
system), not because the mechanism isn't reusable.

Test (one kernel, no fishing):
  Δ = mean(h_cf − h_clean) at (layer L*, val_slot) on donor templates
  Add Δ to a held-out template's clean run at its val_slot.
  If answer log-odds flip toward cf, the mechanism was a reusable direction.

Pre-registered (see CAUSAL_MAPS_LOG.md):
  L* = 2 (layer of max mean IE at val_slot from Variable P1 @7B)
  Primary: donors {X,Y} → target Z; also leave-one-out over all 5 templates
  Controls: within-template Δ (positive); random same-norm directions (null)
  PASS iff cross-template mean ΔIE > 0, beats random null p<0.01, AND
        cross / within ≥ 0.5 (substantial transfer, not a trace effect)
"""
import json
import os

import numpy as np
import torch

from . import variable_pairs
from .logutil import log
from .model_utils import (input_device, last_token_logits,
                          load_model_and_tokenizer, logit_diff)
from .nulls import permutation_pvalue
from .patching import cache_layer_outputs, forward_with_add
from .tensorize import tensorize_pairs

PRIMARY_LAYER = 2
N_NULL = 200
N_NULL_LOO = 50
SCALE = 1.0


def _idx_by_template(templates):
    by = {}
    for i, t in enumerate(templates):
        by.setdefault(t, []).append(i)
    return by


@torch.no_grad()
def _slot_acts(model, batch, layer, pos, cache_to_cpu=True):
    dev = input_device(model)
    c_cache = cache_layer_outputs(
        model, batch["clean"]["input_ids"].to(dev),
        batch["clean"]["attention_mask"].to(dev), to_cpu=cache_to_cpu)
    f_cache = cache_layer_outputs(
        model, batch["cf"]["input_ids"].to(dev),
        batch["cf"]["attention_mask"].to(dev), to_cpu=cache_to_cpu)
    hc = c_cache[layer][:, pos, :].float().cpu()
    hf = f_cache[layer][:, pos, :].float().cpu()
    return hc, hf


def _subset(batch, idx):
    t = torch.tensor(idx, dtype=torch.long)
    return {
        "clean": {"input_ids": batch["clean"]["input_ids"][t],
                  "attention_mask": batch["clean"]["attention_mask"][t]},
        "cf": {"input_ids": batch["cf"]["input_ids"][t],
               "attention_mask": batch["cf"]["attention_mask"][t]},
        "pos_ids": batch["pos_ids"][t],
        "neg_ids": batch["neg_ids"][t],
    }


@torch.no_grad()
def _baselines(model, batch):
    """Cache clean logit_diff and clean greedy-hit once per subset."""
    dev = input_device(model)
    ci = batch["clean"]["input_ids"].to(dev)
    cam = batch["clean"]["attention_mask"].to(dev)
    pos_ids = batch["pos_ids"].to(dev)
    neg_ids = batch["neg_ids"].to(dev)
    logits = last_token_logits(model, ci, cam)
    clean_ld = logit_diff(logits, pos_ids, neg_ids)
    clean_hit = (logits.argmax(-1) == neg_ids)
    return {
        "ci": ci, "cam": cam, "pos_ids": pos_ids, "neg_ids": neg_ids,
        "clean_ld": clean_ld, "clean_hit": clean_hit,
    }


@torch.no_grad()
def _effect_with_baseline(model, base, layer, pos, delta, scale=SCALE):
    """ΔIE and flip-rate given cached clean baselines. One forward only."""
    patched = forward_with_add(
        model, base["ci"], base["cam"], layer, pos, delta, scale=scale)
    ld = logit_diff(patched, base["pos_ids"], base["neg_ids"])
    delta_ld = (ld - base["clean_ld"]).float().cpu().numpy()
    flip = (base["clean_hit"] & (patched.argmax(-1) == base["pos_ids"])).float().cpu().numpy()
    return delta_ld, flip


def run_delta_transfer(model_path, out_dir, quantization="8bit", device_map=None,
                       layer=PRIMARY_LAYER, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    pairs = variable_pairs.make_variable_pairs(50, seed=seed, tok=tok)
    batch = tensorize_pairs(tok, pairs, require_anchor_roles=("val_slot",))
    pos = batch["anchors"]["val_slot"]
    templates = batch["templates"]
    by_t = _idx_by_template(templates)
    names = sorted(by_t.keys())
    log(f"delta_transfer: layer={layer} val_slot={pos} "
        f"templates={ {k: len(v) for k, v in by_t.items()} }")

    hc_all, hf_all = _slot_acts(model, batch, layer, pos)
    delta_per_pair = hf_all - hc_all
    mean_delta = {}
    for t, idxs in by_t.items():
        mean_delta[t] = delta_per_pair[idxs].mean(0)
        log(f"  Δ_{t}: ||Δ||={float(mean_delta[t].norm()):.3f}")

    # Cache baselines per template (one forward each)
    bases = {}
    for t, idxs in by_t.items():
        bases[t] = _baselines(model, _subset(batch, idxs))
        log(f"  baseline {t}: mean clean_ld={float(bases[t]['clean_ld'].mean()):+.3f}")

    rng = np.random.default_rng(seed)
    results = {
        "stage": "delta_transfer",
        "model_path": model_path,
        "layer": int(layer),
        "val_slot": int(pos),
        "scale": SCALE,
        "n_null": N_NULL,
        "templates": {k: len(v) for k, v in by_t.items()},
        "primary": {},
        "leave_one_out": [],
        "within": {},
    }

    for t in names:
        dld, flip = _effect_with_baseline(model, bases[t], layer, pos, mean_delta[t])
        results["within"][t] = {
            "mean_delta_ie": float(dld.mean()),
            "flip_rate": float(flip.mean()),
            "n": len(by_t[t]),
        }
        log(f"  WITHIN {t}: ΔIE={dld.mean():+.3f} flip={flip.mean():.0%}")

    donors = [t for t in ("variable_X", "variable_Y") if t in mean_delta]
    targets = [t for t in ("variable_Z", "variable_W", "variable_K") if t in by_t]
    if donors:
        delta_xy = torch.stack([mean_delta[t] for t in donors]).mean(0)
        primary_rows = []
        for tgt in targets:
            dld, flip = _effect_with_baseline(model, bases[tgt], layer, pos, delta_xy)
            norm = float(delta_xy.norm().clamp(min=1e-8))
            null_means = []
            for i in range(N_NULL):
                v = torch.from_numpy(rng.normal(size=delta_xy.numel()).astype(np.float32))
                v = v / v.norm().clamp(min=1e-8) * norm
                ndld, _ = _effect_with_baseline(model, bases[tgt], layer, pos, v)
                null_means.append(float(ndld.mean()))
                if (i + 1) % 50 == 0:
                    log(f"  null {tgt} {i+1}/{N_NULL}")
            null_means = np.asarray(null_means)
            real = float(dld.mean())
            p = permutation_pvalue(real, null_means, "greater")
            within_ie = results["within"][tgt]["mean_delta_ie"]
            ratio = real / within_ie if abs(within_ie) > 1e-6 else float("nan")
            row = {
                "donors": donors, "target": tgt,
                "mean_delta_ie": real,
                "flip_rate": float(flip.mean()),
                "null_mean": float(null_means.mean()),
                "null_p95": float(np.percentile(null_means, 95)),
                "p": float(p),
                "within_ie": within_ie,
                "cross_over_within": float(ratio),
            }
            primary_rows.append(row)
            log(f"  CROSS {donors}->{tgt}: ΔIE={real:+.3f} flip={flip.mean():.0%} "
                f"null={null_means.mean():+.3f} p={p:.4f} ratio={ratio:.2f}")
        results["primary"] = {
            "donors": donors,
            "rows": primary_rows,
            "mean_cross_ie": float(np.mean([r["mean_delta_ie"] for r in primary_rows])),
            "mean_ratio": float(np.nanmean([r["cross_over_within"] for r in primary_rows])),
            "all_beat_null": all(r["p"] < 0.01 for r in primary_rows),
            "all_ratio_ge_0_5": all(
                (r["cross_over_within"] >= 0.5) for r in primary_rows
                if np.isfinite(r["cross_over_within"])),
        }

    loo = []
    for held in names:
        donors_loo = [t for t in names if t != held]
        delta = torch.stack([mean_delta[t] for t in donors_loo]).mean(0)
        dld, flip = _effect_with_baseline(model, bases[held], layer, pos, delta)
        within_ie = results["within"][held]["mean_delta_ie"]
        real = float(dld.mean())
        ratio = real / within_ie if abs(within_ie) > 1e-6 else float("nan")
        norm = float(delta.norm().clamp(min=1e-8))
        null_means = []
        for _ in range(N_NULL_LOO):
            v = torch.from_numpy(rng.normal(size=delta.numel()).astype(np.float32))
            v = v / v.norm().clamp(min=1e-8) * norm
            ndld, _ = _effect_with_baseline(model, bases[held], layer, pos, v)
            null_means.append(float(ndld.mean()))
        p = permutation_pvalue(real, np.asarray(null_means), "greater")
        loo.append({
            "held_out": held, "donors": donors_loo,
            "mean_delta_ie": real, "flip_rate": float(flip.mean()),
            "within_ie": within_ie, "cross_over_within": float(ratio),
            "p": float(p),
        })
        log(f"  LOO ->{held}: ΔIE={real:+.3f} ratio={ratio:.2f} p={p:.4f}")
    results["leave_one_out"] = loo
    results["loo_mean_ratio"] = float(np.nanmean([r["cross_over_within"] for r in loo]))
    results["loo_mean_ie"] = float(np.mean([r["mean_delta_ie"] for r in loo]))

    prim = results["primary"]
    pass_gate = bool(
        prim.get("mean_cross_ie", 0) > 0
        and prim.get("all_beat_null", False)
        and prim.get("all_ratio_ge_0_5", False)
    )
    results["pass"] = pass_gate
    results["verdict"] = "DIRECTION_REUSABLE" if pass_gate else "NO_TRANSFER_OR_WEAK"
    results["loo_supports"] = bool(
        results["loo_mean_ie"] > 0 and results["loo_mean_ratio"] >= 0.5)

    with open(os.path.join(out_dir, "results_delta_transfer.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    if donors:
        np.savez(os.path.join(out_dir, "delta_vectors.npz"),
                 delta_xy=delta_xy.numpy(),
                 **{f"delta_{t.split('_')[-1]}": mean_delta[t].numpy()
                    for t in names})
    log(f"VERDICT delta_transfer: pass={pass_gate} ({results['verdict']}) "
        f"primary_ie={prim.get('mean_cross_ie')} ratio={prim.get('mean_ratio')} "
        f"loo_ratio={results['loo_mean_ratio']:.2f}")
    return results
