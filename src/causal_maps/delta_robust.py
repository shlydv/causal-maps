"""Kernel 1 full-send: Variable robustness + embed + shuffled-pair controls.

Pre-registered in CAUSAL_MAPS_LOG.md (2026-07-12 FULL SEND). No fishing.
"""
import json
import os

import numpy as np
import torch

from . import variable_pairs
from .direction_transfer import (
    PRIMARY_LAYER, _baselines, _effect_with_baseline, _idx_by_template,
    _slot_acts, _subset,
)
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer
from .nulls import permutation_pvalue
from .tensorize import tensorize_pairs

N_NULL = 100
DONORS = ("variable_X", "variable_Y")
TARGETS = ("variable_Z", "variable_W", "variable_K")
ALPHAS_GATE = (0.5, 1.0, 2.0)
ALPHAS_FLIP = (3.0, 4.0, 5.0)
LAYERS = (1, 2, 3, 4)


def _cell_pass(row):
    return bool(
        row["mean_delta_ie"] > 0
        and row["p"] < 0.01
        and row["cross_over_within"] >= 0.5
    )


def _transfer_cell(model, bases, within_ie, delta, targets, layer, pos, alpha, rng,
                   n_null=N_NULL, tag=""):
    rows = []
    for tgt in targets:
        dld, flip = _effect_with_baseline(
            model, bases[tgt], layer, pos, delta, scale=alpha)
        real = float(dld.mean())
        w = within_ie[tgt]
        ratio = real / w if abs(w) > 1e-6 else float("nan")
        norm = float(delta.norm().clamp(min=1e-8)) * abs(alpha)
        # null at same effective scale: random unit * ||αΔ||
        null_means = []
        for _ in range(n_null):
            v = torch.from_numpy(rng.normal(size=delta.numel()).astype(np.float32))
            v = v / v.norm().clamp(min=1e-8) * (norm / max(abs(alpha), 1e-8))
            # apply with same alpha so effective add = alpha * v_unit * ||Δ||
            ndld, _ = _effect_with_baseline(
                model, bases[tgt], layer, pos, v, scale=alpha)
            null_means.append(float(ndld.mean()))
        null_means = np.asarray(null_means)
        p = permutation_pvalue(real, null_means, "greater")
        row = {
            "target": tgt, "mean_delta_ie": real, "flip_rate": float(flip.mean()),
            "within_ie": w, "cross_over_within": float(ratio),
            "null_mean": float(null_means.mean()),
            "null_p95": float(np.percentile(null_means, 95)),
            "p": float(p), "cell_pass": False,
        }
        row["cell_pass"] = _cell_pass(row)
        rows.append(row)
        log(f"  {tag} ->{tgt}: ΔIE={real:+.3f} flip={flip.mean():.0%} "
            f"ratio={ratio:.2f} p={p:.4f} pass={row['cell_pass']}")
    n_pass = sum(1 for r in rows if r["cell_pass"])
    return {
        "rows": rows,
        "n_pass": n_pass,
        "mean_cross_ie": float(np.mean([r["mean_delta_ie"] for r in rows])),
        "mean_ratio": float(np.nanmean([r["cross_over_within"] for r in rows])),
        "gate_pass": n_pass == len(rows),  # all targets for primary gate cells
    }


@torch.no_grad()
def _embed_delta(model, batch, donor_idx, pos):
    """Mean (embed_cf − embed_clean) at token position pos over donor pairs."""
    emb = model.get_input_embeddings()
    dev = input_device(model)
    ci = batch["clean"]["input_ids"][donor_idx].to(dev)
    fi = batch["cf"]["input_ids"][donor_idx].to(dev)
    # embeddings [B,S,D]; take pos
    ec = emb(ci)[:, pos, :].float()
    ef = emb(fi)[:, pos, :].float()
    return (ef - ec).mean(0).cpu()


def _shuffled_delta(hc, hf, donor_idx, rng):
    """Δ from mismatched pairs: clean[i] with cf[π(i)], π derangement."""
    idx = np.asarray(donor_idx)
    n = len(idx)
    # derangement
    perm = np.arange(n)
    for _ in range(100):
        rng.shuffle(perm)
        if not np.any(perm == np.arange(n)):
            break
    else:
        # fallback: rotate
        perm = np.roll(np.arange(n), 1)
    # hc/hf are [B,D] full batch; index with donor absolute indices
    d = hf[idx[perm]] - hc[idx]
    return d.mean(0)


def run_var_robust(model_path, out_dir, quantization="8bit", device_map=None, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    pairs = variable_pairs.make_variable_pairs(50, seed=seed, tok=tok)
    batch = tensorize_pairs(tok, pairs, require_anchor_roles=("val_slot",))
    pos = batch["anchors"]["val_slot"]
    by_t = _idx_by_template(batch["templates"])
    donors = [t for t in DONORS if t in by_t]
    targets = [t for t in TARGETS if t in by_t]
    donor_idx = [i for t in donors for i in by_t[t]]
    log(f"delta_var_robust: val_slot={pos} donors={donors} targets={targets}")

    bases = {t: _baselines(model, _subset(batch, by_t[t])) for t in by_t}
    rng = np.random.default_rng(seed)

    # Cache acts at each needed layer
    acts = {}
    for L in LAYERS:
        hc, hf = _slot_acts(model, batch, L, pos)
        acts[L] = (hc, hf)
        log(f"  cached L={L}")

    results = {
        "stage": "delta_var_robust",
        "model_path": model_path,
        "val_slot": int(pos),
        "donors": donors,
        "targets": targets,
        "n_null": N_NULL,
        "alpha_grid": {},
        "layer_grid": {},
        "flip_hunt": {},
        "embed_control": {},
        "shuffle_control": {},
    }

    def mean_delta(L, idxs):
        hc, hf = acts[L]
        return (hf[idxs] - hc[idxs]).mean(0)

    def within_at(L, alpha=1.0):
        out = {}
        for t in targets:
            d = mean_delta(L, by_t[t])
            dld, flip = _effect_with_baseline(
                model, bases[t], L, pos, d, scale=alpha)
            out[t] = float(dld.mean())
            log(f"  WITHIN L{L} α={alpha} {t}: ΔIE={out[t]:+.3f} flip={flip.mean():.0%}")
        return out

    # ---- 1a α grid at L=2 ----
    within_L2 = within_at(PRIMARY_LAYER, 1.0)
    delta_L2 = mean_delta(PRIMARY_LAYER, donor_idx)
    for alpha in ALPHAS_GATE:
        # within at this alpha for ratio denominator
        within_a = {}
        for t in targets:
            d = mean_delta(PRIMARY_LAYER, by_t[t])
            dld, _ = _effect_with_baseline(
                model, bases[t], PRIMARY_LAYER, pos, d, scale=alpha)
            within_a[t] = float(dld.mean())
        cell = _transfer_cell(
            model, bases, within_a, delta_L2, targets,
            PRIMARY_LAYER, pos, alpha, rng, tag=f"α={alpha} L2")
        results["alpha_grid"][str(alpha)] = cell

    # ---- 1a layer grid at α=1 ----
    for L in LAYERS:
        within_L = within_at(L, 1.0) if L != PRIMARY_LAYER else within_L2
        d = mean_delta(L, donor_idx)
        cell = _transfer_cell(
            model, bases, within_L, d, targets, L, pos, 1.0, rng,
            tag=f"α=1 L={L}")
        results["layer_grid"][str(L)] = cell

    # ---- flip hunt α=3,4,5 (no null; descriptive) ----
    for alpha in ALPHAS_FLIP:
        rows = []
        for tgt in targets:
            dld, flip = _effect_with_baseline(
                model, bases[tgt], PRIMARY_LAYER, pos, delta_L2, scale=alpha)
            rows.append({
                "target": tgt, "mean_delta_ie": float(dld.mean()),
                "flip_rate": float(flip.mean()), "alpha": alpha,
            })
            log(f"  FLIPHUNT α={alpha} ->{tgt}: ΔIE={dld.mean():+.3f} "
                f"flip={flip.mean():.0%}")
        results["flip_hunt"][str(alpha)] = {
            "rows": rows,
            "any_flip": any(r["flip_rate"] > 0 for r in rows),
        }

    # ---- 1b embed control ----
    delta_emb = _embed_delta(model, batch, donor_idx, pos)
    log(f"  ||Δ_embed||={float(delta_emb.norm()):.3f} ||Δ_L2||={float(delta_L2.norm()):.3f}")
    hc0, hf0 = _slot_acts(model, batch, 0, pos)
    acts[0] = (hc0, hf0)
    within_L0 = {}
    for t in targets:
        d0 = (hf0[by_t[t]] - hc0[by_t[t]]).mean(0)
        dld, _ = _effect_with_baseline(model, bases[t], 0, pos, d0, scale=1.0)
        within_L0[t] = float(dld.mean())
    emb_cell = _transfer_cell(
        model, bases, within_L0, delta_emb, targets, 0, pos, 1.0, rng,
        tag="EMBED@L0")
    # Also compare to L2 cross mean
    l2_cell = results["alpha_grid"]["1.0"]
    ratio_e_l2 = (
        emb_cell["mean_cross_ie"] / l2_cell["mean_cross_ie"]
        if abs(l2_cell["mean_cross_ie"]) > 1e-6 else float("nan")
    )
    emb_n_pass = emb_cell["n_pass"]
    l2_pass = l2_cell["gate_pass"]
    if emb_n_pass >= 2 and ratio_e_l2 >= 0.8:
        embed_verdict = "TRIVIAL"
    elif emb_n_pass >= 2 and l2_pass and 0.3 <= ratio_e_l2 < 0.8:
        embed_verdict = "PARTIAL"
    elif emb_n_pass < 2 and l2_pass:
        embed_verdict = "NONTRIVIAL"
    else:
        embed_verdict = "INCONCLUSIVE"
    results["embed_control"] = {
        **emb_cell,
        "norm_embed": float(delta_emb.norm()),
        "norm_L2": float(delta_L2.norm()),
        "cross_embed_over_cross_L2": float(ratio_e_l2),
        "verdict": embed_verdict,
    }
    log(f"  EMBED verdict={embed_verdict} ratio_e/L2={ratio_e_l2:.3f} "
        f"emb_pass={emb_n_pass}/3 l2_pass={l2_pass}")

    # ---- 1c shuffled pairing ----
    hc2, hf2 = acts[PRIMARY_LAYER]
    delta_shuf = _shuffled_delta(hc2, hf2, donor_idx, rng)
    log(f"  ||Δ_shuf||={float(delta_shuf.norm()):.3f}")
    shuf_cell = _transfer_cell(
        model, bases, within_L2, delta_shuf, targets,
        PRIMARY_LAYER, pos, 1.0, rng, tag="SHUF")
    generic = shuf_cell["n_pass"] >= 2
    results["shuffle_control"] = {
        **shuf_cell,
        "generic_boost": generic,
        "verdict": "GENERIC_BOOST" if generic else "SHUFFLE_FAILS_OK",
    }
    log(f"  SHUFFLE verdict={results['shuffle_control']['verdict']} "
        f"pass={shuf_cell['n_pass']}/3")

    # ---- overall ----
    arxiv_clean = bool(
        l2_pass
        and embed_verdict in ("NONTRIVIAL", "PARTIAL")
        and not generic
    )
    results["l2_alpha1_pass"] = l2_pass
    results["embed_verdict"] = embed_verdict
    results["arxiv_path_clean"] = arxiv_clean
    results["flip_any_alpha_le_2"] = any(
        any(r["flip_rate"] > 0 for r in results["alpha_grid"][str(a)]["rows"])
        for a in ALPHAS_GATE
    )
    results["flip_only_alpha_gt_2"] = (
        not results["flip_any_alpha_le_2"]
        and any(results["flip_hunt"][str(a)]["any_flip"] for a in ALPHAS_FLIP)
    )

    path = os.path.join(out_dir, "results_delta_var_robust.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    np.savez(os.path.join(out_dir, "delta_var_robust_vectors.npz"),
             delta_L2=delta_L2.numpy(), delta_embed=delta_emb.numpy(),
             delta_shuf=delta_shuf.numpy())
    log(f"VERDICT delta_var_robust: embed={embed_verdict} shuffle_ok={not generic} "
        f"l2_pass={l2_pass} arxiv_clean={arxiv_clean}")
    return results
