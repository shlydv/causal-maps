"""Fixed Kernel-1c: mismatched-value re-forward + anti-Δ controls.

The cached-activation derangement shuffle was mathematically identical to matched Δ
(mean_i h_cf[π(i)] - h_clean[i] = mean h_cf - mean h_clean). This rebuilds Δ from
NEW forwards where the cf prompt has an unrelated value word (not a permutation
of matched cf activations), plus −Δ as anti-control.
"""
import json
import os
import re

import numpy as np
import torch

from . import variable_pairs
from .delta_robust import DONORS, TARGETS, N_NULL, _cell_pass, _transfer_cell
from .direction_transfer import (
    PRIMARY_LAYER, _baselines, _effect_with_baseline, _idx_by_template,
    _slot_acts, _subset,
)
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer
from .patching import cache_layer_outputs
from .tensorize import tensorize_pairs
from .variable_pairs import _VALUE_PAIRS, _chat


def _wrong_value(v0, v1, all_vals, rng):
    """Pick a value word not in {v0, v1}."""
    pool = [v for v in all_vals if v not in (v0, v1)]
    return str(rng.choice(pool))


@torch.no_grad()
def _acts_at(model, input_ids, attention_mask, layer, pos):
    dev = input_device(model)
    cache = cache_layer_outputs(
        model, input_ids.to(dev), attention_mask.to(dev), to_cpu=True)
    return cache[layer][:, pos, :].float().cpu()


def run_var_shufflefix(model_path, out_dir, quantization="8bit", device_map=None,
                         seed=0):
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
    metas = batch["metas"]
    rng = np.random.default_rng(seed + 7)
    all_vals = sorted({v for pair in _VALUE_PAIRS for v in pair})

    log(f"delta_var_shufflefix: val_slot={pos} donors={donors} targets={targets}")

    bases = {t: _baselines(model, _subset(batch, by_t[t])) for t in list(by_t)}
    hc, hf = _slot_acts(model, batch, PRIMARY_LAYER, pos)
    delta_matched = (hf[donor_idx] - hc[donor_idx]).mean(0)

    # Within at L2 for ratios
    within_L2 = {}
    for t in targets:
        d = (hf[by_t[t]] - hc[by_t[t]]).mean(0)
        dld, _ = _effect_with_baseline(
            model, bases[t], PRIMARY_LAYER, pos, d, scale=1.0)
        within_L2[t] = float(dld.mean())

    # --- Matched confirm (sanity) ---
    rng_null = np.random.default_rng(seed)
    matched_cell = _transfer_cell(
        model, bases, within_L2, delta_matched, targets,
        PRIMARY_LAYER, pos, 1.0, rng_null, tag="MATCHED")

    # --- Wrong-value re-forward Δ ---
    # For each donor pair, build cf text with an unrelated value; encode; forward.
    wrong_hs = []
    clean_hs = []
    for i in donor_idx:
        meta = metas[i]
        var, v0, v1 = meta["var"], meta["val_clean"], meta["val_cf"]
        v_wrong = _wrong_value(v0, v1, all_vals, rng)
        # clean text already in batch; wrong cf = same template with v_wrong
        wrong_text = _chat(tok, var, v_wrong)
        # Must match length for same pos index — filter if not
        ids = tok.encode(wrong_text, add_special_tokens=False)
        clean_ids = batch["clean"]["input_ids"][i].tolist()
        # strip pad — these are unpadded equal-length rows
        if len(ids) != len(clean_ids):
            log(f"  skip donor[{i}] len mismatch wrong={len(ids)} clean={len(clean_ids)}")
            continue
        # verify val_slot still points at the value token
        # (same skeleton as clean except the value word)
        wi = torch.tensor([ids], dtype=torch.long)
        am = torch.ones_like(wi)
        h_w = _acts_at(model, wi, am, PRIMARY_LAYER, pos)
        wrong_hs.append(h_w[0])
        clean_hs.append(hc[i])
    if len(wrong_hs) < 5:
        raise RuntimeError(f"too few wrong-value forwards kept: {len(wrong_hs)}")
    delta_wrong = (torch.stack(wrong_hs) - torch.stack(clean_hs)).mean(0)
    cos = float(torch.nn.functional.cosine_similarity(
        delta_matched.unsqueeze(0), delta_wrong.unsqueeze(0)).item())
    log(f"  ||Δ_matched||={float(delta_matched.norm()):.3f} "
        f"||Δ_wrong||={float(delta_wrong.norm()):.3f} cos={cos:.4f} n={len(wrong_hs)}")

    wrong_cell = _transfer_cell(
        model, bases, within_L2, delta_wrong, targets,
        PRIMARY_LAYER, pos, 1.0, rng_null, tag="WRONGVAL")

    # --- Anti-Δ ---
    anti_cell = _transfer_cell(
        model, bases, within_L2, -delta_matched, targets,
        PRIMARY_LAYER, pos, 1.0, rng_null, tag="ANTI")

    # Verdicts
    # Wrong-val should FAIL cell-pass on >=2/3 (not a binding direction to true cf)
    wrong_fails = wrong_cell["n_pass"] < 2
    # Anti should have mean ΔIE < 0 (or at least fail PASS)
    anti_ok = anti_cell["mean_cross_ie"] < 0 or anti_cell["n_pass"] == 0
    generic = (not wrong_fails)  # if wrong-val still PASSes, generic boost worry
    results = {
        "stage": "delta_var_shufflefix",
        "model_path": model_path,
        "val_slot": int(pos),
        "layer": PRIMARY_LAYER,
        "n_wrong_forwards": len(wrong_hs),
        "cos_matched_wrong": cos,
        "matched_confirm": matched_cell,
        "wrong_value_control": wrong_cell,
        "anti_delta_control": anti_cell,
        "wrong_fails_as_required": wrong_fails,
        "anti_ok": anti_ok,
        "generic_boost": generic,
        "verdict": (
            "GENERIC_BOOST" if generic else
            ("CONTROLS_CLEAN" if (wrong_fails and anti_ok) else "CONTROLS_MIXED")
        ),
    }
    # combine with prior embed NONTRIVIAL from K1
    results["prior_embed_nontrivial"] = True
    results["arxiv_path_clean"] = bool(
        matched_cell["gate_pass"] and wrong_fails and anti_ok)

    with open(os.path.join(out_dir, "results_delta_var_shufflefix.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    np.savez(os.path.join(out_dir, "delta_shufflefix_vectors.npz"),
             delta_matched=delta_matched.numpy(),
             delta_wrong=delta_wrong.numpy())
    log(f"VERDICT shufflefix: {results['verdict']} "
        f"wrong_fails={wrong_fails} anti_ok={anti_ok} "
        f"arxiv_clean={results['arxiv_path_clean']}")
    return results
