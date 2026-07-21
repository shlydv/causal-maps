"""Kernel 4 — Variable direction transfer across token positions.

Pre-registered: inert prefix only. Both directions mandatory
(p_short→p_long AND p_long→p_short); each must cell-PASS. One direction only
→ position-conditioned; do not average.
"""
import json
import os

import numpy as np
import torch

from . import variable_pairs
from .delta_robust import DONORS, TARGETS, _transfer_cell
from .direction_transfer import (
    PRIMARY_LAYER, _baselines, _effect_with_baseline, _idx_by_template,
    _slot_acts, _subset,
)
from .logutil import log
from .model_utils import load_model_and_tokenizer
from .tensorize import tensorize_pairs

# Inert (task-irrelevant) user-message prefix. Shifts val_slot without changing
# the binding question. Must keep clean/cf equal-length (prefix has no value).
INERT_PREFIX = (
    "Context: this is a practice binder exercise. Ignore this line.\n\n"
)


def _make_pairs(tok, seed, prefix=""):
    """Variable pairs with optional inert prefix on the user content."""
    # Rebuild via the same value grid but with prefixed _chat.
    use_chat = tok is not None
    pairs = []
    for var in variable_pairs._VARS:
        for i, (v0, v1) in enumerate(variable_pairs._VALUE_PAIRS):
            if len(pairs) >= 50:
                break
            pairs.append(_one_prefixed(tok, use_chat, var, v0, v1,
                                       f"{var}{i+1}", prefix))
        if len(pairs) >= 50:
            break
    return pairs


def _one_prefixed(tok, use_chat, var, v0, v1, pid, prefix):
    def _user(val):
        return prefix + variable_pairs._user(var, val)

    if use_chat:
        def _chat(val):
            templated = tok.apply_chat_template(
                [{"role": "user", "content": _user(val)}],
                tokenize=False, add_generation_prompt=True)
            return templated + f"{var} ="
        clean, cf = _chat(v0), _chat(v1)
    else:
        clean = _user(v0) + f"\n{var} ="
        cf = _user(v1) + f"\n{var} ="
    marker = f"Let {var} = "
    off = clean.find(marker) + len(marker)
    return {
        "clean_text": clean, "cf_text": cf,
        "answer_clean": v0, "answer_cf": v1,
        "anchors": {"val_slot": off},
        "template": f"variable_{var}",
        "meta": {"id": pid, "var": var, "val_clean": v0, "val_cf": v1,
                 "prefix": "long" if prefix else "short"},
    }


def _pack(model, tok, prefix, seed):
    pairs = _make_pairs(tok, seed, prefix=prefix)
    batch = tensorize_pairs(tok, pairs, require_anchor_roles=("val_slot",))
    pos = int(batch["anchors"]["val_slot"])
    by_t = _idx_by_template(batch["templates"])
    bases = {t: _baselines(model, _subset(batch, by_t[t])) for t in by_t}
    return batch, pos, by_t, bases


def run_delta_var_crosspos(model_path, out_dir, quantization="8bit",
                          device_map=None, seed=0, layer=PRIMARY_LAYER):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)

    short = _pack(model, tok, "", seed)
    long = _pack(model, tok, INERT_PREFIX, seed)
    b_s, pos_s, by_s, base_s = short
    b_l, pos_l, by_l, base_l = long
    if pos_s == pos_l:
        raise RuntimeError(
            f"inert prefix did not shift val_slot (both={pos_s}); "
            "prefix tokenization failed to lengthen")
    donors = [t for t in DONORS if t in by_s and t in by_l]
    targets = [t for t in TARGETS if t in by_s and t in by_l]
    log(f"delta_var_crosspos: L={layer} pos_short={pos_s} pos_long={pos_l} "
        f"Δpos={pos_l - pos_s} donors={donors} targets={targets}")

    rng = np.random.default_rng(seed)
    results = {
        "stage": "delta_var_crosspos",
        "model_path": model_path,
        "layer": int(layer),
        "pos_short": pos_s,
        "pos_long": pos_l,
        "inert_prefix": INERT_PREFIX,
        "directions": {},
    }

    # Δ from condition A donors @ pos_A; transfer onto condition B targets @ pos_B
    for tag, src, dst, pos_src, pos_dst in (
        ("short_to_long", (b_s, by_s, base_s), (b_l, by_l, base_l), pos_s, pos_l),
        ("long_to_short", (b_l, by_l, base_l), (b_s, by_s, base_s), pos_l, pos_s),
    ):
        b_src, by_src, _ = src
        b_dst, by_dst, base_dst = dst
        donor_idx = [i for t in donors for i in by_src[t]]
        hc, hf = _slot_acts(model, b_src, layer, pos_src)
        delta = (hf[donor_idx] - hc[donor_idx]).mean(0)
        # within on destination condition at destination pos (fair ratio)
        hc_d, hf_d = _slot_acts(model, b_dst, layer, pos_dst)
        within = {}
        for t in targets:
            d = (hf_d[by_dst[t]] - hc_d[by_dst[t]]).mean(0)
            dld, _ = _effect_with_baseline(
                model, base_dst[t], layer, pos_dst, d, scale=1.0)
            within[t] = float(dld.mean())
        cell = _transfer_cell(
            model, base_dst, within, delta, targets,
            layer, pos_dst, 1.0, rng, tag=tag)
        results["directions"][tag] = {
            "pos_src": int(pos_src), "pos_dst": int(pos_dst),
            "||delta||": float(delta.norm()), "within": within, **cell,
        }
        np.savez(os.path.join(out_dir, f"delta_crosspos_{tag}.npz"),
                 delta=delta.numpy())
        log(f"  {tag}: gate_pass={cell['gate_pass']} n_pass={cell['n_pass']}/3 "
            f"mean_ie={cell['mean_cross_ie']:+.3f}")

    s2l = results["directions"]["short_to_long"]
    l2s = results["directions"]["long_to_short"]
    both = bool(s2l["gate_pass"] and l2s["gate_pass"])
    one = bool(s2l["gate_pass"] or l2s["gate_pass"])
    results["pass"] = both
    results["verdict"] = (
        "POSITION_FREE" if both else
        ("POSITION_CONDITIONED" if one else "NO_CROSSPOS_TRANSFER"))

    with open(os.path.join(out_dir, "results_delta_var_crosspos.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT delta_var_crosspos: {results['verdict']} "
        f"s2l={s2l['gate_pass']} l2s={l2s['gate_pass']}")
    return results
