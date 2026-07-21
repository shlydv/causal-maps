"""Kernel 3 — Variable direction transfer at Qwen2.5-1.5B-Instruct.

Pre-registered: L=2 exactly (28 layers; not a formula). Same donors/targets/
cell PASS as 7B Variable. Secondaries L∈{1,3} descriptive only.
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

SECONDARY_LAYERS = (1, 3)


def run_delta_var_1p5b(model_path, out_dir, quantization=None, device_map=None,
                      seed=0):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    n_layers = len(model.model.layers)
    pairs = variable_pairs.make_variable_pairs(50, seed=seed, tok=tok)
    batch = tensorize_pairs(tok, pairs, require_anchor_roles=("val_slot",))
    pos = int(batch["anchors"]["val_slot"])
    by_t = _idx_by_template(batch["templates"])
    donors = [t for t in DONORS if t in by_t]
    targets = [t for t in TARGETS if t in by_t]
    donor_idx = [i for t in donors for i in by_t[t]]
    log(f"delta_var_1p5b: n_layers={n_layers} val_slot={pos} "
        f"donors={donors} targets={targets}")

    bases = {t: _baselines(model, _subset(batch, by_t[t])) for t in list(by_t)}
    rng = np.random.default_rng(seed)
    results = {
        "stage": "delta_var_1p5b",
        "model_path": model_path,
        "n_layers": n_layers,
        "val_slot": pos,
        "templates": {k: len(v) for k, v in by_t.items()},
        "layers": {},
    }

    for L in (PRIMARY_LAYER,) + SECONDARY_LAYERS:
        hc, hf = _slot_acts(model, batch, L, pos)
        within = {}
        for t in targets:
            d = (hf[by_t[t]] - hc[by_t[t]]).mean(0)
            dld, _ = _effect_with_baseline(model, bases[t], L, pos, d, scale=1.0)
            within[t] = float(dld.mean())
        delta = (hf[donor_idx] - hc[donor_idx]).mean(0)
        cell = _transfer_cell(
            model, bases, within, delta, targets, L, pos, 1.0, rng, tag=f"L{L}")
        results["layers"][f"L{L}"] = {
            "layer": L, "||delta||": float(delta.norm()), "within": within, **cell}
        np.savez(os.path.join(out_dir, f"delta_1p5b_L{L}.npz"), delta=delta.numpy())

    primary = results["layers"][f"L{PRIMARY_LAYER}"]
    results["pass"] = bool(primary["gate_pass"])
    results["verdict"] = (
        "SCALE_TRANSFER_OK" if primary["gate_pass"] else "SCALE_TRANSFER_FAIL")

    with open(os.path.join(out_dir, "results_delta_var_1p5b.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT delta_var_1p5b: {results['verdict']} "
        f"L2_pass={primary['gate_pass']} n_pass={primary['n_pass']}/3 "
        f"mean_ie={primary['mean_cross_ie']:+.3f}")
    return results
