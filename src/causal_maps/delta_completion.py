"""Kernel 2 — Completion-state direction transfer.

Pre-registered (CAUSAL_MAPS_LOG FULL SEND):
  Primary slot = 62 (P1 peak, not bit_slot@61). L* = 2.
  Secondary = bit_slot (descriptive). Surfaces A,B → C,D; α=1.
  Shared cell PASS: ΔIE>0 AND p<0.01 vs 100 random same-norm dirs AND
  cross/within ≥ 0.5. Primary gate = all targets cell-PASS at pos62.
"""
import json
import os

import numpy as np
import torch

from . import completion_pairs
from .delta_robust import N_NULL, _cell_pass, _transfer_cell
from .direction_transfer import (
    PRIMARY_LAYER, _baselines, _effect_with_baseline, _idx_by_template,
    _slot_acts, _subset,
)
from .logutil import log
from .model_utils import load_model_and_tokenizer
from .tensorize import tensorize_pairs

DONORS = ("completion_explicit_A", "completion_explicit_B")
TARGETS = ("completion_explicit_C", "completion_explicit_D")
PRIMARY_POS = 62  # frozen from P1 peak
FAMILIES = ("explicit_A", "explicit_B", "explicit_C", "explicit_D")


def run_delta_completion(model_path, out_dir, quantization="8bit", device_map=None,
                         seed=0, layer=PRIMARY_LAYER):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    pairs = completion_pairs.make_completion_pairs(
        40, seed=seed, tok=tok, families=list(FAMILIES))
    batch = tensorize_pairs(tok, pairs, require_anchor_roles=("bit_slot",))
    bit_pos = int(batch["anchors"]["bit_slot"])
    S = int(batch["S"])
    if PRIMARY_POS >= S:
        raise RuntimeError(f"PRIMARY_POS={PRIMARY_POS} >= S={S}")
    by_t = _idx_by_template(batch["templates"])
    donors = [t for t in DONORS if t in by_t]
    targets = [t for t in TARGETS if t in by_t]
    if len(donors) < 2 or len(targets) < 2:
        raise RuntimeError(f"missing templates: by_t={ {k: len(v) for k,v in by_t.items()} }")
    donor_idx = [i for t in donors for i in by_t[t]]
    log(f"delta_completion: L={layer} primary_pos={PRIMARY_POS} bit_slot={bit_pos} "
        f"S={S} donors={donors} targets={targets} "
        f"counts={ {k: len(v) for k, v in by_t.items()} }")

    bases = {t: _baselines(model, _subset(batch, by_t[t])) for t in list(by_t)}
    rng = np.random.default_rng(seed)

    results = {
        "stage": "delta_completion",
        "model_path": model_path,
        "layer": int(layer),
        "primary_pos": PRIMARY_POS,
        "bit_slot": bit_pos,
        "S": S,
        "templates": {k: len(v) for k, v in by_t.items()},
        "slots": {},
    }

    for slot_name, pos in (("primary_62", PRIMARY_POS), ("bit_slot", bit_pos)):
        hc, hf = _slot_acts(model, batch, layer, pos)
        within = {}
        for t in targets + donors:
            d = (hf[by_t[t]] - hc[by_t[t]]).mean(0)
            dld, flip = _effect_with_baseline(
                model, bases[t], layer, pos, d, scale=1.0)
            within[t] = float(dld.mean())
            log(f"  {slot_name} WITHIN {t}: ΔIE={dld.mean():+.3f} flip={flip.mean():.0%}")
        within_tgt = {t: within[t] for t in targets}
        delta = (hf[donor_idx] - hc[donor_idx]).mean(0)
        cell = _transfer_cell(
            model, bases, within_tgt, delta, targets,
            layer, pos, 1.0, rng, tag=f"{slot_name}")
        results["slots"][slot_name] = {
            "pos": int(pos),
            "||delta||": float(delta.norm()),
            "within": within,
            **cell,
        }
        np.savez(os.path.join(out_dir, f"delta_completion_{slot_name}.npz"),
                 delta=delta.numpy())

    primary = results["slots"]["primary_62"]
    results["pass"] = bool(primary["gate_pass"])
    results["verdict"] = (
        "COMPLETION_DIRECTION_REUSABLE" if primary["gate_pass"]
        else "COMPLETION_NO_TRANSFER_OR_WEAK")
    # secondary note only
    bit = results["slots"]["bit_slot"]
    results["bit_slot_also_passes"] = bool(bit["gate_pass"])

    with open(os.path.join(out_dir, "results_delta_completion.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT delta_completion: {results['verdict']} "
        f"primary_pass={primary['gate_pass']} n_pass={primary['n_pass']}/2 "
        f"mean_ie={primary['mean_cross_ie']:+.3f} ratio={primary['mean_ratio']:.2f} "
        f"bit_also={bit['gate_pass']}")
    return results
