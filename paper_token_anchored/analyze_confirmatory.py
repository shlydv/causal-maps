#!/usr/bin/env python3
"""Paired row-bootstrap uncertainty for frozen confirmatory effects."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "evidence" / "confirmatory"
OUT = Path(__file__).resolve().parent / "generated" / "confirmatory_summary.json"


def read(name):
    return json.loads((CONFIRM / name).read_text())


def paired_ratio_ci(cell, seed, draws=10000):
    natural = np.asarray(cell.get("natural_effect_rows", cell.get("natural_rows")),
                         dtype=float)
    write = np.asarray(cell.get("add_effect_rows", cell.get("effect_rows")),
                       dtype=float)
    assert natural.shape == write.shape and natural.ndim == 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(natural), size=(draws, len(natural)))
    ratios = write[idx].mean(1) / natural[idx].mean(1)
    return {"estimate": float(write.mean() / natural.mean()),
            "low": float(np.quantile(ratios, .025)),
            "high": float(np.quantile(ratios, .975)),
            "n": int(len(natural)), "draws": draws}


def summarize(result, seed_offset):
    out = {"model_key": result["model_key"], "matrix": {}, "anchor": {}}
    for seed, block in result["per_seed"].items():
        out["matrix"][seed] = {
            name: paired_ratio_ci(cell, seed_offset + int(seed) * 101 + i)
            for i, (name, cell) in enumerate(block["matrix"]["cells"].items())
            if "natural_effect_rows" in cell
        }
    for i, name in enumerate(("belief_ac", "tell_ac")):
        out["anchor"][name] = paired_ratio_ci(
            result["anchor"]["consequences"][name], seed_offset + 1000 + i)
    return out


def main():
    qwen = read("results_delta_preprint_battery_qwen7b_confirm.json")
    mistral = read("results_delta_preprint_battery_mistral7b_confirm.json")
    result = {"method": "paired row bootstrap", "confidence": .95,
              "qwen": summarize(qwen, 1701),
              "mistral": summarize(mistral, 2903)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
