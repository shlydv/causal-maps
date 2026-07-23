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


def summarize_locus(result, seed_offset):
    """Paired uncertainty for the prespecified, paper-facing locus arms."""
    keep = ("marker_only", "summary_span", "source_anchors",
            "anchors_without_ac", "full_prequery", "full_matched_prefix")
    out = {"model_key": result["model_key"],
           "verdict": result["verdict"], "per_layer": {}}
    natural = result["natural_effect_rows"]
    for layer_i, (layer, block) in enumerate(result["per_layer"].items()):
        out["per_layer"][layer] = {}
        for locus_i, name in enumerate(keep):
            cell = block[name]
            base_seed = seed_offset + 1000 * layer_i + 10 * locus_i
            out["per_layer"][layer][name] = {
                "forward": paired_ratio_ci(
                    {"natural_effect_rows": natural,
                     "effect_rows": cell["forward_effect_rows"]}, base_seed),
                "reverse": paired_ratio_ci(
                    {"natural_effect_rows": natural,
                     "effect_rows": cell["reverse_effect_rows"]}, base_seed + 1),
                "forward_target_acc": cell["forward_target_acc"],
                "reverse_clean_acc": cell["reverse_clean_acc"],
                "sufficient": cell["sufficient"],
            }
    return out


def summarize_closeout(result, seed_offset):
    out = {"model_key": result["model_key"],
           "verdict": result["verdict"], "exact_ac_only": {},
           "naturalized": {"consequences": {}, "trajectory": {}}}
    exact = result["exact_ac_only"]
    for layer_i, (layer, cell) in enumerate(exact["per_layer"].items()):
        out["exact_ac_only"][layer] = {
            "forward": paired_ratio_ci(
                {"natural_effect_rows": cell["natural_effect_rows"],
                 "effect_rows": cell["forward_effect_rows"]},
                seed_offset + 100 * layer_i),
            "reverse": paired_ratio_ci(
                {"natural_effect_rows": cell["natural_effect_rows"],
                 "effect_rows": cell["reverse_effect_rows"]},
                seed_offset + 100 * layer_i + 1),
            "forward_target_acc": cell["forward_target_acc"],
            "reverse_clean_acc": cell["reverse_clean_acc"],
            "sufficient": cell["sufficient"],
        }
    naturalized = result["naturalized"]
    for query_i, query in enumerate(("belief_ac", "tell_ac")):
        out["naturalized"]["consequences"][query] = paired_ratio_ci(
            naturalized["consequences"][query],
            seed_offset + 2000 + query_i)
    for layer_i, (layer, block) in enumerate(
            naturalized["trajectory"]["per_layer"].items()):
        out["naturalized"]["trajectory"][layer] = {}
        for site_i, site in enumerate(("checkpoint", "readout")):
            cell = block[site]
            out["naturalized"]["trajectory"][layer][site] = {
                "forward": paired_ratio_ci(
                    {"natural_effect_rows": cell["natural_effect_rows"],
                     "effect_rows": cell["forward_effect_rows"]},
                    seed_offset + 3000 + 100 * layer_i + 10 * site_i),
                "reverse": paired_ratio_ci(
                    {"natural_effect_rows": cell["natural_effect_rows"],
                     "effect_rows": cell["reverse_effect_rows"]},
                    seed_offset + 3000 + 100 * layer_i + 10 * site_i + 1),
                "forward_target_acc": cell["forward_target_acc"],
                "reverse_clean_acc": cell["reverse_clean_acc"],
                "sufficient": cell["sufficient"],
            }
    return out


def main():
    qwen = read("results_delta_preprint_battery_qwen7b_confirm.json")
    mistral = read("results_delta_preprint_battery_mistral7b_confirm.json")
    qwen14 = read("results_delta_preprint_battery_qwen14b_headline.json")
    deepseek = read("results_delta_preprint_battery_deepseek_r1_llama8b_confirm.json")
    gemma = read("results_delta_preprint_battery_gemma3_12b_confirm.json")
    locus = read("results_delta_preprint_locus_qwen14b_locus_v1.json")
    closeout_path = CONFIRM / "results_delta_paper1_closeout_qwen14b_closeout_v1.json"
    reverse = {}
    for readout, block in qwen14["reverse_base"]["readouts"].items():
        reverse[readout] = {}
        for i, condition in enumerate(("history_reverse", "verbal_reverse",
                                       "both_reverse")):
            reverse[readout][condition] = paired_ratio_ci(
                {"natural_effect_rows": block["natural_effect_rows"],
                 "effect_rows": block[condition]["effect_rows"]},
                4301 + 100 * i + len(reverse))
    result = {"method": "paired row bootstrap", "confidence": .95,
              "qwen": summarize(qwen, 1701),
              "mistral": summarize(mistral, 2903),
              "qwen14": summarize(qwen14, 3701),
              "deepseek_llama": summarize(deepseek, 5101),
              "gemma3": summarize(gemma, 6301),
              "qwen14_locus": summarize_locus(locus, 7501),
              "qwen14_reverse_base": reverse}
    if closeout_path.exists():
        result["qwen14_closeout"] = summarize_closeout(
            json.loads(closeout_path.read_text()), 8903)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
