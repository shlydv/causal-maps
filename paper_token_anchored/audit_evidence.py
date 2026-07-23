#!/usr/bin/env python3
"""Deterministic integrity audit for the frozen Paper 1 evidence package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "evidence" / "confirmatory"
OUT = Path(__file__).resolve().parent / "generated" / "evidence_manifest.json"
RUNTIME = Path(__file__).resolve().parent / "generated" / "runtime_provenance.json"

ARTIFACTS = {
    "qwen7": "results_delta_preprint_battery_qwen7b_confirm.json",
    "mistral7": "results_delta_preprint_battery_mistral7b_confirm.json",
    "qwen14": "results_delta_preprint_battery_qwen14b_headline.json",
    "deepseek_llama8":
        "results_delta_preprint_battery_deepseek_r1_llama8b_confirm.json",
    "gemma3_12": "results_delta_preprint_battery_gemma3_12b_confirm.json",
    "qwen14_locus": "results_delta_preprint_locus_qwen14b_locus_v1.json",
    "qwen14_closeout": "results_delta_paper1_closeout_qwen14b_closeout_v1.json",
}


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _row_key(row):
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


def _require_rows(rows, expected, label):
    assert len(rows) == expected, f"{label}: expected {expected} rows, got {len(rows)}"
    assert len({_row_key(row) for row in rows}) == expected, f"{label}: duplicate rows"


def _audit_battery(key, result):
    assert result["stage"] == "delta_preprint_battery"
    selected = result["structured_world_selection"]["selected"]
    assert result["anchor"]["n_rows"] == selected
    assert len(result["anchor"]["rows"]) == selected
    assert len(result["anchor"]["consequences"]["belief_ac"]["effect_rows"]) == selected
    checkpoint = result["checkpoint"]
    assert checkpoint["g0_clean"] >= .8 and checkpoint["g0_natural"] >= .8
    assert set(checkpoint["layers"]) == {
        int(layer) for layer in checkpoint["per_site"]["checkpoint"]}
    exclusions = []
    audited_cells = 0
    for seed, block in result["per_seed"].items():
        if "matrix" in block:
            expected = int(block["matrix"]["n_rows"])
            for name, cell in block["matrix"]["cells"].items():
                audited_cells += 1
                _require_rows(cell["rows"], expected,
                              f"{key}/s{seed}/matrix/{name}")
                if cell["verdict"] == "INELICITABLE":
                    exclusions.append({"seed": seed, "family": "matrix",
                                       "cell": name, "reason": "behavioral_gate"})
                elif "natural_effect_rows" in cell:
                    assert len(cell["natural_effect_rows"]) == expected
                    assert len(cell["add_effect_rows"]) == expected
        if "entity" in block:
            for family, family_block in block["entity"]["families"].items():
                expected = len(family_block["rows"])
                _require_rows(family_block["rows"], expected,
                              f"{key}/s{seed}/entity/{family}")
                for name, cell in family_block["cells"].items():
                    audited_cells += 1
                    if cell["verdict"] in ("INELICITABLE", "PENDING_TWOHOP_SCALE"):
                        exclusions.append({"seed": seed, "family": family,
                                           "cell": name,
                                           "reason": "behavioral_gate_or_dependency"})
                    if "natural_effect_rows" in cell:
                        assert len(cell["natural_effect_rows"]) == expected
                        assert len(cell["add_effect_rows"]) == expected
    return {"selected_worlds": selected,
            "checkpoint_layers": checkpoint["layers"],
            "checkpoint_verdict": checkpoint["verdict"],
            "anchor_verdict": result["anchor"]["verdict"],
            "audited_cells": audited_cells, "exclusions": exclusions}


def _audit_locus(result):
    assert result["stage"] == "delta_preprint_locus"
    assert result["world_selection"]["selected"] == 30
    assert result["verdict"] == "SOURCE_ANCHORS_SUFFICIENT"
    assert result["random_controls_sufficient"] == 0
    sufficient_layers = []
    for layer, block in result["per_layer"].items():
        exact = block["full_matched_prefix"]
        assert abs(exact["forward_ratio"] - 1) < 1e-6
        assert abs(exact["reverse_ratio"] - 1) < 1e-6
        assert exact["forward_target_acc"] == 1
        assert exact["reverse_clean_acc"] == 1
        if block["source_anchors"]["sufficient"]:
            sufficient_layers.append(int(layer))
        for name, cell in block.items():
            assert len(cell["forward_effect_rows"]) == 30, (layer, name)
            assert len(cell["reverse_effect_rows"]) == 30, (layer, name)
    return {"verdict": result["verdict"],
            "sufficient_layers": sufficient_layers,
            "random_controls_sufficient": result["random_controls_sufficient"]}


def _audit_closeout(result):
    assert result["stage"] == "delta_paper1_closeout"
    assert result["protocol_version"] == "2026-07-22-v1"
    exact = result["exact_ac_only"]
    naturalized = result["naturalized"]
    assert exact["n_rows"] == 30
    assert naturalized["n_rows"] == 30
    assert sum(naturalized["styles"].values()) == 30
    for block in exact["per_layer"].values():
        assert len(block["natural_effect_rows"]) == 30
    for query in ("belief_ac", "tell_ac"):
        assert len(naturalized["consequences"][query]["effect_rows"]) == 30
    for block in naturalized["trajectory"]["per_layer"].values():
        for site in ("checkpoint", "readout"):
            assert len(block[site]["natural_effect_rows"]) == 30
    return {
        "verdict": result["verdict"],
        "exact_ac_verdict": exact["verdict"],
        "naturalized_verdict": naturalized["verdict"],
        "naturalized_null_p": naturalized["null"]["p"],
        "naturalized_checkpoint_max":
            naturalized["trajectory"]["max_abs_checkpoint_ratio"],
        "naturalized_readout_layers":
            naturalized["trajectory"]["readout_sufficient_layers"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-closeout", action="store_true")
    args = parser.parse_args()
    manifest = {"schema": "paper1-evidence-audit-v1", "artifacts": {},
                "checks": {}, "totals": {"bytes": 0, "exclusions": 0}}
    runtime = _read(RUNTIME)
    assert runtime["hardware_guard"] == ["Tesla T4", "Tesla T4"]
    assert runtime["awq_runtime"]["version"] == "7.1.0"
    assert runtime["model_source"].endswith("/14b-instruct-awq/1")
    manifest["runtime"] = runtime
    for key, filename in ARTIFACTS.items():
        path = CONFIRM / filename
        if not path.exists():
            if key == "qwen14_closeout" and not args.require_closeout:
                manifest["artifacts"][key] = {"filename": filename,
                                               "status": "pending"}
                continue
            raise FileNotFoundError(path)
        result = _read(path)
        manifest["artifacts"][key] = {
            "filename": filename, "status": "present",
            "bytes": path.stat().st_size, "sha256": _sha256(path),
            "stage": result.get("stage"), "model_key": result.get("model_key"),
            "verdict": result.get("verdict", result.get("summary")),
        }
        manifest["totals"]["bytes"] += path.stat().st_size
        if result["stage"] == "delta_preprint_battery":
            check = _audit_battery(key, result)
        elif result["stage"] == "delta_preprint_locus":
            check = _audit_locus(result)
        elif result["stage"] == "delta_paper1_closeout":
            check = _audit_closeout(result)
        else:
            raise AssertionError(f"unrecognized stage: {result['stage']}")
        manifest["checks"][key] = check
        manifest["totals"]["exclusions"] += len(check.get("exclusions", []))
    if args.require_closeout:
        closeout_verdict = manifest["checks"]["qwen14_closeout"]["verdict"]
        assert closeout_verdict in {
            "PAPER1_EXPERIMENTS_CLOSED", "PAPER1_CLOSEOUT_BOUNDARY",
        }, f"closeout was not adjudicated: {closeout_verdict}"
        manifest["status"] = (
            "PAPER1_EVIDENCE_FROZEN" if
            closeout_verdict == "PAPER1_EXPERIMENTS_CLOSED" else
            "PAPER1_EVIDENCE_FROZEN_WITH_BOUNDARY")
    else:
        manifest["status"] = "AUDIT_PASS_CLOSEOUT_PENDING"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{manifest['status']}: wrote {OUT}")


if __name__ == "__main__":
    main()
