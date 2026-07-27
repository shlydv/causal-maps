"""Kill test for an intervention-specific DeepSeek binding divergence.

The structured belief world is behaviorally address-ineligible under its
textual counterfactual, so it cannot support a natural-minus-synthetic
mechanism claim.  This module removes belief inference while retaining the
same-valued collision structure and asks whether a synthetic content write
breaks otherwise-correct direct binding.
"""
from __future__ import annotations

import json
import os

import torch

from .delta_anchor_write import LAYER, _neutral_states, _resolve
from .delta_preprint_battery import _last_overlap_token
from .delta_structured_workspace import (
    LOCATIONS,
    _accuracy,
    _common_prefix,
    _switch_metrics,
    _switch_pass,
)
from .delta_trajectory import _forward, _ld
from .logutil import log
from .model_utils import input_device, load_model_and_tokenizer

PROTOCOL_VERSION = "2026-07-23-p2-direct-binding-diagnostic-v1"
PROTOCOL_SHA256 = (
    "56D547DB5B6727059669E795352B4E3EB61B2E3FB6A8B3BC2F76BD9A097953AC")
G_ACC = 0.80
FIELDS = ("ac", "bc", "as", "bs")
QUERY = {
    "ac": "Alice/cube",
    "bc": "Bob/cube",
    "as": "Alice/sphere",
    "bs": "Bob/sphere",
}


def _direct_rows(n_rows=30):
    pairs = [
        (source, target)
        for source in LOCATIONS
        for target in LOCATIONS
        if source != target
    ]
    if not 1 <= int(n_rows) <= len(pairs):
        raise ValueError(f"n_rows must be in [1,{len(pairs)}]")
    rows = []
    for source, target in pairs[:int(n_rows)]:
        rows.append({
            "source": source,
            "target": target,
            "ac": source,
            "bc": source,
            "as": source,
            "bs": target,
        })
    return rows


def _changed(rows, field):
    return [
        {**row, field: row["target"]}
        for row in rows
    ]


def _user(row, field):
    return (
        "Use this direct memory ledger exactly as written.\n"
        f"Alice/cube -> {row['ac']}\n"
        f"Bob/cube -> {row['bc']}\n"
        f"Alice/sphere -> {row['as']}\n"
        f"Bob/sphere -> {row['bs']}\n"
        f"Lookup key: {QUERY[field]}\n"
        "Reply with exactly VALUE, one space, and the stored location. "
        "Do not infer or alter any record."
    )


def _render(tok, row, field):
    return tok.apply_chat_template(
        [{"role": "user", "content": _user(row, field)}],
        tokenize=False, add_generation_prompt=True)


def _batch(tok, rows, field, dev):
    texts = [_render(tok, row, field) for row in rows]
    prefixes, maps = [], []
    for text in texts:
        base = tok.encode(text, add_special_tokens=False)
        continuations = {}
        for location in LOCATIONS:
            full = tok.encode(
                text + f"VALUE {location}", add_special_tokens=False)
            if full[:len(base)] != base:
                raise ValueError("direct-binding answer contract resegmented")
            continuations[location] = full[len(base):]
        common = _common_prefix(list(continuations.values()))
        amap = {}
        for location, continuation in continuations.items():
            if len(continuation) <= len(common):
                raise ValueError("direct-binding answer did not diverge")
            amap[location] = continuation[len(common)]
        if len(set(amap.values())) != len(LOCATIONS):
            raise ValueError("direct-binding answer tokens collide")
        prefixes.append(base + common)
        maps.append(amap)
    if len({len(prefix) for prefix in prefixes}) != 1:
        raise ValueError("direct-binding batch is not length-aligned")
    if any(amap != maps[0] for amap in maps[1:]):
        raise ValueError("direct-binding answer map varies by row")
    ids = torch.tensor(prefixes, dtype=torch.long, device=dev)
    return {
        "texts": texts,
        "ids": ids,
        "am": torch.ones_like(ids),
        "amap": maps[0],
    }


def _anchor_positions(tok, batch, rows):
    positions = {}
    for field in FIELDS:
        needle_key = QUERY[field]
        per_row = [
            _last_overlap_token(
                tok, text, f"{needle_key} -> {row[field]}")
            for text, row in zip(batch["texts"], rows)
        ]
        if len(set(per_row)) != 1:
            raise ValueError(f"{field} address position varies: {per_row}")
        positions[field] = per_row[0]
    if len(set(positions.values())) != len(positions):
        raise ValueError(f"direct-binding positions collide: {positions}")
    return positions


def _endpoint(logits, batch, expected, rival):
    expected_ids = torch.tensor([batch["amap"][x] for x in expected])
    rival_ids = torch.tensor([batch["amap"][x] for x in rival])
    margin = _ld(logits, expected_ids, rival_ids)
    return {
        "accuracy": float(_accuracy(logits, batch, expected)),
        "margin_mean": float(margin.mean()),
        "margin_rows": margin.tolist(),
    }


def _behavior(model, batches, rows, natural_rows):
    out = {}
    for field in FIELDS:
        clean_batch, natural_batch = batches[field]
        clean_logits, _ = _forward(
            model, clean_batch["ids"], clean_batch["am"], ())
        natural_logits, _ = _forward(
            model, natural_batch["ids"], natural_batch["am"], ())
        clean_expected = [row[field] for row in rows]
        natural_expected = [row[field] for row in natural_rows]
        clean_rival = [
            row["target"] if row[field] == row["source"] else row["source"]
            for row in rows
        ]
        natural_rival = [
            row["source"] if row[field] == row["target"] else row["target"]
            for row in natural_rows
        ]
        out[field] = {
            "clean": _endpoint(
                clean_logits, clean_batch, clean_expected, clean_rival),
            "natural": _endpoint(
                natural_logits, natural_batch,
                natural_expected, natural_rival),
        }
    return out


def _verdict(eligible, intended, invariants, wrong_address):
    if not eligible:
        return "BEHAVIORALLY_INELIGIBLE"
    if not intended:
        return "SYNTHETIC_CONTENT_WRITE_FAILED"
    if not invariants:
        return "INTERVENTION_SPECIFIC_DIVERGENCE"
    if wrong_address:
        return "DIRECT_BINDING_SPECIFIC"
    return "WRONG_ADDRESS_CONTROL_FAILED"


@torch.no_grad()
def run_delta_binding_divergence(
        model_path, out_dir, model_key="deepseek_direct_binding_diagnostic",
        quantization="8bit", device_map=None, max_memory=None, n_rows=30):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    rows = _direct_rows(n_rows)
    natural_rows = _changed(rows, "ac")
    bc_natural_rows = _changed(rows, "bc")
    batches = {
        field: (
            _batch(tok, rows, field, dev),
            _batch(tok, natural_rows, field, dev),
        )
        for field in FIELDS
    }
    base_positions = _anchor_positions(
        tok, batches["ac"][0], rows)
    for field in FIELDS:
        clean_batch, natural_batch = batches[field]
        if _anchor_positions(tok, clean_batch, rows) != base_positions:
            raise ValueError("clean address positions vary by query")
        if _anchor_positions(
                tok, natural_batch, natural_rows) != base_positions:
            raise ValueError("natural address positions vary by query")

    states = _neutral_states(
        model, tok, dev, LAYER, LOCATIONS)
    deltas = torch.stack([
        states[row["target"]] - states[row["source"]]
        for row in rows
    ])
    behavior = _behavior(model, batches, rows, natural_rows)
    eligible = bool(min(
        behavior[field][condition]["accuracy"]
        for field in FIELDS
        for condition in ("clean", "natural")) >= G_ACC)
    log("DIRECT BINDING G0 "
        + " ".join(
            f"{field}={behavior[field]['clean']['accuracy']:.0%}/"
            f"{behavior[field]['natural']['accuracy']:.0%}"
            for field in FIELDS)
        + f" eligible={eligible}")

    synthetic = {}
    intended_metrics = None
    for field in FIELDS:
        clean_batch, natural_batch = batches[field]
        clean_logits, _ = _forward(
            model, clean_batch["ids"], clean_batch["am"], ())
        natural_logits, _ = _forward(
            model, natural_batch["ids"], natural_batch["am"], ())
        add_logits, _ = _forward(
            model, clean_batch["ids"], clean_batch["am"], (),
            add=(LAYER, base_positions["ac"], deltas))
        if field == "ac":
            source = [row["source"] for row in rows]
            target = [row["target"] for row in rows]
            intended_metrics = _switch_metrics(
                clean_logits, natural_logits, add_logits,
                clean_batch, source, target)
            intended_metrics["pass"] = _switch_pass(intended_metrics)
            synthetic[field] = intended_metrics
        else:
            expected = [row[field] for row in rows]
            rival = [
                row["target"] if row[field] == row["source"]
                else row["source"]
                for row in rows
            ]
            synthetic[field] = _endpoint(
                add_logits, clean_batch, expected, rival)

    # Same write at Bob/cube must edit Bob/cube and preserve Alice/cube.
    bc_clean = _batch(tok, rows, "bc", dev)
    bc_natural = _batch(tok, bc_natural_rows, "bc", dev)
    bc_clean_logits, _ = _forward(
        model, bc_clean["ids"], bc_clean["am"], ())
    bc_natural_logits, _ = _forward(
        model, bc_natural["ids"], bc_natural["am"], ())
    bc_add_logits, _ = _forward(
        model, bc_clean["ids"], bc_clean["am"], (),
        add=(LAYER, base_positions["bc"], deltas))
    source = [row["source"] for row in rows]
    target = [row["target"] for row in rows]
    wrong_own = _switch_metrics(
        bc_clean_logits, bc_natural_logits, bc_add_logits,
        bc_clean, source, target)
    wrong_own["pass"] = _switch_pass(wrong_own)

    ac_clean = batches["ac"][0]
    ac_wrong_logits, _ = _forward(
        model, ac_clean["ids"], ac_clean["am"], (),
        add=(LAYER, base_positions["bc"], deltas))
    ac_preserved = float(_accuracy(ac_wrong_logits, ac_clean, source))
    wrong_address_pass = bool(
        wrong_own["pass"] and ac_preserved >= G_ACC)

    invariants_pass = bool(min(
        synthetic[field]["accuracy"]
        for field in ("bc", "as", "bs")) >= G_ACC)
    intended_pass = bool(intended_metrics["pass"])
    verdict = _verdict(
        eligible, intended_pass, invariants_pass, wrong_address_pass)
    result = {
        "stage": "delta_binding_divergence",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "n_rows": len(rows),
        "layer": LAYER,
        "rows": rows,
        "positions": base_positions,
        "behavior": behavior,
        "eligible": eligible,
        "synthetic": synthetic,
        "wrong_address": {
            "own": wrong_own,
            "alice_cube_preserved_acc": ac_preserved,
            "pass": wrong_address_pass,
        },
        "gates": {
            "intended": intended_pass,
            "invariants": invariants_pass,
            "wrong_address": wrong_address_pass,
        },
        "verdict": verdict,
    }
    path = os.path.join(
        out_dir, f"results_delta_binding_divergence_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(
        f"DIRECT BINDING synthetic ac={synthetic['ac']['target_acc']:.0%} "
        f"bc={synthetic['bc']['accuracy']:.0%} "
        f"as={synthetic['as']['accuracy']:.0%} "
        f"bs={synthetic['bs']['accuracy']:.0%} "
        f"wrong={wrong_address_pass} verdict={verdict}")
    return result
