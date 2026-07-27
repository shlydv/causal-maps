"""Collision-load pilot for content-equivalence aliasing in direct registers."""
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
from .logutil import Heartbeat, log
from .model_utils import input_device, load_model_and_tokenizer

PROTOCOL_VERSION = "2026-07-24-p2-content-aliasing-v1"
PROTOCOL_SHA256 = (
    "F1183DCD1DC03F99D10A7C419261746F594ED8D86F2D9C44BB08DC32019BAB39")
G_ACC = 0.80
REGISTERS = ("A", "B", "C", "D")


def _collision_rows(load, n_rows=30):
    """Rows with `load` source-valued registers and D fixed to target."""
    if int(load) not in (1, 2, 3):
        raise ValueError("collision load must be 1, 2, or 3")
    pairs = [
        (source, target)
        for source in LOCATIONS
        for target in LOCATIONS
        if source != target
    ]
    if not 1 <= int(n_rows) <= len(pairs):
        raise ValueError(f"n_rows must be in [1,{len(pairs)}]")
    rows = []
    for i, (source, target) in enumerate(pairs[:int(n_rows)]):
        remaining = [
            value for value in LOCATIONS
            if value not in (source, target)
        ]
        filler_1 = remaining[i % len(remaining)]
        filler_2 = remaining[(i + 1) % len(remaining)]
        if filler_2 == filler_1:
            filler_2 = remaining[(i + 2) % len(remaining)]
        row = {
            "source": source,
            "target": target,
            "load": int(load),
            "A": source,
            "D": target,
        }
        if int(load) == 1:
            row.update(B=filler_1, C=filler_2)
        elif int(load) == 2:
            row.update(B=source, C=filler_1)
        else:
            row.update(B=source, C=source)
        rows.append(row)
    return rows


def _natural_rows(rows):
    return [{**row, "A": row["target"]} for row in rows]


def _user(row, register):
    records = "\n".join(
        f"[{name}] = {row[name]}" for name in REGISTERS)
    return (
        "Read the requested memory register literally.\n"
        f"{records}\n"
        f"Requested register: [{register}]\n"
        "Reply with exactly VALUE, one space, and that register's value. "
        "Do not combine or update registers."
    )


def _render(tok, row, register):
    return tok.apply_chat_template(
        [{"role": "user", "content": _user(row, register)}],
        tokenize=False, add_generation_prompt=True)


def _batch(tok, rows, register, dev):
    texts = [_render(tok, row, register) for row in rows]
    prefixes, maps = [], []
    for text in texts:
        base = tok.encode(text, add_special_tokens=False)
        continuations = {}
        for location in LOCATIONS:
            full = tok.encode(
                text + f"VALUE {location}", add_special_tokens=False)
            if full[:len(base)] != base:
                raise ValueError("aliasing answer contract resegmented")
            continuations[location] = full[len(base):]
        common = _common_prefix(list(continuations.values()))
        amap = {}
        for location, continuation in continuations.items():
            if len(continuation) <= len(common):
                raise ValueError("aliasing answer did not diverge")
            amap[location] = continuation[len(common)]
        if len(set(amap.values())) != len(LOCATIONS):
            raise ValueError("aliasing answer tokens collide")
        prefixes.append(base + common)
        maps.append(amap)
    if len({len(prefix) for prefix in prefixes}) != 1:
        raise ValueError("aliasing batch is not length-aligned")
    if any(amap != maps[0] for amap in maps[1:]):
        raise ValueError("aliasing answer map varies by row")
    ids = torch.tensor(prefixes, dtype=torch.long, device=dev)
    return {
        "texts": texts,
        "ids": ids,
        "am": torch.ones_like(ids),
        "amap": maps[0],
    }


def _positions(tok, batch, rows):
    out = {}
    for register in REGISTERS:
        per_row = [
            _last_overlap_token(
                tok, text, f"[{register}] = {row[register]}")
            for text, row in zip(batch["texts"], rows)
        ]
        if len(set(per_row)) != 1:
            raise ValueError(
                f"register {register} position varies: {per_row}")
        out[register] = per_row[0]
    if len(set(out.values())) != len(out):
        raise ValueError(f"register positions collide: {out}")
    return out


def _endpoint(logits, batch, expected, rival):
    expected_ids = torch.tensor([batch["amap"][x] for x in expected])
    rival_ids = torch.tensor([batch["amap"][x] for x in rival])
    margin = _ld(logits, expected_ids, rival_ids)
    return {
        "accuracy": float(_accuracy(logits, batch, expected)),
        "margin_mean": float(margin.mean()),
        "margin_rows": margin.tolist(),
    }


def _pattern_verdict(eligible, intended, blocks):
    if not eligible:
        return "BEHAVIORALLY_INELIGIBLE"
    if not intended:
        return "SYNTHETIC_CONTENT_WRITE_FAILED"

    k1 = blocks["1"]["synthetic"]
    k2 = blocks["2"]["synthetic"]
    k3 = blocks["3"]["synthetic"]
    exact_alias = all((
        k1["B"]["address"]["accuracy"] >= G_ACC,
        k1["C"]["address"]["accuracy"] >= G_ACC,
        k2["B"]["alias"]["accuracy"] >= G_ACC,
        k2["C"]["address"]["accuracy"] >= G_ACC,
        k3["B"]["alias"]["accuracy"] >= G_ACC,
        k3["C"]["alias"]["accuracy"] >= G_ACC,
        *(blocks[str(load)]["synthetic"]["D"]["address"]["accuracy"] >= G_ACC
          for load in (1, 2, 3)),
    ))
    if exact_alias:
        natural_alias = all((
            blocks["2"]["natural_alias"]["B"] >= G_ACC,
            blocks["3"]["natural_alias"]["B"] >= G_ACC,
            blocks["3"]["natural_alias"]["C"] >= G_ACC,
        ))
        subtype = (
            "BEHAVIORAL_AND_CAUSAL"
            if natural_alias else "INTERVENTION_SPECIFIC")
        return f"CONTENT_EQUIVALENCE_ALIASING_{subtype}"

    shared_preserved = all((
        k2["B"]["address"]["accuracy"] >= G_ACC,
        k3["B"]["address"]["accuracy"] >= G_ACC,
        k3["C"]["address"]["accuracy"] >= G_ACC,
    ))
    if shared_preserved:
        return "ADDRESS_SPECIFIC"

    filler_broadcast = any((
        k1["B"]["target_accuracy"] >= G_ACC,
        k1["C"]["target_accuracy"] >= G_ACC,
        k2["C"]["target_accuracy"] >= G_ACC,
    ))
    if filler_broadcast:
        return "GLOBAL_TARGET_BROADCAST"
    return "MIXED_COLLISION_EFFECT"


@torch.no_grad()
def run_delta_content_aliasing(
        model_path, out_dir, model_key="deepseek_content_aliasing_d1",
        quantization="8bit", device_map=None, max_memory=None,
        n_rows=30, loads=(1, 2, 3)):
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(
        _resolve(model_path), quantization=quantization,
        device_map=device_map, max_memory=max_memory)
    dev = input_device(model)
    states = _neutral_states(
        model, tok, dev, LAYER, LOCATIONS)
    selected_loads = tuple(int(load) for load in loads)
    if selected_loads != (1, 2, 3):
        raise ValueError("v1 is frozen to collision loads [1,2,3]")

    result = {
        "stage": "delta_content_aliasing",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_key": model_key,
        "model_path": model_path,
        "quantization": quantization,
        "layer": LAYER,
        "n_rows_per_load": int(n_rows),
        "loads": list(selected_loads),
        "per_load": {},
    }
    hb = Heartbeat(
        len(selected_loads) * len(REGISTERS),
        "content_aliasing", every_sec=30, out_dir=out_dir)
    all_clean_eligible = True
    all_natural_a_eligible = True
    all_intended = True

    for load in selected_loads:
        rows = _collision_rows(load, n_rows)
        natural = _natural_rows(rows)
        deltas = torch.stack([
            states[row["target"]] - states[row["source"]]
            for row in rows
        ])
        batches = {
            register: (
                _batch(tok, rows, register, dev),
                _batch(tok, natural, register, dev),
            )
            for register in REGISTERS
        }
        positions = _positions(tok, batches["A"][0], rows)
        for register in REGISTERS:
            clean_batch, natural_batch = batches[register]
            if _positions(tok, clean_batch, rows) != positions:
                raise ValueError("clean positions vary by query")
            if _positions(tok, natural_batch, natural) != positions:
                raise ValueError("natural positions vary by query")

        behavior = {}
        synthetic = {}
        natural_alias = {}
        intended_metrics = None
        for register in REGISTERS:
            clean_batch, natural_batch = batches[register]
            clean_logits, _ = _forward(
                model, clean_batch["ids"], clean_batch["am"], ())
            natural_logits, _ = _forward(
                model, natural_batch["ids"], natural_batch["am"], ())
            add_logits, _ = _forward(
                model, clean_batch["ids"], clean_batch["am"], (),
                add=(LAYER, positions["A"], deltas))

            clean_expected = [row[register] for row in rows]
            natural_expected = [row[register] for row in natural]
            clean_rival = [
                row["target"] if row[register] != row["target"]
                else row["source"]
                for row in rows
            ]
            natural_rival = [
                row["target"] if row[register] != row["target"]
                else row["source"]
                for row in natural
            ]
            behavior[register] = {
                "clean": _endpoint(
                    clean_logits, clean_batch,
                    clean_expected, clean_rival),
                "natural": _endpoint(
                    natural_logits, natural_batch,
                    natural_expected, natural_rival),
            }

            address_expected = clean_expected
            alias_expected = [
                row["target"] if row[register] == row["source"]
                else row[register]
                for row in rows
            ]
            address_rival = [
                row["target"] if value != row["target"]
                else row["source"]
                for row, value in zip(rows, address_expected)
            ]
            alias_rival = [
                row["source"] if value == row["target"]
                else row["target"]
                for row, value in zip(rows, alias_expected)
            ]
            synthetic[register] = {
                "address": _endpoint(
                    add_logits, clean_batch,
                    address_expected, address_rival),
                "alias": _endpoint(
                    add_logits, clean_batch,
                    alias_expected, alias_rival),
                "target_accuracy": float(_accuracy(
                    add_logits, clean_batch,
                    [row["target"] for row in rows])),
            }
            natural_alias[register] = float(_accuracy(
                natural_logits, natural_batch, alias_expected))

            if register == "A":
                intended_metrics = _switch_metrics(
                    clean_logits, natural_logits, add_logits,
                    clean_batch,
                    [row["source"] for row in rows],
                    [row["target"] for row in rows])
                intended_metrics["pass"] = _switch_pass(
                    intended_metrics)
                synthetic["A"]["switch"] = intended_metrics
            hb.step(extra=f"k={load} register={register}")

        clean_ok = min(
            behavior[register]["clean"]["accuracy"]
            for register in REGISTERS) >= G_ACC
        natural_a_ok = (
            behavior["A"]["natural"]["accuracy"] >= G_ACC)
        intended_ok = bool(intended_metrics["pass"])
        all_clean_eligible &= clean_ok
        all_natural_a_eligible &= natural_a_ok
        all_intended &= intended_ok
        result["per_load"][str(load)] = {
            "rows": rows,
            "positions": positions,
            "behavior": behavior,
            "natural_alias": natural_alias,
            "synthetic": synthetic,
            "gates": {
                "clean": clean_ok,
                "natural_A": natural_a_ok,
                "intended": intended_ok,
            },
        }
        log(
            f"ALIAS k={load} clean={clean_ok} naturalA={natural_a_ok} "
            f"intended={intended_ok} B="
            f"{synthetic['B']['address']['accuracy']:.0%}/"
            f"{synthetic['B']['alias']['accuracy']:.0%} C="
            f"{synthetic['C']['address']['accuracy']:.0%}/"
            f"{synthetic['C']['alias']['accuracy']:.0%}")
    hb.done()

    eligible = bool(all_clean_eligible and all_natural_a_eligible)
    verdict = _pattern_verdict(
        eligible, all_intended, result["per_load"])
    result["gates"] = {
        "eligible": eligible,
        "intended_all_loads": bool(all_intended),
    }
    result["verdict"] = verdict
    path = os.path.join(
        out_dir, f"results_delta_content_aliasing_{model_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(f"CONTENT ALIASING verdict={verdict} artifact={path}")
    return result
