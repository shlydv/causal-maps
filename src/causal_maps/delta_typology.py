"""Typology prediction (INTENT_DIRECTION_PLAN.md Part B / H2).

Pre-registered computational-type → steerability profile on held-out skills:

  H2a  ROUTE type  — boolean NOT of a stored bit   [v3; echo_kth permanently stopped]
       Predict: early/strong  (sig layers ⊆ {2,8,14})
  H2b  CREATE type — which of two numbers is larger
       Predict: late/weak     (sig layers ⊆ {20,26})

v1: echo 0% (no primer) → INELICITABLE.
v2: echo fixed menu all failed (max 6%) → ECHO_STOPPED_PERMANENT.
v3: switch to plan's alternate route skill (boolean NOT); compare unchanged.

Gate T2: both skills' sig layers on predicted side of early/late split.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .logutil import Heartbeat, log
from .model_utils import (input_device, last_token_logits, load_model_and_tokenizer,
                          single_token_id)
from .nulls import permutation_pvalue
from .patching import cache_layer_outputs, forward_with_add

N_NULL = 100
LAYERS = [2, 8, 14, 20, 26]
BEHAV_GATE = 0.80
N_TRAIN = 8
N_TEST = 8
EARLY = {2, 8, 14}
LATE = {20, 26}
VERSION = 3

PREDICTIONS = {
    "not_bit": {
        "type": "route",
        "predict_profile": "early_strong",
        "predict_sig_side": "early",
        "note": "plan alternate after echo_kth permanent stop",
    },
    "compare_larger": {
        "type": "create",
        "predict_profile": "late_weak",
        "predict_sig_side": "late",
    },
}


def _chat(tok, user, primer=""):
    t = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    return t + primer


def _encode(tok, texts, device):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ---------------------------------------------------------------------------
# Skill A — route: boolean NOT of a stored bit
# ---------------------------------------------------------------------------
def _not_text(tok, bit):
    # bit is 0 or 1; answer is 1-bit
    user = (
        f"Let flag = {bit}. "
        f"Output the boolean NOT of flag (0 becomes 1, 1 becomes 0)."
    )
    return _chat(tok, user, primer="Answer: ")


def _run_not_bit(model, tok, device, rng, layers, n_null, out_dir):
    # Balanced train/test of bit=0 and bit=1
    bits_tr = [0, 1] * (N_TRAIN // 2)
    bits_te = [0, 1] * (N_TEST // 2)
    rng.shuffle(bits_tr)
    rng.shuffle(bits_te)

    @torch.no_grad()
    def greedy(ids, am):
        return last_token_logits(model, ids, am).argmax(-1).cpu().numpy()

    texts_tr = [_not_text(tok, b) for b in bits_tr]
    texts_te = [_not_text(tok, b) for b in bits_te]
    lens = {len(tok.encode(t, add_special_tokens=False)) for t in texts_tr + texts_te}
    assert len(lens) == 1, f"not_bit length nonuniform {lens}"
    ids_tr, am_tr = _encode(tok, texts_tr, device)
    ids_te, am_te = _encode(tok, texts_te, device)

    id0 = single_token_id(tok, "0", leading_space=False)
    id1 = single_token_id(tok, "1", leading_space=False)
    ans_tr = np.array([id1 if b == 0 else id0 for b in bits_tr])
    ans_te = np.array([id1 if b == 0 else id0 for b in bits_te])
    rate = float((greedy(ids_te, am_te) == ans_te).mean())
    rate_tr = float((greedy(ids_tr, am_tr) == ans_tr).mean())
    log(f"  not_bit behav: train={rate_tr:.0%} test={rate:.0%}")
    if rate < BEHAV_GATE:
        return {
            "skill": "not_bit", "type": "route",
            "predict_profile": PREDICTIONS["not_bit"]["predict_profile"],
            "behav": rate, "verdict": "TYPOLOGY_INELICITABLE",
            "stop_reason": "behav_below_gate", "per_layer": [],
        }

    # Δ = mean(h | bit=1) − mean(h | bit=0) at last pos — routing/transform of stored bit
    # Actually for NOT: contrast prompts that store 0 vs 1; effect = raise NOT(bit).
    # Train caches by bit value:
    idx0 = [i for i, b in enumerate(bits_tr) if b == 0]
    idx1 = [i for i, b in enumerate(bits_tr) if b == 1]
    cache = cache_layer_outputs(model, ids_tr, am_tr, to_cpu=True)
    last = ids_tr.shape[1] - 1
    Delta = {}
    for L in layers:
        h = cache[L][:, last, :].float()
        Delta[L] = h[idx1].mean(0) - h[idx0].mean(0)

    # On held-out bit=0 prompts, +Δ should raise logit(1)−logit(0) (the NOT answer)
    # On bit=1, −Δ should raise logit(0)−logit(1). Primary: add Δ on bit=0 pool.
    te0 = [i for i, b in enumerate(bits_te) if b == 0]
    assert len(te0) >= 2
    ids0 = ids_te[te0]
    am0 = am_te[te0]

    @torch.no_grad()
    def not_ld(ids, am, delta=None, layer=None, pos=None):
        """logit(1) − logit(0): higher means preferring NOT(0)=1."""
        if delta is None:
            lg = last_token_logits(model, ids, am)
        else:
            lg = forward_with_add(model, ids, am, layer, pos, delta)
        n = ids.shape[0]
        return (lg[torch.arange(n, device=device), id1]
                - lg[torch.arange(n, device=device), id0]).float().cpu().numpy()

    pos = ids0.shape[1] - 1
    base = not_ld(ids0, am0)
    hb = Heartbeat(len(layers) * n_null, "typology_not", every_sec=15, out_dir=out_dir)
    per_layer = []
    for L in layers:
        d = Delta[L]
        effect = float((not_ld(ids0, am0, d, L, pos) - base).mean())
        ns = float(d.norm().clamp(min=1e-8))
        nulls = []
        for _ in range(n_null):
            r = torch.from_numpy(rng.normal(size=d.numel()).astype(np.float32))
            r = r / r.norm().clamp(min=1e-8) * ns
            nulls.append(float((not_ld(ids0, am0, r, L, pos) - base).mean()))
            hb.step()
        p = permutation_pvalue(effect, np.array(nulls), "greater")
        per_layer.append({"layer": int(L), "effect": effect, "p": float(p),
                          "delta_norm": ns})
        log(f"  not_bit L{L}: effect={effect:+.2f}(p={p:.3f})")
    hb.done()
    return {
        "skill": "not_bit", "type": "route",
        "predict_profile": PREDICTIONS["not_bit"]["predict_profile"],
        "predict_sig_side": "early",
        "behav": rate, "behav_train": rate_tr,
        "per_layer": per_layer, "confounded": False,
        "note": "Δ = h(bit=1)−h(bit=0); +Δ on bit=0 should raise NOT=1",
    }


# ---------------------------------------------------------------------------
# Skill B — create: which number is larger
# ---------------------------------------------------------------------------
def _compare_pool(rng, n):
    rows, seen = [], set()
    guard = 0
    while len(rows) < n and guard < 5000:
        guard += 1
        a, b = int(rng.integers(1, 9)), int(rng.integers(1, 9))
        if a == b:
            continue
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        larger = a if a > b else b
        smaller = b if a > b else a
        rows.append({"a": a, "b": b, "larger": larger, "smaller": smaller})
    return rows


def _compare_text(tok, row):
    user = (
        f"X={row['a']}, Y={row['b']}. "
        f"Output only the larger number."
    )
    return _chat(tok, user, primer="Answer: ")


def _run_compare(model, tok, device, rng, layers, n_null, out_dir):
    rows = _compare_pool(rng, N_TRAIN + N_TEST)
    train, test = rows[:N_TRAIN], rows[N_TRAIN:]
    texts_tr = [_compare_text(tok, r) for r in train]
    texts_te = [_compare_text(tok, r) for r in test]
    ids_tr, am_tr = _encode(tok, texts_tr, device)
    ids_te, am_te = _encode(tok, texts_te, device)

    @torch.no_grad()
    def greedy(ids, am):
        return last_token_logits(model, ids, am).argmax(-1).cpu().numpy()

    ans_ids = np.array([single_token_id(tok, str(r["larger"]), leading_space=False)
                        for r in test])
    tr_ids = np.array([single_token_id(tok, str(r["larger"]), leading_space=False)
                       for r in train])
    rate = float((greedy(ids_te, am_te) == ans_ids).mean())
    rate_tr = float((greedy(ids_tr, am_tr) == tr_ids).mean())
    log(f"  compare_larger behav: train={rate_tr:.0%} test={rate:.0%}")
    if rate < BEHAV_GATE:
        return {
            "skill": "compare_larger", "type": "create",
            "predict_profile": PREDICTIONS["compare_larger"]["predict_profile"],
            "behav": rate, "verdict": "TYPOLOGY_INELICITABLE",
            "stop_reason": "behav_below_gate", "per_layer": [],
        }

    cache = cache_layer_outputs(model, ids_tr, am_tr, to_cpu=True)
    last = ids_tr.shape[1] - 1
    digits = sorted({r["larger"] for r in train} | {r["larger"] for r in test}
                    | {r["smaller"] for r in train} | {r["smaller"] for r in test})

    def _smaller_text(row):
        user = (
            f"X={row['a']}, Y={row['b']}. "
            f"Output only the smaller number."
        )
        return _chat(tok, user, primer="Answer: ")

    ids_foil, am_foil = _encode(tok, [_smaller_text(r) for r in train], device)
    assert ids_foil.shape[1] == ids_tr.shape[1]
    cache_f = cache_layer_outputs(model, ids_foil, am_foil, to_cpu=True)
    Delta = {L: (cache[L][:, last, :].float().mean(0)
                 - cache_f[L][:, last, :].float().mean(0))
             for L in layers}

    foil_te = test
    ids_ft, am_ft = _encode(tok, [_smaller_text(r) for r in foil_te], device)
    large_ids = np.array([single_token_id(tok, str(r["larger"]), leading_space=False)
                          for r in foil_te])
    small_ids = np.array([single_token_id(tok, str(r["smaller"]), leading_space=False)
                          for r in foil_te])

    @torch.no_grad()
    def create_ld(ids, am, delta=None, layer=None, pos=None):
        if delta is None:
            lg = last_token_logits(model, ids, am)
        else:
            lg = forward_with_add(model, ids, am, layer, pos, delta)
        hi = torch.tensor(large_ids, device=device)
        lo = torch.tensor(small_ids, device=device)
        return (lg[torch.arange(len(foil_te), device=device), hi]
                - lg[torch.arange(len(foil_te), device=device), lo]).float().cpu().numpy()

    pos = ids_ft.shape[1] - 1
    base = create_ld(ids_ft, am_ft)
    hb = Heartbeat(len(layers) * n_null, "typology_compare", every_sec=15, out_dir=out_dir)
    per_layer = []
    for L in layers:
        d = Delta[L]
        effect = float((create_ld(ids_ft, am_ft, d, L, pos) - base).mean())
        ns = float(d.norm().clamp(min=1e-8))
        nulls = []
        for _ in range(n_null):
            r = torch.from_numpy(rng.normal(size=d.numel()).astype(np.float32))
            r = r / r.norm().clamp(min=1e-8) * ns
            nulls.append(float((create_ld(ids_ft, am_ft, r, L, pos) - base).mean()))
            hb.step()
        p = permutation_pvalue(effect, np.array(nulls), "greater")
        per_layer.append({"layer": int(L), "effect": effect, "p": float(p),
                          "delta_norm": ns})
        log(f"  compare L{L}: effect={effect:+.2f}(p={p:.3f})")
    hb.done()
    return {
        "skill": "compare_larger", "type": "create",
        "predict_profile": PREDICTIONS["compare_larger"]["predict_profile"],
        "predict_sig_side": "late",
        "behav": rate, "behav_train": rate_tr,
        "per_layer": per_layer, "confounded": False,
        "digits": digits,
    }


def _sig_layers(per_layer):
    return [r["layer"] for r in per_layer
            if r.get("p", 1) < 0.01 and r.get("effect", 0) > 0]


def _side_ok(sig, side):
    if not sig:
        return False
    s = set(sig)
    if side == "early":
        return s.issubset(EARLY) and len(s & EARLY) > 0
    if side == "late":
        return s.issubset(LATE) and len(s & LATE) > 0
    return False


def run_delta_typology(model_path, out_dir, quantization="8bit", device_map=None,
                       seed=0, layers=None, n_null=N_NULL):
    os.makedirs(out_dir, exist_ok=True)
    layers = list(layers or LAYERS)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    device = input_device(model)
    rng = np.random.default_rng(seed)
    n_layers = model.config.num_hidden_layers
    layers = [L for L in layers if 0 <= L < n_layers]

    log(f"delta_typology H2 v{VERSION}: layers={layers} n_null={n_null} "
        f"predictions={ {k: v['predict_profile'] for k, v in PREDICTIONS.items()} } "
        f"(echo permanently stopped; route=not_bit)")

    route = _run_not_bit(model, tok, device, rng, layers, n_null, out_dir)
    compare = _run_compare(model, tok, device, rng, layers, n_null, out_dir)

    for skill in (route, compare):
        if skill.get("per_layer"):
            skill["sig_layers"] = _sig_layers(skill["per_layer"])
            skill["side_confirmed"] = _side_ok(
                skill["sig_layers"], skill.get("predict_sig_side", ""))
            best = max(skill["per_layer"], key=lambda r: r["effect"])
            skill["best_layer"] = best["layer"]
            skill["best_effect"] = best["effect"]
            if skill.get("type") == "create":
                skill["observed_profile"] = (
                    "late_weak" if skill["side_confirmed"] else "MISS")
            else:
                skill["observed_profile"] = (
                    "early_strong" if skill["side_confirmed"] else "MISS")

    elicitable = all(s.get("per_layer") for s in (route, compare))
    if not elicitable:
        verdict = "TYPOLOGY_INELICITABLE"
    elif route.get("side_confirmed") and compare.get("side_confirmed"):
        verdict = "TYPOLOGY_CONFIRMED"
    elif route.get("side_confirmed") or compare.get("side_confirmed"):
        verdict = "TYPOLOGY_PARTIAL"
    else:
        verdict = "TYPOLOGY_FALSIFIED"

    results = {
        "stage": "delta_typology", "version": VERSION, "model_path": model_path,
        "layers": layers, "n_null": n_null, "behav_gate": BEHAV_GATE,
        "predictions": PREDICTIONS,
        "skills": {"not_bit": route, "compare_larger": compare},
        "echo_status": "permanently_stopped_v2",
        "verdict": verdict,
        "gate_T2": "both skills' sig layers on predicted side of early/late split",
        "framing": {
            "TYPOLOGY_CONFIRMED": "a-priori type→profile prediction held for both skills",
            "TYPOLOGY_PARTIAL": "one skill confirmed, one miss — report both",
            "TYPOLOGY_FALSIFIED": "type theory miss — finding, not a failure to bury",
            "TYPOLOGY_INELICITABLE": "behav gate failed — harness/elicitation, not science",
        }[verdict],
    }
    path = os.path.join(out_dir, "results_delta_typology.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT delta_typology: {verdict} | "
        f"not_sig={route.get('sig_layers')} confirm={route.get('side_confirmed')} | "
        f"compare_sig={compare.get('sig_layers')} confirm={compare.get('side_confirmed')}")
    return results
