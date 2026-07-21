"""Computational chaining: BIND → ROUTE → PREDICATE (one pre-registered kernel).

Hypothesis (CHAIN_PROTOCOL.md): independently extracted dirs form dataflow
C(B(A(·))) — not additive knobs, not mere protocol communication.

G0 ABSOLUTE: bind/route/predicate native each ≥80% at L2, else
CHAIN_INELICITABLE — stop. No replacement donors, layer sweep, or template search.

Gates (frozen here):
  G1  CS = oriented(p_FULL − p_FLIP) > 0 vs null on B (p < 0.01)
       FULL = A + (+Δ_route) + C; FLIP = A + (−Δ_route) + C
  G2  Ablate B: drop_B = oriented(FULL) − oriented(noB) ≥ 0.5 × CS
  G3  Ablate A@X: drop_A ≥ 0.5 × max(oriented(FULL) − oriented(empty), ε)
  G4  |oriented(empty)| < 0.5 × max(|oriented(FULL)|, ε)  OR  |empty| < 1.0

Pre-registered CAUSAL_MAPS_LOG.md 2026-07-13. Novelty CLEAR (exact hyp untested).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .delta_crossskill import _variable_directions
from .delta_multislot import forward_add_multi
from .delta_select import SELECT_TEMPLATES, _encode_pool, _pair_pool, _render
from .direction_transfer import PRIMARY_LAYER
from .logutil import Heartbeat, log
from .model_utils import (input_device, last_token_logits,
                          load_model_and_tokenizer, single_token_id)
from .nulls import permutation_pvalue
from .patching import cache_layer_outputs
from .tensorize import _anchor_token_index

N_NULL = 100
BEHAV_GATE = 0.80
N_TRAIN_ROUTE = 8
N_TRAIN_PRED = 8
N_TEST_PRED = 8
N_TRIALS = 12
LAYER = PRIMARY_LAYER  # L2 only — no sweep
VERSION = 1
ROUTE_TMPL_NAME = "value_of"
EPS = 1e-6

# Frozen offline partitions (single-token under Qwen2.5; bare & spaced).
# Chain binds use Variable Δ vocabulary (cf-values): only "cat" is animal there —
# that is enough for balanced animal@X / animal@Y pairing; not a donor swap.
ANIMALS = [
    "dog", "cat", "fox", "owl", "ram", "pig", "cow", "ant", "bee", "bat",
    "hen", "bug", "fly", "ape", "duck", "wolf", "bear", "lion", "deer",
    "frog", "fish", "bird", "moth",
]
NON_ANIMALS = [
    "red", "blue", "hot", "cold", "open", "left", "right", "north", "south",
    "true", "false", "apple", "one", "two", "alpha", "beta", "gold", "pink",
    "lime", "gray",
]


def _chat(tok, user, primer):
    t = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    return t + primer


def _value_of_tmpl():
    for t in SELECT_TEMPLATES:
        if t["name"] == ROUTE_TMPL_NAME:
            return t
    raise RuntimeError(f"missing {ROUTE_TMPL_NAME}")


def _filter_pool(tok, words):
    out = []
    for w in words:
        try:
            single_token_id(tok, w, leading_space=True)
            single_token_id(tok, w, leading_space=False)
            out.append(w)
        except Exception:
            continue
    return out


def _pred_text(tok, v):
    user = f"Value: {v}. Is this an animal?"
    # Trailing space → next token is spaced YES/NO (matched readout).
    return _chat(tok, user, primer="Answer: ")


def _carrier_text(tok, v0, flag: int):
    user = (
        f"Let X = {v0}. Let Y = {v0}. "
        f"If flag=1, output the value of X. If flag=0, output the value of Y. "
        f"flag={flag}. "
        f"Is the answer an animal?"
    )
    text = _chat(tok, user, primer="Answer: ")
    off_x = text.find("Let X = ") + len("Let X = ")
    off_y = text.find("Let Y = ") + len("Let Y = ")
    xslot = _anchor_token_index(tok, text, off_x)
    yslot = _anchor_token_index(tok, text, off_y)
    needle = f"flag={flag}"
    start = text.rfind(needle)
    prefix = text[: start + len("flag=")]
    pre = tok.encode(prefix, add_special_tokens=False)
    full = tok.encode(text, add_special_tokens=False)
    fpos = len(pre)
    expect = single_token_id(tok, str(flag), leading_space=False)
    assert full[fpos] == expect
    return text, xslot, yslot, fpos


def _extract_routing_delta(model, tok, device, seed, layer):
    tmpl = _value_of_tmpl()
    rng = np.random.default_rng(seed + 17)
    pairs = _pair_pool(tok, tmpl, rng, N_TRAIN_ROUTE)
    assert len(pairs) >= N_TRAIN_ROUTE
    train = pairs[:N_TRAIN_ROUTE]

    def build(rows, f):
        texts, keys, flags = [], [], []
        for va, vb in rows:
            t, k, ff = _render(tok, tmpl, va, vb, f)
            texts.append(t); keys.append(k); flags.append(ff)
        return _encode_pool(tok, texts, keys, flags, device)

    ids0, am0, fpos = build(train, 0)
    ids1, am1, _ = build(train, 1)
    c0 = cache_layer_outputs(model, ids0, am0, to_cpu=True)
    c1 = cache_layer_outputs(model, ids1, am1, to_cpu=True)
    d = c1[layer][:, fpos, :].float().mean(0) - c0[layer][:, fpos, :].float().mean(0)
    return d


def _extract_predicate_delta(model, tok, device, animals_tr, non_tr, layer):
    texts_a = [_pred_text(tok, v) for v in animals_tr]
    texts_n = [_pred_text(tok, v) for v in non_tr]
    assert len({len(tok.encode(t, add_special_tokens=False)) for t in texts_a + texts_n}) == 1
    ids_a = torch.tensor([tok.encode(t, add_special_tokens=False) for t in texts_a],
                         dtype=torch.long, device=device)
    ids_n = torch.tensor([tok.encode(t, add_special_tokens=False) for t in texts_n],
                         dtype=torch.long, device=device)
    am_a, am_n = torch.ones_like(ids_a), torch.ones_like(ids_n)
    ca = cache_layer_outputs(model, ids_a, am_a, to_cpu=True)
    cn = cache_layer_outputs(model, ids_n, am_n, to_cpu=True)
    last = ids_a.shape[1] - 1
    return ca[layer][:, last, :].float().mean(0) - cn[layer][:, last, :].float().mean(0)


@torch.no_grad()
def _greedy_id(model, text, tok, device):
    e = tok.encode(text, add_special_tokens=False)
    ii = torch.tensor([e], dtype=torch.long, device=device)
    return int(last_token_logits(model, ii, torch.ones_like(ii)).argmax(-1).item())


@torch.no_grad()
def _p_yes_no(model, ids, am, layer, pos_deltas, yes_id, no_id, device):
    if pos_deltas:
        lg = forward_add_multi(model, ids, am, layer, pos_deltas)
    else:
        lg = last_token_logits(model, ids, am)
    n = ids.shape[0]
    return (lg[torch.arange(n, device=device), yes_id]
            - lg[torch.arange(n, device=device), no_id]).float().cpu().numpy()


def run_delta_chain(model_path, out_dir, quantization="8bit", device_map=None,
                    layer=LAYER, seed=0, n_trials=N_TRIALS, n_null=N_NULL):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    device = input_device(model)
    rng = np.random.default_rng(seed)

    yes_id = single_token_id(tok, "YES", leading_space=True)
    no_id = single_token_id(tok, "NO", leading_space=True)
    animals = _filter_pool(tok, ANIMALS)
    nons = _filter_pool(tok, NON_ANIMALS)
    assert len(animals) >= N_TRAIN_PRED + 2 and len(nons) >= N_TRAIN_PRED + 2

    log(f"delta_chain v{VERSION}: layer={layer} G0_gate={BEHAV_GATE:.0%} "
        f"n_animals={len(animals)} n_non={len(nons)} (no donor fishing)")

    # --- G0: donor behavioral gates (HARD STOP) ---
    Delta, values = _variable_directions(model, tok, layer, seed)

    bind_ok = 0
    bind_n = min(8, len(values))
    for v in values[:bind_n]:
        user = f"Let X = {v}. What is the value of X?"
        text = _chat(tok, user, primer="X = ")
        gold = single_token_id(tok, v, leading_space=True)
        if _greedy_id(model, text, tok, device) == gold:
            bind_ok += 1
    behav_bind = bind_ok / max(bind_n, 1)

    tmpl = _value_of_tmpl()
    route_pairs = _pair_pool(tok, tmpl, rng, 8)
    r_ok0 = r_ok1 = 0
    n_route = max(min(8, len(route_pairs)), 1)
    for va, vb in route_pairs[:n_route]:
        t0, _, _ = _render(tok, tmpl, va, vb, 0)
        t1, _, _ = _render(tok, tmpl, va, vb, 1)
        if _greedy_id(model, t0, tok, device) == single_token_id(tok, vb, leading_space=True):
            r_ok0 += 1
        if _greedy_id(model, t1, tok, device) == single_token_id(tok, va, leading_space=True):
            r_ok1 += 1
    behav_route = 0.5 * (r_ok0 + r_ok1) / n_route

    rng.shuffle(animals)
    rng.shuffle(nons)
    pred_tr_a = animals[:N_TRAIN_PRED]
    pred_te_a = animals[N_TRAIN_PRED:N_TRAIN_PRED + N_TEST_PRED]
    pred_tr_n = nons[:N_TRAIN_PRED]
    pred_te_n = nons[N_TRAIN_PRED:N_TRAIN_PRED + N_TEST_PRED]
    while len(pred_te_a) < N_TEST_PRED:
        pred_te_a.append(animals[len(pred_te_a) % len(animals)])
    while len(pred_te_n) < N_TEST_PRED:
        pred_te_n.append(nons[len(pred_te_n) % len(nons)])

    p_ok = p_n = 0
    for v in pred_te_a:
        p_n += 1
        if _greedy_id(model, _pred_text(tok, v), tok, device) == yes_id:
            p_ok += 1
    for v in pred_te_n:
        p_n += 1
        if _greedy_id(model, _pred_text(tok, v), tok, device) == no_id:
            p_ok += 1
    behav_pred = p_ok / max(p_n, 1)

    g0 = {
        "bind": behav_bind, "route": behav_route, "predicate": behav_pred,
        "pass_bind": behav_bind >= BEHAV_GATE,
        "pass_route": behav_route >= BEHAV_GATE,
        "pass_predicate": behav_pred >= BEHAV_GATE,
    }
    g0["pass"] = g0["pass_bind"] and g0["pass_route"] and g0["pass_predicate"]
    log(f"  G0 behav: bind={behav_bind:.0%} route={behav_route:.0%} "
        f"predicate={behav_pred:.0%} pass={g0['pass']}")

    if not g0["pass"]:
        results = {
            "stage": "delta_chain", "version": VERSION, "model_path": model_path,
            "layer": layer, "verdict": "CHAIN_INELICITABLE",
            "G0": g0, "stop_reason": "G0_hard_stop_no_donor_fishing",
            "framing": "predicate/bind/route failed behav gate — stop; no fishing",
            "gates_frozen": {
                "G1": "CS>0 vs null on B p<0.01",
                "G2": "drop_B >= 0.5 * CS",
                "G3": "drop_A >= 0.5 * (full - empty)",
                "G4": "|empty| < 0.5*|full| OR |empty| < 1.0",
            },
        }
        with open(os.path.join(out_dir, "results_delta_chain.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        log("VERDICT delta_chain: CHAIN_INELICITABLE (G0 absolute stop)")
        return results

    # --- extract C and B (A = Delta already) ---
    d_pred = _extract_predicate_delta(model, tok, device, pred_tr_a, pred_tr_n, layer)
    d_route = _extract_routing_delta(model, tok, device, seed, layer)
    log(f"  ‖Δ_pred‖={float(d_pred.norm()):.2f} ‖Δ_route‖={float(d_route.norm()):.2f}")

    bindable = set(Delta.keys())
    animal_set = set(animals)
    animals_b = [v for v in values if v in animal_set]
    nons_b = [v for v in values if v not in animal_set]
    # Variable cf-pool has only "cat" as animal — ≥1 animal + ≥2 nons is enough.
    if len(animals_b) < 1 or len(nons_b) < 2:
        results = {
            "stage": "delta_chain", "version": VERSION, "model_path": model_path,
            "layer": layer, "verdict": "CHAIN_INELICITABLE",
            "G0": g0,
            "stop_reason": "insufficient_overlap_binding_pool_with_animal_partition",
            "n_animals_bindable": len(animals_b), "n_non_bindable": len(nons_b),
            "framing": "cannot form animal/non pairs inside Variable Δ vocabulary",
        }
        with open(os.path.join(out_dir, "results_delta_chain.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        log("VERDICT delta_chain: CHAIN_INELICITABLE (binding∩partition too small)")
        return results

    v0_pool = [v for v in values if v not in animals_b]
    if not v0_pool:
        v0_pool = list(values)

    trials, used = [], set()
    tries = 0
    while len(trials) < n_trials and tries < 5000:
        tries += 1
        reverse = (len(trials) % 2 == 1)
        if reverse:
            u = str(rng.choice(nons_b))
            w = str(rng.choice(animals_b))
            animal_at_x = False
        else:
            u = str(rng.choice(animals_b))
            w = str(rng.choice(nons_b))
            animal_at_x = True
        v0 = str(rng.choice(v0_pool))
        if u == w or u == v0 or w == v0:
            continue
        key = (u, w, reverse)
        if key in used:
            continue
        used.add(key)
        trials.append({"u": u, "w": w, "v0": v0, "animal_at_x": animal_at_x})

    assert len(trials) >= 4, f"only {len(trials)} trials"
    log(f"  n_trials={len(trials)} animals_b={animals_b} n_nons_b={len(nons_b)}")

    texts, xslots, yslots, fposs = [], [], [], []
    for tr in trials:
        text, xs, ys, fp = _carrier_text(tok, tr["v0"], 0)
        texts.append(text); xslots.append(xs); yslots.append(ys); fposs.append(fp)
    assert len(set(len(tok.encode(t, add_special_tokens=False)) for t in texts)) == 1
    assert len(set(xslots)) == 1 and len(set(yslots)) == 1 and len(set(fposs)) == 1
    xslot, yslot, fpos = xslots[0], yslots[0], fposs[0]
    last = len(tok.encode(texts[0], add_special_tokens=False)) - 1
    ids = torch.tensor([tok.encode(t, add_special_tokens=False) for t in texts],
                       dtype=torch.long, device=device)
    am = torch.ones_like(ids)

    dX = torch.stack([Delta[tr["u"]] for tr in trials]).to(device)
    dY = torch.stack([Delta[tr["w"]] for tr in trials]).to(device)
    dR = d_route.to(device)
    dP = d_pred.to(device)
    binds = [(xslot, dX), (yslot, dY)]

    def pack(extra):
        return _p_yes_no(model, ids, am, layer, extra, yes_id, no_id, device)

    # Surface flag=0 → Y. +Δ_route → X; −Δ_route → Y.
    p_base = pack([])
    p_full = pack(binds + [(fpos, dR), (last, dP)])
    p_flip = pack(binds + [(fpos, -dR), (last, dP)])
    p_noB = pack(binds + [(last, dP)])                 # ablate B
    p_noA = pack([(yslot, dY), (fpos, dR), (last, dP)])  # ablate A@X
    p_noC = pack(binds + [(fpos, dR)])
    p_empty = pack([(fpos, dR), (last, dP)])

    signs = np.array([1.0 if tr["animal_at_x"] else -1.0 for tr in trials])

    def orient(p):
        return float((signs * p).mean())

    p_full_m, p_flip_m = orient(p_full), orient(p_flip)
    p_noB_m, p_noA_m = orient(p_noB), orient(p_noA)
    p_noC_m, p_empty_m, p_base_m = orient(p_noC), orient(p_empty), orient(p_base)
    CS = float((signs * (p_full - p_flip)).mean())

    log(f"  oriented prefs: full={p_full_m:+.2f} flip={p_flip_m:+.2f} CS={CS:+.2f}")
    log(f"  ablate: noB={p_noB_m:+.2f} noA={p_noA_m:+.2f} noC={p_noC_m:+.2f} "
        f"empty={p_empty_m:+.2f}")

    ns = float(d_route.norm().clamp(min=1e-8))
    hb = Heartbeat(n_null, "delta_chain_null", every_sec=15, out_dir=out_dir)
    null_cs = []
    for _ in range(n_null):
        r = torch.from_numpy(rng.normal(size=d_route.numel()).astype(np.float32))
        r = (r / r.norm().clamp(min=1e-8) * ns).to(device)
        p_r = pack(binds + [(fpos, r), (last, dP)])
        null_cs.append(float((signs * (p_r - p_flip)).mean()))
        hb.step()
    hb.done()
    p_cs = permutation_pvalue(CS, np.array(null_cs), "greater")
    g1 = bool(CS > 0 and p_cs < 0.01)

    drop_B = p_full_m - p_noB_m
    g2 = bool(CS > 0 and drop_B >= 0.5 * CS)
    drop_A = p_full_m - p_noA_m
    g3 = bool(drop_A >= 0.5 * max(p_full_m - p_empty_m, EPS))
    g4 = bool(abs(p_empty_m) < 0.5 * max(abs(p_full_m), EPS) or abs(p_empty_m) < 1.0)

    log(f"  G1 CS={CS:+.2f}(p={p_cs:.3f}) pass={g1}")
    log(f"  G2 drop_B={drop_B:+.2f} need≥{0.5 * max(CS, 0):.2f} pass={g2}")
    log(f"  G3 drop_A={drop_A:+.2f} pass={g3}")
    log(f"  G4 empty={p_empty_m:+.2f} pass={g4}")

    if g1 and g2 and g3 and g4:
        verdict = "CHAIN_PRIMITIVE"
    elif g1 and not g2:
        verdict = "CHAIN_PROTOCOL_ONLY"
    elif not g1:
        verdict = "CHAIN_KNOBS"
    else:
        verdict = "CHAIN_PROTOCOL_ONLY"

    results = {
        "stage": "delta_chain", "version": VERSION, "model_path": model_path,
        "layer": layer, "n_null": n_null, "n_trials": len(trials),
        "G0": g0,
        "gates": {"G1": g1, "G2": g2, "G3": g3, "G4": g4},
        "gates_frozen": {
            "G1": "CS>0 vs null on B p<0.01; FULL=+route FLIP=-route",
            "G2": "drop_B >= 0.5 * CS",
            "G3": "drop_A >= 0.5 * (full - empty)",
            "G4": "|empty| < 0.5*|full| OR |empty| < 1.0",
        },
        "CS": {"value": CS, "p": float(p_cs), "pass": g1},
        "oriented_prefs": {
            "full": p_full_m, "flip": p_flip_m, "noB": p_noB_m,
            "noA": p_noA_m, "noC": p_noC_m, "empty": p_empty_m, "base": p_base_m,
        },
        "dependency": {
            "drop_B": drop_B, "drop_A": drop_A,
            "g2_rule": "drop_B >= 0.5 * CS",
            "g3_rule": "drop_A >= 0.5 * (full - empty)",
        },
        "verdict": verdict,
        "framing": {
            "CHAIN_PRIMITIVE": "C computes on B(A); bypass controls hold",
            "CHAIN_PROTOCOL_ONLY": "route-conditional signal without forced use of B",
            "CHAIN_KNOBS": "no null-controlled chain sensitivity",
            "CHAIN_INELICITABLE": "G0 failed",
        }[verdict],
        "trials": trials,
        "animals_bindable": animals_b,
        "slots": {"xslot": int(xslot), "yslot": int(yslot),
                  "flag_pos": int(fpos), "pred_pos": int(last)},
        "readout": {"YES": int(yes_id), "NO": int(no_id), "leading_space": True},
    }
    with open(os.path.join(out_dir, "results_delta_chain.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT delta_chain: {verdict} | CS={CS:+.2f}(p={p_cs:.3f}) | "
        f"G1–G4={[g1, g2, g3, g4]}")
    return results
