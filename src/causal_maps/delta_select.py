"""Boundary mapping — Level 2 (Selection), v2.

v1 (if-flag / Answer =) was L2_INELICITABLE (behav ~0%/6%).
v2: ONE fixed hand-designed template menu. First template with both
flag0→B and flag1→A greedy ≥ BEHAV_GATE wins and proceeds to causal.
If NONE clear: verdict L2_STOPPED_PERMANENT — no further Selection
prompt-fishing (Sahil stop condition, 2026-07-13).

Causal protocol unchanged: Δ_sel(L)=mean(h_f1−h_f0) at flag digit;
ADD on held-out flag=0; route + selectivity vs null; layer sweep; graded.
"""
import json
import os

import numpy as np
import torch

from . import variable_pairs
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

# Fixed menu (order = search order). Every template MUST contain `{key}={f}`
# with a bare digit so injection site is well-defined (same trap as transform
# space+digit). No post-hoc additions after seeing causal results.
SELECT_TEMPLATES = [
    {
        "name": "v1_if_flag",  # known fail; kept as first check
        "key": "flag",
        "primer": "Answer =",
        "user": lambda va, vb, f, key: (
            f"Let A = {va}. Let B = {vb}. "
            f"If the flag is 1, answer A. If the flag is 0, answer B. "
            f"{key}={f}. What is the answer?"),
    },
    {
        "name": "value_of",
        "key": "flag",
        "primer": "Answer =",
        "user": lambda va, vb, f, key: (
            f"Let A = {va}. Let B = {vb}. "
            f"If {key}=1, output the value of A. If {key}=0, output the value of B. "
            f"{key}={f}. What is the answer?"),
    },
    {
        "name": "rules_answer_var",  # Completion-like rules; A/B are variable names
        "key": "flag",
        "primer": "Answer =",
        "user": lambda va, vb, f, key: (
            f"Let A = {va}. Let B = {vb}.\n"
            f"Rules:\n"
            f"- if {key} is 0, answer B\n"
            f"- if {key} is 1, answer A\n"
            f"{key}={f}\n"
            f"What is the answer?"),
    },
    {
        "name": "xy_selector",
        "key": "sel",
        "primer": "Answer =",
        "user": lambda va, vb, f, key: (
            f"Let X = {va}. Let Y = {vb}. "
            f"When {key}=1 report X. When {key}=0 report Y. "
            f"{key}={f}. What is the answer?"),
    },
    {
        "name": "read_slot",
        "key": "slot",
        "primer": "Answer =",
        "user": lambda va, vb, f, key: (
            # slot=1 → A, slot=0 → B  (same Δ 0→1 semantics)
            f"Slot 1 holds {va}. Slot 0 holds {vb}. "
            f"Read {key}={f}. What value is in that slot?"),
    },
    {
        "name": "switch_on_off",
        "key": "switch",
        "primer": "Answer =",
        "user": lambda va, vb, f, key: (
            f"A = {va}. B = {vb}. "
            f"If the {key} is 1, say the value of A. If the {key} is 0, say the value of B. "
            f"{key}={f}."),
    },
    {
        "name": "return_instr",
        "key": "flag",
        "primer": "Answer =",
        "user": lambda va, vb, f, key: (
            f"Variables: A={va}, B={vb}. "
            f"Instruction: return A if {key} is 1 else return B. "
            f"{key}={f}. Return:"),
    },
    {
        "name": "choose_label",
        "key": "flag",
        "primer": "Answer =",
        "user": lambda va, vb, f, key: (
            f"Option A is {va}. Option B is {vb}. "
            f"Choose A when {key}=1; choose B when {key}=0. "
            f"{key}={f}. Your choice's value:"),
    },
    {
        "name": "bit_state",
        "key": "bit",
        "primer": "Answer =",
        "user": lambda va, vb, f, key: (
            f"Let A = {va}. Let B = {vb}.\n"
            f"State:\n{key}={f}\n"
            f"If {key} is 1 the answer is A, otherwise the answer is B. "
            f"What is the answer?"),
    },
    {
        "name": "fewshot_one",
        "key": "flag",
        "primer": "Answer =",
        "user": lambda va, vb, f, key: (
            f"Example: Let A = red. Let B = blue. {key}=1. Answer = red\n"
            f"Now: Let A = {va}. Let B = {vb}. {key}={f}. What is the answer?"),
    },
    {
        "name": "direct_ask",
        "key": "flag",
        "primer": "Answer =",
        "user": lambda va, vb, f, key: (
            f"A holds {va}. B holds {vb}. {key}={f}. "
            f"Give me A if {key} is 1, else give me B."),
    },
    {
        "name": "which_binding",
        "key": "flag",
        "primer": "Answer =",
        "user": lambda va, vb, f, key: (
            f"Bindings: A={va}; B={vb}. "
            f"Active binding index {key}={f} (1=A, 0=B). "
            f"Emit the active binding's value."),
    },
]


def _render(tok, tmpl, va, vb, f):
    user = tmpl["user"](va, vb, f, tmpl["key"])
    templated = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return templated + tmpl["primer"], tmpl["key"], f


def _flag_digit_index(tok, text, key, f):
    needle = f"{key}={f}"
    start = text.rfind(needle)
    assert start >= 0, f"missing {needle!r} in text"
    prefix = text[:start + len(f"{key}=")]
    pre = tok.encode(prefix, add_special_tokens=False)
    full = tok.encode(text, add_special_tokens=False)
    idx = len(pre)
    assert full[:idx] == pre and idx < len(full), f"unstable {key}= prefix"
    expect = single_token_id(tok, str(f), leading_space=False)
    assert full[idx] == expect, (
        f"{key} digit mismatch: got {tok.convert_ids_to_tokens([full[idx]])} "
        f"expected {tok.convert_ids_to_tokens([expect])}")
    return idx


def _encode_pool(tok, texts, keys, flags, dev):
    enc = [tok.encode(t, add_special_tokens=False) for t in texts]
    lens = sorted(set(len(e) for e in enc))
    assert len(lens) == 1, f"non-uniform select pool length: {lens}"
    flag_pos = [_flag_digit_index(tok, t, k, f) for t, k, f in zip(texts, keys, flags)]
    assert len(set(flag_pos)) == 1, f"non-uniform flag_pos {set(flag_pos)}"
    ids = torch.tensor(enc, dtype=torch.long, device=dev)
    return ids, torch.ones_like(ids), flag_pos[0]


def _grade_l2(per_layer):
    if not per_layer:
        return {"verdict": "L2_INELICITABLE", "best_layer": None,
                "sig_layers": [], "best_route": float("nan"),
                "best_selectivity": float("nan")}
    ok = [r for r in per_layer
          if r["route"]["p"] < 0.01 and r["route"]["effect"] > 0
          and r["selectivity"]["p"] < 0.01 and r["selectivity"]["effect"] > 0]
    pos_route = any(r["route"]["effect"] > 0 for r in per_layer)
    best = max(per_layer, key=lambda r: r["route"]["effect"])
    n_sig, n_tot = len(ok), len(per_layer)
    if not ok:
        verdict = "L2_PARTIAL" if pos_route else "L2_BOUNDARY"
    elif n_sig == n_tot:
        verdict = "L2_STRONG"
    elif n_sig >= 2 or (n_sig == 1 and best["layer"] <= 14):
        verdict = "L2_LAYER_DEPENDENT_STRONG" if best["layer"] <= 14 else "L2_LAYER_DEPENDENT_WEAK"
    else:
        verdict = "L2_LAYER_DEPENDENT_WEAK"
    return {
        "verdict": verdict,
        "sig_layers": [r["layer"] for r in ok],
        "best_layer": best["layer"],
        "best_route": float(best["route"]["effect"]),
        "best_selectivity": float(best["selectivity"]["effect"]),
    }


def _value_pool(tok):
    vals = []
    for a, b in variable_pairs._VALUE_PAIRS:
        for w in (a, b):
            try:
                single_token_id(tok, w)
                if w not in vals:
                    vals.append(w)
            except ValueError:
                continue
    return vals


def _pair_pool(tok, tmpl, rng, n_need):
    """Distinct (va,vb) sharing one sequence length under tmpl (for batching)."""
    vals = _value_pool(tok)
    pairs = [(vals[i], vals[j]) for i in range(len(vals)) for j in range(len(vals)) if i != j]
    rng.shuffle(pairs)
    by_len = {}
    for va, vb in pairs:
        try:
            t0, k, _ = _render(tok, tmpl, va, vb, 0)
            t1, _, _ = _render(tok, tmpl, va, vb, 1)
            _flag_digit_index(tok, t0, k, 0)
            _flag_digit_index(tok, t1, k, 1)
            L0 = len(tok.encode(t0, add_special_tokens=False))
            L1 = len(tok.encode(t1, add_special_tokens=False))
            if L0 != L1:
                continue
            by_len.setdefault(L0, []).append((va, vb))
            if len(by_len[L0]) >= n_need:
                return by_len[L0][:n_need]
        except Exception:
            continue
    if not by_len:
        return []
    best = max(by_len.values(), key=len)
    return best[:n_need] if len(best) >= n_need else []


def run_delta_select(model_path, out_dir, quantization="8bit", device_map=None,
                     seed=0, layers=None, n_null=N_NULL, behav_gate=BEHAV_GATE):
    os.makedirs(out_dir, exist_ok=True)
    layers = list(layers or LAYERS)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    dev = input_device(model)
    rng = np.random.default_rng(seed)
    n_need = N_TRAIN + N_TEST

    log(f"delta_select v2: layers={layers} behav_gate={behav_gate:.0%} "
        f"n_templates={len(SELECT_TEMPLATES)} n_null={n_null}")

    template_scores = []
    winner = None
    train = test = None

    @torch.no_grad()
    def greedy_ids(ids, am):
        return last_token_logits(model, ids, am).argmax(-1).cpu().numpy()

    for tmpl in SELECT_TEMPLATES:
        pairs = _pair_pool(tok, tmpl, rng, n_need)
        if len(pairs) < n_need:
            log(f"  template {tmpl['name']}: SKIP (only {len(pairs)} uniform pairs)")
            template_scores.append({"name": tmpl["name"], "flag0": None, "flag1": None,
                                    "error": f"pairs={len(pairs)}"})
            continue
        tr, te = pairs[:N_TRAIN], pairs[N_TRAIN:]

        def build(rows, f):
            texts, keys, flags, vas, vbs = [], [], [], [], []
            for va, vb in rows:
                t, k, ff = _render(tok, tmpl, va, vb, f)
                texts.append(t); keys.append(k); flags.append(ff)
                vas.append(va); vbs.append(vb)
            ids, am, fpos = _encode_pool(tok, texts, keys, flags, dev)
            return ids, am, fpos, vas, vbs

        try:
            all_rows = tr + te
            ids0, am0, fpos, vas, vbs = build(all_rows, 0)
            ids1, am1, _, _, _ = build(all_rows, 1)
            va_ids = np.array([single_token_id(tok, v) for v in vas])
            vb_ids = np.array([single_token_id(tok, v) for v in vbs])
            rate0 = float((greedy_ids(ids0, am0) == vb_ids).mean())
            rate1 = float((greedy_ids(ids1, am1) == va_ids).mean())
        except Exception as e:
            log(f"  template {tmpl['name']}: SKIP ({e})")
            template_scores.append({"name": tmpl["name"], "flag0": None, "flag1": None,
                                    "error": str(e)})
            continue

        template_scores.append({
            "name": tmpl["name"], "flag0": rate0, "flag1": rate1,
            "flag_pos": int(fpos), "S": int(ids0.shape[1]),
        })
        log(f"  template {tmpl['name']}: flag0→B={rate0:.0%} flag1→A={rate1:.0%} "
            f"(S={ids0.shape[1]} flag_pos={fpos})")
        if winner is None and rate0 >= behav_gate and rate1 >= behav_gate:
            winner = tmpl
            train, test = tr, te
            # keep searching remaining for the log

    if winner is None:
        log("  no template cleared behav_gate — L2_STOPPED_PERMANENT (no more Selection fishing)")
        results = {
            "stage": "delta_select", "version": 2, "model_path": model_path,
            "layers": layers, "behav_gate": behav_gate, "confounded": True,
            "verdict": "L2_STOPPED_PERMANENT",
            "template_scores": template_scores, "chosen_template": None,
            "per_layer": [], "stop_reason": "no_template_cleared_behav_gate",
        }
        with open(os.path.join(out_dir, "results_delta_select.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        log("VERDICT delta_select: L2_STOPPED_PERMANENT")
        return results

    # --- causal on winner ---
    tmpl = winner
    log(f"  chosen_template={tmpl['name']} — proceeding to causal sweep")

    def build(rows, f):
        texts, keys, flags, vas, vbs = [], [], [], [], []
        for va, vb in rows:
            t, k, ff = _render(tok, tmpl, va, vb, f)
            texts.append(t); keys.append(k); flags.append(ff)
            vas.append(va); vbs.append(vb)
        ids, am, fpos = _encode_pool(tok, texts, keys, flags, dev)
        return ids, am, fpos, vas, vbs

    ids_tr0, am_tr0, fpos, _, _ = build(train, 0)
    ids_tr1, am_tr1, _, _, _ = build(train, 1)
    cache0 = cache_layer_outputs(model, ids_tr0, am_tr0, to_cpu=True)
    cache1 = cache_layer_outputs(model, ids_tr1, am_tr1, to_cpu=True)
    n_layers = model.config.num_hidden_layers
    layers = [L for L in layers if 0 <= L < n_layers]
    Delta = {L: (cache1[L][:, fpos, :].float().mean(0)
                 - cache0[L][:, fpos, :].float().mean(0))
             for L in layers}

    ids_te, am_te, _, te_va, te_vb = build(test, 0)
    vals = _value_pool(tok)
    val_list = sorted(set(te_va) | set(te_vb) | set(vals[:10]))
    for v in list(te_va) + list(te_vb):
        if v not in val_list:
            val_list.append(v)
    v_ids = [single_token_id(tok, v) for v in val_list]
    va_idx = np.array([val_list.index(v) for v in te_va])
    vb_idx = np.array([val_list.index(v) for v in te_vb])

    @torch.no_grad()
    def val_logits(ids, am, delta=None, layer=None):
        if delta is None:
            lg = last_token_logits(model, ids, am)
        else:
            lg = forward_with_add(model, ids, am, layer, fpos, delta)
        return lg[:, v_ids].float().cpu().numpy()

    base = val_logits(ids_te, am_te)
    ids_te1, am_te1, _, te_va1, te_vb1 = build(test, 1)
    base1 = val_logits(ids_te1, am_te1)
    va1 = np.array([val_list.index(v) for v in te_va1])
    vb1 = np.array([val_list.index(v) for v in te_vb1])

    # recover behav rates for winner from scores
    wscore = next(s for s in template_scores if s["name"] == tmpl["name"])
    rate0, rate1 = wscore["flag0"], wscore["flag1"]

    hb = Heartbeat(len(layers) * n_null, "delta_select", every_sec=15, out_dir=out_dir)
    per_layer = []
    for L in layers:
        d = Delta[L]
        md = val_logits(ids_te, am_te, d, L) - base
        route = float((md[np.arange(len(va_idx)), va_idx]
                       - md[np.arange(len(vb_idx)), vb_idx]).mean())
        tgt = md[np.arange(len(va_idx)), va_idx]
        others = (md.sum(1) - tgt) / (md.shape[1] - 1)
        sel = float((tgt - others).mean())
        ns = float(d.norm().clamp(min=1e-8))
        null_route, null_sel = [], []
        for _ in range(n_null):
            r = torch.from_numpy(rng.normal(size=d.numel()).astype(np.float32))
            r = r / r.norm().clamp(min=1e-8) * ns
            md_r = val_logits(ids_te, am_te, r, L) - base
            null_route.append(float((md_r[np.arange(len(va_idx)), va_idx]
                                     - md_r[np.arange(len(vb_idx)), vb_idx]).mean()))
            tgt_r = md_r[np.arange(len(va_idx)), va_idx]
            oth_r = (md_r.sum(1) - tgt_r) / (md_r.shape[1] - 1)
            null_sel.append(float((tgt_r - oth_r).mean()))
            hb.step()
        p_r = permutation_pvalue(route, np.array(null_route), "greater")
        p_s = permutation_pvalue(sel, np.array(null_sel), "greater")
        md_anti = val_logits(ids_te1, am_te1, -d, L) - base1
        anti_route = float((md_anti[np.arange(len(va1)), va1]
                            - md_anti[np.arange(len(vb1)), vb1]).mean())
        row = {
            "layer": int(L),
            "route": {"effect": route, "p": float(p_r)},
            "selectivity": {"effect": sel, "p": float(p_s)},
            "anti_route_on_flag1": anti_route,
            "delta_norm": ns,
        }
        per_layer.append(row)
        log(f"  L{L}: route={route:+.2f}(p={p_r:.3f}) sel={sel:+.2f}(p={p_s:.3f}) "
            f"anti={anti_route:+.2f}")
    hb.done()

    grade = _grade_l2(per_layer)
    results = {
        "stage": "delta_select", "version": 2, "model_path": model_path,
        "layers": layers, "n_null": n_null, "behav_gate": behav_gate,
        "behav_flag0": rate0, "behav_flag1": rate1, "confounded": False,
        "template_scores": template_scores, "chosen_template": tmpl["name"],
        "train_pairs": train, "test_pairs": test, "flag_pos": int(fpos),
        "n_train": len(train), "n_test": len(test),
        "per_layer": per_layer, **grade,
    }
    with open(os.path.join(out_dir, "results_delta_select.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT delta_select: {grade['verdict']} | template={tmpl['name']} "
        f"sig={grade['sig_layers']}/{layers} best L{grade['best_layer']} "
        f"route={grade['best_route']:+.2f} | behav={rate0:.0%}/{rate1:.0%}")
    return results
