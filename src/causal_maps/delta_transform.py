"""Boundary mapping — Level 3 (Transformation): do COMPUTED values carry a
transferable, value-selective readout direction, the way STORED values do?

Taxonomy (CAUSAL_MAPS_LOG.md 2026-07-13):
  L1 Representation  X = cat        (store / retrieve)  — established
  L2 Selection       if f: X else Y (route among stored) — delta_select (next)
  L3 Transformation  X = X + 1      (create a NEW value) — THIS module

v2 (2026-07-13): v1 was CONFOUNDED (computed_greedy=0% on rewrite-bind).
Before the causal sweep, try a fixed template menu; require
computed_greedy>=0.80 and stored_greedy>=0.80. If no template clears:
L3_INELICITABLE (skip causal). Else run the graded L3 protocol on the winner.

Method (FV-style; readout last-position for both store and compute):
  Per-answer direction at (L, readout):
      Δ_d(L) = mean_{answer=d} h(L)  -  mean_{answer!=d} h(L)
  ADD Δ_d at (L, readout) of HELD-OUT prompts; transfer + selectivity on digits.
  Four cells: ss, cc, sc, cs + cos(store,comp). Graded verdict via _grade_l3.

Pre-registered CAUSAL_MAPS_LOG.md 2026-07-13 (v2 elicitation fix).
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .logutil import Heartbeat, log
from .model_utils import (input_device, last_token_logits,
                          load_model_and_tokenizer, single_token_id)
from .nulls import permutation_pvalue
from .patching import cache_layer_outputs, forward_with_add

N_NULL = 100                       # p-floor = 1/(N+1) = 0.0099 (< 0.01 gate)
LAYERS = [2, 8, 14, 20, 26]        # early -> near-final (28-layer 7B)
MAX_COMBOS = 3                     # computed (a,b) combos per answer digit
RATIO_STRONG = 0.7                 # comp/stored selectivity to call it "STRONG"
BEHAV_GATE = 0.80                  # both store and compute must clear


# Primer ends with a TRAILING SPACE on purpose: in Qwen2.5 " 7" tokenizes as
# [' ', '7'] (space is its own token 220), so "X = 7" is [X, ' =', ' ', '7'].
# Priming "X = " (ending in that space token) makes the very next token the bare
# digit '7' (no leading space) — in-distribution, and single-token.
_PRIMER_X = "{name} = "
_PRIMER_ANSWER = "Answer: "
_PRIMER_SUM = "The sum is "


def _chat(tok, user, primer):
    templated = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return templated + primer


def _stored_text(tok, d, name="X"):
    user = f"Let {name} = {d}. What is the value of {name}?"
    return _chat(tok, user, _PRIMER_X.format(name=name))


# Fixed menu (order = search order; no post-hoc fishing after causal).
# Each entry: name, user(a,b,name)->str, primer(name)->str
COMPUTED_TEMPLATES = [
    {
        "name": "rewrite_bind",  # v1 — known 0% on 7B greedy; kept as first check
        "user": lambda a, b, name: (
            f"Let {name} = {a}. Then let {name} = {name} + {b}. "
            f"What is the value of {name}?"),
        "primer": lambda name: _PRIMER_X.format(name=name),
    },
    {
        "name": "add_to_x",
        "user": lambda a, b, name: (
            f"Let {name} = {a}. Add {b} to {name}. "
            f"What is the value of {name}?"),
        "primer": lambda name: _PRIMER_X.format(name=name),
    },
    {
        "name": "inline_sum",
        "user": lambda a, b, name: (
            f"Let {name} = {a} + {b}. What is the value of {name}?"),
        "primer": lambda name: _PRIMER_X.format(name=name),
    },
    {
        "name": "direct_sum",
        "user": lambda a, b, name: (
            f"What is {a} + {b}? Reply with only the single digit."),
        "primer": lambda name: _PRIMER_ANSWER,
    },
    {
        "name": "equals_phrase",
        "user": lambda a, b, name: f"Compute {a}+{b}.",
        "primer": lambda name: _PRIMER_SUM,
    },
]


def _computed_text(tok, a, b, template, name="X"):
    return _chat(tok, template["user"](a, b, name), template["primer"](name))


def _encode_pool(tok, texts, dev):
    enc = [tok.encode(t, add_special_tokens=False) for t in texts]
    lens = sorted(set(len(e) for e in enc))
    assert len(lens) == 1, f"non-uniform pool token length: {lens}"
    ids = torch.tensor(enc, dtype=torch.long, device=dev)
    return ids, torch.ones_like(ids)


def _delta_dir(h, answers, d):
    """One-vs-rest mean-difference direction for answer d. h [B,D], answers [B]."""
    ans = np.asarray(answers)
    return h[ans == d].mean(0) - h[ans != d].mean(0)      # [D]


def _grade_l3(per_layer, ratio_strong=RATIO_STRONG):
    """Pure graded-verdict decision tree over the per-layer profile (no model).
    Returns a summary dict. store->store (ss) is the L1 control; comp->comp (cc)
    is the L3 test. Categories (Sahil): STRONG/WEAK/LAYER_DEPENDENT/PARTIAL/
    BOUNDARY, plus CONTROL_FAILED if the stored control never transfers."""
    store_ok = any(r["ss"]["p"] < 0.01 and r["ss"]["selectivity"] > 0 for r in per_layer)
    cc_sig = [r for r in per_layer if r["cc"]["p"] < 0.01 and r["cc"]["selectivity"] > 0]
    cc_pos_transfer = any(r["cc"]["transfer"] > 0 for r in per_layer)
    best = max(per_layer, key=lambda r: r["cc"]["selectivity"])
    ratio_best = best["ratio_cc_over_ss"]
    n_sig, n_tot = len(cc_sig), len(per_layer)
    good_ratio = np.isfinite(ratio_best) and ratio_best >= ratio_strong

    if not store_ok:
        verdict = "CONTROL_FAILED"
    elif not cc_sig:
        verdict = "L3_PARTIAL" if cc_pos_transfer else "L3_BOUNDARY"
    elif n_sig == n_tot and good_ratio:
        verdict = "L3_STRONG"
    elif good_ratio:
        verdict = "L3_LAYER_DEPENDENT_STRONG"
    elif n_sig < n_tot:
        verdict = "L3_LAYER_DEPENDENT_WEAK"
    else:
        verdict = "L3_WEAK"
    mean_cos = float(np.mean([r["cos_store_comp"] for r in per_layer]))
    cross_ok = (best["sc"]["selectivity"] > 0 and best["cs"]["selectivity"] > 0
                and mean_cos > 0.3)
    return {
        "verdict": verdict,
        "store_control_ok": bool(store_ok),
        "cc_significant_layers": [r["layer"] for r in cc_sig],
        "best_layer": best["layer"],
        "best_ratio_cc_over_ss": float(ratio_best),
        "mean_cos_store_comp": mean_cos,
        "representation_shared": bool(cross_ok),
    }


def _digit_pool():
    combos = {}
    for a in range(1, 9):
        for b in range(1, 9):
            if a + b <= 9:
                combos.setdefault(a + b, []).append((a, b))
    dtargets = sorted(d for d in combos if len(combos[d]) >= 2)   # 3..9
    return combos, dtargets


def run_delta_transform(model_path, out_dir, quantization="8bit", device_map=None,
                        seed=0, layers=None, n_null=N_NULL, behav_gate=BEHAV_GATE):
    os.makedirs(out_dir, exist_ok=True)
    layers = list(layers or LAYERS)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    dev = input_device(model)
    rng = np.random.default_rng(seed)

    combos, dtargets = _digit_pool()
    val_ids = [single_token_id(tok, str(d), leading_space=False) for d in dtargets]
    didx = {d: i for i, d in enumerate(dtargets)}

    stored_ans = list(dtargets)
    stored_txt = [_stored_text(tok, d) for d in stored_ans]
    ids_s, am_s = _encode_pool(tok, stored_txt, dev)
    comp_specs = [(d, a, b) for d in dtargets for (a, b) in combos[d][:MAX_COMBOS]]
    comp_ans = [d for (d, a, b) in comp_specs]

    n_layers = model.config.num_hidden_layers
    layers = [L for L in layers if 0 <= L < n_layers]

    @torch.no_grad()
    def greedy_rate(ids, am, answers):
        pred = last_token_logits(model, ids, am).argmax(-1).cpu().numpy()
        gold = np.array([single_token_id(tok, str(a), leading_space=False)
                         for a in answers])
        return float((pred == gold).mean()), (pred == gold)

    store_greedy, _ = greedy_rate(ids_s, am_s, stored_ans)
    log(f"delta_transform v2: layers={layers} dtargets={dtargets} "
        f"stored={len(stored_ans)} n_comp_specs={len(comp_ans)} "
        f"stored_greedy={store_greedy:.0%} behav_gate={behav_gate:.0%} n_null={n_null}")

    # --- fixed template search (behavioral only; no causal until a winner) ---
    template_scores = []
    winner = None
    ids_c = am_c = None
    for tmpl in COMPUTED_TEMPLATES:
        try:
            comp_txt = [_computed_text(tok, a, b, tmpl) for (d, a, b) in comp_specs]
            ids_t, am_t = _encode_pool(tok, comp_txt, dev)
        except AssertionError as e:
            log(f"  template {tmpl['name']}: SKIP non-uniform length ({e})")
            template_scores.append({"name": tmpl["name"], "computed_greedy": None,
                                    "error": str(e)})
            continue
        cg, _ = greedy_rate(ids_t, am_t, comp_ans)
        template_scores.append({"name": tmpl["name"], "computed_greedy": cg,
                                "S": int(ids_t.shape[1])})
        log(f"  template {tmpl['name']}: computed_greedy={cg:.0%} (S={ids_t.shape[1]})")
        if winner is None and cg >= behav_gate:
            winner = tmpl
            ids_c, am_c = ids_t, am_t
            # keep searching remaining templates for the log, but lock winner = first clear

    if store_greedy < behav_gate:
        log(f"  FAIL store control greedy {store_greedy:.0%} < {behav_gate:.0%}")
        results = {
            "stage": "delta_transform", "version": 2, "model_path": model_path,
            "layers": layers, "dtargets": dtargets, "n_null": n_null,
            "store_greedy": store_greedy, "computed_greedy": None,
            "confounded": True, "verdict": "CONTROL_FAILED",
            "template_scores": template_scores, "chosen_template": None,
            "behav_gate": behav_gate, "per_layer": [],
        }
        with open(os.path.join(out_dir, "results_delta_transform.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        log(f"VERDICT delta_transform: CONTROL_FAILED (store_greedy)")
        return results

    if winner is None:
        log("  no computed template cleared behav_gate — L3_INELICITABLE; skip causal")
        results = {
            "stage": "delta_transform", "version": 2, "model_path": model_path,
            "layers": layers, "dtargets": dtargets, "n_null": n_null,
            "n_stored": len(stored_ans), "n_computed": len(comp_ans),
            "store_greedy": store_greedy, "computed_greedy": 0.0,
            "confounded": True, "verdict": "L3_INELICITABLE",
            "template_scores": template_scores, "chosen_template": None,
            "behav_gate": behav_gate, "per_layer": [],
            "store_control_ok": False, "cc_significant_layers": [],
            "best_layer": None, "best_ratio_cc_over_ss": float("nan"),
            "mean_cos_store_comp": float("nan"), "representation_shared": False,
        }
        with open(os.path.join(out_dir, "results_delta_transform.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        log("VERDICT delta_transform: L3_INELICITABLE")
        return results

    comp_greedy = next(t["computed_greedy"] for t in template_scores
                       if t["name"] == winner["name"])
    log(f"  chosen_template={winner['name']} computed_greedy={comp_greedy:.0%} "
        f"— proceeding to causal sweep")

    # --- cache readout residual at each swept layer (2 forwards total) ---
    sc = cache_layer_outputs(model, ids_s, am_s, to_cpu=True)
    cc = cache_layer_outputs(model, ids_c, am_c, to_cpu=True)
    last_s, last_c = ids_s.shape[1] - 1, ids_c.shape[1] - 1
    h_store = {L: sc[L][:, last_s, :].float() for L in layers}   # [Bs,D]
    h_comp = {L: cc[L][:, last_c, :].float() for L in layers}    # [Bc,D]

    @torch.no_grad()
    def digit_logits(ids, am, delta=None, layer=None):
        if delta is None:
            lg = last_token_logits(model, ids, am)
        else:
            lg = forward_with_add(model, ids, am, layer, ids.shape[1] - 1, delta)
        return lg[:, val_ids].float().cpu().numpy()               # [B,nD]

    def transfer_sel(ids, am, base, delta, layer, ti):
        md = (digit_logits(ids, am, delta, layer) - base).mean(0)  # [nD]
        nD = md.shape[0]
        t = float(md[ti])
        others = float((md.sum() - md[ti]) / (nD - 1))
        return t, t - others

    hb = Heartbeat(len(layers) * len(dtargets) * n_null * 2, "delta_transform",
                   every_sec=15, out_dir=out_dir)
    per_layer = []
    for L in layers:
        hs, hc = h_store[L], h_comp[L]
        cells = {k: {"t": [], "sel": []} for k in ("ss", "cc", "sc", "cs")}
        null_ss, null_cc, coss = [], [], []
        for d in dtargets:
            ti = didx[d]
            d_store, d_comp = _delta_dir(hs, stored_ans, d), _delta_dir(hc, comp_ans, d)
            coss.append(float(F.cosine_similarity(d_store[None], d_comp[None]).item()))
            s_idx = [i for i, a in enumerate(stored_ans) if a != d]
            c_idx = [i for i, a in enumerate(comp_ans) if a != d]
            ish, amsh = ids_s[s_idx], am_s[s_idx]
            ich, amch = ids_c[c_idx], am_c[c_idx]
            base_s, base_c = digit_logits(ish, amsh), digit_logits(ich, amch)
            for key, ids_, am_, base_, delta_ in (
                    ("ss", ish, amsh, base_s, d_store),
                    ("cc", ich, amch, base_c, d_comp),
                    ("sc", ich, amch, base_c, d_store),
                    ("cs", ish, amsh, base_s, d_comp)):
                t, se = transfer_sel(ids_, am_, base_, delta_, L, ti)
                cells[key]["t"].append(t)
                cells[key]["sel"].append(se)
            ns_s = float(d_store.norm().clamp(min=1e-8))
            ns_c = float(d_comp.norm().clamp(min=1e-8))
            row_ss, row_cc = [], []
            for _ in range(n_null):
                r = torch.from_numpy(rng.normal(size=d_store.numel()).astype(np.float32))
                r = r / r.norm().clamp(min=1e-8) * ns_s
                row_ss.append(transfer_sel(ish, amsh, base_s, r, L, ti)[1])
                hb.step()
                r2 = torch.from_numpy(rng.normal(size=d_comp.numel()).astype(np.float32))
                r2 = r2 / r2.norm().clamp(min=1e-8) * ns_c
                row_cc.append(transfer_sel(ich, amch, base_c, r2, L, ti)[1])
                hb.step()
            null_ss.append(row_ss)
            null_cc.append(row_cc)

        def m(key, fld):
            return float(np.mean(cells[key][fld]))

        nss = np.asarray(null_ss).mean(0)
        ncc = np.asarray(null_cc).mean(0)
        p_ss = permutation_pvalue(m("ss", "sel"), nss, "greater")
        p_cc = permutation_pvalue(m("cc", "sel"), ncc, "greater")
        row = {
            "layer": int(L),
            "ss": {"transfer": m("ss", "t"), "selectivity": m("ss", "sel"), "p": float(p_ss)},
            "cc": {"transfer": m("cc", "t"), "selectivity": m("cc", "sel"), "p": float(p_cc)},
            "sc": {"transfer": m("sc", "t"), "selectivity": m("sc", "sel")},
            "cs": {"transfer": m("cs", "t"), "selectivity": m("cs", "sel")},
            "cos_store_comp": float(np.mean(coss)),
            "ratio_cc_over_ss": (m("cc", "sel") / m("ss", "sel")
                                 if abs(m("ss", "sel")) > 1e-6 else float("nan")),
            "cc_sel_per_digit": [float(x) for x in cells["cc"]["sel"]],
        }
        per_layer.append(row)
        log(f"  L{L}: ss sel={row['ss']['selectivity']:+.2f}(p={p_ss:.3f}) "
            f"cc sel={row['cc']['selectivity']:+.2f}(p={p_cc:.3f}) "
            f"ratio={row['ratio_cc_over_ss']:.2f} | cross sc={row['sc']['selectivity']:+.2f} "
            f"cs={row['cs']['selectivity']:+.2f} cos={row['cos_store_comp']:+.2f}")
    hb.done()

    grade = _grade_l3(per_layer)
    results = {
        "stage": "delta_transform", "version": 2, "model_path": model_path,
        "layers": layers, "dtargets": dtargets, "n_null": n_null,
        "n_stored": len(stored_ans), "n_computed": len(comp_ans),
        "store_greedy": store_greedy, "computed_greedy": float(comp_greedy),
        "confounded": False, "behav_gate": behav_gate,
        "template_scores": template_scores, "chosen_template": winner["name"],
        "per_layer": per_layer,
        **grade,
    }
    with open(os.path.join(out_dir, "results_delta_transform.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT delta_transform: {grade['verdict']} | "
        f"template={winner['name']} store_ok={grade['store_control_ok']} "
        f"cc_sig_layers={grade['cc_significant_layers']}/{layers} "
        f"best L{grade['best_layer']} ratio={grade['best_ratio_cc_over_ss']:.2f} | "
        f"computed_greedy={comp_greedy:.0%} cos={grade['mean_cos_store_comp']:+.2f} "
        f"shared={grade['representation_shared']}")
    return results
