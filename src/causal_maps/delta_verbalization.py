"""Verbalization write-back — Arm A (teacher-forced V), Floor 3.

Pre-registered VERBALIZATION_PROTOCOL.md + CAUSAL_MAPS_LOG.md 2026-07-15.
In the G0-validated false-belief world, insert a verbalization line
V = "Alice believes the cube is in {loc}." between STATECHECK and the
question, and measure the CAUSAL LOAD LEDGER:

  lam_t(r) = effect(prototype edit at t) / effect(full natural counterfactual)

Conditions: no-V baseline; V-consistent with edits at the history anchor
(lam_hist — also the conflict cell: edited history vs V), at the V anchor
(lam_cot — reverse conflict), at both; irrelevant-V' (sphere) control for
content-specific shadowing (same word "Paris", different ROLE, similar
position — kills the recency confound); V-position variation; textual
inconsistency baselines (which source wins in TEXT). Readouts: belief_ac and
tell_ac. Frozen gates; graded verdicts per H1-H3; every branch reportable.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .delta_anchor_write import _neutral_states, _resolve
from .delta_structured_workspace import (MARKER, _accuracy, _batch,
                                         _counterfactual, _locations, _rows,
                                         _user)
from .delta_trajectory import _forward, _ld
from .logutil import Heartbeat, log
from .model_utils import input_device, load_model_and_tokenizer
from .nulls import permutation_pvalue

N_NULL = 30
LAYER = 2
G0 = 0.80
GATE_P = 0.04
SOURCE, TARGET = "Paris", "Rome"
LAM_HIGH, LAM_LOW = 0.7, 0.3
READOUTS = ("belief_ac", "tell_ac")


def _reverse_base_verdict(primary):
    """Frozen M5 verdict for the Rome-base quorum/prior discriminator."""
    if min(primary.get("g0", (0.0, 0.0))) < G0:
        return "REVERSE_BASE_INELICITABLE"
    both = primary["both_reverse"]
    if both["paris_acc"] < G0 or not (0.6 <= both["lam"] <= 1.4):
        return "REVERSE_BASE_SANITY_FAIL"
    hist, verbal = primary["history_reverse"], primary["verbal_reverse"]
    if hist["rome_acc"] >= G0 and verbal["rome_acc"] >= G0:
        return "QUORUM_REPLICATES_REVERSE_BASE"
    if hist["paris_acc"] >= G0 and verbal["paris_acc"] >= G0:
        return "PARIS_PRIOR_REPLICATES_REVERSE_BASE"
    return "REVERSE_BASE_MIXED"


def _vline_cube(loc):
    return f"Alice believes the cube is in {loc}."


def _vline_sphere(loc):
    return f"Alice believes the sphere is in {loc}."


def _render_v(tok, row, query, vline, pos="after"):
    user = _user(row, query, "narrative")
    if vline is None:
        text = user
    elif pos == "after":
        text = user.replace(f"{MARKER}.\n", f"{MARKER}. {vline}\n", 1)
    else:                                    # before the marker
        text = user.replace(f" {MARKER}.", f" {vline} {MARKER}.", 1)
    assert vline is None or vline in text, "V insertion failed"
    return tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True)


def _uniform_diff(b1, b2):
    """Single differing position, uniform across rows, between two batches."""
    i1, i2 = b1["ids"], b2["ids"]
    assert i1.shape == i2.shape, "batch shapes differ"
    pos = []
    for r in range(i1.shape[0]):
        d = (i1[r] != i2[r]).nonzero().flatten().tolist()
        assert len(d) == 1, f"row {r}: {len(d)} differing positions"
        pos.append(d[0])
    assert len(set(pos)) == 1, f"non-uniform positions: {pos}"
    return pos[0]


@torch.no_grad()
def run_delta_verbalization(model_path, out_dir, quantization="awq",
                            device_map=None, seed=0, n_null=N_NULL):
    if quantization != "awq" or seed != 0:
        raise ValueError("verbalization Arm A is frozen to 14B-AWQ, seed 0")
    os.makedirs(out_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(_resolve(model_path),
                                          quantization=quantization)
    dev = input_device(model)
    rng = np.random.default_rng(seed)
    hb = Heartbeat(2 * n_null, "verbalization", every_sec=20, out_dir=out_dir)

    rows = (_rows(SOURCE, TARGET, "ac", "train")
            + _rows(SOURCE, TARGET, "ac", "test"))          # n = 10
    nat_rows = _counterfactual(rows, {"ac": TARGET})
    Z = _neutral_states(model, tok, dev, LAYER, (SOURCE, TARGET))
    delta = Z[TARGET] - Z[SOURCE]
    log(f"verbalization: n={len(rows)} |delta|={float(delta.norm()):.2f}")

    def B(rws, query, vline, pos="after"):
        rf = (None if vline is None
              else (lambda row: _render_v(tok, row, query, vline, pos)))
        return _batch(tok, rws, query, "narrative", dev, render_fn=rf)

    result = {"stage": "delta_verbalization", "arm": "A_teacher_forced",
              "layer": LAYER, "seed": seed, "n_null": n_null,
              "n_rows": len(rows), "model_path": model_path, "readouts": {}}

    for query in READOUTS:
        # ---- batches ------------------------------------------------------
        P_noV = B(rows, query, None)
        R_noV = B(nat_rows, query, None)
        P_V = B(rows, query, _vline_cube(SOURCE))
        R_V = B(nat_rows, query, _vline_cube(TARGET))       # consistent Rome
        X_hV = B(nat_rows, query, _vline_cube(SOURCE))      # text: hist=R, V=P
        X_vV = B(rows, query, _vline_cube(TARGET))          # text: hist=P, V=R
        P_Vs = B(rows, query, _vline_sphere(SOURCE))        # irrelevant V'
        R_Vs = B(nat_rows, query, _vline_sphere(SOURCE))
        P_V2 = B(rows, query, _vline_cube(SOURCE), pos="before")
        R_V2 = B(nat_rows, query, _vline_cube(TARGET), pos="before")

        # ---- anchor positions ----------------------------------------------
        a_h_noV = _uniform_diff(P_noV, R_noV)
        a_h_V = _uniform_diff(P_V, X_hV)                    # history token
        a_v_V = _uniform_diff(P_V, X_vV)                    # V token
        a_h_Vs = _uniform_diff(P_Vs, R_Vs)
        a_v_V2 = _uniform_diff(P_V2, B(rows, query, _vline_cube(TARGET),
                                       pos="before"))
        assert a_h_V != a_v_V

        src_exp = _locations(rows, query)                   # all "Paris"
        tgt_exp = _locations(nat_rows, query)               # all "Rome"
        sid = torch.tensor([P_V["amap"][x] for x in src_exp])
        tid = torch.tensor([P_V["amap"][x] for x in tgt_exp])

        def logits(batch, add=None):
            lg, _ = _forward(model, batch["ids"], batch["am"],
                             (batch["marker"],), add=add)
            return lg

        def m(lg):
            return _ld(lg, tid, sid)

        # ---- G0 -------------------------------------------------------------
        lg_PnoV, lg_RnoV = logits(P_noV), logits(R_noV)
        lg_PV, lg_RV = logits(P_V), logits(R_V)
        lg_PVs, lg_RVs = logits(P_Vs), logits(R_Vs)
        lg_PV2, lg_RV2 = logits(P_V2), logits(R_V2)
        g0 = {
            "noV": (_accuracy(lg_PnoV, P_noV, src_exp),
                    _accuracy(lg_RnoV, R_noV, tgt_exp)),
            "V": (_accuracy(lg_PV, P_V, src_exp),
                  _accuracy(lg_RV, R_V, tgt_exp)),
            "Vs": (_accuracy(lg_PVs, P_Vs, src_exp),
                   _accuracy(lg_RVs, R_Vs, tgt_exp)),
            "V2": (_accuracy(lg_PV2, P_V2, src_exp),
                   _accuracy(lg_RV2, R_V2, tgt_exp)),
        }
        out = {"g0": {k: [float(a), float(b)] for k, (a, b) in g0.items()},
               "anchors": {"a_h_noV": a_h_noV, "a_h_V": a_h_V,
                           "a_v_V": a_v_V, "a_h_Vs": a_h_Vs}}
        core_ok = min(g0["V"]) >= G0 and min(g0["noV"]) >= G0
        if not core_ok:
            out["verdict"] = "VERBALIZATION_INELICITABLE"
            result["readouts"][query] = out
            log(f"  [{query}] G0 fail {g0}")
            continue

        # ---- textual inconsistency priors (no edits) ------------------------
        lg_XhV, lg_XvV = logits(X_hV), logits(X_vV)
        out["textual_prior"] = {
            "hist_R_V_P": {"acc_hist": _accuracy(lg_XhV, X_hV, tgt_exp),
                           "acc_V": _accuracy(lg_XhV, X_hV, src_exp)},
            "hist_P_V_R": {"acc_hist": _accuracy(lg_XvV, X_vV, src_exp),
                           "acc_V": _accuracy(lg_XvV, X_vV, tgt_exp)},
        }

        # ---- causal ledger ---------------------------------------------------
        def lam(P_lg_clean, batch, anchor, ref_effect, d=delta):
            lg = logits(batch, add=(LAYER, anchor, d))
            eff = float((m(lg) - m(P_lg_clean)).mean())
            acc = _accuracy(lg, batch, tgt_exp)
            return {"effect": eff, "lam": eff / ref_effect,
                    "target_acc": float(acc)}

        eff_noV = float((m(lg_RnoV) - m(lg_PnoV)).mean())
        eff_V = float((m(lg_RV) - m(lg_PV)).mean())
        eff_Vs = float((m(lg_RVs) - m(lg_PVs)).mean())
        eff_V2 = float((m(lg_RV2) - m(lg_PV2)).mean())
        out["natural_effects"] = {"noV": eff_noV, "V": eff_V,
                                  "Vs": eff_Vs, "V2": eff_V2}

        out["cond1_hist_noV"] = lam(lg_PnoV, P_noV, a_h_noV, eff_noV)
        out["cond2_hist_underV"] = lam(lg_PV, P_V, a_h_V, eff_V)
        out["cond3_cot"] = lam(lg_PV, P_V, a_v_V, eff_V)
        # cond 4: both edits in one forward
        lg_both, _ = _forward(model, P_V["ids"], P_V["am"], (P_V["marker"],),
                              add=[(LAYER, a_h_V, delta),
                                   (LAYER, a_v_V, delta)])
        out["cond4_both"] = {
            "effect": float((m(lg_both) - m(lg_PV)).mean()),
            "lam": float((m(lg_both) - m(lg_PV)).mean()) / eff_V,
            "target_acc": float(_accuracy(lg_both, P_V, tgt_exp))}
        out["cond6_hist_underVs"] = lam(lg_PVs, P_Vs, a_h_Vs, eff_Vs)
        out["cond7_cot_beforeMarker"] = lam(lg_PV2, P_V2, a_v_V2, eff_V2)

        # ---- nulls (belief readout only; primary) ----------------------------
        if query == "belief_ac":
            norm = float(delta.norm().clamp(min=1e-8))
            nulls_h, nulls_v = [], []
            base = m(lg_PV)
            for _ in range(n_null):
                r = torch.from_numpy(
                    rng.normal(size=delta.numel()).astype(np.float32))
                r = r / r.norm().clamp(min=1e-8) * norm
                nulls_h.append(float((m(logits(P_V, add=(LAYER, a_h_V, r)))
                                      - base).mean()))
                nulls_v.append(float((m(logits(P_V, add=(LAYER, a_v_V, r)))
                                      - base).mean()))
                hb.step(); hb.step()
            out["null_p_hist"] = float(permutation_pvalue(
                out["cond2_hist_underV"]["effect"], np.asarray(nulls_h),
                "greater"))
            out["null_p_cot"] = float(permutation_pvalue(
                out["cond3_cot"]["effect"], np.asarray(nulls_v), "greater"))

        log(f"  [{query}] lam_hist(noV)={out['cond1_hist_noV']['lam']:.3f} "
            f"lam_hist(V)={out['cond2_hist_underV']['lam']:.3f} "
            f"lam_cot={out['cond3_cot']['lam']:.3f} "
            f"lam_both={out['cond4_both']['lam']:.3f} "
            f"lam_hist(V')={out['cond6_hist_underVs']['lam']:.3f} "
            f"lam_cot(beforeM)={out['cond7_cot_beforeMarker']['lam']:.3f}")
        result["readouts"][query] = out
    hb.done()

    # ---- verdicts on the primary readout -----------------------------------
    b = result["readouts"].get("belief_ac", {})
    if b.get("verdict") == "VERBALIZATION_INELICITABLE" or "cond3_cot" not in b:
        result["verdict"] = "VERBALIZATION_INELICITABLE"
    else:
        lam_cot = b["cond3_cot"]["lam"]
        lam_hist = b["cond2_hist_underV"]["lam"]
        lam_hist_ctrl = b["cond6_hist_underVs"]["lam"]
        h1 = (lam_cot >= LAM_HIGH and b["cond3_cot"]["target_acc"] >= G0
              and b.get("null_p_cot", 1.0) < GATE_P)
        h2 = lam_hist <= LAM_LOW and lam_hist_ctrl >= LAM_HIGH
        if h1 and h2:
            verdict = "WRITE_BACK_SHADOWING"
        elif lam_cot >= LAM_HIGH and lam_hist >= LAM_HIGH:
            verdict = "WRITE_BACK_REDUNDANT"
        elif lam_cot < LAM_LOW:
            verdict = "COT_DECORATIVE"
        else:
            verdict = "LEDGER_MIXED"
        result["gates"] = {"H1_register": bool(h1), "H2_migration": bool(h2)}
        result["verdict"] = verdict
    with open(os.path.join(out_dir, "results_delta_verbalization.json"),
              "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(f"VERDICT verbalization armA: {result['verdict']}")
    return result


@torch.no_grad()
def run_reverse_base_quorum(model_path, out_dir, quantization="awq",
                            device_map=None, seed=0, model=None, tok=None,
                            clean_rows=None):
    """M5: resolve quorum versus Paris-prior with the base value reversed.

    The base prompt has Rome at both the history and verbalization anchors.
    A Rome->Paris write is applied at one anchor or both. The untouched-witness
    account predicts Rome after either single edit and Paris only after the
    double edit. A Paris prior predicts Paris after both single edits.

    This cell reuses the already null-validated L2 neutral-carrier write. It
    adds no new direction search, coefficient, layer sweep, or random null.
    """
    if quantization != "awq" or seed != 0:
        raise ValueError("reverse-base M5 is frozen to 14B-AWQ, seed 0")
    os.makedirs(out_dir, exist_ok=True)
    if model is None or tok is None:
        model, tok = load_model_and_tokenizer(
            _resolve(model_path), quantization=quantization,
            device_map=device_map)
    dev = input_device(model)
    rows = (list(clean_rows) if clean_rows is not None else
            _rows(SOURCE, TARGET, "ac", "train") +
            _rows(SOURCE, TARGET, "ac", "test"))
    rome_rows = _counterfactual(rows, {"ac": TARGET})
    Z = _neutral_states(model, tok, dev, LAYER, (SOURCE, TARGET))
    reverse_delta = Z[SOURCE] - Z[TARGET]
    result = {"stage": "delta_verbalization_reverse_base",
              "protocol": "M5_frozen_2026-07-21",
              "model_path": model_path, "quantization": quantization,
              "layer": LAYER, "seed": seed, "n_rows": len(rows),
              "readouts": {}}

    def B(rws, query, vline):
        return _batch(
            tok, rws, query, "narrative", dev,
            render_fn=lambda row: _render_v(tok, row, query, vline, "after"))

    for query in READOUTS:
        paris = B(rows, query, _vline_cube(SOURCE))       # P history, P V
        rome = B(rome_rows, query, _vline_cube(TARGET))   # R history, R V
        hist_p_v_r = B(rows, query, _vline_cube(TARGET))  # P history, R V
        hist_r_v_p = B(rome_rows, query, _vline_cube(SOURCE))
        a_h = _uniform_diff(rome, hist_p_v_r)
        a_v = _uniform_diff(rome, hist_r_v_p)
        assert a_h != a_v

        paris_expected = _locations(rows, query)
        rome_expected = _locations(rome_rows, query)
        sid = torch.tensor([rome["amap"][x] for x in rome_expected])
        tid = torch.tensor([rome["amap"][x] for x in paris_expected])

        def forward(batch, add=None):
            logits, _ = _forward(model, batch["ids"], batch["am"],
                                 (batch["marker"],), add=add)
            return logits

        def margin(logits):
            return _ld(logits, tid, sid)  # Paris minus Rome

        lg_paris, lg_rome = forward(paris), forward(rome)
        natural_rows = margin(lg_paris) - margin(lg_rome)
        natural_effect = float(natural_rows.mean())

        def score(logits):
            effect_rows = margin(logits) - margin(lg_rome)
            return {
                "effect": float(effect_rows.mean()),
                "effect_rows": effect_rows.detach().cpu().tolist(),
                "lam": float(effect_rows.mean()) / natural_effect,
                "paris_acc": float(_accuracy(
                    logits, rome, paris_expected)),
                "rome_acc": float(_accuracy(
                    logits, rome, rome_expected)),
                "positive_fraction": float((effect_rows > 0).float().mean()),
            }

        lg_hist = forward(rome, add=(LAYER, a_h, reverse_delta))
        lg_verbal = forward(rome, add=(LAYER, a_v, reverse_delta))
        lg_both = forward(rome, add=[(LAYER, a_h, reverse_delta),
                                    (LAYER, a_v, reverse_delta)])
        lg_text_hist = forward(hist_p_v_r)
        lg_text_verbal = forward(hist_r_v_p)
        out = {
            "g0": [float(_accuracy(lg_paris, paris, paris_expected)),
                   float(_accuracy(lg_rome, rome, rome_expected))],
            "anchors": {"history": a_h, "verbalization": a_v},
            "natural_effect": natural_effect,
            "natural_effect_rows": natural_rows.detach().cpu().tolist(),
            "history_reverse": score(lg_hist),
            "verbal_reverse": score(lg_verbal),
            "both_reverse": score(lg_both),
            "textual_conflicts": {
                "history_Paris_V_Rome": {
                    "paris_acc": float(_accuracy(
                        lg_text_hist, hist_p_v_r, paris_expected)),
                    "rome_acc": float(_accuracy(
                        lg_text_hist, hist_p_v_r, rome_expected)),
                },
                "history_Rome_V_Paris": {
                    "paris_acc": float(_accuracy(
                        lg_text_verbal, hist_r_v_p, paris_expected)),
                    "rome_acc": float(_accuracy(
                        lg_text_verbal, hist_r_v_p, rome_expected)),
                },
            },
        }
        out["verdict"] = _reverse_base_verdict(out)
        result["readouts"][query] = out
        log(f"  [reverse-base/{query}] hist Rome={out['history_reverse']['rome_acc']:.0%} "
            f"V Rome={out['verbal_reverse']['rome_acc']:.0%} "
            f"both Paris={out['both_reverse']['paris_acc']:.0%} "
            f"both lam={out['both_reverse']['lam']:.3f} verdict={out['verdict']}")

    result["verdict"] = result["readouts"]["belief_ac"]["verdict"]
    with open(os.path.join(out_dir, "results_reverse_base_quorum.json"),
              "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(f"VERDICT reverse-base M5: {result['verdict']}")
    return result
