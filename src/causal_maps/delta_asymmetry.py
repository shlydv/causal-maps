"""Route–binding asymmetry: redundancy vs incomplete basis vs bottleneck.

ASYMMETRY_PROTOCOL.md — study why route necessity ≫ bind necessity.
Do NOT print a “bus” claim; only account verdicts A/B/C.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from . import completion_pairs
from .delta_crossskill import _variable_directions
from .delta_select import SELECT_TEMPLATES, _encode_pool, _pair_pool, _render
from .direction_transfer import PRIMARY_LAYER
from .logutil import Heartbeat, log
from .model_utils import (input_device, last_token_logits,
                          load_model_and_tokenizer, single_token_id)
from .nulls import permutation_pvalue
from .patching import (cache_layer_outputs, forward_with_mean_knock,
                       forward_with_project)
from .tensorize import _anchor_token_index

N_NULL = 100
BEHAV_GATE = 0.80
N_BIND = 12
N_ROUTE = 8
N_UTIL = 8
LAYER = PRIMARY_LAYER
VERSION = 1
ROUTE_TMPL_NAME = "value_of"
N_TRAIN_ROUTE = 8
EPS = 1e-6


def _value_of_tmpl():
    for t in SELECT_TEMPLATES:
        if t["name"] == ROUTE_TMPL_NAME:
            return t
    raise RuntimeError(f"missing {ROUTE_TMPL_NAME}")


def _basis_1d(v: torch.Tensor) -> torch.Tensor:
    u = v.float().reshape(-1)
    return (u / u.norm().clamp(min=1e-8)).unsqueeze(1)


def _basis_span(dirs) -> torch.Tensor:
    M = torch.stack([d.float().reshape(-1) for d in dirs], dim=1)
    Q, _ = torch.linalg.qr(M, mode="reduced")
    return Q


def _extract_routing_delta(model, tok, device, seed, layer):
    tmpl = _value_of_tmpl()
    rng = np.random.default_rng(seed + 17)
    pairs = _pair_pool(tok, tmpl, rng, N_TRAIN_ROUTE)
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
    return (c1[layer][:, fpos, :].float().mean(0)
            - c0[layer][:, fpos, :].float().mean(0))


@torch.no_grad()
def _greedy_id(model, text, tok, device):
    e = tok.encode(text, add_special_tokens=False)
    ii = torch.tensor([e], dtype=torch.long, device=device)
    return int(last_token_logits(model, ii, torch.ones_like(ii)).argmax(-1).item())


@torch.no_grad()
def _pref_vs_pool(logits, gold_idx, pool_ids_t):
    B = logits.shape[0]
    g = logits[torch.arange(B, device=logits.device), gold_idx]
    mean_all = logits[:, pool_ids_t].mean(-1)
    return (g - mean_all).float().cpu().numpy()


def run_delta_asymmetry(model_path, out_dir, quantization="8bit", device_map=None,
                        layer=LAYER, seed=0, n_null=N_NULL):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    device = input_device(model)
    rng = np.random.default_rng(seed)

    log(f"delta_asymmetry v{VERSION}: layer={layer} n_null={n_null} "
        f"(accounts A/B/C only — no bus claim)")

    Delta, values = _variable_directions(model, tok, layer, seed)
    d_route = _extract_routing_delta(model, tok, device, seed, layer)
    vals = list(values)
    assert len(vals) >= 4
    val_ids = torch.tensor([single_token_id(tok, v) for v in vals],
                           dtype=torch.long, device=device)
    Ddim = int(d_route.numel())
    k_span = len(vals)
    log(f"  nV={len(vals)} ‖Δ_route‖={float(d_route.norm()):.2f}")

    # --- G0 ---
    bind_rows = []
    for v in vals:
        user = f"Let X = {v}. What is the value of X?"
        text = tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True) + "X ="
        off = text.find("Let X = ") + len("Let X = ")
        slot = _anchor_token_index(tok, text, off)
        if slot is None:
            continue
        gold = single_token_id(tok, v, leading_space=True)
        ok = _greedy_id(model, text, tok, device) == gold
        bind_rows.append({"v": v, "text": text, "slot": slot, "gold": gold, "ok": ok})
    behav_bind = sum(r["ok"] for r in bind_rows) / max(len(bind_rows), 1)

    tmpl = _value_of_tmpl()
    route_pairs = _pair_pool(tok, tmpl, rng, N_ROUTE)[:N_ROUTE]
    route_rows = []
    r_ok0 = r_ok1 = 0
    for va, vb in route_pairs:
        t0, _, _ = _render(tok, tmpl, va, vb, 0)
        t1, _, _ = _render(tok, tmpl, va, vb, 1)
        g0 = single_token_id(tok, vb, leading_space=True)
        g1 = single_token_id(tok, va, leading_space=True)
        ok0 = _greedy_id(model, t0, tok, device) == g0
        ok1 = _greedy_id(model, t1, tok, device) == g1
        r_ok0 += int(ok0); r_ok1 += int(ok1)
        route_rows.append({"va": va, "vb": vb, "t0": t0, "t1": t1, "g0": g0, "g1": g1})
    behav_route = 0.5 * (r_ok0 + r_ok1) / max(len(route_rows), 1)
    g0_pass = behav_bind >= BEHAV_GATE and behav_route >= BEHAV_GATE
    log(f"  G0: bind={behav_bind:.0%} route={behav_route:.0%} pass={g0_pass}")

    if not g0_pass:
        results = {
            "stage": "delta_asymmetry", "version": VERSION, "model_path": model_path,
            "layer": layer, "verdict": "ASYM_INELICITABLE",
            "G0": {"bind": behav_bind, "route": behav_route, "pass": False},
        }
        with open(os.path.join(out_dir, "results_delta_asymmetry.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        log("VERDICT delta_asymmetry: ASYM_INELICITABLE")
        return results

    # Bind batch
    bind_ok_rows = [r for r in bind_rows if r["ok"]] or bind_rows
    lens = [len(tok.encode(r["text"], add_special_tokens=False)) for r in bind_ok_rows]
    maj = max(set(lens), key=lens.count)
    bind_ok_rows = [r for r, L in zip(bind_ok_rows, lens) if L == maj][:N_BIND]
    assert len(bind_ok_rows) >= 4 and len({r["slot"] for r in bind_ok_rows}) == 1
    b_slot = bind_ok_rows[0]["slot"]
    b_ids = torch.tensor([tok.encode(r["text"], add_special_tokens=False)
                          for r in bind_ok_rows], dtype=torch.long, device=device)
    b_am = torch.ones_like(b_ids)
    b_gold = torch.tensor([r["gold"] for r in bind_ok_rows],
                          dtype=torch.long, device=device)
    b_true_v = [r["v"] for r in bind_ok_rows]

    # Route batch
    r_texts0 = [r["t0"] for r in route_rows]
    r_texts1 = [r["t1"] for r in route_rows]
    ids0, am0, fpos = _encode_pool(
        tok, r_texts0, [tmpl["key"]] * len(r_texts0), [0] * len(r_texts0), device)
    ids1, am1, fpos1 = _encode_pool(
        tok, r_texts1, [tmpl["key"]] * len(r_texts1), [1] * len(r_texts1), device)
    assert fpos == fpos1
    g0s = torch.tensor([r["g0"] for r in route_rows], dtype=torch.long, device=device)
    g1s = torch.tensor([r["g1"] for r in route_rows], dtype=torch.long, device=device)
    va_ids = torch.tensor([single_token_id(tok, r["va"], leading_space=True)
                           for r in route_rows], dtype=torch.long, device=device)
    n_rr = len(route_rows)
    idx_r = torch.arange(n_rr, device=device)

    # Completion utility
    comp_pairs = completion_pairs.make_completion_pairs(
        20, seed=seed + 3, tok=tok, families=["explicit_A"])
    comp_rows = []
    for p in comp_pairs:
        try:
            gold = single_token_id(tok, p["answer_clean"], leading_space=True)
        except ValueError:
            continue
        off_bit = p["anchors"].get("bit_slot")
        if off_bit is None:
            continue
        slot = _anchor_token_index(tok, p["clean_text"], off_bit)
        if slot is None:
            continue
        comp_rows.append({"text": p["clean_text"], "gold": gold, "slot": slot})
        if len(comp_rows) >= N_UTIL:
            break
    clens = [len(tok.encode(r["text"], add_special_tokens=False)) for r in comp_rows]
    cmaj = max(set(clens), key=clens.count)
    comp_rows = [r for r, L in zip(comp_rows, clens) if L == cmaj][:N_UTIL]
    assert len(comp_rows) >= 4 and len({r["slot"] for r in comp_rows}) == 1
    c_slot = comp_rows[0]["slot"]
    c_ids = torch.tensor([tok.encode(r["text"], add_special_tokens=False)
                          for r in comp_rows], dtype=torch.long, device=device)
    c_am = torch.ones_like(c_ids)
    c_gold = torch.tensor([r["gold"] for r in comp_rows], dtype=torch.long, device=device)

    def acc(lg, gold):
        return float((lg.argmax(-1) == gold).float().mean().item())

    def pref_bind(lg):
        return float(_pref_vs_pool(lg, b_gold, val_ids).mean())

    def project(ids, am, pos, basis):
        return forward_with_project(model, ids, am, layer, pos, basis, alpha=1.0)

    def knock(ids, am, pos):
        return forward_with_mean_knock(model, ids, am, layer, pos)

    # ----- BIND -----
    lg_clean = last_token_logits(model, b_ids, b_am)
    pref_clean = pref_bind(lg_clean)
    acc_clean = acc(lg_clean, b_gold)

    def bind_pref_dir():
        prefs = []
        for i, v in enumerate(b_true_v):
            lg = project(b_ids[i:i + 1], b_am[i:i + 1], b_slot, _basis_1d(Delta[v]))
            prefs.append(float(_pref_vs_pool(lg, b_gold[i:i + 1], val_ids).mean()))
        return float(np.mean(prefs))

    span = _basis_span([Delta[v] for v in vals])
    pref_dir = bind_pref_dir()
    pref_span = pref_bind(project(b_ids, b_am, b_slot, span))
    pref_knock = pref_bind(knock(b_ids, b_am, b_slot))
    drop_dir = pref_clean - pref_dir
    drop_span = pref_clean - pref_span
    drop_knock_b = pref_clean - pref_knock
    R_span = drop_span / max(drop_dir, EPS)
    R_knock = drop_knock_b / max(drop_dir, EPS)

    # null: random k-D span drops
    hb = Heartbeat(n_null, "asym_bind_null", every_sec=15, out_dir=out_dir)
    null_span, null_knock = [], []
    for _ in range(n_null):
        R = torch.from_numpy(rng.normal(size=(Ddim, k_span)).astype(np.float32))
        Q, _ = torch.linalg.qr(R, mode="reduced")
        null_span.append(pref_clean - pref_bind(project(b_ids, b_am, b_slot, Q)))
        # knock has no random analog of equal form — use random 1D project as soft null for knock magnitude
        r = torch.from_numpy(rng.normal(size=Ddim).astype(np.float32))
        null_knock.append(pref_clean - pref_bind(project(b_ids, b_am, b_slot, _basis_1d(r))))
        hb.step()
    hb.done()
    p_span = permutation_pvalue(drop_span, np.array(null_span), "greater")
    p_knock_b = permutation_pvalue(drop_knock_b, np.array(null_knock), "greater")

    A1 = bool(R_span >= 1.5 and drop_span > 0 and p_span < 0.01)
    B1 = bool(R_span < 1.5 and R_knock >= 2.0 and drop_knock_b > 0 and p_knock_b < 0.01)
    log(f"  BIND: drop_dir={drop_dir:+.2f} drop_span={drop_span:+.2f} "
        f"drop_knock={drop_knock_b:+.2f}")
    log(f"  R_span={R_span:.2f}(p={p_span:.3f}) R_knock={R_knock:.2f}(p={p_knock_b:.3f}) "
        f"A1={A1} B1={B1}")

    # ----- ROUTE -----
    def route_gap(lg0, lg1):
        p0 = (lg0[idx_r, g0s] - lg0[idx_r, va_ids]).float().mean().item()
        p1 = (lg1[idx_r, g1s] - lg1[idx_r, g0s]).float().mean().item()
        return 0.5 * (p0 + p1)

    gap_clean = route_gap(last_token_logits(model, ids0, am0),
                          last_token_logits(model, ids1, am1))
    gap_dir = route_gap(project(ids0, am0, fpos, _basis_1d(d_route)),
                        project(ids1, am1, fpos, _basis_1d(d_route)))
    gap_knock = route_gap(knock(ids0, am0, fpos), knock(ids1, am1, fpos))
    drop_route = gap_clean - gap_dir
    drop_knock_r = gap_clean - gap_knock
    beta_route = drop_route / max(drop_knock_r, EPS)
    beta_bind = drop_dir / max(drop_knock_b, EPS)

    hb = Heartbeat(n_null, "asym_route_null", every_sec=15, out_dir=out_dir)
    null_rd = []
    for _ in range(n_null):
        r = torch.from_numpy(rng.normal(size=Ddim).astype(np.float32))
        g = route_gap(project(ids0, am0, fpos, _basis_1d(r)),
                      project(ids1, am1, fpos, _basis_1d(r)))
        null_rd.append(gap_clean - g)
        hb.step()
    hb.done()
    p_route = permutation_pvalue(drop_route, np.array(null_rd), "greater")

    C1 = bool(beta_route >= 0.7 and beta_bind <= 0.4 and drop_route > 0 and p_route < 0.01)
    log(f"  ROUTE: drop_dir={drop_route:+.2f} drop_knock={drop_knock_r:+.2f} "
        f"β_route={beta_route:.2f} β_bind={beta_bind:.2f} p={p_route:.3f} C1={C1}")

    # ----- Utility -----
    comp_clean = acc(last_token_logits(model, c_ids, c_am), c_gold)
    drop_comp_bind_knock = comp_clean - acc(knock(c_ids, c_am, c_slot), c_gold)
    drop_comp_route_knock = drop_comp_bind_knock  # same mean-knock at bit_slot
    # Route-dir project on completion as extra (report); U uses mean-knock at bit
    drop_comp_route_dir = comp_clean - acc(
        project(c_ids, c_am, c_slot, _basis_1d(d_route)), c_gold)

    def u_ok(drop_util, drop_target_acc):
        if drop_util <= 0.10:
            return True
        return drop_util <= 0.25 * max(drop_target_acc, 0.0)

    acc_knock_b = acc(knock(b_ids, b_am, b_slot), b_gold)
    drop_knock_b_acc = acc_clean - acc_knock_b
    acc_r_clean = 0.5 * (acc(last_token_logits(model, ids0, am0), g0s)
                         + acc(last_token_logits(model, ids1, am1), g1s))
    acc_r_knock = 0.5 * (acc(knock(ids0, am0, fpos), g0s)
                         + acc(knock(ids1, am1, fpos), g1s))
    drop_knock_r_acc = acc_r_clean - acc_r_knock

    U = (u_ok(drop_comp_bind_knock, drop_knock_b_acc)
         and u_ok(drop_comp_route_knock, drop_knock_r_acc))
    log(f"  U: compΔ knock={drop_comp_bind_knock:+.0%} "
        f"route-dir={drop_comp_route_dir:+.0%} pass={U}")

    # ----- verdict (no bus) -----
    if C1 and B1 and U:
        verdict = "ASYM_BOTTLENECK_AND_INCOMPLETE"
    elif C1 and U:
        verdict = "ASYM_BOTTLENECK"
    elif A1 and U:
        verdict = "ASYM_SPAN_REDUNDANCY"
    elif B1 and U:
        verdict = "ASYM_INCOMPLETE_BASIS"
    elif A1 or B1 or C1:
        verdict = "ASYM_UNCLEAR"  # account signal but U fail
        if not U:
            verdict = "ASYM_UNCLEAR"
    else:
        verdict = "ASYM_UNCLEAR"

    # refine: if accounts fire without U
    if (A1 or B1 or C1) and not U:
        verdict = "ASYM_UNCLEAR"

    results = {
        "stage": "delta_asymmetry", "version": VERSION, "model_path": model_path,
        "layer": layer, "n_null": n_null,
        "G0": {"bind": behav_bind, "route": behav_route, "pass": True},
        "bind": {
            "pref_clean": pref_clean, "pref_dir": pref_dir, "pref_span": pref_span,
            "pref_knock": pref_knock,
            "drop_dir": drop_dir, "drop_span": drop_span, "drop_knock": drop_knock_b,
            "R_span": R_span, "R_knock": R_knock,
            "p_span": float(p_span), "p_knock": float(p_knock_b),
            "A1": A1, "B1": B1, "slot": int(b_slot), "n": len(bind_ok_rows),
        },
        "route": {
            "gap_clean": gap_clean, "gap_dir": gap_dir, "gap_knock": gap_knock,
            "drop_dir": drop_route, "drop_knock": drop_knock_r,
            "beta_route": beta_route, "beta_bind": beta_bind,
            "p_dir": float(p_route), "C1": C1,
            "flag_pos": int(fpos), "n": len(route_rows),
        },
        "utility": {
            "comp_clean": comp_clean,
            "drop_comp_bind_knock": drop_comp_bind_knock,
            "drop_comp_route_knock": drop_comp_route_knock,
            "drop_comp_route_dir": drop_comp_route_dir,
            "drop_knock_bind_acc": drop_knock_b_acc,
            "drop_knock_route_acc": drop_knock_r_acc,
            "U": U,
        },
        "accounts": {"A_span_redundancy": A1, "B_incomplete_basis": B1,
                     "C_bottleneck_asymmetry": C1, "U": U},
        "verdict": verdict,
        "framing": {
            "ASYM_BOTTLENECK_AND_INCOMPLETE": "route≈site bottleneck; bind dirs miss most of site write",
            "ASYM_BOTTLENECK": "route direction ≈ flag site; bind direction ≪ val site",
            "ASYM_SPAN_REDUNDANCY": "extracted Δ span ≫ single Δ_v",
            "ASYM_INCOMPLETE_BASIS": "site knockout ≫ extracted dirs; span not redundant",
            "ASYM_UNCLEAR": "no account cleared with utility",
            "ASYM_INELICITABLE": "G0 fail",
        }[verdict],
        "elevation": "bus_interpretation_forbidden_until_A_or_C_and_B",
        "hard_stop": "no_layer_expansion_no_bus_claim",
    }
    path = os.path.join(out_dir, "results_delta_asymmetry.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT delta_asymmetry: {verdict} | A={A1} B={B1} C={C1} U={U}")
    return results
