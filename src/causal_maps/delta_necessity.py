"""Causal necessity: site-matched directional ablation of bind/route dirs.

NECESSITY_PROTOCOL.md v1.1 (Sahil approved). Sufficiency → compatibility → necessity.

PRIMARY evidence = selectivity + utility (S1/S2/U). Target magnitude T secondary.
α-sweep support; failure demotes NECESSARY → PARTIAL.
Site-local only; fail → SUFFICIENT_ONLY. No layer expansion / rescue.

Gates frozen in NECESSITY_PROTOCOL.md.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from . import completion_pairs
from .delta_crossskill import _variable_directions
from .delta_select import SELECT_TEMPLATES, _encode_pool, _pair_pool, _render
from .direction_transfer import PRIMARY_LAYER
from .logutil import Heartbeat, log
from .model_utils import (input_device, last_token_logits,
                          load_model_and_tokenizer, single_token_id)
from .nulls import permutation_pvalue
from .patching import cache_layer_outputs, forward_with_project
from .tensorize import _anchor_token_index

N_NULL = 100
BEHAV_GATE = 0.80
N_BIND = 12
N_ROUTE = 8
N_UTIL = 8
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
LAYER = PRIMARY_LAYER
VERSION = 1
ROUTE_TMPL_NAME = "value_of"
N_TRAIN_ROUTE = 8


def _value_of_tmpl():
    for t in SELECT_TEMPLATES:
        if t["name"] == ROUTE_TMPL_NAME:
            return t
    raise RuntimeError(f"missing {ROUTE_TMPL_NAME}")


def _basis_1d(v: torch.Tensor) -> torch.Tensor:
    u = v.float().reshape(-1)
    return (u / u.norm().clamp(min=1e-8)).unsqueeze(1)  # [D,1]


def _basis_span(dirs) -> torch.Tensor:
    """Orthonormal column basis [D, K] for span of dirs ([D] each)."""
    M = torch.stack([d.float().reshape(-1) for d in dirs], dim=1)  # [D, K]
    Q, _ = torch.linalg.qr(M, mode="reduced")
    return Q


def _spearman(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return 0.0
    rho, _ = spearmanr(xs, ys)
    return float(rho) if np.isfinite(rho) else 0.0


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
    d = c1[layer][:, fpos, :].float().mean(0) - c0[layer][:, fpos, :].float().mean(0)
    return d


@torch.no_grad()
def _greedy_id(model, text, tok, device):
    e = tok.encode(text, add_special_tokens=False)
    ii = torch.tensor([e], dtype=torch.long, device=device)
    return int(last_token_logits(model, ii, torch.ones_like(ii)).argmax(-1).item())


@torch.no_grad()
def _pref_vs_pool(logits, gold_idx, pool_ids_t):
    """mean over batch of logit(gold) − mean_{others in pool} logit."""
    # logits [B,V]; gold_idx [B]; pool_ids_t [nV]
    B = logits.shape[0]
    g = logits[torch.arange(B), gold_idx]
    pool = logits[:, pool_ids_t]  # [B, nV]
    # exclude gold from mean others when present
    mean_all = pool.mean(-1)
    # approximate: pref = gold - mean(pool); if gold in pool this understates — OK
    return (g - mean_all).float().cpu().numpy()


def _util_pass(drop_util_acc, drop_target_pref_or_acc, target_is_pref=True):
    """U: drop_util ≤ 0.25×drop_target OR drop_util ≤ 10pp accuracy."""
    # util always accuracy pp (0–100 scale as fraction*100 or fraction — use fraction)
    # freeze: both drops as fractions in [0,1] for acc; pref drops unbounded —
    # for pref targets use OR with 10pp only on util side: drop_util_acc ≤ 0.10
    # OR drop_util_acc ≤ 0.25 * max(drop_target, 0) when target also acc.
    if drop_util_acc <= 0.10:
        return True
    if not target_is_pref:
        return drop_util_acc <= 0.25 * max(drop_target_pref_or_acc, 0.0)
    # target is preference: only the absolute 10pp util slack applies as primary,
    # plus relative if we convert — protocol: OR 10pp. Prefer absolute for pref.
    return drop_util_acc <= 0.10


def run_delta_necessity(model_path, out_dir, quantization="8bit", device_map=None,
                        layer=LAYER, seed=0, n_null=N_NULL):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    device = input_device(model)
    rng = np.random.default_rng(seed)

    log(f"delta_necessity v{VERSION}: layer={layer} alphas={list(ALPHAS)} "
        f"n_null={n_null} (site-local only; no locus fishing)")

    # --- extract dirs ---
    Delta, values = _variable_directions(model, tok, layer, seed)
    d_route = _extract_routing_delta(model, tok, device, seed, layer)
    vals = list(values)
    assert len(vals) >= 4
    vidx = {v: i for i, v in enumerate(vals)}
    val_ids = torch.tensor([single_token_id(tok, v) for v in vals],
                           dtype=torch.long, device=device)
    log(f"  nV={len(vals)} ‖Δ_route‖={float(d_route.norm()):.2f}")

    # ========== G0: native behavioral ==========
    # Variable retrieve
    bind_ok = 0
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
        bind_ok += int(ok)
        bind_rows.append({"v": v, "text": text, "slot": slot, "gold": gold, "ok": ok})
    behav_bind = bind_ok / max(len(bind_rows), 1)

    tmpl = _value_of_tmpl()
    route_pairs = _pair_pool(tok, tmpl, rng, N_ROUTE)
    route_pairs = route_pairs[:N_ROUTE]
    r_ok0 = r_ok1 = 0
    route_rows = []
    for va, vb in route_pairs:
        t0, k, _ = _render(tok, tmpl, va, vb, 0)
        t1, _, _ = _render(tok, tmpl, va, vb, 1)
        g0 = single_token_id(tok, vb, leading_space=True)
        g1 = single_token_id(tok, va, leading_space=True)
        ok0 = _greedy_id(model, t0, tok, device) == g0
        ok1 = _greedy_id(model, t1, tok, device) == g1
        r_ok0 += int(ok0); r_ok1 += int(ok1)
        route_rows.append({"va": va, "vb": vb, "t0": t0, "t1": t1,
                           "g0": g0, "g1": g1})
    n_r = max(len(route_rows), 1)
    behav_route = 0.5 * (r_ok0 + r_ok1) / n_r
    g0_pass = behav_bind >= BEHAV_GATE and behav_route >= BEHAV_GATE
    log(f"  G0: bind={behav_bind:.0%} route={behav_route:.0%} pass={g0_pass}")

    if not g0_pass:
        results = {
            "stage": "delta_necessity", "version": VERSION, "model_path": model_path,
            "layer": layer, "verdict": "NEC_INELICITABLE",
            "G0": {"bind": behav_bind, "route": behav_route, "pass": False},
            "stop_reason": "G0_native_behav",
        }
        with open(os.path.join(out_dir, "results_delta_necessity.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        log("VERDICT delta_necessity: NEC_INELICITABLE")
        return results

    # Keep only successful bind rows for causal (or all if enough)
    bind_ok_rows = [r for r in bind_rows if r["ok"]] or bind_rows
    bind_ok_rows = bind_ok_rows[:N_BIND]
    # length-uniform bind batch
    lens = [len(tok.encode(r["text"], add_special_tokens=False)) for r in bind_ok_rows]
    maj = max(set(lens), key=lens.count)
    bind_ok_rows = [r for r, L in zip(bind_ok_rows, lens) if L == maj][:N_BIND]
    assert len(bind_ok_rows) >= 4
    assert len({r["slot"] for r in bind_ok_rows}) == 1
    b_slot = bind_ok_rows[0]["slot"]
    b_ids = torch.tensor([tok.encode(r["text"], add_special_tokens=False)
                          for r in bind_ok_rows], dtype=torch.long, device=device)
    b_am = torch.ones_like(b_ids)
    b_gold = torch.tensor([r["gold"] for r in bind_ok_rows], dtype=torch.long, device=device)
    b_true_v = [r["v"] for r in bind_ok_rows]
    # foil: different value for each row
    b_foil_v = []
    for v in b_true_v:
        cands = [w for w in vals if w != v]
        b_foil_v.append(str(rng.choice(cands)))

    # Route batch (flag=0 and flag=1 separately for accuracy; shared geometry)
    r_texts0, r_texts1, keys, flags = [], [], [], []
    for row in route_rows:
        r_texts0.append(row["t0"]); r_texts1.append(row["t1"])
    # may differ length between flag0/1 if digit same length — both "0"/"1" single token
    ids0, am0, fpos = _encode_pool(
        tok, r_texts0, [tmpl["key"]] * len(r_texts0), [0] * len(r_texts0), device)
    ids1, am1, fpos1 = _encode_pool(
        tok, r_texts1, [tmpl["key"]] * len(r_texts1), [1] * len(r_texts1), device)
    assert fpos == fpos1
    g0s = torch.tensor([row["g0"] for row in route_rows], dtype=torch.long, device=device)
    g1s = torch.tensor([row["g1"] for row in route_rows], dtype=torch.long, device=device)

    # Completion utility batch (explicit_A, bit=0 → act0)
    comp_pairs = completion_pairs.make_completion_pairs(
        20, seed=seed + 3, tok=tok, families=["explicit_A"])
    comp_rows = []
    for p in comp_pairs:
        # clean = bit 0
        text = p["clean_text"]
        act0 = p["answer_clean"]
        try:
            gold = single_token_id(tok, act0, leading_space=True)
        except ValueError:
            continue
        off = text.find(" = ")  # bit assignment — fragile; use anchors if present
        # bit_slot from pair anchors
        off_bit = p["anchors"].get("bit_slot")
        if off_bit is None:
            continue
        slot = _anchor_token_index(tok, text, off_bit)
        if slot is None:
            continue
        comp_rows.append({"text": text, "gold": gold, "slot": slot})
        if len(comp_rows) >= N_UTIL:
            break
    assert len(comp_rows) >= 4, "completion util pool too small"
    # uniform length + slot
    clens = [len(tok.encode(r["text"], add_special_tokens=False)) for r in comp_rows]
    cmaj = max(set(clens), key=clens.count)
    comp_rows = [r for r, L in zip(comp_rows, clens) if L == cmaj][:N_UTIL]
    assert len({r["slot"] for r in comp_rows}) == 1
    c_slot = comp_rows[0]["slot"]
    c_ids = torch.tensor([tok.encode(r["text"], add_special_tokens=False)
                          for r in comp_rows], dtype=torch.long, device=device)
    c_am = torch.ones_like(c_ids)
    c_gold = torch.tensor([r["gold"] for r in comp_rows], dtype=torch.long, device=device)

    def acc_from_logits(lg, gold):
        pred = lg.argmax(-1)
        return float((pred == gold).float().mean().item())

    def pref_bind(lg):
        return float(_pref_vs_pool(lg, b_gold, val_ids).mean())

    def project(ids, am, pos, basis, alpha):
        return forward_with_project(model, ids, am, layer, pos, basis, alpha=alpha)

    # ----- BIND arm -----
    log("  --- bind arm ---")
    # per-row true/foil bases differ — must loop or stack per-example projection.
    # forward_with_project uses shared basis. For true/foil, run per-row or
    # group by value. Simpler: sequential per-row for true/foil at α=1; batch
    # for clean and for random/full-span (shared).

    def bind_pref_ablate_per_row(dir_for_row, alpha):
        prefs = []
        for i, r in enumerate(bind_ok_rows):
            d = dir_for_row(i)
            ii = b_ids[i:i + 1]; aa = b_am[i:i + 1]
            lg = project(ii, aa, b_slot, _basis_1d(d), alpha)
            prefs.append(float(_pref_vs_pool(lg, b_gold[i:i + 1], val_ids).mean()))
        return float(np.mean(prefs))

    def bind_acc_ablate_per_row(dir_for_row, alpha):
        oks = []
        for i, r in enumerate(bind_ok_rows):
            d = dir_for_row(i)
            lg = project(b_ids[i:i + 1], b_am[i:i + 1], b_slot, _basis_1d(d), alpha)
            oks.append(int(lg.argmax(-1).item() == int(b_gold[i].item())))
        return float(np.mean(oks))

    lg_clean = last_token_logits(model, b_ids, b_am)
    pref_clean = pref_bind(lg_clean)
    acc_clean = acc_from_logits(lg_clean, b_gold)
    log(f"  bind clean pref={pref_clean:+.2f} acc={acc_clean:.0%}")

    pref_true = bind_pref_ablate_per_row(lambda i: Delta[b_true_v[i]], 1.0)
    pref_foil = bind_pref_ablate_per_row(lambda i: Delta[b_foil_v[i]], 1.0)
    acc_true = bind_acc_ablate_per_row(lambda i: Delta[b_true_v[i]], 1.0)
    acc_foil = bind_acc_ablate_per_row(lambda i: Delta[b_foil_v[i]], 1.0)
    drop_true = pref_clean - pref_true
    drop_foil = pref_clean - pref_foil
    drop_true_acc = acc_clean - acc_true
    drop_foil_acc = acc_clean - acc_foil

    # full span support
    span = _basis_span([Delta[v] for v in vals])
    lg_span = project(b_ids, b_am, b_slot, span, 1.0)
    pref_span = pref_bind(lg_span)
    drop_span = pref_clean - pref_span

    # random null at α=1 (shared random dir per draw, mean drop)
    hb = Heartbeat(n_null, "nec_bind_null", every_sec=15, out_dir=out_dir)
    null_drops = []
    Ddim = int(d_route.numel())
    for _ in range(n_null):
        r = torch.from_numpy(rng.normal(size=Ddim).astype(np.float32))
        lg = project(b_ids, b_am, b_slot, _basis_1d(r), 1.0)
        null_drops.append(pref_clean - pref_bind(lg))
        hb.step()
    hb.done()
    p_bind = permutation_pvalue(drop_true, np.array(null_drops), "greater")

    s1 = bool(drop_true > 0
              and (drop_true - drop_foil) >= 0.5 * drop_true
              and p_bind < 0.01)
    if pref_clean > 0:
        t_bind = bool(drop_true >= 0.5 * pref_clean)
    else:
        t_bind = bool(drop_true_acc >= 0.40)

    # α-sweep bind true/foil
    sweep_true, sweep_foil = [], []
    for a in ALPHAS:
        sweep_true.append(pref_clean - bind_pref_ablate_per_row(
            lambda i: Delta[b_true_v[i]], a))
        sweep_foil.append(pref_clean - bind_pref_ablate_per_row(
            lambda i: Delta[b_foil_v[i]], a))
    rho_true = _spearman(ALPHAS, sweep_true)
    rho_foil = _spearman(ALPHAS, sweep_foil)
    alpha_bind_ok = bool(rho_true >= 0.7 and abs(rho_foil) <= 0.3)

    # Utility bind: Select with bind-dir @ flag; Completion with bind span @ bit
    # Use mean Δ over values as "binding direction" at wrong site
    d_bind_mean = torch.stack([Delta[v] for v in vals]).mean(0)
    # Select clean acc
    def route_acc(basis, alpha, pos=fpos):
        a0 = acc_from_logits(project(ids0, am0, pos, basis, alpha), g0s)
        a1 = acc_from_logits(project(ids1, am1, pos, basis, alpha), g1s)
        return 0.5 * (a0 + a1)

    route_clean_acc = 0.5 * (
        acc_from_logits(last_token_logits(model, ids0, am0), g0s)
        + acc_from_logits(last_token_logits(model, ids1, am1), g1s))
    route_under_bind = route_acc(_basis_1d(d_bind_mean), 1.0)
    drop_route_under_bind = route_clean_acc - route_under_bind

    comp_clean = acc_from_logits(last_token_logits(model, c_ids, c_am), c_gold)
    comp_under_bind = acc_from_logits(
        project(c_ids, c_am, c_slot, _basis_1d(d_bind_mean), 1.0), c_gold)
    drop_comp_bind = comp_clean - comp_under_bind

    u_bind = (_util_pass(drop_route_under_bind, drop_true, target_is_pref=True)
              and _util_pass(drop_comp_bind, drop_true, target_is_pref=True))

    bind_pass = bool(s1 and t_bind and u_bind)
    log(f"  S1 drop_true={drop_true:+.2f} drop_foil={drop_foil:+.2f} "
        f"p={p_bind:.3f} pass={s1}")
    log(f"  T_bind={t_bind} U_bind={u_bind} "
        f"(routeΔacc={drop_route_under_bind:+.0%} compΔ={drop_comp_bind:+.0%})")
    log(f"  α-sweep ρ_true={rho_true:+.2f} ρ_foil={rho_foil:+.2f} ok={alpha_bind_ok}")
    log(f"  BIND_ARM pass={bind_pass}")

    # ----- ROUTE arm -----
    log("  --- route arm ---")
    b_route = _basis_1d(d_route)

    va_ids = torch.tensor([single_token_id(tok, row["va"], leading_space=True)
                           for row in route_rows], dtype=torch.long, device=device)
    vb_ids = g0s
    n_rr = len(route_rows)
    idx_r = torch.arange(n_rr, device=device)

    def route_pref_gap(basis, alpha):
        if alpha == 0:
            lg0 = last_token_logits(model, ids0, am0)
            lg1 = last_token_logits(model, ids1, am1)
        else:
            lg0 = project(ids0, am0, fpos, basis, alpha)
            lg1 = project(ids1, am1, fpos, basis, alpha)
        p0 = (lg0[idx_r, vb_ids] - lg0[idx_r, va_ids]).float().mean().item()
        p1 = (lg1[idx_r, g1s] - lg1[idx_r, vb_ids]).float().mean().item()
        return 0.5 * (p0 + p1)

    gap_clean = route_pref_gap(b_route, 0.0)
    gap_route = route_pref_gap(b_route, 1.0)
    # random
    rvec = torch.from_numpy(rng.normal(size=Ddim).astype(np.float32))
    gap_rand = route_pref_gap(_basis_1d(rvec), 1.0)
    gap_binddir = route_pref_gap(_basis_1d(d_bind_mean), 1.0)
    drop_route = gap_clean - gap_route
    drop_rand = gap_clean - gap_rand
    drop_binddir = gap_clean - gap_binddir
    drop_wrong = max(drop_rand, drop_binddir)

    acc_r_clean = route_clean_acc
    acc_r_ablate = route_acc(b_route, 1.0)
    drop_route_acc = acc_r_clean - acc_r_ablate

    hb = Heartbeat(n_null, "nec_route_null", every_sec=15, out_dir=out_dir)
    null_rdrops = []
    for _ in range(n_null):
        r = torch.from_numpy(rng.normal(size=Ddim).astype(np.float32))
        null_rdrops.append(gap_clean - route_pref_gap(_basis_1d(r), 1.0))
        hb.step()
    hb.done()
    p_route = permutation_pvalue(drop_route, np.array(null_rdrops), "greater")

    s2 = bool(drop_route > 0
              and (drop_route - drop_wrong) >= 0.5 * drop_route
              and p_route < 0.01)
    t_route = bool(drop_route_acc >= 0.40 or (
        gap_clean > 0 and drop_route >= 0.5 * gap_clean))

    sweep_route, sweep_rfoil = [], []
    for a in ALPHAS:
        sweep_route.append(gap_clean - route_pref_gap(b_route, a))
        sweep_rfoil.append(gap_clean - route_pref_gap(_basis_1d(d_bind_mean), a))
    rho_route = _spearman(ALPHAS, sweep_route)
    rho_rfoil = _spearman(ALPHAS, sweep_rfoil)
    alpha_route_ok = bool(rho_route >= 0.7 and abs(rho_rfoil) <= 0.3)

    # Utility route: Variable with route @ val_slot; Completion with route @ bit
    pref_var_under_route = pref_bind(project(b_ids, b_am, b_slot, b_route, 1.0))
    drop_var_under_route = pref_clean - pref_var_under_route
    # For U with pref target on route arm, util Variable: use acc drop
    acc_var_under_route = acc_from_logits(project(b_ids, b_am, b_slot, b_route, 1.0), b_gold)
    drop_var_acc = acc_clean - acc_var_under_route
    comp_under_route = acc_from_logits(
        project(c_ids, c_am, c_slot, b_route, 1.0), c_gold)
    drop_comp_route = comp_clean - comp_under_route

    u_route = (_util_pass(drop_var_acc, drop_route_acc, target_is_pref=False)
               and _util_pass(drop_comp_route, drop_route_acc, target_is_pref=False))

    route_pass = bool(s2 and t_route and u_route)
    log(f"  S2 drop_route={drop_route:+.2f} wrong={drop_wrong:+.2f} "
        f"p={p_route:.3f} pass={s2}")
    log(f"  T_route={t_route} U_route={u_route} "
        f"(varΔacc={drop_var_acc:+.0%} compΔ={drop_comp_route:+.0%})")
    log(f"  α-sweep ρ_route={rho_route:+.2f} ρ_foil={rho_rfoil:+.2f} ok={alpha_route_ok}")
    log(f"  ROUTE_ARM pass={route_pass}")

    # ----- verdict -----
    alpha_ok = alpha_bind_ok and alpha_route_ok
    if bind_pass and route_pass and alpha_ok:
        verdict = "NECESSARY"
    elif bind_pass and route_pass and not alpha_ok:
        verdict = "PARTIAL"  # α-sweep demotion
    elif route_pass and not bind_pass:
        verdict = "ROUTE_NECESSARY_ONLY"
    elif bind_pass and not route_pass:
        verdict = "BIND_NECESSARY_ONLY"
    elif (t_bind or t_route or s1 or s2) and not (bind_pass or route_pass):
        verdict = "PARTIAL"
    else:
        # G0 passed; site-local necessity null
        verdict = "SUFFICIENT_ONLY"

    results = {
        "stage": "delta_necessity", "version": VERSION, "model_path": model_path,
        "layer": layer, "n_null": n_null, "alphas": list(ALPHAS),
        "G0": {"bind": behav_bind, "route": behav_route, "pass": True},
        "bind": {
            "pref_clean": pref_clean, "pref_true": pref_true, "pref_foil": pref_foil,
            "pref_span": pref_span,
            "drop_true": drop_true, "drop_foil": drop_foil, "drop_span": drop_span,
            "acc_clean": acc_clean, "acc_true": acc_true, "acc_foil": acc_foil,
            "drop_true_acc": drop_true_acc,
            "p_null": float(p_bind),
            "S1": s1, "T": t_bind, "U": u_bind, "pass": bind_pass,
            "alpha_sweep": {"true": sweep_true, "foil": sweep_foil,
                            "rho_true": rho_true, "rho_foil": rho_foil,
                            "pass": alpha_bind_ok},
            "utility": {"drop_select_acc": drop_route_under_bind,
                        "drop_completion_acc": drop_comp_bind},
            "n": len(bind_ok_rows), "slot": int(b_slot),
        },
        "route": {
            "gap_clean": gap_clean, "gap_route": gap_route,
            "gap_rand": gap_rand, "gap_binddir": gap_binddir,
            "drop_route": drop_route, "drop_wrong": drop_wrong,
            "acc_clean": acc_r_clean, "acc_ablate": acc_r_ablate,
            "drop_acc": drop_route_acc,
            "p_null": float(p_route),
            "S2": s2, "T": t_route, "U": u_route, "pass": route_pass,
            "alpha_sweep": {"route": sweep_route, "foil": sweep_rfoil,
                            "rho_route": rho_route, "rho_foil": rho_rfoil,
                            "pass": alpha_route_ok},
            "utility": {"drop_variable_acc": drop_var_acc,
                        "drop_completion_acc": drop_comp_route},
            "n": len(route_rows), "flag_pos": int(fpos),
        },
        "gates": {
            "S1": s1, "S2": s2, "U_bind": u_bind, "U_route": u_route,
            "T_bind": t_bind, "T_route": t_route,
            "alpha_ok": alpha_ok,
        },
        "verdict": verdict,
        "framing": {
            "NECESSARY": "both arms selective+utility+magnitude; α-sweep ok",
            "BIND_NECESSARY_ONLY": "bind arm only",
            "ROUTE_NECESSARY_ONLY": "route arm only",
            "PARTIAL": "damage without clean selectivity/utility/α",
            "SUFFICIENT_ONLY": "site-local necessity null; dirs install but native unused at site",
            "NEC_INELICITABLE": "G0 fail",
        }.get(verdict, verdict),
        "hard_stop": "no_layer_expansion",
    }
    path = os.path.join(out_dir, "results_delta_necessity.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT delta_necessity: {verdict} | "
        f"bind={bind_pass} route={route_pass} alpha={alpha_ok}")
    return results
