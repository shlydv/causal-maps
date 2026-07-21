"""What does Δ_route read — span{Δ} write or ambient residual?

ROUTER_READ_PROTOCOL.md. Native protocol carrier + write-site span/residual edits.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .delta_crossskill import _variable_directions
from .delta_protocol import (BEHAV_GATE, _extract_routing_delta, _native_text,
                             _pref_uw)
from .direction_transfer import PRIMARY_LAYER
from .logutil import Heartbeat, log
from .model_utils import (get_decoder_layers, input_device, last_token_logits,
                          load_model_and_tokenizer, single_token_id)
from .nulls import permutation_pvalue
from .patching import _split_output

N_NULL = 100
N_TRIALS = 12
LAYER = PRIMARY_LAYER
VERSION = 1
EPS = 1e-6


def _basis_span(dirs) -> torch.Tensor:
    M = torch.stack([d.float().reshape(-1) for d in dirs], dim=1)
    Q, _ = torch.linalg.qr(M, mode="reduced")
    return Q


@torch.no_grad()
def _forward_ops(model, ids, am, layer, ops):
    """ops: list of dicts with keys kind in {add, keep_span, keep_res}.

    add: {kind, pos, delta}  delta [D] or [B,D]
    keep_span: {kind, pos, basis}  basis [D,K] → h ← P h
    keep_res: {kind, pos, basis} → h ← h - P h
    """
    lyr = get_decoder_layers(model)[layer]

    def hook(module, inp, out):
        hs, rebuild = _split_output(out)
        hs = hs.clone()
        for op in ops:
            pos = op["pos"]
            h = hs[:, pos, :].float()
            kind = op["kind"]
            if kind == "add":
                d = op["delta"].to(dtype=h.dtype, device=h.device)
                if d.dim() == 1:
                    h = h + d
                else:
                    h = h + d
            else:
                U = op["basis"].float().to(device=h.device)
                if U.dim() == 1:
                    U = (U / U.norm().clamp(min=1e-8)).unsqueeze(1)
                proj = (h @ U) @ U.T
                if kind == "keep_span":
                    h = proj
                elif kind == "keep_res":
                    h = h - proj
                else:
                    raise ValueError(kind)
            hs[:, pos, :] = h.to(dtype=hs.dtype)
        return rebuild(hs)

    handle = lyr.register_forward_hook(hook)
    try:
        out = model(input_ids=ids, attention_mask=am, use_cache=False)
    finally:
        handle.remove()
    return out.logits[:, -1, :]


@torch.no_grad()
def _logits_pool(model, ids, am, layer, ops, val_ids_t):
    if ops:
        lg = _forward_ops(model, ids, am, layer, ops)
    else:
        lg = last_token_logits(model, ids, am)
    return lg[:, val_ids_t].float().cpu().numpy()


def run_delta_router_read(model_path, out_dir, quantization="8bit", device_map=None,
                          layer=LAYER, seed=0, n_trials=N_TRIALS, n_null=N_NULL):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    device = input_device(model)
    rng = np.random.default_rng(seed)

    log(f"delta_router_read v{VERSION}: layer={layer} — span vs residual for route")

    Delta, values = _variable_directions(model, tok, layer, seed)
    d_route, _, route_src = _extract_routing_delta(model, tok, device, seed, layer)
    vals = list(values)
    span = _basis_span([Delta[v] for v in vals]).to(device)
    dR = d_route.to(device)
    log(f"  nV={len(vals)} route_src={route_src} ‖Δ_route‖={float(d_route.norm()):.2f}")

    # trials
    trials, used = [], set()
    tries = 0
    while len(trials) < n_trials and tries < 2000:
        tries += 1
        u, w, v0 = (str(z) for z in rng.choice(vals, size=3, replace=False))
        if (u, w) in used:
            continue
        t0, *_ = _native_text(tok, u, w, 0)
        t1, *_ = _native_text(tok, u, w, 1)
        if len(tok.encode(t0, add_special_tokens=False)) != len(
                tok.encode(t1, add_special_tokens=False)):
            continue
        used.add((u, w))
        trials.append((u, w, v0))
    assert len(trials) >= 4
    log(f"  n_trials={len(trials)}")

    # native flag=0 batch (real bindings)
    texts, xslots, yslots, fposs = [], [], [], []
    for u, w, v0 in trials:
        text, xs, ys, fp = _native_text(tok, u, w, 0)
        texts.append(text); xslots.append(xs); yslots.append(ys); fposs.append(fp)
    assert len(set(len(tok.encode(t, add_special_tokens=False)) for t in texts)) == 1
    assert len(set(xslots)) == 1 and len(set(yslots)) == 1 and len(set(fposs)) == 1
    xslot, yslot, fpos = xslots[0], yslots[0], fposs[0]
    ids = torch.tensor([tok.encode(t, add_special_tokens=False) for t in texts],
                       dtype=torch.long, device=device)
    am = torch.ones_like(ids)

    val_list = sorted(set(vals))
    vidx = {v: i for i, v in enumerate(val_list)}
    val_ids_t = torch.tensor([single_token_id(tok, v) for v in val_list],
                             dtype=torch.long, device=device)
    u_idx = np.array([vidx[u] for u, w, v0 in trials])
    w_idx = np.array([vidx[w] for u, w, v0 in trials])

    # G0 native behav
    @torch.no_grad()
    def greedy_one(text):
        e = tok.encode(text, add_special_tokens=False)
        ii = torch.tensor([e], dtype=torch.long, device=device)
        return int(last_token_logits(model, ii, torch.ones_like(ii)).argmax(-1).item())

    n0 = n1 = 0
    for u, w, v0 in trials:
        t0, *_ = _native_text(tok, u, w, 0)
        t1, *_ = _native_text(tok, u, w, 1)
        if greedy_one(t0) == single_token_id(tok, w):
            n0 += 1
        if greedy_one(t1) == single_token_id(tok, u):
            n1 += 1
    behav0, behav1 = n0 / len(trials), n1 / len(trials)
    g0 = behav0 >= BEHAV_GATE and behav1 >= BEHAV_GATE
    log(f"  G0: flag0→Y={behav0:.0%} flag1→X={behav1:.0%} pass={g0}")
    if not g0:
        results = {
            "stage": "delta_router_read", "version": VERSION, "model_path": model_path,
            "layer": layer, "verdict": "ROUTER_READ_INELICITABLE",
            "G0": {"flag0": behav0, "flag1": behav1, "pass": False},
        }
        with open(os.path.join(out_dir, "results_delta_router_read.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        log("VERDICT delta_router_read: ROUTER_READ_INELICITABLE")
        return results

    def pack(ops):
        return _logits_pool(model, ids, am, layer, ops, val_ids_t)

    def write_ops(mode):
        """mode: none | span | res"""
        if mode == "none":
            return []
        kind = "keep_span" if mode == "span" else "keep_res"
        return [
            {"kind": kind, "pos": xslot, "basis": span},
            {"kind": kind, "pos": yslot, "basis": span},
        ]

    def with_route(mode, route_delta=None):
        rd = dR if route_delta is None else route_delta
        return write_ops(mode) + [{"kind": "add", "pos": fpos, "delta": rd}]

    # preferences
    pref_base = _pref_uw(pack(write_ops("none")), u_idx, w_idx)
    pref_full = _pref_uw(pack(with_route("none")), u_idx, w_idx)
    pref_span0 = _pref_uw(pack(write_ops("span")), u_idx, w_idx)
    pref_span = _pref_uw(pack(with_route("span")), u_idx, w_idx)
    pref_res0 = _pref_uw(pack(write_ops("res")), u_idx, w_idx)
    pref_res = _pref_uw(pack(with_route("res")), u_idx, w_idx)

    RS_full = pref_full - pref_base
    RS_span = pref_span - pref_span0
    RS_res = pref_res - pref_res0
    log(f"  pref: base={pref_base:+.2f} full={pref_full:+.2f} "
        f"span0={pref_span0:+.2f} span={pref_span:+.2f} "
        f"res0={pref_res0:+.2f} res={pref_res:+.2f}")
    log(f"  RS: full={RS_full:+.2f} span={RS_span:+.2f} res={RS_res:+.2f}")

    # nulls on each surface
    ns = float(d_route.norm().clamp(min=1e-8))
    Ddim = int(d_route.numel())

    def null_rs(mode, n=n_null):
        hb = Heartbeat(n, f"router_read_null_{mode}", every_sec=15, out_dir=out_dir)
        base_p = _pref_uw(pack(write_ops(mode)), u_idx, w_idx)
        out = []
        for _ in range(n):
            r = torch.from_numpy(rng.normal(size=Ddim).astype(np.float32))
            r = (r / r.norm().clamp(min=1e-8) * ns).to(device)
            p = _pref_uw(pack(with_route(mode, route_delta=r)), u_idx, w_idx)
            out.append(p - base_p)
            hb.step()
        hb.done()
        return np.array(out)

    null_full = null_rs("none")
    null_span = null_rs("span")
    null_res = null_rs("res")
    p_full = permutation_pvalue(RS_full, null_full, "greater")
    p_span = permutation_pvalue(RS_span, null_span, "greater")
    p_res = permutation_pvalue(RS_res, null_res, "greater")

    G1 = bool(RS_full > 0 and p_full < 0.01)
    SPAN = bool(G1 and RS_span > 0 and RS_span >= 0.5 * RS_full and p_span < 0.01)
    RES = bool(G1 and RS_res > 0 and RS_res >= 0.5 * RS_full and p_res < 0.01)
    log(f"  G1={G1}(p={p_full:.3f}) SPAN={SPAN}(p={p_span:.3f}) RES={RES}(p={p_res:.3f})")

    if G1 and SPAN and RES:
        verdict = "ROUTER_READS_BOTH"
    elif G1 and SPAN and not RES:
        verdict = "ROUTER_READS_SPAN"
    elif G1 and RES and not SPAN:
        verdict = "ROUTER_READS_RESIDUAL"
    elif G1:
        verdict = "ROUTER_READS_NEITHER"
    else:
        verdict = "ROUTER_READ_WEAK"

    results = {
        "stage": "delta_router_read", "version": VERSION, "model_path": model_path,
        "layer": layer, "n_null": n_null, "n_trials": len(trials),
        "route_src": route_src,
        "G0": {"flag0": behav0, "flag1": behav1, "pass": True},
        "prefs": {
            "base": pref_base, "full": pref_full,
            "span0": pref_span0, "span": pref_span,
            "res0": pref_res0, "res": pref_res,
        },
        "RS": {
            "full": RS_full, "span": RS_span, "res": RS_res,
            "p_full": float(p_full), "p_span": float(p_span), "p_res": float(p_res),
        },
        "gates": {"G1": G1, "SPAN": SPAN, "RES": RES},
        "slots": {"xslot": int(xslot), "yslot": int(yslot), "flag_pos": int(fpos)},
        "verdict": verdict,
        "framing": {
            "ROUTER_READS_SPAN": "route consumes span{Δ} write; residual epiphenomenal for routing",
            "ROUTER_READS_RESIDUAL": "route consumes ambient residual; Δ install is correlated",
            "ROUTER_READS_BOTH": "redundant codes — span and residual both suffice",
            "ROUTER_READS_NEITHER": "neither slice carries RS under these ops",
            "ROUTER_READ_WEAK": "route does not reliably move native write",
            "ROUTER_READ_INELICITABLE": "G0 fail",
        }[verdict],
        "hard_stop": "no_layer_expansion",
    }
    path = os.path.join(out_dir, "results_delta_router_read.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT delta_router_read: {verdict} | "
        f"RS_full={RS_full:+.2f} span={RS_span:+.2f} res={RS_res:+.2f}")
    return results
