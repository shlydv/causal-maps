"""Activation-distribution control for ROUTER_READS_RESIDUAL.

ROUTER_OOD_PROTOCOL.md — is keep_span OOD-harsh vs keep_res?
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .delta_crossskill import _variable_directions
from .delta_protocol import BEHAV_GATE, _extract_routing_delta, _native_text
from .delta_router_read import _basis_span
from .direction_transfer import PRIMARY_LAYER
from .logutil import log
from .model_utils import (input_device, last_token_logits, load_model_and_tokenizer,
                          single_token_id)
from .patching import cache_layer_outputs

N_TRIALS = 12
LAYER = PRIMARY_LAYER
VERSION = 1


@torch.no_grad()
def _acts_at(model, ids, am, layer, positions):
    """Return dict pos -> [B,D] float cpu."""
    cache = cache_layer_outputs(model, ids, am, to_cpu=True)
    return {p: cache[layer][:, p, :].float() for p in positions}


@torch.no_grad()
def _acts_after_ops(model, ids, am, layer, ops, positions):
    """Run forward with ops; capture layer outputs at positions."""
    from .model_utils import get_decoder_layers
    from .patching import _split_output
    lyr = get_decoder_layers(model)[layer]
    captured = {}

    def hook(module, inp, out):
        hs, rebuild = _split_output(out)
        hs = hs.clone()
        for op in ops:
            pos = op["pos"]
            h = hs[:, pos, :].float()
            kind = op["kind"]
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
        for p in positions:
            captured[p] = hs[:, p, :].detach().float().cpu()
        return rebuild(hs)

    handle = lyr.register_forward_hook(hook)
    try:
        model(input_ids=ids, attention_mask=am, use_cache=False)
    finally:
        handle.remove()
    return captured


def _pair_stats(H, Hp):
    """H, Hp [B,D] → mean rel_norm, cos, rel_disp over batch."""
    B = H.shape[0]
    rn, cs, rd = [], [], []
    for i in range(B):
        h = H[i]; hp = Hp[i]
        nh = float(h.norm().clamp(min=1e-8))
        nhp = float(hp.norm().clamp(min=1e-8))
        rn.append(nhp / nh)
        cs.append(float(torch.nn.functional.cosine_similarity(
            h.unsqueeze(0), hp.unsqueeze(0)).item()))
        rd.append(float((hp - h).norm() / nh))
    return float(np.mean(rn)), float(np.mean(cs)), float(np.mean(rd))


def run_delta_router_ood(model_path, out_dir, quantization="8bit", device_map=None,
                         layer=LAYER, seed=0, n_trials=N_TRIALS):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    device = input_device(model)
    rng = np.random.default_rng(seed)

    log(f"delta_router_ood v{VERSION}: layer={layer} — span vs res distribution")

    Delta, values = _variable_directions(model, tok, layer, seed)
    vals = list(values)
    span = _basis_span([Delta[v] for v in vals]).to(device)
    _ = _extract_routing_delta(model, tok, device, seed, layer)  # parity with router_read seed path

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

    texts, xslots, yslots = [], [], []
    for u, w, v0 in trials:
        text, xs, ys, _fp = _native_text(tok, u, w, 0)
        texts.append(text); xslots.append(xs); yslots.append(ys)
    assert len(set(xslots)) == 1 and len(set(yslots)) == 1
    xslot, yslot = xslots[0], yslots[0]
    ids = torch.tensor([tok.encode(t, add_special_tokens=False) for t in texts],
                       dtype=torch.long, device=device)
    am = torch.ones_like(ids)
    positions = [xslot, yslot]

    # G0 light check
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
    log(f"  G0: {behav0:.0%}/{behav1:.0%} pass={g0}")
    if not g0:
        results = {
            "stage": "delta_router_ood", "version": VERSION, "verdict": "OOD_INELICITABLE",
            "G0": {"flag0": behav0, "flag1": behav1, "pass": False},
        }
        with open(os.path.join(out_dir, "results_delta_router_ood.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        log("VERDICT delta_router_ood: OOD_INELICITABLE")
        return results

    H = _acts_at(model, ids, am, layer, positions)
    # energy in span on clean
    energy = []
    for p in positions:
        for i in range(ids.shape[0]):
            h = H[p][i].to(device)
            proj = (h.unsqueeze(0) @ span) @ span.T
            energy.append(float(proj.norm() ** 2 / h.norm().clamp(min=1e-8) ** 2))
    energy_frac = float(np.mean(energy))

    span_ops = [
        {"kind": "keep_span", "pos": xslot, "basis": span},
        {"kind": "keep_span", "pos": yslot, "basis": span},
    ]
    res_ops = [
        {"kind": "keep_res", "pos": xslot, "basis": span},
        {"kind": "keep_res", "pos": yslot, "basis": span},
    ]
    Hs = _acts_after_ops(model, ids, am, layer, span_ops, positions)
    Hr = _acts_after_ops(model, ids, am, layer, res_ops, positions)

    def agg(Hop):
        rns, css, rds = [], [], []
        for p in positions:
            rn, cs, rd = _pair_stats(H[p], Hop[p])
            rns.append(rn); css.append(cs); rds.append(rd)
        return float(np.mean(rns)), float(np.mean(css)), float(np.mean(rds))

    n_s, c_s, d_s = agg(Hs)
    n_r, c_r, d_r = agg(Hr)
    log(f"  energy_frac(span in clean)={energy_frac:.3f}")
    log(f"  SPAN: rel_norm={n_s:.3f} cos={c_s:.3f} rel_disp={d_s:.3f}")
    log(f"  RES:  rel_norm={n_r:.3f} cos={c_r:.3f} rel_disp={d_r:.3f}")

    fail_norm = bool(n_s < 0.5 * n_r)
    fail_cos = bool(c_s < c_r - 0.30)
    fail_disp = bool(d_s > d_r + 0.30)
    ood_fail = fail_norm or fail_cos or fail_disp
    verdict = "OOD_FAIL" if ood_fail else "OOD_PASS"

    results = {
        "stage": "delta_router_ood", "version": VERSION, "model_path": model_path,
        "layer": layer, "n_trials": len(trials),
        "G0": {"flag0": behav0, "flag1": behav1, "pass": True},
        "energy_frac_span_in_clean": energy_frac,
        "span": {"rel_norm": n_s, "cos": c_s, "rel_disp": d_s},
        "res": {"rel_norm": n_r, "cos": c_r, "rel_disp": d_r},
        "fail_flags": {
            "norm": fail_norm, "cos": fail_cos, "disp": fail_disp,
        },
        "verdict": verdict,
        "implication": (
            "ROUTER_READS_RESIDUAL remains in play"
            if verdict == "OOD_PASS"
            else "demote router-read to ROUTER_READ_AMBIGUOUS — SPAN OOD confound"
        ),
        "hard_stop": "no_rerun_router_read",
    }
    path = os.path.join(out_dir, "results_delta_router_ood.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT delta_router_ood: {verdict} | "
        f"fail_norm={fail_norm} fail_cos={fail_cos} fail_disp={fail_disp}")
    return results
