"""Protocol compatibility: independently-extracted binding + routing directions.

Crisp question (CAUSAL_MAPS_LOG.md 2026-07-13):
  Do independently discovered intervention directions write representations in a
  format that independently discovered downstream mechanisms can consume?

Not metaphors ("algebra"). Not mere readability. Primary signatures:
  (1) Routing sensitivity RS — interaction / flip under injected bindings.
  (2) Dependency — ablate upstream write ⇒ downstream preference collapses;
      scale α on the write ⇒ preference tracks α.

v2: P2 metric fix only (pre-registered). Compare dep_gap to routed content
(full−empty), not to RS. One rerun then stop.

Donor protocols (unchanged):
  Binding Δ_v from Variable pairs at (L2, val_slot).
  Routing Δ_route from Select value_of at (L2, flag digit).

Carrier (neutral third protocol):
  Let X=v0. Let Y=v0. If flag=1 output X; if flag=0 output Y. flag=0.
  Primer: Answer =

Pre-registered verdicts: PROTOCOL_COMPATIBLE / COMPATIBLE_WEAK /
KNOBS_NOT_PROTOCOL / INCOMPATIBLE.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from . import variable_pairs
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
N_TRIALS = 12
SCALE_ALPHAS = (0.0, 0.5, 1.0, 1.5)
VERSION = 2

# Select winner used for Δ_route extraction (frozen from delta_select v2).
ROUTE_TMPL_NAME = "value_of"


def _value_of_tmpl():
    for t in SELECT_TEMPLATES:
        if t["name"] == ROUTE_TMPL_NAME:
            return t
    raise RuntimeError(f"missing template {ROUTE_TMPL_NAME}")


def _carrier_text(tok, v0, flag: int):
    """Neutral carrier with X/Y + flag. Returns text, xslot, yslot, flag_pos."""
    user = (
        f"Let X = {v0}. Let Y = {v0}. "
        f"If flag=1, output the value of X. If flag=0, output the value of Y. "
        f"flag={flag}. What is the answer?"
    )
    templated = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    text = templated + "Answer ="
    off_x = text.find(f"Let X = ") + len("Let X = ")
    off_y = text.find(f"Let Y = ") + len("Let Y = ")
    xslot = _anchor_token_index(tok, text, off_x)
    yslot = _anchor_token_index(tok, text, off_y)
    # flag digit index (bare)
    needle = f"flag={flag}"
    start = text.rfind(needle)
    assert start >= 0
    prefix = text[: start + len("flag=")]
    pre = tok.encode(prefix, add_special_tokens=False)
    full = tok.encode(text, add_special_tokens=False)
    fpos = len(pre)
    expect = single_token_id(tok, str(flag), leading_space=False)
    assert full[fpos] == expect, (
        f"flag digit mismatch {tok.convert_ids_to_tokens([full[fpos]])}")
    return text, xslot, yslot, fpos


def _native_text(tok, u, w, flag: int):
    """Same structure with stated bindings (native baseline)."""
    user = (
        f"Let X = {u}. Let Y = {w}. "
        f"If flag=1, output the value of X. If flag=0, output the value of Y. "
        f"flag={flag}. What is the answer?"
    )
    templated = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    text = templated + "Answer ="
    off_x = text.find(f"Let X = ") + len("Let X = ")
    off_y = text.find(f"Let Y = ") + len("Let Y = ")
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
    """Δ_route at flag digit from Select value_of (flag1 − flag0)."""
    tmpl = _value_of_tmpl()
    rng = np.random.default_rng(seed + 17)
    pairs = _pair_pool(tok, tmpl, rng, N_TRAIN_ROUTE)
    assert len(pairs) >= N_TRAIN_ROUTE, f"route train pool {len(pairs)}"
    train = pairs[:N_TRAIN_ROUTE]

    def build(rows, f):
        texts, keys, flags = [], [], []
        for va, vb in rows:
            t, k, ff = _render(tok, tmpl, va, vb, f)
            texts.append(t); keys.append(k); flags.append(ff)
        return _encode_pool(tok, texts, keys, flags, device)

    ids0, am0, fpos = build(train, 0)
    ids1, am1, _ = build(train, 1)
    cache0 = cache_layer_outputs(model, ids0, am0, to_cpu=True)
    cache1 = cache_layer_outputs(model, ids1, am1, to_cpu=True)
    d = cache1[layer][:, fpos, :].float().mean(0) - cache0[layer][:, fpos, :].float().mean(0)
    return d, fpos, tmpl["name"]


@torch.no_grad()
def _logits_vals(model, ids, am, layer, pos_deltas, val_ids):
    if pos_deltas:
        lg = forward_add_multi(model, ids, am, layer, pos_deltas)
    else:
        lg = last_token_logits(model, ids, am)
    return lg[:, val_ids].float().cpu().numpy()


def _pref_uw(logits, u_idx, w_idx):
    """mean logit(u) − logit(w) over trials."""
    return float((logits[np.arange(len(u_idx)), u_idx]
                  - logits[np.arange(len(w_idx)), w_idx]).mean())


def run_delta_protocol(model_path, out_dir, quantization="8bit", device_map=None,
                       layer=PRIMARY_LAYER, seed=0, n_trials=N_TRIALS,
                       n_null=N_NULL):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    device = input_device(model)
    rng = np.random.default_rng(seed)

    log(f"delta_protocol v{VERSION}: layer={layer} n_trials={n_trials} "
        f"n_null={n_null} question=protocol_compatibility")

    # --- extract independently ---
    Delta, values = _variable_directions(model, tok, layer, seed)
    d_route, _, route_src = _extract_routing_delta(model, tok, device, seed, layer)
    log(f"  binding nV={len(values)} route_src={route_src} "
        f"‖Δ_route‖={float(d_route.norm()):.2f}")

    # --- trials: distinct (u, w, v0) ---
    trials, used = [], set()
    tries = 0
    while len(trials) < n_trials and tries < 2000:
        tries += 1
        u, w, v0 = (str(z) for z in rng.choice(values, size=3, replace=False))
        if (u, w) in used:
            continue
        # length-uniform carrier under v0
        t0, *_ = _carrier_text(tok, v0, 0)
        t1, *_ = _carrier_text(tok, v0, 1)
        if len(tok.encode(t0, add_special_tokens=False)) != len(
                tok.encode(t1, add_special_tokens=False)):
            continue
        used.add((u, w))
        trials.append((u, w, v0))
    assert len(trials) >= max(4, n_trials // 2), f"only {len(trials)} trials"
    log(f"  n_trials_ok={len(trials)}")

    # Build carrier batch (flag=0 surface) — shared geometry
    texts, xslots, yslots, fposs = [], [], [], []
    for u, w, v0 in trials:
        text, xs, ys, fp = _carrier_text(tok, v0, 0)
        texts.append(text); xslots.append(xs); yslots.append(ys); fposs.append(fp)
    assert len(set(len(tok.encode(t, add_special_tokens=False)) for t in texts)) == 1
    assert len(set(xslots)) == 1 and len(set(yslots)) == 1 and len(set(fposs)) == 1
    xslot, yslot, fpos = xslots[0], yslots[0], fposs[0]
    enc = [tok.encode(t, add_special_tokens=False) for t in texts]
    ids = torch.tensor(enc, dtype=torch.long, device=device)
    am = torch.ones_like(ids)

    # Value id table (leading-space, Variable convention)
    val_list = sorted(set(values))
    vidx = {v: i for i, v in enumerate(val_list)}
    val_ids = [single_token_id(tok, v) for v in val_list]
    u_idx = np.array([vidx[u] for u, w, v0 in trials])
    w_idx = np.array([vidx[w] for u, w, v0 in trials])

    def pack(pos_deltas):
        return _logits_vals(model, ids, am, layer, pos_deltas, val_ids)

    def binds_for(alpha_u=1.0):
        """Per-trial (pos, delta) lists for bind u@X, w@Y."""
        # forward_add_multi expects shared pos with [D] or [B,D]; build [B,D]
        dX = torch.stack([alpha_u * Delta[u] for u, w, v0 in trials])  # [B,D]
        dY = torch.stack([Delta[w] for u, w, v0 in trials])
        return [(xslot, dX.to(device)), (yslot, dY.to(device))]

    dR = d_route.to(device)

    # --- P0 native behavioral gate ---
    native0_texts, native1_texts = [], []
    for u, w, v0 in trials:
        t0, *_ = _native_text(tok, u, w, 0)
        t1, *_ = _native_text(tok, u, w, 1)
        native0_texts.append(t0); native1_texts.append(t1)
    # length check — may vary with u,w; evaluate greedy per-row if needed
    @torch.no_grad()
    def greedy_one(text):
        e = tok.encode(text, add_special_tokens=False)
        ii = torch.tensor([e], dtype=torch.long, device=device)
        aa = torch.ones_like(ii)
        return int(last_token_logits(model, ii, aa).argmax(-1).item())

    n_ok0 = n_ok1 = 0
    for (u, w, v0), t0, t1 in zip(trials, native0_texts, native1_texts):
        gold_w = single_token_id(tok, w)
        gold_u = single_token_id(tok, u)
        if greedy_one(t0) == gold_w:
            n_ok0 += 1
        if greedy_one(t1) == gold_u:
            n_ok1 += 1
    behav0 = n_ok0 / len(trials)
    behav1 = n_ok1 / len(trials)
    p0_pass = behav0 >= BEHAV_GATE and behav1 >= BEHAV_GATE
    log(f"  P0 native behav: flag0→Y={behav0:.0%} flag1→X={behav1:.0%} "
        f"pass={p0_pass}")

    # --- conditions on injected carrier ---
    base = pack([])
    bind = pack(binds_for(1.0))
    full = pack(binds_for(1.0) + [(fpos, dR)])
    ablate = pack(  # no bind@X; only w@Y + route
        [(yslot, torch.stack([Delta[w] for u, w, v0 in trials]).to(device)),
         (fpos, dR)])
    empty = pack([(fpos, dR)])  # route only

    pref_base = _pref_uw(base, u_idx, w_idx)
    pref_bind = _pref_uw(bind, u_idx, w_idx)
    pref_full = _pref_uw(full, u_idx, w_idx)
    pref_ablate = _pref_uw(ablate, u_idx, w_idx)
    pref_empty = _pref_uw(empty, u_idx, w_idx)
    RS = pref_full - pref_bind
    RS_ablate = pref_ablate - pref_bind  # route without u@X vs binds-only baseline
    # Better ablate metric: preference for u under (route + only Y) vs (route + XY)
    dep_gap = pref_full - pref_ablate  # how much u@X contributes under routing

    log(f"  prefs: base={pref_base:+.2f} bind={pref_bind:+.2f} full={pref_full:+.2f} "
        f"ablate={pref_ablate:+.2f} empty={pref_empty:+.2f}")
    log(f"  RS={RS:+.2f} dep_gap(full−ablate)={dep_gap:+.2f}")

    # --- null: random same-norm route directions ---
    ns = float(d_route.norm().clamp(min=1e-8))
    hb = Heartbeat(n_null, "delta_protocol_null", every_sec=15, out_dir=out_dir)
    null_rs = []
    for _ in range(n_null):
        r = torch.from_numpy(rng.normal(size=d_route.numel()).astype(np.float32))
        r = (r / r.norm().clamp(min=1e-8) * ns).to(device)
        full_r = pack(binds_for(1.0) + [(fpos, r)])
        null_rs.append(_pref_uw(full_r, u_idx, w_idx) - pref_bind)
        hb.step()
    hb.done()
    p_rs = permutation_pvalue(RS, np.array(null_rs), "greater")
    p1_pass = bool(RS > 0 and p_rs < 0.01)
    log(f"  P1 RS vs null: RS={RS:+.2f} p={p_rs:.3f} pass={p1_pass}")

    # --- P2v2 dependency (ablate) — metric fix, pre-registered 2026-07-13 ---
    # Compare write contribution to *routed content*, not to RS (which is
    # inflated by leaving the read-Y / prefer-w baseline).
    content = pref_full - pref_empty
    p2_pass = bool(
        content > 0
        and dep_gap > 0.5 * content
        and pref_ablate < pref_full
    )
    log(f"  P2v2 ablate/dependency: dep_gap={dep_gap:+.2f} content(full−empty)={content:+.2f} "
        f"need>{0.5 * max(content, 0):.2f} pref_ablate={pref_ablate:+.2f} pass={p2_pass}")

    # --- P3 scale sweep ---
    scale_prefs = []
    for a in SCALE_ALPHAS:
        lg = pack(binds_for(a) + [(fpos, dR)])
        scale_prefs.append(_pref_uw(lg, u_idx, w_idx))
    # Spearman α vs pref
    xs = np.asarray(SCALE_ALPHAS, dtype=float)
    ys = np.asarray(scale_prefs, dtype=float)
    if ys.std() < 1e-8:
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(xs, ys)[0, 1])  # Pearson on ranks≈ for 4 pts; use rank
        rx = xs.argsort().argsort().astype(float)
        ry = ys.argsort().argsort().astype(float)
        spearman = float(np.corrcoef(rx, ry)[0, 1]) if ry.std() > 0 else 0.0
    p3_pass = bool(spearman > 0)
    log(f"  P3 scale: alphas={list(SCALE_ALPHAS)} prefs={[round(p,2) for p in scale_prefs]} "
        f"spearman={spearman:+.2f} pass={p3_pass}")

    # --- P4 empty ---
    p4_pass = bool(abs(pref_empty) < 0.5 * max(abs(RS), 1e-6) or abs(pref_empty) < 1.0)
    log(f"  P4 empty: pref_empty={pref_empty:+.2f} pass={p4_pass}")

    # --- Native RS (sanity, not a claim gate) ---
    # Rebuild native flag=0 batch if uniform; else skip
    native_rs = None
    try:
        nt, nxs, nys, nfs = [], [], [], []
        for u, w, v0 in trials:
            text, xs, ys, fp = _native_text(tok, u, w, 0)
            nt.append(text); nxs.append(xs); nys.append(ys); nfs.append(fp)
        if (len(set(len(tok.encode(t, add_special_tokens=False)) for t in nt)) == 1
                and len(set(nxs)) == 1 and len(set(nfs)) == 1):
            nids = torch.tensor([tok.encode(t, add_special_tokens=False) for t in nt],
                                dtype=torch.long, device=device)
            nam = torch.ones_like(nids)
            nb = _logits_vals(model, nids, nam, layer, [], val_ids)
            nf = _logits_vals(model, nids, nam, layer, [(nfs[0], dR)], val_ids)
            native_rs = _pref_uw(nf, u_idx, w_idx) - _pref_uw(nb, u_idx, w_idx)
            log(f"  native RS (sanity)={native_rs:+.2f}")
    except Exception as e:
        log(f"  native RS skipped: {e}")

    # --- Verdict ---
    knobs = (native_rs is not None and native_rs > 1.0 and not p1_pass)
    if p0_pass and p1_pass and p2_pass and p3_pass and p4_pass:
        verdict = "PROTOCOL_COMPATIBLE"
    elif p0_pass and p1_pass and (p2_pass or p3_pass):
        verdict = "COMPATIBLE_WEAK"
    elif knobs or (p0_pass and not p1_pass and native_rs is not None and native_rs > 0):
        verdict = "KNOBS_NOT_PROTOCOL"
    else:
        verdict = "INCOMPATIBLE"

    results = {
        "stage": "delta_protocol", "version": VERSION, "model_path": model_path,
        "layer": layer, "n_null": n_null, "n_trials": len(trials),
        "route_src": route_src, "behav_gate": BEHAV_GATE,
        "question": ("Do independently discovered intervention directions write "
                     "representations that independently discovered downstream "
                     "mechanisms can consume?"),
        "P0_native_behav": {"flag0_to_Y": behav0, "flag1_to_X": behav1,
                            "pass": p0_pass},
        "prefs": {
            "base": pref_base, "bind": pref_bind, "full": pref_full,
            "ablate": pref_ablate, "empty": pref_empty,
        },
        "RS": {"value": RS, "p": float(p_rs), "pass": p1_pass},
        "dependency": {
            "dep_gap_full_minus_ablate": dep_gap,
            "content_full_minus_empty": content,
            "p2_rule": "dep_gap > 0.5 * (full - empty) and ablate < full",
            "pass_ablate": p2_pass,
            "scale_alphas": list(SCALE_ALPHAS), "scale_prefs": scale_prefs,
            "spearman": spearman, "pass_scale": p3_pass,
        },
        "empty_control": {"pref": pref_empty, "pass": p4_pass},
        "native_RS_sanity": native_rs,
        "gates": {"P0": p0_pass, "P1": p1_pass, "P2": p2_pass,
                  "P3": p3_pass, "P4": p4_pass},
        "verdict": verdict,
        "framing": {
            "PROTOCOL_COMPATIBLE": "injected write is consumed and depended on",
            "COMPATIBLE_WEAK": "RS holds; dependency partial",
            "KNOBS_NOT_PROTOCOL": "native routing works; injected write not re-read",
            "INCOMPATIBLE": "no protocol-compat signal under this design",
        }[verdict],
        "trials": [{"u": u, "w": w, "v0": v0} for u, w, v0 in trials],
        "slots": {"xslot": int(xslot), "yslot": int(yslot), "flag_pos": int(fpos)},
    }
    path = os.path.join(out_dir, "results_delta_protocol.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT delta_protocol: {verdict} | RS={RS:+.2f}(p={p_rs:.3f}) "
        f"dep_gap={dep_gap:+.2f} spearman={spearman:+.2f} | "
        f"gates P0–P4={[p0_pass,p1_pass,p2_pass,p3_pass,p4_pass]}")
    return results
