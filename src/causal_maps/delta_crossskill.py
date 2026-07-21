"""Cross-skill composition: Variable + Completion directions in one prompt.

The deepest composition test — do HETEROGENEOUS computational primitives compose?
One joint prompt exercises both skills; we add the Variable value-direction at the
value slot AND the Completion bit-direction at the completion site (both at L2),
and read out value and next-action independently.

    Let X = {v0}.
    Rules:
    - if {flag} is 0, next action is {act0}
    - if {flag} is 1, next action is {act1}
    State:
    {flag} = 0
    <value readout>  What is the value of X?      primed "X ="        -> a value
    <action readout> What is the next action?      primed "The next action is" -> an action

Δ_V(v) = per-value Variable direction (L2, val_slot). Δ_C = Completion bit-flip
direction (L2, bit_slot+1; delta_completion's frozen peak was one token after the
bit). Base has value v0 and bit 0. Adding Δ_V(vX) should install vX for the value
readout; adding Δ_C should flip the action readout 0->1 (act0->act1).

Pre-registered (CAUSAL_MAPS_LOG.md). Gates:
  X1 each primitive transfers into the joint prompt (V-only steers value; C-only steers action)
  X2 simultaneous (add-both steers BOTH)
  X3 independence (retention >=0.7x each; cross-talk ~0)  -> COMPOSES_CROSS_SKILL
"""
import json
import os

import numpy as np
import torch

from . import completion_pairs, variable_pairs
from .direction_transfer import PRIMARY_LAYER, _slot_acts
from .logutil import Heartbeat, log
from .model_utils import (input_device, last_token_logits,
                          load_model_and_tokenizer, logit_diff, single_token_id)
from .nulls import permutation_pvalue
from .delta_multislot import forward_add_multi
from .tensorize import _anchor_token_index, tensorize_pairs

N_NULL = 100
COMPLETION_SITE_OFFSET = 1  # inject Δ_C at bit_slot + 1 (delta_completion peak)


def _variable_directions(model, tok, layer, seed):
    pairs = variable_pairs.make_variable_pairs(50, seed=seed, tok=tok)
    batch = tensorize_pairs(tok, pairs, require_anchor_roles=("val_slot",))
    pos = batch["anchors"]["val_slot"]; metas = batch["metas"]
    values = []
    for _, v1 in variable_pairs._VALUE_PAIRS:
        try:
            single_token_id(tok, v1)
        except ValueError:
            continue
        if sum(1 for m in metas if m["val_cf"] == v1) >= 2 and v1 not in values:
            values.append(v1)
    hc, hf = _slot_acts(model, batch, layer, pos)
    dpp = hf - hc
    Delta = {v: dpp[[i for i, m in enumerate(metas) if m["val_cf"] == v]].mean(0)
             for v in values}
    return Delta, values


def _completion_direction(model, tok, layer, seed):
    pairs = completion_pairs.make_completion_pairs(
        40, seed=seed, tok=tok, families=["explicit_A", "explicit_B", "explicit_C", "explicit_D"])
    batch = tensorize_pairs(tok, pairs, require_anchor_roles=("bit_slot",))
    bit = batch["anchors"]["bit_slot"]
    site = bit + COMPLETION_SITE_OFFSET
    hc, hf = _slot_acts(model, batch, layer, site)
    return (hf - hc).mean(0)  # bit 0->1 direction (installs act1)


def _joint_text(tok, v0, flag, act0, act1, query):
    ctx = (f"Let X = {v0}.\nRules:\n- if {flag} is 0, next action is {act0}\n"
           f"- if {flag} is 1, next action is {act1}\nState:\n{flag} = 0\n\n")
    q = "What is the value of X?" if query == "value" else "What is the next action?"
    user = ctx + q
    templated = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    primer = "X =" if query == "value" else "The next action is"
    text = templated + primer
    off_val = text.find("Let X = ") + len("Let X = ")
    off_bit = text.find(f"{flag} = ") + len(f"{flag} = ")
    return text, off_val, off_bit


def run_delta_crossskill(model_path, out_dir, quantization="8bit", device_map=None,
                         layer=PRIMARY_LAYER, seed=0, n_trials=10):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    Delta, values = _variable_directions(model, tok, layer, seed)
    dC = _completion_direction(model, tok, layer, seed)
    val_ids = [single_token_id(tok, v) for v in values]
    vidx = {v: i for i, v in enumerate(values)}
    dev = input_device(model)
    log(f"delta_crossskill: L={layer} nvalues={len(values)} ||dC||={float(dC.norm()):.3f}")

    # trials: completion rows with single-token acts, paired with target/base values
    rng = np.random.default_rng(seed)
    rows = [(f, a0, a1) for (f, a0, a1) in completion_pairs._ROWS
            if _both_single_token(tok, a0, a1)]
    trials = []
    for i, (flag, act0, act1) in enumerate(rows):
        if len(trials) >= n_trials:
            break
        vX, v0 = (str(z) for z in rng.choice(values, size=2, replace=False))
        trials.append({"flag": flag, "act0": act0, "act1": act1, "vX": vX, "v0": v0})
    log(f"  n_trials={len(trials)} (rows w/ single-token acts={len(rows)})")

    def build(query):
        ids_list, ov, ob = [], [], []
        for t in trials:
            text, off_val, off_bit = _joint_text(tok, t["v0"], t["flag"], t["act0"], t["act1"], query)
            vi = _anchor_token_index(tok, text, off_val)
            bi = _anchor_token_index(tok, text, off_bit)
            ids_list.append(tok.encode(text, add_special_tokens=False)); ov.append(vi); ob.append(bi)
        S = len(ids_list[0])
        assert all(len(r) == S for r in ids_list), "non-uniform joint-prompt length"
        assert len(set(ov)) == 1 and len(set(ob)) == 1, f"non-uniform slots {set(ov)} {set(ob)}"
        t_ids = torch.tensor(ids_list, dtype=torch.long, device=dev)
        return t_ids, torch.ones_like(t_ids), ov[0], ob[0] + COMPLETION_SITE_OFFSET

    v_ids, v_am, v_vslot, v_cpos = build("value")
    a_ids, a_am, a_vslot, a_cpos = build("action")
    dV = torch.stack([Delta[t["vX"]] for t in trials])         # [B,D] per-trial value dir
    vX_idx = np.array([vidx[t["vX"]] for t in trials])
    act0_ids = torch.tensor([single_token_id(tok, t["act0"]) for t in trials])
    act1_ids = torch.tensor([single_token_id(tok, t["act1"]) for t in trials])

    @torch.no_grad()
    def val_effect(pos_deltas):
        lg = (last_token_logits(model, v_ids, v_am) if pos_deltas is None
              else forward_add_multi(model, v_ids, v_am, layer, pos_deltas))
        return lg[:, val_ids].float().cpu().numpy()

    @torch.no_grad()
    def act_ld(pos_deltas):
        lg = (last_token_logits(model, a_ids, a_am) if pos_deltas is None
              else forward_add_multi(model, a_ids, a_am, layer, pos_deltas))
        return logit_diff(lg, act1_ids, act0_ids).float().cpu().numpy()  # act1 - act0

    base_v = val_effect(None)
    base_a = act_ld(None)

    def v_sel(pos_deltas):  # value selectivity for vX (mean over trials)
        dl = val_effect(pos_deltas) - base_v
        tgt = dl[np.arange(len(vX_idx)), vX_idx]
        others = (dl.sum(1) - tgt) / (dl.shape[1] - 1)
        return float((tgt - others).mean())

    def a_eff(pos_deltas):  # Δ(act1 - act0) toward act1
        return float((act_ld(pos_deltas) - base_a).mean())

    def rnd_like(d):
        r = torch.randn(d.shape if d.dim() == 2 else (len(trials), d.shape[0]))
        n = (d.norm(dim=1, keepdim=True) if d.dim() == 2 else d.norm())
        return r / r.norm(dim=1, keepdim=True).clamp(min=1e-8) * n

    hb = Heartbeat(4 * N_NULL, "delta_crossskill", every_sec=15, out_dir=out_dir)

    def null_of(measure, make_pd):
        out = []
        for _ in range(N_NULL):
            out.append(measure(make_pd()))
            hb.step()
        return np.array(out)

    # real effects
    v_Vonly = v_sel([(v_vslot, dV)])
    v_both = v_sel([(v_vslot, dV), (v_cpos, dC)])
    v_Conly = v_sel([(v_cpos, dC)])                    # cross-talk: C should NOT install a value
    a_Conly = a_eff([(a_cpos, dC)])
    a_both = a_eff([(a_vslot, dV), (a_cpos, dC)])
    a_Vonly = a_eff([(a_vslot, dV)])                   # cross-talk: V should NOT flip the action

    # nulls (one forward each -> one hb.step each; total = 4*N_NULL)
    p_vV = permutation_pvalue(v_Vonly, null_of(v_sel, lambda: [(v_vslot, rnd_like(dV))]), "greater")
    p_vB = permutation_pvalue(v_both, null_of(v_sel, lambda: [(v_vslot, rnd_like(dV)), (v_cpos, rnd_like(dC))]), "greater")
    p_aC = permutation_pvalue(a_Conly, null_of(a_eff, lambda: [(a_cpos, rnd_like(dC))]), "greater")
    p_aB = permutation_pvalue(a_both, null_of(a_eff, lambda: [(a_vslot, rnd_like(dV)), (a_cpos, rnd_like(dC))]), "greater")
    hb.done()

    ret_v = v_both / v_Vonly if abs(v_Vonly) > 1e-6 else float("nan")
    ret_a = a_both / a_Conly if abs(a_Conly) > 1e-6 else float("nan")
    X1 = v_Vonly > 0 and p_vV < 0.01 and a_Conly > 0 and p_aC < 0.01
    X2 = v_both > 0 and p_vB < 0.01 and a_both > 0 and p_aB < 0.01
    X3 = (ret_v >= 0.7 and ret_a >= 0.7
          and abs(v_Conly) < 0.5 * v_Vonly and abs(a_Vonly) < 0.5 * a_Conly)
    verdict = "COMPOSES_CROSS_SKILL" if (X1 and X2 and X3) else "DOES_NOT_COMPOSE_CLEANLY"

    results = {
        "stage": "delta_crossskill", "model_path": model_path, "layer": int(layer),
        "n_trials": len(trials), "n_null": N_NULL, "completion_site_offset": COMPLETION_SITE_OFFSET,
        "value": {"V_only": v_Vonly, "p_V": float(p_vV), "both": v_both, "p_both": float(p_vB),
                  "C_only_crosstalk": v_Conly, "retention_both_over_single": ret_v},
        "action": {"C_only": a_Conly, "p_C": float(p_aC), "both": a_both, "p_both": float(p_aB),
                   "V_only_crosstalk": a_Vonly, "retention_both_over_single": ret_a},
        "gates": {"X1_each_transfers": bool(X1), "X2_simultaneous": bool(X2),
                  "X3_independence": bool(X3)},
        "verdict": verdict,
    }
    with open(os.path.join(out_dir, "results_delta_crossskill.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT delta_crossskill: {verdict} | X1={X1} X2={X2} X3={X3}")
    log(f"  value: Vonly={v_Vonly:+.3f}(p={p_vV:.3f}) both={v_both:+.3f}(p={p_vB:.3f}) "
        f"Conly_xtalk={v_Conly:+.3f} ret={ret_v:.2f}")
    log(f"  action: Conly={a_Conly:+.3f}(p={p_aC:.3f}) both={a_both:+.3f}(p={p_aB:.3f}) "
        f"Vonly_xtalk={a_Vonly:+.3f} ret={ret_a:.2f}")
    return results


def _both_single_token(tok, a0, a1):
    try:
        single_token_id(tok, a0); single_token_id(tok, a1)
        return True
    except ValueError:
        return False
