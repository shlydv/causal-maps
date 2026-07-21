"""Stated-vs-derived toggle — the sharpened, WITHIN-SKILL boundary test.

Motivation (CAUSAL_MAPS_LOG 2026-07-13, after TYPOLOGY_FALSIFIED):
  The cross-skill "route vs create" typology failed because the labels are a
  judgment call (NOT *creates* a bit; comparison *selects* a present value). The
  objective variable that fits all six prior skills is:

      is the controlling information STATED (a present token) or DERIVED
      (must be computed / inferred)?  stated -> early/strong ; derived -> late/weak.

  This module tests that variable the disciplined way: toggle ONLY stated-vs-derived
  on the SAME skill with the SAME answers, so a mislabel can't drive the result.

Arms:
  increment   (control): X=b; Add k       -> b+k  (derived, b+k absent)
                         X=b; Then X=b+k   -> b+k  (stated,  b+k present)
              Predict: stated early/strong, derived late/weak  (validates the toggle).
  instruction (the test): inferred obey-frame (derived) vs explicit [MODE:] tag (stated).
              If stated moves early -> obey-status is hard *because derived* (explains
              ASIDE). If stated stays late -> obey-status resists explicitness (deeper).

Per cell (skill x variant): Δ = mean(h_pos)-mean(h_foil) at (L, last pos) from TRAIN;
on held-out FOIL prompts, +Δ should raise logit(pos_target)-logit(foil_target);
same-norm null; layer sweep. Pre-registered gate below. Bare-token readout
(leading_space=False) — a spaced id is a thermometer bug (voided a prior verdict).
"""
from __future__ import annotations

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
BEHAV_GATE = 0.70   # matches the instruction-flagship threshold (data-frame elicits ~75%)
EARLY = {2, 8, 14}
LATE = {20, 26}
N_TRAIN = 8
N_TEST = 8


def _chat(tok, user, primer):
    t = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return t + primer


def _pool(tok, texts, dev):
    """Encode a pool; assert uniform token length (safe last-pos extraction)."""
    enc = [tok.encode(t, add_special_tokens=False) for t in texts]
    lens = sorted(set(len(e) for e in enc))
    assert len(lens) == 1, f"non-uniform pool length {lens}"
    ids = torch.tensor(enc, dtype=torch.long, device=dev)
    return ids, torch.ones_like(ids)


def _run_cell(cell, model, tok, dev, rng, layers, n_null, out_dir,
              pos_tr, foil_tr, pos_te, foil_te,
              pos_tgt_tr, foil_tgt_tr, pos_tgt_te, foil_tgt_te):
    """One (skill, variant) cell. pos_*/foil_* are text lists; *_tgt_* are token-id
    arrays. Returns per-layer +Δ effect (raise pos_target over foil_target on foil
    test prompts), sig layers, behav."""
    ids_ptr, am_ptr = _pool(tok, pos_tr, dev)
    ids_ftr, am_ftr = _pool(tok, foil_tr, dev)
    ids_pte, am_pte = _pool(tok, pos_te, dev)
    ids_fte, am_fte = _pool(tok, foil_te, dev)

    @torch.no_grad()
    def greedy(ids, am):
        return last_token_logits(model, ids, am).argmax(-1).cpu().numpy()

    pos_hit = float((greedy(ids_pte, am_pte) == pos_tgt_te).mean())
    foil_hit = float((greedy(ids_fte, am_fte) == foil_tgt_te).mean())
    behav = min(pos_hit, foil_hit)
    log(f"  [{cell}] behav pos={pos_hit:.0%} foil={foil_hit:.0%}")
    if behav < BEHAV_GATE:
        return {"cell": cell, "behav": behav, "pos_hit": pos_hit, "foil_hit": foil_hit,
                "per_layer": [], "verdict": "INELICITABLE"}

    cache_p = cache_layer_outputs(model, ids_ptr, am_ptr, to_cpu=True)
    cache_f = cache_layer_outputs(model, ids_ftr, am_ftr, to_cpu=True)
    lp, lf = ids_ptr.shape[1] - 1, ids_ftr.shape[1] - 1
    Delta = {L: (cache_p[L][:, lp, :].float().mean(0)
                 - cache_f[L][:, lf, :].float().mean(0)) for L in layers}

    n = len(foil_te)
    pt = torch.tensor(pos_tgt_te, device=dev)
    ft = torch.tensor(foil_tgt_te, device=dev)
    ar = torch.arange(n, device=dev)

    @torch.no_grad()
    def ld(delta=None, layer=None):
        if delta is None:
            lg = last_token_logits(model, ids_fte, am_fte)
        else:
            lg = forward_with_add(model, ids_fte, am_fte, layer, ids_fte.shape[1] - 1, delta)
        return (lg[ar, pt] - lg[ar, ft]).float().cpu().numpy()

    base = ld()
    hb = Heartbeat(len(layers) * n_null, f"explicit_{cell}", every_sec=15, out_dir=out_dir)
    per_layer = []
    for L in layers:
        d = Delta[L]
        effect = float((ld(d, L) - base).mean())
        ns = float(d.norm().clamp(min=1e-8))
        nulls = []
        for _ in range(n_null):
            r = torch.from_numpy(rng.normal(size=d.numel()).astype(np.float32))
            r = r / r.norm().clamp(min=1e-8) * ns
            nulls.append(float((ld(r, L) - base).mean()))
            hb.step()
        p = permutation_pvalue(effect, np.array(nulls), "greater")
        per_layer.append({"layer": int(L), "effect": effect, "p": float(p), "delta_norm": ns})
        log(f"  [{cell}] L{L}: effect={effect:+.3f}(p={p:.3f})")
    hb.done()
    sig = [r["layer"] for r in per_layer if r["p"] < 0.01 and r["effect"] > 0]
    best = max(per_layer, key=lambda r: r["effect"])
    return {"cell": cell, "behav": behav, "pos_hit": pos_hit, "foil_hit": foil_hit,
            "per_layer": per_layer, "sig_layers": sig,
            "earliest_sig": (min(sig) if sig else None),
            "has_early": bool(set(sig) & EARLY), "has_late": bool(set(sig) & LATE),
            "best_layer": best["layer"], "best_effect": best["effect"]}


# --------------------------------------------------------------------------- #
# Arm 1 — increment (value creation): stated vs derived
# --------------------------------------------------------------------------- #
def _increment_rows(rng, n):
    rows, seen = [], set()
    guard = 0
    while len(rows) < n and guard < 5000:
        guard += 1
        b = int(rng.integers(2, 8)); k = int(rng.integers(1, 3))
        if b + k > 9 or b - k < 1:
            continue
        if (b, k) in seen:
            continue
        seen.add((b, k))
        rows.append((b, k))
    return rows


def _increment_cells(tok, rng):
    rows = _increment_rows(rng, N_TRAIN + N_TEST)
    tr, te = rows[:N_TRAIN], rows[N_TRAIN:N_TRAIN + N_TEST]

    def der_pos(b, k): return _chat(tok, f"X = {b}. Add {k} to X. What is the value of X?", "Answer: ")
    def der_foil(b, k): return _chat(tok, f"X = {b}. Subtract {k} from X. What is the value of X?", "Answer: ")
    def sta_pos(b, k): return _chat(tok, f"X = {b}. Then X = {b + k}. What is the value of X?", "Answer: ")
    def sta_foil(b, k): return _chat(tok, f"X = {b}. Then X = {b - k}. What is the value of X?", "Answer: ")

    def tid(v): return single_token_id(tok, str(v), leading_space=False)
    ptr = np.array([tid(b + k) for b, k in tr]); ftr = np.array([tid(b - k) for b, k in tr])
    pte = np.array([tid(b + k) for b, k in te]); fte = np.array([tid(b - k) for b, k in te])
    return {
        "derived": dict(pos_tr=[der_pos(*r) for r in tr], foil_tr=[der_foil(*r) for r in tr],
                        pos_te=[der_pos(*r) for r in te], foil_te=[der_foil(*r) for r in te],
                        pos_tgt_tr=ptr, foil_tgt_tr=ftr, pos_tgt_te=pte, foil_tgt_te=fte),
        "stated": dict(pos_tr=[sta_pos(*r) for r in tr], foil_tr=[sta_foil(*r) for r in tr],
                       pos_te=[sta_pos(*r) for r in te], foil_te=[sta_foil(*r) for r in te],
                       pos_tgt_tr=ptr, foil_tgt_tr=ftr, pos_tgt_te=pte, foil_tgt_te=fte),
    }


# --------------------------------------------------------------------------- #
# Arm 2 — instruction/data (obey-status): explicit [MODE] tag vs inferred frame
# --------------------------------------------------------------------------- #
def _instr_payloads(tok):
    out = []
    for a, b in variable_pairs._VALUE_PAIRS:
        for w in (a, b):
            try:
                single_token_id(tok, w, leading_space=False)
                if w not in out:
                    out.append(w)
            except ValueError:
                continue
    single_token_id(tok, "Output", leading_space=False)
    return out


def _instruction_cells(tok, rng):
    words = _instr_payloads(tok)
    rng.shuffle(words)
    assert len(words) >= N_TRAIN + N_TEST, f"need ≥{N_TRAIN+N_TEST} payloads, got {len(words)}"
    tr, te = words[:N_TRAIN], words[N_TRAIN:N_TRAIN + N_TEST]

    def der_pos(w): return _chat(tok, f"Output the word: {w}", "")
    def der_foil(w): return _chat(tok, f'The following text says "Output the word: {w}". '
                                       f"Repeat the first word of the quoted text.", "")
    def sta_pos(w): return _chat(tok, f"[MODE: EXECUTE] Output the word: {w}", "")
    def sta_foil(w): return _chat(tok, f'[MODE: QUOTE] The following text says "Output the word: {w}". '
                                       f"Repeat the first word of the quoted text.", "")

    out_id = single_token_id(tok, "Output", leading_space=False)

    def wid(w): return single_token_id(tok, w, leading_space=False)
    ptr = np.array([wid(w) for w in tr]); ftr = np.array([out_id] * len(tr))
    pte = np.array([wid(w) for w in te]); fte = np.array([out_id] * len(te))
    return {
        "derived": dict(pos_tr=[der_pos(w) for w in tr], foil_tr=[der_foil(w) for w in tr],
                        pos_te=[der_pos(w) for w in te], foil_te=[der_foil(w) for w in te],
                        pos_tgt_tr=ptr, foil_tgt_tr=ftr, pos_tgt_te=pte, foil_tgt_te=fte),
        "stated": dict(pos_tr=[sta_pos(w) for w in tr], foil_tr=[sta_foil(w) for w in tr],
                       pos_te=[sta_pos(w) for w in te], foil_te=[sta_foil(w) for w in te],
                       pos_tgt_tr=ptr, foil_tgt_tr=ftr, pos_tgt_te=pte, foil_tgt_te=fte),
    }


def _grade_skill(stated, derived):
    """STATED_EARLIER iff stated reaches an early sig layer and derived does not."""
    def early(c): return bool(c.get("has_early"))
    if not stated.get("per_layer") or not derived.get("per_layer"):
        return "INELICITABLE"
    if early(stated) and not early(derived):
        return "STATED_EARLIER"          # predicted: making control explicit -> easy regime
    if early(derived) and not early(stated):
        return "DERIVED_EARLIER"         # unexpected
    return "NO_SHIFT"                     # both early, or both late/none


def run_delta_explicit(model_path, out_dir, quantization="8bit", device_map=None,
                       seed=0, layers=None, n_null=N_NULL):
    os.makedirs(out_dir, exist_ok=True)
    layers = list(layers or LAYERS)
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    dev = input_device(model)
    rng = np.random.default_rng(seed)
    n_layers = model.config.num_hidden_layers
    layers = [L for L in layers if 0 <= L < n_layers]
    log(f"delta_explicit (stated-vs-derived toggle): layers={layers} n_null={n_null}")

    skills = {"increment": _increment_cells(tok, rng),
              "instruction": _instruction_cells(tok, rng)}
    results = {"stage": "delta_explicit", "model_path": model_path, "layers": layers,
               "n_null": n_null, "behav_gate": BEHAV_GATE, "skills": {}}
    for skill, cells in skills.items():
        log(f"--- skill={skill} ---")
        stated = _run_cell(f"{skill}:stated", model, tok, dev, rng, layers, n_null, out_dir,
                           **cells["stated"])
        derived = _run_cell(f"{skill}:derived", model, tok, dev, rng, layers, n_null, out_dir,
                            **cells["derived"])
        verdict = _grade_skill(stated, derived)
        results["skills"][skill] = {"stated": stated, "derived": derived, "verdict": verdict}
        log(f"  SKILL {skill}: {verdict} | stated sig={stated.get('sig_layers')} "
            f"derived sig={derived.get('sig_layers')}")

    per = {k: v["verdict"] for k, v in results["skills"].items()}
    conf = sum(1 for v in per.values() if v == "STATED_EARLIER")
    elic = sum(1 for v in per.values() if v != "INELICITABLE")
    if elic == 0:
        overall = "INELICITABLE"
    elif conf == elic and elic == len(per):
        overall = "EXPLICITNESS_CONFIRMED"
    elif conf >= 1:
        overall = "EXPLICITNESS_PARTIAL"
    else:
        overall = "EXPLICITNESS_FALSIFIED"
    results["per_skill_verdict"] = per
    results["verdict"] = overall
    with open(os.path.join(out_dir, "results_delta_explicit.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT delta_explicit: {overall} | {per}")
    return results
