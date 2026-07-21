"""Multi-slot binding composition (Variable skill).

Do two value directions compose? In a two-variable prompt, install X=vX at X's
value slot AND Y=vY at Y's slot simultaneously by ADDING their directions, and
retrieve both independently. Tests compositional working-memory via directions,
which is stronger than single-output vector arithmetic.

Pre-registered in CAUSAL_MAPS_LOG.md (2026-07-12). Gates:
  M1 single-slot install transfers to the 2-var template (add-X ⇒ query-X selects vX)
  M2 simultaneous install (add-both ⇒ query-X selects vX AND query-Y selects vY)
  M3 independence (add-both keeps ≥0.7x single-slot selectivity; add-Y doesn't raise vX)
  Verdict COMPOSES ⟺ M1∧M2∧M3.
"""
import json
import os

import numpy as np
import torch

from . import variable_pairs
from .direction_transfer import PRIMARY_LAYER, _slot_acts
from .logutil import Heartbeat, log
from .model_utils import (input_device, last_token_logits,
                          load_model_and_tokenizer, single_token_id)
from .nulls import permutation_pvalue
from .patching import _split_output, get_decoder_layers
from .tensorize import _anchor_token_index, tensorize_pairs

N_NULL = 100


@torch.no_grad()
def forward_add_multi(model, ids, am, layer, pos_deltas):
    """Forward adding each `delta` ([B,D] or [D]) at its `pos`, all at `layer`.
    pos_deltas: list of (pos, delta). Returns last-token logits [B, V]."""
    lyr = get_decoder_layers(model)[layer]

    def hook(module, inp, out):
        hs, rebuild = _split_output(out)
        hs = hs.clone()
        for p, d in pos_deltas:
            hs[:, p, :] = hs[:, p, :] + d.to(dtype=hs.dtype, device=hs.device)
        return rebuild(hs)

    h = lyr.register_forward_hook(hook)
    try:
        out = model(input_ids=ids, attention_mask=am, use_cache=False)
    finally:
        h.remove()
    return out.logits[:, -1, :]


def _two_var_text(tok, x, y, a, b, q):
    user = f"Let {x} = {a}. Let {y} = {b}. What is the value of {q}?"
    templated = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    text = templated + f"{q} ="
    off_a = text.find(f"Let {x} = ") + len(f"Let {x} = ")
    off_b = text.find(f"Let {y} = ") + len(f"Let {y} = ")
    return text, off_a, off_b


def _selectivity(dlogit, target_idx):
    """dlogit [B, nV], target_idx [B]. Returns per-trial [Δ(target) − mean(others)]."""
    nV = dlogit.shape[1]
    tgt = dlogit[np.arange(len(target_idx)), target_idx]
    others = (dlogit.sum(1) - tgt) / (nV - 1)
    return tgt - others


def run_delta_multislot(model_path, out_dir, quantization="8bit", device_map=None,
                        layer=PRIMARY_LAYER, seed=0, n_trials=12):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)  # reproducible null directions
    model, tok = load_model_and_tokenizer(model_path, device_map=device_map,
                                          quantization=quantization)
    # --- per-value directions from single-var pairs (as in delta_decompose) ---
    pairs = variable_pairs.make_variable_pairs(50, seed=seed, tok=tok)
    batch = tensorize_pairs(tok, pairs, require_anchor_roles=("val_slot",))
    pos = batch["anchors"]["val_slot"]
    metas = batch["metas"]
    values = []
    for _, v1 in variable_pairs._VALUE_PAIRS:
        try:
            single_token_id(tok, v1)
        except ValueError:
            continue
        if sum(1 for m in metas if m["val_cf"] == v1) >= 2 and v1 not in values:
            values.append(v1)
    vidx = {v: i for i, v in enumerate(values)}
    val_ids = [single_token_id(tok, v) for v in values]
    hc, hf = _slot_acts(model, batch, layer, pos)
    dpp = hf - hc
    Delta = {v: dpp[[i for i, m in enumerate(metas) if m["val_cf"] == v]].mean(0)
             for v in values}
    dev = input_device(model)

    # --- trials: distinct (vX, vY, v0) ---
    rng = np.random.default_rng(seed)
    trials, used = [], set()
    tries = 0
    while len(trials) < n_trials and tries < 1000:
        tries += 1
        vX, vY, v0 = (str(z) for z in rng.choice(values, size=3, replace=False))
        if (vX, vY) in used:
            continue
        used.add((vX, vY))
        trials.append((vX, vY, v0))
    log(f"delta_multislot: layer={layer} nV={len(values)} n_trials={len(trials)}")

    def build_batch(query):  # query in {"X","Y"}
        ids_list, xs, ys = [], [], []
        for (vX, vY, v0) in trials:
            text, off_a, off_b = _two_var_text(tok, "X", "Y", v0, v0, query)
            xi = _anchor_token_index(tok, text, off_a)
            yi = _anchor_token_index(tok, text, off_b)
            ids = tok.encode(text, add_special_tokens=False)
            ids_list.append(ids); xs.append(xi); ys.append(yi)
        S = len(ids_list[0])
        assert all(len(r) == S for r in ids_list), "non-uniform length in 2-var prompts"
        assert len(set(xs)) == 1 and len(set(ys)) == 1, f"non-uniform slots xs={set(xs)} ys={set(ys)}"
        t = torch.tensor(ids_list, dtype=torch.long, device=dev)
        return t, torch.ones_like(t), xs[0], ys[0]

    def logits_at(ids, am, pos_deltas):
        if pos_deltas is None:
            lg = last_token_logits(model, ids, am)
        else:
            lg = forward_add_multi(model, ids, am, layer, pos_deltas)
        return lg[:, val_ids].float().cpu().numpy()

    results = {"stage": "delta_multislot", "model_path": model_path, "layer": int(layer),
               "n_trials": len(trials), "values": values, "n_null": N_NULL, "queries": {}}
    hb = Heartbeat(2 * 2 * N_NULL, "delta_multislot", every_sec=15, out_dir=out_dir)

    per_query = {}
    for query in ("X", "Y"):
        ids, am, xslot, yslot = build_batch(query)
        tgt_idx = np.array([vidx[t[0] if query == "X" else t[1]] for t in trials])
        # target's OWN slot / other slot
        own_slot = xslot if query == "X" else yslot
        oth_slot = yslot if query == "X" else xslot
        dX = torch.stack([Delta[t[0]] for t in trials])   # install-X deltas [B,D]
        dY = torch.stack([Delta[t[1]] for t in trials])
        d_own = dX if query == "X" else dY
        d_oth = dY if query == "X" else dX

        base = logits_at(ids, am, None)
        sel_own = _selectivity(logits_at(ids, am, [(own_slot, d_own)]) - base, tgt_idx)
        sel_both = _selectivity(
            logits_at(ids, am, [(xslot, dX), (yslot, dY)]) - base, tgt_idx)
        sel_othonly = _selectivity(logits_at(ids, am, [(oth_slot, d_oth)]) - base, tgt_idx)

        # nulls: random same-per-trial-norm dir at own slot (for M1) and both (for M2)
        norms_own = d_own.norm(dim=1, keepdim=True)
        n_own, n_both = [], []
        for _ in range(N_NULL):
            r = torch.randn(d_own.shape, generator=None)
            r = r / r.norm(dim=1, keepdim=True).clamp(min=1e-8) * norms_own
            n_own.append(float(_selectivity(
                logits_at(ids, am, [(own_slot, r)]) - base, tgt_idx).mean()))
            hb.step()
            r2 = torch.randn(d_oth.shape); r2 = r2 / r2.norm(dim=1, keepdim=True).clamp(min=1e-8) * d_oth.norm(dim=1, keepdim=True)
            n_both.append(float(_selectivity(
                logits_at(ids, am, [(own_slot, r), (oth_slot, r2)]) - base, tgt_idx).mean()))
            hb.step()
        m_own, m_both, m_oth = float(sel_own.mean()), float(sel_both.mean()), float(sel_othonly.mean())
        p_own = permutation_pvalue(m_own, np.array(n_own), "greater")
        p_both = permutation_pvalue(m_both, np.array(n_both), "greater")
        per_query[query] = {
            "sel_single_slot": m_own, "p_single": float(p_own),
            "sel_both": m_both, "p_both": float(p_both),
            "sel_otherslot_only": m_oth,
            "retention_both_over_single": (m_both / m_own) if abs(m_own) > 1e-6 else float("nan"),
        }
        log(f"  query {query}: single={m_own:+.3f}(p={p_own:.3f}) both={m_both:+.3f}"
            f"(p={p_both:.3f}) otheronly={m_oth:+.3f} ret={per_query[query]['retention_both_over_single']:.2f}")
    hb.done()
    results["queries"] = per_query

    M1 = all(per_query[q]["sel_single_slot"] > 0 and per_query[q]["p_single"] < 0.01 for q in "XY")
    M2 = all(per_query[q]["sel_both"] > 0 and per_query[q]["p_both"] < 0.01 for q in "XY")
    M3 = all(per_query[q]["retention_both_over_single"] >= 0.7 for q in "XY") and \
        all(abs(per_query[q]["sel_otherslot_only"]) < 0.5 * per_query[q]["sel_single_slot"] for q in "XY")
    results["gates"] = {"M1_single_transfers": bool(M1), "M2_simultaneous": bool(M2),
                        "M3_independence": bool(M3)}
    results["verdict"] = "COMPOSES" if (M1 and M2 and M3) else "DOES_NOT_COMPOSE_CLEANLY"

    with open(os.path.join(out_dir, "results_delta_multislot.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"VERDICT delta_multislot: {results['verdict']} | M1={M1} M2={M2} M3={M3} | "
        f"X:{per_query['X']} Y:{per_query['Y']}")
    return results
