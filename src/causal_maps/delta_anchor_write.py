"""Anchor-write test — the token-anchored theory's positive arm.

Pre-registered CAUSAL_MAPS_LOG.md 2026-07-15. The report_only_v1 run showed
the derived six-variable belief world is NOT stored at a query-independent
checkpoint token (full-state swap moves ~0.1% of the natural effect at every
layer). The token-anchored theory says the state lives at the STATED anchor
tokens and is re-derived at query time. Prediction: a neutral-carrier write
z(Rome)-z(Paris) at the ANCHOR token (Alice's cube-event location in the
history) edits her belief with all and only its consequences — success
exactly where the checkpoint edit got nothing, in the SAME world.

Reuses the structured-workspace world, batching, and metrics verbatim; the
write recipe (neutral carrier + L2 + prototype difference) is the one
validated by the entity matrix at 14B-AWQ.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .delta_structured_workspace import (_accuracy, _batch, _counterfactual,
                                         _locations, _rows, _switch_metrics,
                                         _switch_pass)
from .delta_trajectory import _forward
from .logutil import Heartbeat, log
from .model_utils import input_device, load_model_and_tokenizer
from .nulls import permutation_pvalue

N_NULL = 30
LAYER = 2
G0 = 0.80
GATE_P = 0.04
SOURCE, TARGET = "Paris", "Rome"


def _resolve(model_path):
    if "*" in model_path:
        import glob as _glob
        hits = sorted(_glob.glob(model_path, recursive=True))
        assert hits, f"model_path glob matched nothing: {model_path}"
        mp = hits[0]
        if os.path.basename(mp) == "config.json":
            mp = os.path.dirname(mp)
        return mp
    return model_path


def _anchor_position(clean_batch, natural_batch):
    """The single position where clean/natural prompts differ, uniform
    across rows (the CF changes exactly one stated location token)."""
    ci, ni = clean_batch["ids"], natural_batch["ids"]
    assert ci.shape == ni.shape, "clean/natural shapes differ"
    positions = []
    for r in range(ci.shape[0]):
        d = (ci[r] != ni[r]).nonzero().flatten().tolist()
        assert len(d) == 1, f"row {r}: CF changed {len(d)} positions"
        positions.append(d[0])
    assert len(set(positions)) == 1, f"non-uniform anchors: {positions}"
    return positions[0]


@torch.no_grad()
def _neutral_states(model, tok, dev, layer, locations):
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": f"Here is a word: {loc}."}],
        tokenize=False, add_generation_prompt=True) for loc in locations]
    enc = [tok.encode(t, add_special_tokens=False) for t in texts]
    assert len({len(e) for e in enc}) == 1, "neutral pool non-uniform"
    d = [i for i, (x, y) in enumerate(zip(enc[0], enc[1])) if x != y]
    assert len(d) == 1, "neutral prompts differ at more than the word token"
    ids = torch.tensor(enc, dtype=torch.long, device=dev)
    _, cache = _forward(model, ids, torch.ones_like(ids), (d[0],), (layer,))
    return {loc: cache[layer][i, 0].float()
            for i, loc in enumerate(locations)}


@torch.no_grad()
def run_delta_anchor_write(model_path, out_dir, quantization="awq",
                           device_map=None, seed=0, n_null=N_NULL,
                           n_rows=None, model=None, tok=None,
                           battery=False, clean_rows=None):
    # `battery=True` marks the pre-registered preprint widening protocol
    # (PREPRINT_PLAN.md M1/M2): any model/quantization, widened rows, seeds.
    # The original frozen pair stays enforced for standalone runs.
    if not battery and (quantization != "awq" or seed != 0):
        raise ValueError("anchor-write is frozen to the 14B-AWQ pair, seed 0")
    os.makedirs(out_dir, exist_ok=True)
    if model is None or tok is None:
        model, tok = load_model_and_tokenizer(_resolve(model_path),
                                              quantization=quantization)
    dev = input_device(model)
    rng = np.random.default_rng(seed)
    hb = Heartbeat(n_null, "anchor_write", every_sec=20, out_dir=out_dir)

    if clean_rows is None:
        clean_rows = _rows(SOURCE, TARGET, "ac", "test", n_rows=n_rows)
    else:
        clean_rows = [dict(row) for row in clean_rows]
    natural_rows = _counterfactual(clean_rows, {"ac": TARGET})
    bc_natural_rows = _counterfactual(clean_rows, {"bc": TARGET})
    Z = _neutral_states(model, tok, dev, LAYER, (SOURCE, TARGET))
    delta = Z[TARGET] - Z[SOURCE]
    log(f"anchor_write: |delta|={float(delta.norm()):.2f} layer={LAYER}")

    result = {"stage": "delta_anchor_write", "layer": LAYER, "seed": seed,
              "n_null": n_null, "source": SOURCE, "target": TARGET,
              "model_path": model_path, "n_rows": len(clean_rows),
              "rows": clean_rows}

    # ---- consequences: belief_ac + tell_ac at the ac anchor ----------------
    consequences = {}
    anchor_ac = None
    base_margin = add_margin = None
    for query in ("belief_ac", "tell_ac"):
        cb = _batch(tok, clean_rows, query, "narrative", dev)
        nb = _batch(tok, natural_rows, query, "narrative", dev)
        pos = _anchor_position(cb, nb)
        anchor_ac = pos if anchor_ac is None else anchor_ac
        assert pos == anchor_ac, "anchor differs across queries"
        cl, _ = _forward(model, cb["ids"], cb["am"], (pos,))
        nl, _ = _forward(model, nb["ids"], nb["am"], (pos,))
        g0c = _accuracy(cl, cb, _locations(clean_rows, query))
        g0n = _accuracy(nl, nb, _locations(natural_rows, query))
        add, _ = _forward(model, cb["ids"], cb["am"], (pos,),
                          add=(LAYER, pos, delta))
        m = _switch_metrics(cl, nl, add, cb, _locations(clean_rows, query),
                            _locations(natural_rows, query))
        m["g0_clean"], m["g0_natural"] = g0c, g0n
        m["pass"] = bool(min(g0c, g0n) >= G0 and _switch_pass(m))
        consequences[query] = m
        log(f"  [{query}] g0={g0c:.0%}/{g0n:.0%} nat={m['natural_effect']:+.1f} "
            f"eff={m['effect']:+.1f} ratio={m['ratio']} acc={m['target_acc']:.0%} "
            f"pass={m['pass']}")
        if query == "belief_ac":
            base_margin, add_margin = cl, add
            belief_batch, belief_pos = cb, pos
            belief_src = _locations(clean_rows, query)
            belief_tgt = _locations(natural_rows, query)
    result["consequences"] = consequences

    # ---- nulls on the belief_ac margin -------------------------------------
    nulls = []
    norm = float(delta.norm().clamp(min=1e-8))
    cl = base_margin
    from .delta_trajectory import _ld
    tgt_ids = torch.tensor([belief_batch["amap"][x] for x in belief_tgt])
    src_ids = torch.tensor([belief_batch["amap"][x] for x in belief_src])
    base_ld = _ld(cl, tgt_ids, src_ids)
    add_eff = float((_ld(add_margin, tgt_ids, src_ids) - base_ld).mean())
    for _ in range(n_null):
        r = torch.from_numpy(rng.normal(size=delta.numel()).astype(np.float32))
        r = r / r.norm().clamp(min=1e-8) * norm
        lg, _ = _forward(model, belief_batch["ids"], belief_batch["am"],
                         (belief_pos,), add=(LAYER, belief_pos, r))
        nulls.append(float((_ld(lg, tgt_ids, src_ids) - base_ld).mean()))
        hb.step()
    hb.done()
    p = permutation_pvalue(add_eff, np.asarray(nulls), "greater")
    result["null"] = {"p": float(p), "add_effect": add_eff,
                      "null_mean": float(np.mean(nulls))}
    log(f"  null: add_eff={add_eff:+.2f} null_mean={np.mean(nulls):+.2f} p={p:.3f}")

    # ---- invariants under the ac-anchor write ------------------------------
    invariants = {}
    for query in ("belief_as", "belief_bc", "truth_cube", "truth_sphere"):
        b = _batch(tok, clean_rows, query, "narrative", dev)
        base, _ = _forward(model, b["ids"], b["am"], (anchor_ac,))
        add, _ = _forward(model, b["ids"], b["am"], (anchor_ac,),
                          add=(LAYER, anchor_ac, delta))
        expected = _locations(clean_rows, query)
        invariants[query] = {"clean_acc": _accuracy(base, b, expected),
                             "add_acc": _accuracy(add, b, expected)}
        log(f"  [invariant {query}] clean={invariants[query]['clean_acc']:.0%} "
            f"under_write={invariants[query]['add_acc']:.0%}")
    result["invariants"] = invariants

    # ---- wrong-address positive control: same delta at BOB's cube anchor ---
    wb_c = _batch(tok, clean_rows, "belief_bc", "narrative", dev)
    wb_n = _batch(tok, bc_natural_rows, "belief_bc", "narrative", dev)
    pos_bc = _anchor_position(wb_c, wb_n)
    cl, _ = _forward(model, wb_c["ids"], wb_c["am"], (pos_bc,))
    nl, _ = _forward(model, wb_n["ids"], wb_n["am"], (pos_bc,))
    add, _ = _forward(model, wb_c["ids"], wb_c["am"], (pos_bc,),
                      add=(LAYER, pos_bc, delta))
    own = _switch_metrics(cl, nl, add, wb_c, _locations(clean_rows, "belief_bc"),
                          _locations(bc_natural_rows, "belief_bc"))
    ab = _batch(tok, clean_rows, "belief_ac", "narrative", dev)
    base, _ = _forward(model, ab["ids"], ab["am"], (pos_bc,))
    addp, _ = _forward(model, ab["ids"], ab["am"], (pos_bc,),
                       add=(LAYER, pos_bc, delta))
    preserved = _accuracy(addp, ab, _locations(clean_rows, "belief_ac"))
    result["wrong_address"] = {"own": own, "own_pass": bool(_switch_pass(own)),
                               "target_preserved_acc": preserved}
    log(f"  [wrong-address bc] own acc={own['target_acc']:.0%} "
        f"ratio={own['ratio']} | belief_ac preserved={preserved:.0%}")

    # ---- verdict ------------------------------------------------------------
    cons_ok = all(consequences[q]["pass"] for q in ("belief_ac", "tell_ac"))
    inv_ok = all(v["add_acc"] >= G0 for v in invariants.values())
    wa_ok = (result["wrong_address"]["own_pass"] and preserved >= G0)
    null_ok = p < GATE_P
    if cons_ok and inv_ok and wa_ok and null_ok:
        verdict = "TOKEN_ANCHORED_CONFIRMED"
    elif not cons_ok and consequences["belief_ac"]["target_acc"] < 0.2:
        verdict = "TOKEN_ANCHORED_FALSIFIED"
    else:
        verdict = "TOKEN_ANCHORED_PARTIAL"
    result["gates"] = {"consequences": cons_ok, "invariants": inv_ok,
                       "wrong_address": wa_ok, "null": null_ok}
    result["verdict"] = verdict
    with open(os.path.join(out_dir, "results_delta_anchor_write.json"),
              "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(f"VERDICT anchor_write: {verdict} | cons={cons_ok} inv={inv_ok} "
        f"wa={wa_ok} null={null_ok}")
    return result
