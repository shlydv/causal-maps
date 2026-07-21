"""Offline correctness tests — NO model download, NO real model run.

Validates the instrument logic against a tiny toy model that reproduces the
transformers-5.x calling convention (decoder layers in `model.model.layers`,
each layer forward returning a BARE tensor) and a causal token-mixing block so
that patching an earlier position propagates to the final logits (as it must in
a real causal transformer). Run: `.venv/bin/python tests/test_core.py`.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from causal_maps import nulls  # noqa: E402
from causal_maps.model_utils import (  # noqa: E402
    _bnb_config_kwargs, _load_dtype_and_map, logit_diff, single_token_id,
    validate_single_token)
from causal_maps.patching import cache_layer_outputs, forward_with_patch, sweep_ie  # noqa: E402
from causal_maps.tensorize import tensorize_pairs  # noqa: E402
from causal_maps import binding_pairs, rule_world  # noqa: E402

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}", flush=True)


# ---------------- toy model (transformers-5.x-like) -------------------------
def _causal_avg(hidden_states):
    """Causal running-average over positions so that changing position j affects
    all positions >= j (mimics attention's forward information flow)."""
    B, S, D = hidden_states.shape
    w = torch.tril(torch.ones(S, S, device=hidden_states.device))
    w = w / torch.arange(1, S + 1, device=hidden_states.device).float().view(-1, 1)
    return torch.einsum("ij,bjd->bid", w, hidden_states)


class CausalMixBlock(nn.Module):
    """Bare-tensor return (transformers 5.x)."""
    def forward(self, hidden_states, **kw):
        return _causal_avg(hidden_states)


class TupleBlock(nn.Module):
    """Same causal mixing but TUPLE return (4.x style) — exercises the tuple
    branch of the hooks WITH real propagation."""
    def forward(self, hidden_states, **kw):
        return (_causal_avg(hidden_states), None)


class ToyInner(nn.Module):
    def __init__(self, V, D, N, tuple_style=False):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        blk = TupleBlock if tuple_style else CausalMixBlock
        self.layers = nn.ModuleList([blk() for _ in range(N)])

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for layer in self.layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
        return h


class ToyLM(nn.Module):
    def __init__(self, V=12, D=8, N=3, tuple_style=False, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.model = ToyInner(V, D, N, tuple_style)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.config = SimpleNamespace(num_hidden_layers=N, hidden_size=D)

    def get_input_embeddings(self):
        return self.model.embed

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.lm_head(self.model(input_ids)))


# ---------------- pure-math tests -------------------------------------------
def test_logit_diff():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    ld = logit_diff(logits, pos_id=0, neg_id=2)  # 2.0 - 0.0
    check("logit_diff scalar ids == raw logit difference", abs(ld.item() - 2.0) < 1e-6)
    # equals log p(pos) - log p(neg)
    lp = torch.log_softmax(logits, -1)
    check("logit_diff == log p(pos) - log p(neg)",
          abs(ld.item() - (lp[0, 0] - lp[0, 2]).item()) < 1e-5)
    pos = torch.tensor([0, 3]); neg = torch.tensor([1, 2])
    logits2 = torch.randn(2, 4)
    ld2 = logit_diff(logits2, pos, neg)
    manual = torch.tensor([logits2[0, 0] - logits2[0, 1], logits2[1, 3] - logits2[1, 2]])
    check("logit_diff per-example tensor ids", torch.allclose(ld2, manual, atol=1e-6))


def test_quantization_recipe():
    q8 = _bnb_config_kwargs("8bit", torch.bfloat16)
    q4 = _bnb_config_kwargs("4bit", torch.bfloat16)
    check("8-bit recipe is unambiguous", q8 == {"load_in_8bit": True})
    check("4-bit recipe uses NF4", q4["load_in_4bit"] is True
          and q4["bnb_4bit_quant_type"] == "nf4")
    check("4-bit recipe preserves bf16 compute",
          q4["bnb_4bit_compute_dtype"] == torch.bfloat16)
    try:
        _bnb_config_kwargs("3bit", torch.bfloat16)
    except ValueError:
        bad_rejected = True
    else:
        bad_rejected = False
    check("unknown quantization rejected", bad_rejected)
    awq_dtype, awq_map = _load_dtype_and_map("awq", torch.bfloat16, None)
    check("AWQ uses checkpoint dtype", awq_dtype == "auto")
    check("AWQ dispatches directly", awq_map == "auto")
    explicit_dtype, explicit_map = _load_dtype_and_map(
        "awq", torch.bfloat16, "balanced")
    check("AWQ preserves explicit device map", explicit_dtype == "auto"
          and explicit_map == "balanced")


def test_nulls():
    check("permutation_pvalue smoothing (real above all)",
          abs(nulls.permutation_pvalue(10.0, np.zeros(99)) - 1 / 100) < 1e-9)
    check("permutation_pvalue (real below all) ~ 1",
          nulls.permutation_pvalue(-10.0, np.zeros(99)) > 0.98)
    a = np.arange(12).reshape(3, 4).astype(float)
    check("spearman_grid monotone == 1", abs(nulls.spearman_grid(a, 2 * a + 1) - 1.0) < 1e-9)
    # random_position_null: spike the expected column -> real must beat null
    ie = np.random.RandomState(0).randn(4, 6, 5) * 0.01
    ie[:, 2, :] += 5.0  # position 2 is the signal
    real, layer, null = nulls.random_position_null(
        ie, expected_positions=[2],
        candidate_positions=[0, 1, 3, 4, 5], n_draws=200, seed=1)
    check("random_position_null: real >> null when signal at expected pos",
          real > null.max())


# ---------------- hook / sweep tests ----------------------------------------
def _batch(clean_ids, cf_ids):
    return (
        {"input_ids": clean_ids, "attention_mask": torch.ones_like(clean_ids)},
        {"input_ids": cf_ids, "attention_mask": torch.ones_like(cf_ids)},
    )


def test_hooks_and_sweep(tuple_style=False):
    tag = "tuple" if tuple_style else "tensor"
    V, D, N, S, B = 12, 8, 3, 5, 4
    model = ToyLM(V, D, N, tuple_style=tuple_style, seed=1).eval()
    torch.manual_seed(2)
    clean_ids = torch.randint(0, V, (B, S))
    cf_ids = clean_ids.clone()
    cf_ids[:, 1] = (clean_ids[:, 1] + 3) % V  # differ ONLY at position 1
    clean, cf = _batch(clean_ids, cf_ids)

    cache = cache_layer_outputs(model, cf["input_ids"], cf["attention_mask"])
    check(f"[{tag}] cache has N layers", len(cache) == N)
    check(f"[{tag}] cache tensor shape [B,S,D]", tuple(cache[0].shape) == (B, S, D))

    pos = torch.randint(0, V, (B,)); neg = torch.randint(0, V, (B,))
    ie_mean, ie_all, clean_ld, meta = sweep_ie(
        model, clean, cf, pos, neg, out_dir=None, tag=f"toy_{tag}")
    check(f"[{tag}] ie_all shape [L,S,B]", ie_all.shape == (N, S, B))

    # position 0: clean and cf identical up to pos0 => cf residual == clean => IE ~ 0
    check(f"[{tag}] IE ~ 0 at position 0 (no divergence yet)",
          np.nanmax(np.abs(ie_all[:, 0, :])) < 1e-4)
    # position 1 (the divergence) must have real effect at some layer
    check(f"[{tag}] IE != 0 at position 1 (divergence propagates)",
          np.nanmax(np.abs(ie_all[:, 1, :])) > 1e-3)

    # manual recompute of one (layer, position) matches the sweep exactly
    L, P = N - 1, 1
    from causal_maps.model_utils import last_token_logits
    base = logit_diff(last_token_logits(model, clean["input_ids"], clean["attention_mask"]), pos, neg)
    val = cache[L][:, P, :]
    patched = forward_with_patch(model, clean["input_ids"], clean["attention_mask"], L, P, val)
    ie_manual = (logit_diff(patched, pos, neg) - base).detach().numpy()
    check(f"[{tag}] manual IE matches sweep at (L={L},P={P})",
          np.allclose(ie_manual, ie_all[L, P, :], atol=1e-5))


def test_patch_overwrites_value():
    """forward_with_patch must actually replace the residual at (layer,pos).
    We read layer-0's patched output as the INPUT to layer 1 (a forward-pre-hook
    on layer 1 fires after layer 0's patch hook, so it sees the patched value)."""
    V, D, N, S, B = 12, 8, 2, 4, 1
    model = ToyLM(V, D, N, seed=3).eval()
    ids = torch.randint(0, V, (B, S)); am = torch.ones_like(ids)
    sentinel = torch.full((B, D), 99.0)
    captured = {}

    def grab_pre(m, args, kwargs):
        hs = args[0] if args else kwargs.get("hidden_states")
        captured["hs"] = hs.detach().clone()

    hh = model.model.layers[1].register_forward_pre_hook(grab_pre, with_kwargs=True)
    forward_with_patch(model, ids, am, layer_idx=0, position=2, value=sentinel)
    hh.remove()
    check("patched residual propagates to next layer input at pos2",
          torch.allclose(captured["hs"][:, 2, :], sentinel, atol=1e-4))
    check("other positions not overwritten by the sentinel",
          not torch.allclose(captured["hs"][:, 0, :], sentinel, atol=1e-4))


# ---------------- tensorize tests (fake tokenizer, offline) -----------------
class FakeTok:
    """Whitespace tokenizer: one id per whitespace-delimited word. Leading space
    on encode path is handled by callers via ' word' strings for single_token_id."""
    def __init__(self):
        self.vocab = {}

    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab) + 1)

    def encode(self, text, add_special_tokens=False):
        import re
        return [self._id(w) for w in re.split(r"\s+", text.strip()) if w]


def test_tensorize():
    tok = FakeTok()
    check("single_token_id leading space", isinstance(single_token_id(tok, "red"), int))
    got = validate_single_token(tok, ["red", "blue two"])  # 'blue two' is 2 tokens
    check("validate_single_token drops multi-token", set(got.keys()) == {"red"})

    bp = binding_pairs.make_binding_pairs(n=20, seed=0)
    batch = tensorize_pairs(tok, bp, require_anchor_roles=("a1_slot", "a2_slot"))
    B = len(batch["kept_indices"])
    check("binding: kept all pairs (fake tok single-token)", B == 20)
    check("binding: clean/cf equal length", batch["clean"]["input_ids"].shape == batch["cf"]["input_ids"].shape)
    check("binding: a1_slot < a2_slot < S", batch["anchors"]["a1_slot"] < batch["anchors"]["a2_slot"] < batch["S"])
    # clean and cf differ exactly at the two attribute slots
    diff = (batch["clean"]["input_ids"] != batch["cf"]["input_ids"])
    diff_positions = set(torch.where(diff.any(0))[0].tolist())
    check("binding: exactly the two attribute slots differ clean vs cf",
          diff_positions == {batch["anchors"]["a1_slot"], batch["anchors"]["a2_slot"]})

    cp = rule_world.make_copy_pairs(n=15, seed=0)
    cbatch = tensorize_pairs(tok, cp, require_anchor_roles=("val_slot",))
    cdiff = (cbatch["clean"]["input_ids"] != cbatch["cf"]["input_ids"])
    cdiff_positions = set(torch.where(cdiff.any(0))[0].tolist())
    check("copy: only the value slot differs clean vs cf",
          cdiff_positions == {cbatch["anchors"]["val_slot"]})

    from causal_maps import completion_pairs, variable_pairs, instruction_pairs
    comp = completion_pairs.make_completion_pairs(n=10, chat=False)
    check("completion hand10 count", len(comp) == 10)
    cbat = tensorize_pairs(tok, comp, require_anchor_roles=("bit_slot",))
    cdiff2 = (cbat["clean"]["input_ids"] != cbat["cf"]["input_ids"])
    check("completion: only bit_slot differs",
          set(torch.where(cdiff2.any(0))[0].tolist()) == {cbat["anchors"]["bit_slot"]})
    check("completion: answers differ act0/act1",
          all(p["answer_clean"] != p["answer_cf"] for p in comp))

    var = variable_pairs.make_variable_pairs(n=10, chat=False)
    check("variable hand10 count", len(var) == 10)
    vbat = tensorize_pairs(tok, var, require_anchor_roles=("val_slot",))
    vdiff = (vbat["clean"]["input_ids"] != vbat["cf"]["input_ids"])
    check("variable: only val_slot differs",
          set(torch.where(vdiff.any(0))[0].tolist()) == {vbat["anchors"]["val_slot"]})

    ins = instruction_pairs.make_instruction_pairs(n=10, chat=False)
    check("instruction hand10 count", len(ins) == 10)
    check("instruction: natural framing (Execute vs document)",
          all("Execute the following instruction." in p["clean_text"]
              and "appeared in a document" in p["cf_text"] for p in ins))
    check("instruction: no explicit Treat-as mode token",
          all("Treat as:" not in p["clean_text"] and "Treat as:" not in p["cf_text"]
              for p in ins))
    check("instruction: identical payload in both arms",
          all(p["meta"]["payload"] in p["clean_text"]
              and p["meta"]["payload"] in p["cf_text"] for p in ins))
    check("instruction: marked allow_len_mismatch",
          all(p.get("allow_len_mismatch") for p in ins))


def test_forward_add_multi():
    """delta_multislot.forward_add_multi must add each delta at its own position."""
    from causal_maps.delta_multislot import forward_add_multi
    V, D, N, S, B = 12, 8, 2, 5, 2
    model = ToyLM(V, D, N, seed=5).eval()
    ids = torch.randint(0, V, (B, S)); am = torch.ones_like(ids)
    cap = []

    def grab(m, args, kwargs):
        hs = args[0] if args else kwargs.get("hidden_states")
        cap.append(hs.detach().clone())

    h = model.model.layers[1].register_forward_pre_hook(grab, with_kwargs=True)
    model(input_ids=ids, attention_mask=am)                       # cap[0]: layer0 out, no add
    d2 = torch.full((B, D), 5.0); d3 = torch.full((B, D), -3.0)
    forward_add_multi(model, ids, am, 0, [(2, d2), (3, d3)])      # cap[1]: with two adds
    h.remove()
    diff = cap[1] - cap[0]
    check("multi-add: pos2 += d2", torch.allclose(diff[:, 2, :], d2, atol=1e-4))
    check("multi-add: pos3 += d3", torch.allclose(diff[:, 3, :], d3, atol=1e-4))
    check("multi-add: pos0 unchanged", torch.allclose(diff[:, 0, :], torch.zeros(B, D), atol=1e-4))


def main():
    print("== pure math =="); test_logit_diff(); test_nulls(); test_quantization_recipe()
    print("== hooks/sweep (tensor return, 5.x) =="); test_hooks_and_sweep(False)
    print("== hooks/sweep (tuple return, 4.x) =="); test_hooks_and_sweep(True)
    print("== patch overwrite =="); test_patch_overwrites_value()
    print("== tensorize =="); test_tensorize()
    print("== multi-add =="); test_forward_add_multi()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
