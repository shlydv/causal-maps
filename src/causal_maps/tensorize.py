"""Tokenize text minimal pairs into aligned tensors + anchor token indices.

Tokenizer-only (no model weights). Enforces the harness invariants that make a
layer x position map well-defined and the answer metric valid:
  - answer words are single-token (with leading space),
  - clean and cf tokenize to equal length,
  - all retained pairs share one sequence length S (positions align across batch),
  - each anchor char-offset maps to a VERIFIED token index (guards the
    leading-space tokenization trap by checking prefix tokenization is stable).
Pairs violating any invariant are dropped and counted (exclusions logged).
"""
from collections import Counter

import torch

from .logutil import log
from .model_utils import single_token_id


def _anchor_token_index(tokenizer, text, char_off):
    """Token index of the space-preceded word starting at char_off. Verified:
    the prefix (minus its trailing space) must be an exact token-prefix of the
    full encoding. Returns idx or None (unstable / not space-preceded)."""
    prefix = text[:char_off]
    if not prefix.endswith(" "):
        return None
    pre_ids = tokenizer.encode(prefix[:-1], add_special_tokens=False)
    full_ids = tokenizer.encode(text, add_special_tokens=False)
    idx = len(pre_ids)
    if full_ids[:idx] != pre_ids or idx >= len(full_ids):
        return None
    return idx


def tensorize_pairs(tokenizer, pairs, require_anchor_roles=()):
    """Build aligned tensors from text pairs.

    Returns dict:
      clean/cf: {input_ids [B,S], attention_mask [B,S]}
      pos_ids [B] (cf answer), neg_ids [B] (clean answer)
      anchors {role: token_index}  (uniform across pairs)
      S, kept_indices, exclusions
    """
    excl = {"answer_multitoken": 0, "len_mismatch": 0, "anchor_unstable": 0,
            "len_nonuniform": 0}
    rows = []  # (orig_idx, clean_ids, cf_ids, pos_id, neg_id, anchors, len)

    for k, p in enumerate(pairs):
        try:
            neg = single_token_id(tokenizer, p["answer_clean"])
            pos = single_token_id(tokenizer, p["answer_cf"])
        except ValueError:
            excl["answer_multitoken"] += 1
            continue
        ci = tokenizer.encode(p["clean_text"], add_special_tokens=False)
        fi = tokenizer.encode(p["cf_text"], add_special_tokens=False)
        if len(ci) != len(fi):
            excl["len_mismatch"] += 1
            continue
        anc, ok = {}, True
        for role, off in p.get("anchors", {}).items():
            idx = _anchor_token_index(tokenizer, p["clean_text"], off)
            if idx is None:
                ok = False
                break
            anc[role] = idx
        if not ok or any(r not in anc for r in require_anchor_roles):
            excl["anchor_unstable"] += 1
            continue
        rows.append((k, ci, fi, pos, neg, anc, len(ci)))

    if not rows:
        raise ValueError(f"no pairs survived tensorization; exclusions={excl}")

    # Enforce one uniform S (keep the majority length).
    S = Counter(r[6] for r in rows).most_common(1)[0][0]
    rows2 = []
    for r in rows:
        if r[6] == S:
            rows2.append(r)
        else:
            excl["len_nonuniform"] += 1

    # Anchors must be uniform across pairs (fixed template). Fail loud if not
    # (that would signal a template/tokenization bug, per protocol rule 6).
    anchors = {}
    for role in rows2[0][5]:
        vals = {r[5][role] for r in rows2}
        if len(vals) != 1:
            raise ValueError(f"anchor {role!r} not uniform across pairs: {sorted(vals)}")
        anchors[role] = rows2[0][5][role]

    clean_ids = torch.tensor([r[1] for r in rows2], dtype=torch.long)
    cf_ids = torch.tensor([r[2] for r in rows2], dtype=torch.long)
    templates = [pairs[r[0]].get("template", "unknown") for r in rows2]
    metas = [pairs[r[0]].get("meta", {}) for r in rows2]
    batch = {
        "clean": {"input_ids": clean_ids, "attention_mask": torch.ones_like(clean_ids)},
        "cf": {"input_ids": cf_ids, "attention_mask": torch.ones_like(cf_ids)},
        "pos_ids": torch.tensor([r[3] for r in rows2], dtype=torch.long),
        "neg_ids": torch.tensor([r[4] for r in rows2], dtype=torch.long),
        "anchors": anchors,
        "S": S,
        "kept_indices": [r[0] for r in rows2],
        "templates": templates,
        "metas": metas,
        "exclusions": excl,
    }
    log(f"tensorized {len(rows2)}/{len(pairs)} pairs | S={S} | anchors={anchors} "
        f"| exclusions={excl}")
    return batch
