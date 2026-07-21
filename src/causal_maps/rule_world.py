"""Minimal synthetic minimal-pairs for the PLUMBING check (skill-agnostic).

'copy' task: one fact whose single-token value is queried. Patching the value
token's residual from the counterfactual flips the answer. Used only to verify
the instrument (hooks read/write, IE sign, grid assembly) before the real gates
- not a scientific result.
"""
import random

# Single-token candidates (validated at tensorization; non-single-token dropped).
CODES = ["red", "blue", "green", "black", "white", "gold", "gray", "pink",
         "seven", "three", "north", "south", "east", "west", "one", "two",
         "four", "five", "left", "right"]

TEMPLATE = "copy_v1"


def make_copy_pairs(n=40, seed=0, codes=None, tok=None):
    codes = codes or CODES  # tok accepted for a uniform build() signature; unused here
    rng = random.Random(seed)
    pairs, seen = [], set()
    tries = 0
    while len(pairs) < n and tries < 50 * n:
        tries += 1
        if len(codes) < 2:
            break
        a1, a2 = rng.sample(codes, 2)
        if (a1, a2) in seen:
            continue
        seen.add((a1, a2))
        prefix = "The code is "          # value starts right here
        off = len(prefix)
        clean_text = prefix + a1 + ". The code is"
        cf_text = prefix + a2 + ". The code is"
        pairs.append({
            "clean_text": clean_text, "cf_text": cf_text,
            "answer_clean": a1, "answer_cf": a2,
            "anchors": {"val_slot": off}, "template": TEMPLATE,
        })
    return pairs
