"""Skill 2 — Variable substitution: one stored symbol→value.

Five templates × 10 value-pairs = 50. One shared skeleton so S and val_slot
are uniform (required for a single IE grid). Template identity = which
variable name group (X/Y/Z/W/K) — disjoint for Gate P1.

    Let {var} = {val}.
    What is the value of {var}?

Assistant primed with: "{var} ="
BIT: the bound value of {var}. DIFF: `{val}` only.
Expected site: val_slot.
"""

# 10 value pairs (likely single-token under Qwen2.5; drops logged on Kaggle).
_VALUE_PAIRS = [
    ("dog", "cat"),
    ("red", "blue"),
    ("one", "two"),
    ("hot", "cold"),
    ("apple", "grape"),
    ("north", "south"),
    ("open", "shut"),
    ("left", "right"),
    ("alpha", "beta"),
    ("true", "false"),
]

# Five template groups (= variable names).
_VARS = ("X", "Y", "Z", "W", "K")

TEMPLATE_PREFIX = "variable_"


def _user(var, val):
    return f"Let {var} = {val}.\nWhat is the value of {var}?"


def _chat(tok, var, val):
    templated = tok.apply_chat_template(
        [{"role": "user", "content": _user(var, val)}],
        tokenize=False, add_generation_prompt=True)
    return templated + f"{var} ="


def make_variable_pairs(n=50, seed=0, tok=None, chat=True, hand=None,
                        vars=None, **_kw):
    use_chat = chat and tok is not None
    if hand is not None:
        pairs = []
        for row in hand:
            pairs.append(_one(tok, use_chat, row["var"], row["val_clean"],
                              row["val_cf"], row.get("id")))
        return pairs[:n]

    var_list = list(vars) if vars is not None else list(_VARS)
    pairs = []
    for var in var_list:
        for i, (v0, v1) in enumerate(_VALUE_PAIRS):
            if len(pairs) >= n:
                return pairs
            pairs.append(_one(tok, use_chat, var, v0, v1, f"{var}{i+1}"))
    return pairs


def _one(tok, use_chat, var, v0, v1, pid):
    if use_chat:
        clean, cf = _chat(tok, var, v0), _chat(tok, var, v1)
    else:
        clean = _user(var, v0) + f"\n{var} ="
        cf = _user(var, v1) + f"\n{var} ="
    marker = f"Let {var} = "
    off = clean.find(marker) + len(marker)
    return {
        "clean_text": clean, "cf_text": cf,
        "answer_clean": v0, "answer_cf": v1,
        "anchors": {"val_slot": off},
        "template": f"{TEMPLATE_PREFIX}{var}",
        "meta": {"id": pid, "var": var, "val_clean": v0, "val_cf": v1},
    }


HAND10 = [
    {"id": f"V{i+1}", "var": _VARS[i % len(_VARS)],
     "val_clean": v0, "val_cf": v1}
    for i, (v0, v1) in enumerate(_VALUE_PAIRS)
]
