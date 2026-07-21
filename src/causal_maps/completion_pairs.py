"""Skill 1 — Completion state: one boolean bit flips next-action.

Five surface families × 10 instantiations = 50 pairs.

  F1–F4  *explicit* — identical rules skeleton; only a single-token Surface tag
         differs (A/B/C/D). Same sequence length + bit_slot → one IE grid.
  F5     *implicit* — options listed, NO bit→act rule. May differ in length;
         behav-checked always; included in IE sweep only if S/anchors match.

Assistant primed with: "The next action is"
BIT: the boolean {flag}. DIFF: `0`↔`1` after `= `.
Expected site: bit_slot.
"""

_ROWS = [
    ("read_file", "READ", "DONE"),
    ("invoice_loaded", "LOAD", "OPEN"),
    ("search_done", "FIND", "SEND"),
    ("draft_ready", "EDIT", "WAIT"),
    ("batch_ready", "FETCH", "DROP"),
    ("model_cached", "WRITE", "CLOSE"),
    ("task_flag", "BUILD", "QUERY"),
    ("flag_a", "LOCK", "FREE"),
    ("flag_b", "GO", "STOP"),
    ("slot_a", "HOLD", "SKIP"),
]

# Four explicit families: same body, Surface tag ∈ {A,B,C,D}.
_EXPLICIT_TAGS = ("A", "B", "C", "D")
_IMPLICIT_TAG = "Q"

_PRIMER = "The next action is"
FAMILY_IDS = [f"explicit_{t}" for t in _EXPLICIT_TAGS] + ["implicit"]


def _user_explicit(tag, flag, act0, act1, bit):
    return (
        f"Surface: {tag}\n"
        f"Rules:\n"
        f"- if {flag} is 0, next action is {act0}\n"
        f"- if {flag} is 1, next action is {act1}\n"
        f"\n"
        f"State:\n"
        f"{flag} = {bit}\n"
        f"\n"
        f"Next action:"
    )


def _user_implicit(flag, act0, act1, bit):
    """Options only — does not say which bit selects which action."""
    return (
        f"Surface: {_IMPLICIT_TAG}\n"
        f"Options:\n"
        f"- action label {act0}\n"
        f"- action label {act1}\n"
        f"\n"
        f"State:\n"
        f"{flag} = {bit}\n"
        f"\n"
        f"Next action:"
    )


def _chat(tok, user):
    templated = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    return templated + _PRIMER


def _one(tok, use_chat, family, flag, act0, act1, pid):
    if family == "implicit":
        user0 = _user_implicit(flag, act0, act1, "0")
        user1 = _user_implicit(flag, act0, act1, "1")
    else:
        tag = family.split("_", 1)[1]  # explicit_A -> A
        user0 = _user_explicit(tag, flag, act0, act1, "0")
        user1 = _user_explicit(tag, flag, act0, act1, "1")
    if use_chat:
        clean, cf = _chat(tok, user0), _chat(tok, user1)
    else:
        clean = user0 + "\n" + _PRIMER
        cf = user1 + "\n" + _PRIMER
    marker = f"{flag} = "
    off = clean.find(marker) + len(marker)
    return {
        "clean_text": clean, "cf_text": cf,
        "answer_clean": act0, "answer_cf": act1,
        "anchors": {"bit_slot": off},
        "template": f"completion_{family}",
        "meta": {"id": pid, "family": family, "flag": flag,
                 "act0": act0, "act1": act1, "implicit": family == "implicit"},
    }


def make_completion_pairs(n=50, seed=0, tok=None, chat=True, hand=None,
                          families=None, **_kw):
    use_chat = chat and tok is not None
    if hand is not None:
        out = []
        for row in hand:
            fam = row.get("family", "explicit_A")
            out.append(_one(tok, use_chat, fam, row["flag"], row["act0"],
                            row["act1"], row.get("id")))
        return out[:n]

    fams = list(families) if families is not None else FAMILY_IDS
    pairs = []
    for fam in fams:
        for i, (flag, act0, act1) in enumerate(_ROWS):
            if len(pairs) >= n:
                return pairs
            pairs.append(_one(tok, use_chat, fam, flag, act0, act1, f"{fam}:{i+1}"))
    return pairs


# Hand-10 backward compat.
HAND10 = [
    {"id": f"C{i+1}", "flag": f, "act0": a0, "act1": a1, "family": "explicit_A"}
    for i, (f, a0, a1) in enumerate(_ROWS)
]
