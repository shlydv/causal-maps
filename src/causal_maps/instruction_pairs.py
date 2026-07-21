"""Instruction vs data pairs — isolated framing (redesign 2026-07-13).

Old hand10 (command→OK / mention→text) failed greedy elicitation.
New design keeps the embedded directive identical; only the frame changes:

  D(W) = "Output the word: {W}"
  Instruction framing: execute D(W)           → answer W
  Data framing: quote D(W), repeat first word → answer "Output"

Used by delta_instruction.py. This module exposes pair construction for
inspection / future patching stages; Stage 1–2 build texts inline.
"""
from .variable_pairs import _VALUE_PAIRS

TEMPLATE = "instr_data_isolated_v1"


def _directive(w):
    return f"Output the word: {w}"


def _user_instruction(w):
    return _directive(w)


def _user_data(w):
    d = _directive(w)
    return (f'The following text says "{d}". '
            f"Repeat the first word of the quoted text.")


def _chat(tok, user):
    return tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)


def make_instruction_pairs(n=16, seed=0, tok=None, chat=True, payloads=None, **_kw):
    """Return pairs: clean=instruction frame, cf=data frame.
    answer_clean=W, answer_cf=Output. allow_len_mismatch=True (frames differ)."""
    if payloads is None:
        payloads = []
        for a, b in _VALUE_PAIRS:
            for w in (a, b):
                if w not in payloads:
                    payloads.append(w)
        payloads = payloads[:n]
    use_chat = chat and tok is not None
    pairs = []
    for i, w in enumerate(payloads):
        if use_chat:
            clean, cf = _chat(tok, _user_instruction(w)), _chat(tok, _user_data(w))
        else:
            clean, cf = _user_instruction(w), _user_data(w)
        pairs.append({
            "clean_text": clean, "cf_text": cf,
            "answer_clean": w, "answer_cf": "Output",
            "anchors": {},
            "template": TEMPLATE,
            "meta": {"id": f"ID{i+1}", "payload": w},
            "allow_len_mismatch": True,
        })
    return pairs
