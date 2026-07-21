"""Offline tests for delta_instruction — pure logic / tokenization, no model."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from causal_maps.instruction_pairs import (  # noqa: E402
    _directive, _user_data, _user_instruction, make_instruction_pairs,
)

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}", flush=True)


def test_isolation():
    w = "cat"
    d = _directive(w)
    check("directive embeds W", w in d)
    check("instr user == directive only", _user_instruction(w) == d)
    check("data user quotes same directive", d in _user_data(w))
    check("data asks for first word", "first word" in _user_data(w).lower())
    pairs = make_instruction_pairs(n=3, tok=None, chat=False, payloads=["cat", "dog", "two"])
    check("3 pairs", len(pairs) == 3)
    check("answers W vs Output", pairs[0]["answer_clean"] == "cat" and pairs[0]["answer_cf"] == "Output")
    check("len mismatch allowed", pairs[0]["allow_len_mismatch"] is True)


def test_bare_readout_ids():
    """Regression: assistant\\n readout must use bare tokens, not Ġ-prefixed."""
    try:
        from transformers import AutoTokenizer
        from causal_maps.model_utils import single_token_id
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True, local_files_only=True)
    except Exception as e:
        print(f"  [SKIP] bare readout ({e})")
        return
    bare = single_token_id(tok, "cat", leading_space=False)
    spaced = single_token_id(tok, "cat", leading_space=True)
    check("bare ≠ spaced for cat", bare != spaced)
    out_b = single_token_id(tok, "Output", leading_space=False)
    out_s = single_token_id(tok, "Output", leading_space=True)
    check("bare ≠ spaced for Output", out_b != out_s)
    check("bare cat is single-token string 'cat'",
          tok.convert_ids_to_tokens([bare])[0] == "cat")


def main():
    print("== isolation design =="); test_isolation()
    print("== bare readout regression =="); test_bare_readout_ids()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
