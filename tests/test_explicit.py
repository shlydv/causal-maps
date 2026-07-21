"""Offline tests for delta_explicit grade logic (pure, no model)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from causal_maps.delta_explicit import _grade_skill  # noqa: E402

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}", flush=True)


def cell(has_early, elicited=True):
    return {"has_early": has_early, "per_layer": ([1] if elicited else [])}


def test_grade():
    check("stated early, derived late -> STATED_EARLIER",
          _grade_skill(cell(True), cell(False)) == "STATED_EARLIER")
    check("both early -> NO_SHIFT",
          _grade_skill(cell(True), cell(True)) == "NO_SHIFT")
    check("both late -> NO_SHIFT",
          _grade_skill(cell(False), cell(False)) == "NO_SHIFT")
    check("derived early, stated late -> DERIVED_EARLIER",
          _grade_skill(cell(False), cell(True)) == "DERIVED_EARLIER")
    check("stated not elicited -> INELICITABLE",
          _grade_skill(cell(True, elicited=False), cell(False)) == "INELICITABLE")
    check("derived not elicited -> INELICITABLE",
          _grade_skill(cell(True), cell(False, elicited=False)) == "INELICITABLE")


def main():
    print("== _grade_skill =="); test_grade()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL); sys.exit(1)


if __name__ == "__main__":
    main()
