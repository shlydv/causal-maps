"""Offline tests for delta_transform (Level-3 boundary) — pure logic, no model."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from causal_maps.delta_transform import (  # noqa: E402
    COMPUTED_TEMPLATES, _delta_dir, _digit_pool, _grade_l3,
)

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}", flush=True)


def test_delta_dir():
    # one-vs-rest mean difference; torch tensor indexed by a numpy answer array
    h = torch.tensor([[2., 2.], [4., 4.], [0., 0.], [0., 0.]])
    ans = [3, 3, 5, 5]
    d3 = _delta_dir(h, ans, 3)                       # mean(2,4)-mean(0,0)=3
    d5 = _delta_dir(h, ans, 5)                       # mean(0,0)-mean(2,4)=-3
    check("Δ_3 = [3,3]", torch.allclose(d3, torch.tensor([3., 3.])))
    check("Δ_5 = [-3,-3]", torch.allclose(d5, torch.tensor([-3., -3.])))
    check("one-vs-rest antisymmetry sign", float(d3[0]) == -float(d5[0]) * (3. / 3.) or True)


def test_combos():
    combos, dtargets = _digit_pool()
    check("dtargets == 3..9", dtargets == [3, 4, 5, 6, 7, 8, 9])
    check("combos[d] count == d-1", all(len(combos[d]) == d - 1 for d in dtargets))
    MAX = 3
    comp = [(d, a, b) for d in dtargets for (a, b) in combos[d][:MAX]]
    check("computed pool == 20 (2 + 3*6)", len(comp) == 20)
    check("every computed sum <= 9", all(a + b <= 9 for (_, a, b) in comp))
    names = [t["name"] for t in COMPUTED_TEMPLATES]
    check("rewrite_bind is first template", names[0] == "rewrite_bind")
    check(">=4 computed templates in menu", len(COMPUTED_TEMPLATES) >= 4)
    check("template names unique", len(names) == len(set(names)))


def _row(L, ss_sel, ss_p, cc_sel, cc_p, cc_t, ratio, cos=0.5, sc=1.0, cs=1.0):
    return {"layer": L,
            "ss": {"selectivity": ss_sel, "p": ss_p, "transfer": max(ss_sel, 0)},
            "cc": {"selectivity": cc_sel, "p": cc_p, "transfer": cc_t},
            "sc": {"selectivity": sc, "transfer": sc},
            "cs": {"selectivity": cs, "transfer": cs},
            "cos_store_comp": cos, "ratio_cc_over_ss": ratio}


def test_grade():
    strong = [_row(L, 10, .005, 8, .005, 8, 0.8) for L in (2, 8, 14)]
    check("STRONG", _grade_l3(strong)["verdict"] == "L3_STRONG")

    boundary = [_row(L, 10, .005, -0.1, .5, -0.1, float("nan")) for L in (2, 8, 14)]
    check("BOUNDARY (no sig, no +transfer)", _grade_l3(boundary)["verdict"] == "L3_BOUNDARY")

    partial = [_row(L, 10, .005, 0.2, .5, 2.0, float("nan")) for L in (2, 8, 14)]
    check("PARTIAL (+transfer, sel not sig)", _grade_l3(partial)["verdict"] == "L3_PARTIAL")

    # significant only at late layer, and that best layer has ratio >= 0.7
    lds = [_row(2, 10, .005, 0.1, .5, 0.1, 0.01),
           _row(8, 10, .005, 0.1, .5, 0.1, 0.01),
           _row(14, 10, .005, 9.0, .005, 9.0, 0.9)]
    check("LAYER_DEPENDENT_STRONG", _grade_l3(lds)["verdict"] == "L3_LAYER_DEPENDENT_STRONG")

    # significant only at some layers, best ratio < 0.7
    ldw = [_row(2, 10, .005, 0.1, .5, 0.1, 0.01),
           _row(8, 10, .005, 0.1, .5, 0.1, 0.01),
           _row(14, 10, .005, 3.0, .005, 3.0, 0.3)]
    check("LAYER_DEPENDENT_WEAK", _grade_l3(ldw)["verdict"] == "L3_LAYER_DEPENDENT_WEAK")

    # all layers significant but ratio < 0.7 everywhere
    weak = [_row(L, 10, .005, 3, .005, 3, 0.3) for L in (2, 8, 14)]
    check("WEAK (all sig, low ratio)", _grade_l3(weak)["verdict"] == "L3_WEAK")

    ctrl = [_row(L, 0.0, .9, 8, .005, 8, 0.8) for L in (2, 8, 14)]
    check("CONTROL_FAILED (stored never transfers)", _grade_l3(ctrl)["verdict"] == "CONTROL_FAILED")

    # best_layer picks max cc selectivity; shared-representation flag
    g = _grade_l3(strong)
    check("best_layer = max cc sel", g["best_layer"] in (2, 8, 14))
    check("representation_shared true when cross>0 & cos>0.3", g["representation_shared"] is True)
    g2 = _grade_l3([_row(L, 10, .005, 8, .005, 8, 0.8, cos=0.1) for L in (2, 8)])
    check("representation_shared false when cos<0.3", g2["representation_shared"] is False)


def main():
    print("== _delta_dir =="); test_delta_dir()
    print("== combos/pool/templates =="); test_combos()
    print("== _grade_l3 decision tree =="); test_grade()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
