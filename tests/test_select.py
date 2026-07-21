"""Offline tests for delta_select — pure logic, no model."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from causal_maps.delta_select import SELECT_TEMPLATES, _grade_l2  # noqa: E402

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}", flush=True)


def _row(L, route, rp, sel, sp):
    return {"layer": L,
            "route": {"effect": route, "p": rp},
            "selectivity": {"effect": sel, "p": sp}}


def test_grade():
    strong = [_row(L, 5, .005, 4, .005) for L in (2, 8, 14, 20, 26)]
    check("STRONG", _grade_l2(strong)["verdict"] == "L2_STRONG")
    boundary = [_row(L, -0.1, .5, -0.1, .5) for L in (2, 8, 14)]
    check("BOUNDARY", _grade_l2(boundary)["verdict"] == "L2_BOUNDARY")
    late = [_row(2, 0.1, .5, 0.1, .5), _row(26, 8.0, .005, 7.0, .005)]
    check("LAYER_DEPENDENT_WEAK", _grade_l2(late)["verdict"] == "L2_LAYER_DEPENDENT_WEAK")
    check("empty → INELICITABLE", _grade_l2([])["verdict"] == "L2_INELICITABLE")


def test_menu():
    names = [t["name"] for t in SELECT_TEMPLATES]
    check("v1_if_flag first", names[0] == "v1_if_flag")
    check("menu size in 10..20", 10 <= len(SELECT_TEMPLATES) <= 20)
    check("names unique", len(names) == len(set(names)))
    check("every template has key+primer+user",
          all("key" in t and "primer" in t and callable(t["user"]) for t in SELECT_TEMPLATES))


def main():
    print("== _grade_l2 =="); test_grade()
    print("== template menu =="); test_menu()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
