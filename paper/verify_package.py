#!/usr/bin/env python3
"""Fail-fast checks for the frozen publication package."""

from __future__ import annotations

import json
import re
from pathlib import Path


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent


def load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual} != {expected}")


def main() -> None:
    main_tex = (PAPER / "main.tex").read_text()
    bib = (PAPER / "references.bib").read_text()
    thread = (PAPER / "X_THREAD.md").read_text()
    summary = load(PAPER / "generated/results_summary.json")

    required = [
        PAPER / "main.pdf",
        PAPER / "figures/coordinate_system.pdf",
        PAPER / "figures/transfer_ratios.pdf",
        PAPER / "figures/controls.pdf",
        PAPER / "generated/results_macros.tex",
        PAPER / "LITERATURE_AUDIT.md",
    ]
    for path in required:
        assert path.exists() and path.stat().st_size > 0, f"missing/empty: {path}"

    # Independent source-to-summary checks for every headline result.
    p1 = load(ROOT / "runs/p1_7b/results_p1_both_7b.json")["skills"]
    frag = load(ROOT / "runs/p1_7b/fragility_variable_p1.json")
    transfer = load(ROOT / "runs/delta_transfer/results_delta_transfer.json")
    completion = load(ROOT / "runs/delta_completion/results_delta_completion.json")
    scale = load(ROOT / "runs/delta_var_1p5b/results_delta_var_1p5b.json")
    crosspos = load(ROOT / "runs/delta_var_crosspos/results_delta_var_crosspos.json")
    controls = load(ROOT / "runs/delta_var_shufflefix/results_delta_var_shufflefix.json")
    robust = load(ROOT / "runs/delta_var_robust/results_delta_var_robust.json")

    close(summary["p1"]["variable_gate_rho"], p1["variable_p1"]["p1"]["rho"])
    close(summary["p1"]["completion_gate_rho"], p1["completion_p1"]["p1"]["rho"])
    close(summary["p1"]["variable_expected_column_r"], frag["mean_expected_col_r"])
    close(summary["transfer"]["variable_7b_mean_ratio"], transfer["primary"]["mean_ratio"])
    close(
        summary["transfer"]["completion_mean_ratio"],
        completion["slots"]["primary_62"]["mean_ratio"],
    )
    close(summary["transfer"]["variable_1p5b_mean_ratio"], scale["layers"]["L2"]["mean_ratio"])
    close(
        summary["transfer"]["short_to_long_mean_ratio"],
        crosspos["directions"]["short_to_long"]["mean_ratio"],
    )
    close(
        summary["transfer"]["long_to_short_mean_ratio"],
        crosspos["directions"]["long_to_short"]["mean_ratio"],
    )
    close(summary["controls"]["wrong_mean_ratio"], controls["wrong_value_control"]["mean_ratio"])
    close(summary["controls"]["matched_wrong_cosine"], controls["cos_matched_wrong"])
    close(
        summary["controls"]["embed_over_l2"],
        robust["embed_control"]["cross_embed_over_cross_L2"],
    )
    close(summary["controls"]["anti_mean_ie"], controls["anti_delta_control"]["mean_cross_ie"])

    # Citation integrity.
    cited: set[str] = set()
    for command in re.findall(r"\\cite[tp]?\{([^}]+)\}", main_tex):
        cited.update(key.strip() for key in command.split(","))
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing = cited - bib_keys
    assert not missing, f"citations missing from bibliography: {sorted(missing)}"

    # Public-claim guardrails.
    lower = main_tex.lower()
    assert "not a value-specific binder" in lower
    assert lower.count("generic\\_boost") >= 3
    assert "unique natural mechanism" in lower
    assert "completion was tested at 7b; the variable protocol was replicated" in lower
    assert "not cross-model vector transfer" in lower
    assert "wrong-value" in main_tex
    assert "grant2026divergent" in main_tex and "makelov2024illusion" in main_tex
    assert "opie" in bib.lower() and "2026" in bib
    assert "\u2014" not in thread, "X thread contains an em dash"
    assert "generic_boost" in thread.lower()
    assert thread.lower().count("wrong") >= 2
    assert "[paper link]" in thread and "[repo link]" in thread

    # Build log must be clean.
    log = (PAPER / "main.log").read_text()
    bad = [
        line for line in log.splitlines()
        if re.search(
            r"(^!|LaTeX Warning|Package \S+ Warning|"
            r"Overfull|Underfull|Undefined control sequence)",
            line,
            re.I,
        )
    ]
    assert not bad, "TeX build warnings:\n" + "\n".join(bad)

    print(f"verified {len(cited)} citations and {len(required)} publication artifacts")
    print("all frozen-result, claim-guardrail, and TeX-log checks passed")


if __name__ == "__main__":
    main()
