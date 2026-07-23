#!/usr/bin/env python3
"""Fast source-level checks when a TeX distribution is unavailable."""
from __future__ import annotations

import re
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
tex = (HERE / "main.tex").read_text()
bib = (HERE / "references.bib").read_text()
macros = (HERE / "generated" / "pilot_results.tex").read_text()

assert "TODO:" not in tex, "unresolved TODO in manuscript"
assert "Running/final gate" not in tex, "stale running status in manuscript"
assert tex.count("\\begin{") == tex.count("\\end{"), "environment count differs"

required_claim_guards = [
    "alternative or distributed state representations are absent",
    "failure of that gate is an identification boundary",
    "not a generic patching failure",
    "does not imply that state is literally stored in one token",
    "no prompt or model rescue is run",
    "PAPER1\\_EVIDENCE\\_FROZEN\\_WITH\\_BOUNDARY",
]
missing_guards = [text for text in required_claim_guards if text not in tex]
assert not missing_guards, f"missing claim guardrails: {missing_guards}"

forbidden_overclaims = [
    "no consolidated world-state buffer exists",
    "state is stored in a single token",
    "address specificity is universal",
    "proves that the model has no distributed representation",
]
present_overclaims = [text for text in forbidden_overclaims if text in tex]
assert not present_overclaims, f"forbidden overclaims: {present_overclaims}"

bib_keys = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.M))
cited = set()
for block in re.findall(r"\\cite[tp]?\{([^}]+)\}", tex):
    cited.update(key.strip() for key in block.split(","))
missing = sorted(cited - bib_keys)
assert not missing, f"missing bibliography keys: {missing}"

defined = set(re.findall(r"\\newcommand\{\\((?:Pilot|Confirm)\w+)\}", macros))
used = set(re.findall(r"\\((?:Pilot|Confirm)\w+)", tex))
undefined = sorted(used - defined)
assert not undefined, f"undefined generated macros: {undefined}"

for figure in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex):
    assert (HERE / figure).exists(), f"missing manuscript figure: {figure}"

manifest = json.loads((HERE / "generated" / "evidence_manifest.json").read_text())
assert manifest["status"] in {
    "PAPER1_EVIDENCE_FROZEN", "PAPER1_EVIDENCE_FROZEN_WITH_BOUNDARY",
}, f"evidence is not frozen: {manifest['status']}"
assert "ConfirmCloseoutVerdict" in tex, "closeout boundary missing from paper"
assert manifest["artifacts"]["qwen14_closeout"]["verdict"] == (
    "PAPER1_CLOSEOUT_BOUNDARY"), "closeout verdict drifted"
assert manifest["artifacts"]["qwen14_locus"]["verdict"] == (
    "SOURCE_ANCHORS_SUFFICIENT"), "locus verdict drifted"
assert len(re.findall(r"\\includegraphics", tex)) == 3, (
    "expected exactly three data-derived figures")
assert len(cited) >= 20, "citation coverage regressed"

required = [
    HERE.parent / "PREPRINT_CONFIRMATORY_PROTOCOL.md",
    HERE.parent / "PREPRINT_LAUNCH.md",
    HERE.parent / "MULTITOKEN_LOCUS_PROTOCOL.md",
    HERE.parent / "PAPER1_CLOSEOUT_PROTOCOL.md",
    HERE.parent / "PAPER1_EVIDENCE_FREEZE.md",
    HERE.parent / "src" / "causal_maps" / "delta_preprint_battery.py",
    HERE.parent / "src" / "causal_maps" / "delta_preprint_probe.py",
    HERE.parent / "src" / "causal_maps" / "delta_preprint_locus.py",
    HERE.parent / "src" / "causal_maps" / "delta_paper1_closeout.py",
]
assert all(path.exists() for path in required), "required preprint artifact missing"
print(f"preprint source checks: PASS ({len(cited)} citations, {len(used)} macros)")
