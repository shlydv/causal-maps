#!/usr/bin/env python3
"""Fast source-level checks when a TeX distribution is unavailable."""
from __future__ import annotations

import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
tex = (HERE / "main.tex").read_text()
bib = (HERE / "references.bib").read_text()
macros = (HERE / "generated" / "pilot_results.tex").read_text()

assert "TODO:" not in tex, "unresolved TODO in manuscript"
assert tex.count("\\begin{") == tex.count("\\end{"), "environment count differs"

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

required = [
    HERE.parent / "PREPRINT_CONFIRMATORY_PROTOCOL.md",
    HERE.parent / "PREPRINT_LAUNCH.md",
    HERE.parent / "src" / "causal_maps" / "delta_preprint_battery.py",
    HERE.parent / "src" / "causal_maps" / "delta_preprint_probe.py",
]
assert all(path.exists() for path in required), "required preprint artifact missing"
print(f"preprint source checks: PASS ({len(cited)} citations, {len(used)} macros)")
