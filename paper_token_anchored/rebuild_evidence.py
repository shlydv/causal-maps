#!/usr/bin/env python3
"""One-command regeneration and integrity gate for all Paper 1 evidence."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run(script, *args):
    subprocess.run([sys.executable, str(HERE / script), *args], check=True)


def main():
    run("analyze_confirmatory.py")
    run("build_evidence.py")
    run("build_figures.py")
    run("audit_evidence.py", "--require-closeout")
    print("Paper 1 evidence, figures, and audit passed")


if __name__ == "__main__":
    main()
