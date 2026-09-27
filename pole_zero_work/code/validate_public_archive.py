#!/usr/bin/env python3
"""Run a lean public validation subset for the reproducibility archive.

The full internal release suite includes packaging, metadata, and journal-review
checks that are intentionally excluded from this public archive.  This script
runs the scientific and numerical checks that are meaningful for users.
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMANDS = [
    [sys.executable, "code/validate_solver.py"],
    [sys.executable, "code/validate_prl_reinforcement.py"],
    [sys.executable, "code/validate_high_level_reinforcement.py"],
    [sys.executable, "code/validate_prl_editorial_claims.py"],
    [sys.executable, "code/validate_pole_zero_reinforcement.py"],
    [sys.executable, "code/validate_multimode_tomography.py"],
    [sys.executable, "code/validate_observability_map.py"],
    [sys.executable, "code/symbolic_pole_zero_proofs.py"],
    [sys.executable, "code/validate_competitor_crossover.py"],
    [sys.executable, "code/tests/run_gauge_completion_checks.py"],
    [sys.executable, "code/full_bz_completion/validate_completion.py"],
]

def main() -> None:
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    passed = 0
    for cmd in COMMANDS:
        print("[run]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        passed += 1
    print(f"[ok] {passed} public validation commands completed.")

if __name__ == "__main__":
    main()
