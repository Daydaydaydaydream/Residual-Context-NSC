#!/usr/bin/env python3
"""Verify the starter adapter against the course baseline.

Two claims are checked over all 200 archived configurations:

1. ablation(config) equals the starter's nsc.original_nsc. The beta=0
   ablation is the course baseline, not merely an approximation.
2. score(config) with NSC_BETA=0 equals that same baseline, in both the main
   scope and the two controls. This checks that each mechanism collapses at
   zero before any development-set comparison.

Usage (repo root):

    python scripts/verify_adapter.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER = REPO_ROOT / "course" / "nsc-starter"
SRC = REPO_ROOT / "src"
for path in (SRC, STARTER):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import nsc  # noqa: E402  course baseline

from cs5491_nsc import score_config, submission  # noqa: E402

TOLERANCE = 1e-10


def _load_configs() -> dict:
    path = STARTER / "data" / "architectures.json"
    return json.loads(path.read_text())


def _max_relative_difference(configs, baseline, compute):
    worst_diff = 0.0
    worst_case = None
    for key, config in configs.items():
        expected = baseline(config)
        actual = compute(config)
        if not math.isfinite(actual):
            raise AssertionError(f"{key}: non-finite score {actual!r}")
        diff = (
            abs(actual - expected) / abs(expected)
            if expected
            else abs(actual - expected)
        )
        if diff > worst_diff:
            worst_diff = diff
            worst_case = (key, expected, actual)
    return worst_diff, worst_case


def main() -> int:
    configs = _load_configs()
    os.environ["NSC_BETA"] = "0"

    def run_control(form: str, scope: str):
        def compute(config):
            return score_config(config, beta=0.0, form=form, scope=scope)

        return compute

    checks = {
        "ablation": submission.ablation,
        "score(beta=0, signed/ffn)": submission.score,
        "score(beta=0, signed/joint)": run_control("signed", "joint"),
        "score(beta=0, exponential/ffn)": run_control("exponential", "ffn"),
    }

    print(f"configurations checked: {len(configs)}")
    ok = True
    for name, compute in checks.items():
        diff, case = _max_relative_difference(configs, nsc.original_nsc, compute)
        print(f"{name:<36} max relative difference: {diff:.3e}")
        if case is not None:
            key, expected, actual = case
            print(f"  worst case {key}: starter={expected:.12f} adapter={actual:.12f}")
        ok = ok and diff < TOLERANCE

    # A non-zero beta must remain finite on every archived configuration.
    for beta in (-1000.0, -1.0, 1.0, 1000.0):
        values = [score_config(config, beta=beta) for config in configs.values()]
        if not all(math.isfinite(value) for value in values):
            print(f"FAIL: non-finite score at beta={beta}")
            ok = False
    print("PASS" if ok else "FAIL", f"(tolerance {TOLERANCE:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
