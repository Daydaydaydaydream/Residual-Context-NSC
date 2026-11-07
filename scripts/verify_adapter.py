"""Check that the adapter reproduces the starter's baseline before any experiment.

Two claims are verified over all 200 archived configurations:

1. ``starter_adapter.ablation`` equals the starter's ``nsc.original_nsc`` — the
   additive ablation really is the course baseline, not merely similar to it.
2. ``starter_adapter.score`` with ``NSC_ALPHA=0`` equals the same baseline — the
   project score degenerates correctly.

A difference here invalidates every downstream comparison, so this runs first.

Usage (repo root):

    python scripts/verify_adapter.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER = REPO_ROOT / "course" / "nsc-starter"
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

os.environ["NSC_ALPHA"] = "0"

import nsc  # noqa: E402  course baseline
import starter_adapter  # noqa: E402

TOLERANCE = 1e-10


def compare(configs, name, compute):
    """Worst relative difference between ``compute`` and the starter baseline."""
    worst_diff, worst_case = 0.0, None
    for key, config in configs.items():
        baseline = nsc.original_nsc(config)
        value = compute(config)
        diff = abs(value - baseline) / abs(baseline) if baseline else abs(value)
        if diff > worst_diff:
            worst_diff, worst_case = diff, (key, baseline, value)
    return name, worst_diff, worst_case


def main():
    configs = json.loads((STARTER / "data" / "architectures.json").read_text())
    results = [
        compare(configs, "ablation", starter_adapter.ablation),
        compare(configs, "score(alpha=0)", starter_adapter.score),
    ]
    print(f"configurations checked: {len(configs)}")
    for name, diff, case in results:
        print(f"{name:<16} max relative difference: {diff:.3e}")
        if case is not None:
            key, baseline, value = case
            print(f"  worst case {key}: starter={baseline:.12f} adapter={value:.12f}")

    ok = all(diff < TOLERANCE for _, diff, _ in results)
    print("PASS" if ok else "FAIL", f"(tolerance {TOLERANCE:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
