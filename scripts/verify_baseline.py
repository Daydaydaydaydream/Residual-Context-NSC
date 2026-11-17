#!/usr/bin/env python3
"""Reproduce the development-split baseline and check it against the reference.

Two scoring paths are compared with ``course/nsc-starter/expected_development.json``:

1. the starter's own template, where ``student`` is the original NSC;
2. the project adapter at ``NSC_BETA=0``, which must collapse to that baseline.

The comparison ignores only the timing fields the reference omits, and allows
floating-point noise of 1e-12 relative. Everything else — rank correlations,
the 180 size-controlled pairs, the eight budget/k selections — must match.

Usage (repository root):

    python scripts/verify_baseline.py            # check and refresh the JSON
    python scripts/verify_baseline.py --no-write # check only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER = REPO_ROOT / "course" / "nsc-starter"
ADAPTER = REPO_ROOT / "scripts" / "student_score.py"
RESULT_DIR = REPO_ROOT / "results" / "baseline"
IGNORED_FIELDS = {"scoring_seconds", "timing_note"}
REL_TOLERANCE = 1e-12


def load_starter_evaluate():
    """Import the archived evaluator without installing anything."""
    for path in (STARTER, REPO_ROOT / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(
        "starter_evaluate", STARTER / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(expected, actual, path="") -> list[tuple]:
    """Return the differences, tolerating only floating-point noise."""
    diffs: list[tuple] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key in IGNORED_FIELDS:
                continue
            if key not in expected or key not in actual:
                diffs.append(
                    (f"{path}/{key}", "missing", key in expected, key in actual)
                )
                continue
            diffs += compare(expected[key], actual[key], f"{path}/{key}")
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            diffs.append((path, "length", len(expected), len(actual)))
        for index, (left, right) in enumerate(zip(expected, actual)):
            diffs += compare(left, right, f"{path}[{index}]")
    elif isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            diffs.append((path, "value", expected, actual))
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(expected, actual, rel_tol=REL_TOLERANCE, abs_tol=0.0):
            diffs.append((path, "value", expected, actual))
    elif expected != actual:
        diffs.append((path, "value", expected, actual))
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-write", action="store_true", help="check without writing JSON"
    )
    args = parser.parse_args()

    evaluate = load_starter_evaluate()
    expected = json.loads((STARTER / "expected_development.json").read_text())

    runs = {}
    os.environ.pop("NSC_BETA", None)
    runs["development_baseline"] = evaluate.evaluate(
        "development", STARTER / "student_score.py"
    )
    os.environ["NSC_BETA"] = "0"
    runs["development_adapter_beta0"] = evaluate.evaluate("development", ADAPTER)
    os.environ.pop("NSC_BETA", None)

    reference = runs["development_baseline"]
    print(f"architectures: {reference['architectures']}")
    print(
        "size-controlled pairs: "
        f"{reference['ranking']['original_nsc']['size_controlled_pairs']['pairs']}"
    )
    print(f"budget/k cells: {len(reference['selection'])}")
    print(
        "original_nsc: "
        f"rho={reference['ranking']['original_nsc']['spearman_rho']:.6f} "
        f"tau={reference['ranking']['original_nsc']['all_pairs']['tau_b']:.6f} "
        f"controlled_tau="
        f"{reference['ranking']['original_nsc']['size_controlled_pairs']['tau_b']:.6f}"
    )

    ok = True
    for name, result in runs.items():
        diffs = compare(expected, result)
        print(f"\n{name} vs expected_development.json: {len(diffs)} difference(s)")
        for diff in diffs[:10]:
            print(f"  {diff}")
        if len(diffs) > 10:
            print(f"  ... {len(diffs) - 10} more")
        ok = ok and not diffs

    adapter_diffs = compare(reference, runs["development_adapter_beta0"])
    print(f"\nbaseline vs adapter(beta=0): {len(adapter_diffs)} difference(s)")
    ok = ok and not adapter_diffs

    if not args.no_write:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        for name, result in runs.items():
            path = RESULT_DIR / f"{name}.json"
            path.write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            print(f"wrote {path.relative_to(REPO_ROOT)}")

    print("\nPASS" if ok else "\nFAIL", f"(floating-point tolerance {REL_TOLERANCE:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
