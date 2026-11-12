#!/usr/bin/env python3
"""Development-split beta sweep for the three declared scoring settings.

Main setting        : NSC_FORM=signed,      NSC_SCOPE=ffn
Joint control       : NSC_FORM=signed,      NSC_SCOPE=joint
Exponential control : NSC_FORM=exponential, NSC_SCOPE=ffn

Eleven beta values times three settings is 33 runs, development split only.
Selection follows the project plan: the highest Spearman rho against negative
validation PPL wins; ties go to the smaller |beta|, then to the negative sign.

Usage (repository root):

    python scripts/sweep_beta.py
    python scripts/sweep_beta.py --output-dir results/beta-sweep
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER = REPO_ROOT / "course" / "nsc-starter"
STUDENT = REPO_ROOT / "scripts" / "student_score.py"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "beta-sweep"

BETAS = (0.0, -0.1, 0.1, -1.0, 1.0, -10.0, 10.0, -100.0, 100.0, -1000.0, 1000.0)
SETTINGS = (
    ("main", "signed", "ffn"),
    ("joint", "signed", "joint"),
    ("exponential", "exponential", "ffn"),
)
FIELDS = (
    "setting",
    "form",
    "scope",
    "beta",
    "spearman_rho",
    "tau_b_all_pairs",
    "tau_b_size_controlled",
    "size_controlled_pairs",
    "score_only_ties",
    "scoring_seconds",
    "sweep_wall_seconds",
)


def load_evaluator():
    """Import the archived evaluator without installing the package."""
    for path in (STARTER, REPO_ROOT / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(
        "starter_evaluate", STARTER / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = module.__version__
    return versions


def run_setting(evaluate, form: str, scope: str, beta: float) -> tuple[dict, float]:
    os.environ["NSC_FORM"] = form
    os.environ["NSC_SCOPE"] = scope
    os.environ["NSC_BETA"] = repr(beta)
    start = time.perf_counter()
    result = evaluate("development", STUDENT)
    return result, time.perf_counter() - start


def make_row(
    setting: str, form: str, scope: str, beta: float, result: dict, wall: float
) -> dict:
    student = result["ranking"]["student"]
    controlled = student["size_controlled_pairs"]
    return {
        "setting": setting,
        "form": form,
        "scope": scope,
        "beta": beta,
        "spearman_rho": student["spearman_rho"],
        "tau_b_all_pairs": student["all_pairs"]["tau_b"],
        "tau_b_size_controlled": controlled["tau_b"],
        "size_controlled_pairs": controlled["pairs"],
        "score_only_ties": student["all_pairs"]["score_only_ties"],
        "scoring_seconds": result["scoring_seconds"]["student"],
        "sweep_wall_seconds": wall,
    }


def select_betas(rows: list[dict]) -> dict[str, dict]:
    """Pick one beta per setting; ties prefer smaller |beta|, then the negative."""
    selected = {}
    for setting, _, _ in SETTINGS:
        candidates = [row for row in rows if row["setting"] == setting]
        selected[setting] = max(
            candidates,
            key=lambda row: (
                row["spearman_rho"],
                -abs(row["beta"]),
                0 if row["beta"] < 0 else 1,
            ),
        )
    return selected


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in FIELDS})


def write_markdown(path: Path, rows: list[dict], selected: dict[str, dict]) -> None:
    lines = [
        "# beta 扫描结果（开发集，33 组）",
        "",
        "本文件由 `python scripts/sweep_beta.py` 生成，不要手工编辑。",
        "",
        "| 设置 | form | scope | β | Spearman ρ | τb（全部对） | τb（180 受控配对） | 评分并列对 | 耗时 s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {setting} | {form} | {scope} | {beta:g} | {spearman_rho:.9f} | "
            "{tau_b_all_pairs:.9f} | {tau_b_size_controlled:.9f} | "
            "{score_only_ties} | {sweep_wall_seconds:.3f} |".format(**row)
        )
    lines += [
        "",
        "按计划的选择规则（ρ 最高；并列先取 |β| 小者，再优先负号）：",
        "",
        "| 设置 | 选中 β | Spearman ρ | τb（全部对） | τb（180 受控配对） |",
        "|---|---:|---:|---:|---:|",
    ]
    for setting, _, _ in SETTINGS:
        row = selected[setting]
        lines.append(
            "| {setting} | **{beta:g}** | {spearman_rho:.9f} | {tau_b_all_pairs:.9f} | "
            "{tau_b_size_controlled:.9f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    evaluate = load_evaluator()
    rows = []
    started = time.perf_counter()
    for setting, form, scope in SETTINGS:
        for beta in BETAS:
            result, wall = run_setting(evaluate, form, scope, beta)
            rows.append(make_row(setting, form, scope, beta, result, wall))
            row = rows[-1]
            print(
                f"{setting:12s} beta={beta:>8g} rho={row['spearman_rho']:.9f} "
                f"tau={row['tau_b_all_pairs']:.9f} "
                f"controlled_tau={row['tau_b_size_controlled']:.9f}"
            )
    elapsed = time.perf_counter() - started

    selected = select_betas(rows)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "beta_sweep.csv", rows)
    write_markdown(output_dir / "beta_sweep.md", rows, selected)
    payload = {
        "meta": {
            "split": "development",
            "architectures": 100,
            "settings": [list(item) for item in SETTINGS],
            "betas": list(BETAS),
            "selection_rule": (
                "highest Spearman rho against negative validation PPL; "
                "ties prefer smaller |beta|, then the negative sign"
            ),
            "selection_note": (
                "the reported rho of a selected beta is optimistic: the same "
                "development split was used to choose it"
            ),
            "random_seed": (
                "starter fixed schedule, seed = 202627 + budget + k, used only "
                "to break score ties in the selection section"
            ),
            "command": "python scripts/sweep_beta.py",
            "environment": library_versions(),
            "total_seconds": elapsed,
        },
        "rows": rows,
        "selected": selected,
    }
    (output_dir / "beta_sweep.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print("\nselected:")
    for setting, _, _ in SETTINGS:
        row = selected[setting]
        print(
            f"  {setting:12s} beta={row['beta']:g} rho={row['spearman_rho']:.9f} "
            f"tau={row['tau_b_all_pairs']:.9f} "
            f"controlled_tau={row['tau_b_size_controlled']:.9f}"
        )
    print(f"\ntotal seconds: {elapsed:.1f}")
    print(f"wrote {output_dir.relative_to(REPO_ROOT)}/beta_sweep{{.csv,.md,.json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
