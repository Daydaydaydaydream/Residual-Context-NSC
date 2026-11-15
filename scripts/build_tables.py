#!/usr/bin/env python3
"""Build the two development-split result tables from the frozen settings.

The frozen settings are the ones recorded in ``docs/selected-settings.md``:

    main          signed / ffn            beta = -100
    joint         signed / joint          beta = -1
    exponential   exponential / ffn       beta = -100

This script does not choose them. It runs the course evaluator four times on the
development split (the three settings plus ``beta = 0``, which yields the
original NSC, the ablation, the position example, the size/FLOP references and
the uniform-random baseline) and writes the ranking table, the budget table and
the plotting input for the rho-vs-|beta| curve.

The final split is never touched here.

Usage (repository root):

    python scripts/build_tables.py
    python scripts/build_tables.py --output-dir results/tables
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
SWEEP_CSV = REPO_ROOT / "results" / "beta-sweep" / "beta_sweep.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "tables"

#: Frozen settings, in the order the report introduces them.
SETTINGS = (
    ("baseline", "signed", "ffn", 0.0),
    ("main", "signed", "ffn", -100.0),
    ("joint", "signed", "joint", -1.0),
    ("exponential", "exponential", "ffn", -100.0),
)

RANKING_ROWS = (
    ("original NSC", "baseline", "original_nsc"),
    ("ablation (beta=0)", "baseline", "ablation"),
    ("position example", "baseline", "position_example"),
    ("main beta=-100", "main", "student"),
    ("joint beta=-1", "joint", "student"),
    ("exponential beta=-100", "exponential", "student"),
    ("reported parameters", "baseline", "reported_parameters"),
    ("reference FLOPs", "baseline", "reference_flops"),
)

BUDGET_ROWS = (
    ("original NSC", "baseline", "original_nsc"),
    ("ablation (beta=0)", "baseline", "ablation"),
    ("position example", "baseline", "position_example"),
    ("main beta=-100", "main", "student"),
    ("joint beta=-1", "joint", "student"),
    ("exponential beta=-100", "exponential", "student"),
    ("reported parameters", "baseline", "reported_parameters"),
    ("reference FLOPs", "baseline", "reference_flops"),
    ("uniform random", "baseline", "random"),
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


def run_setting(evaluate, form: str, scope: str, beta: float) -> tuple[dict, float]:
    """Evaluate one setting and return the panel plus its wall time."""
    os.environ["NSC_FORM"] = form
    os.environ["NSC_SCOPE"] = scope
    os.environ["NSC_BETA"] = repr(beta)
    start = time.perf_counter()
    result = evaluate("development", STUDENT)
    return result, time.perf_counter() - start


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = module.__version__
    return versions


def ranking_table(runs: dict[str, dict]) -> str:
    """Seven-method ranking comparison against negative validation PPL."""
    base = runs["baseline"]["ranking"]["original_nsc"]["spearman_rho"]
    lines = [
        "| method | Spearman rho | Kendall tau-b (all pairs) | tau-b (180 "
        "size-controlled pairs) | score ties | delta rho vs original NSC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, run, key in RANKING_ROWS:
        row = runs[run]["ranking"][key]
        delta = "--" if label == "original NSC" else f"{row['spearman_rho'] - base:+.6f}"
        lines.append(
            f"| {label} | {row['spearman_rho']:.6f} | "
            f"{row['all_pairs']['tau_b']:.6f} | "
            f"{row['size_controlled_pairs']['tau_b']:.6f} | "
            f"{row['all_pairs']['score_only_ties']} | {delta} |"
        )
    return "\n".join(lines)


def budget_header(runs: dict[str, dict]) -> list[str]:
    header = ["method"]
    for cell in runs["baseline"]["selection"]:
        cap = cell["parameter_cap"] // 1_000_000
        header.append(f"{cap}M / k={cell['k']}")
    return header


def best_ppl_table(runs: dict[str, dict]) -> str:
    """Best revealed validation PPL, with the absolute regret in parentheses."""
    header = budget_header(runs)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for label, run, key in BUDGET_ROWS:
        cells = []
        for cell in runs[run]["selection"]:
            stats = cell["methods"][key]
            cells.append(f"{stats['mean_best_ppl']:.4f} ({stats['mean_regret']:.4f})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def regret_table(runs: dict[str, dict]) -> str:
    """The same table as regrets only, the form the report uses."""
    header = budget_header(runs)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for label, run, key in BUDGET_ROWS:
        cells = [
            f"{cell['methods'][key]['mean_regret']:.4f}"
            for cell in runs[run]["selection"]
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pool_table(runs: dict[str, dict]) -> str:
    """Eligible architecture count and the pool oracle for each budget cell."""
    lines = [
        "| parameter cap | k | eligible architectures | pool oracle PPL |",
        "|---|---:|---:|---:|",
    ]
    for cell in runs["baseline"]["selection"]:
        lines.append(
            f"| {cell['parameter_cap'] // 1_000_000}M | {cell['k']} | "
            f"{cell['eligible']} | {cell['oracle_ppl']:.4f} |"
        )
    return "\n".join(lines)


def write_beta_curve(path: Path) -> bool:
    """Export the sweep as plotting input, sorted by |beta| within a setting."""
    if not SWEEP_CSV.is_file():
        return False
    rows = list(csv.DictReader(SWEEP_CSV.open(encoding="utf-8")))
    rows.sort(key=lambda row: (row["setting"], abs(float(row["beta"]))))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "setting",
                "form",
                "scope",
                "beta",
                "abs_beta",
                "spearman_rho",
                "tau_b_all_pairs",
                "tau_b_size_controlled",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row["setting"],
                    row["form"],
                    row["scope"],
                    row["beta"],
                    f"{abs(float(row['beta'])):g}",
                    row["spearman_rho"],
                    row["tau_b_all_pairs"],
                    row["tau_b_size_controlled"],
                )
            )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    evaluate = load_evaluator()
    runs: dict[str, dict] = {}
    timings: dict[str, float] = {}
    total_start = time.perf_counter()
    for name, form, scope, beta in SETTINGS:
        runs[name], timings[name] = run_setting(evaluate, form, scope, beta)
        student = runs[name]["ranking"]["student"]
        print(
            f"{name:12s} beta={beta:>7g} rho={student['spearman_rho']:.9f} "
            f"tau={student['all_pairs']['tau_b']:.9f} "
            f"controlled_tau={student['size_controlled_pairs']['tau_b']:.9f}"
        )
    total = time.perf_counter() - total_start
    for name in ("NSC_FORM", "NSC_SCOPE", "NSC_BETA"):
        os.environ.pop(name, None)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    header = (
        "数据：`course/nsc-starter/data/`，development split 100 个架构；"
        "选择使用 starter 固定种子 `202627 + budget + k`，200 次重复。\n"
        "设置取自 `docs/selected-settings.md`：main -100、joint -1、"
        "exponential -100。\n"
    )
    (output_dir / "ranking.md").write_text(
        f"# 排序对比（对负验证 PPL）\n\n{header}\n{ranking_table(runs)}\n",
        encoding="utf-8",
    )
    (output_dir / "budget.md").write_text(
        "# 预算选择\n\n"
        f"{header}\n"
        "## 入选最佳 PPL（括号内为绝对遗憾 = 选中最优 PPL - 池内最优 PPL）\n\n"
        f"{best_ppl_table(runs)}\n\n"
        "## 遗憾值\n\n"
        "数值为 `mean_best_ppl - oracle_ppl`，绝对差值，不是相对遗憾；"
        "若要相对遗憾需另算 `mean_regret / oracle_ppl`。\n\n"
        f"{regret_table(runs)}\n\n"
        "## 预算池\n\n"
        f"{pool_table(runs)}\n",
        encoding="utf-8",
    )

    payload = {
        "meta": {
            "split": "development",
            "architectures": runs["baseline"]["architectures"],
            "settings": [list(item) for item in SETTINGS],
            "selection_rule_note": (
                "betas are frozen from docs/selected-settings.md; this script "
                "does not choose them"
            ),
            "random_seed": "starter fixed schedule, seed = 202627 + budget + k",
            "regret_definition": "mean_best_ppl - oracle_ppl (absolute)",
            "command": "python scripts/build_tables.py",
            "environment": library_versions(),
            "timings_seconds": timings,
            "total_seconds": total,
        },
        "ranking": {name: run["ranking"] for name, run in runs.items()},
        "selection": {name: run["selection"] for name, run in runs.items()},
    }
    (output_dir / "tables.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    wrote_curve = write_beta_curve(output_dir / "beta-curve.csv")

    files = ["ranking.md", "budget.md", "tables.json"]
    if wrote_curve:
        files.append("beta-curve.csv")
    (output_dir / "README.md").write_text(
        "# 开发集结果表（11/16 交付）\n\n"
        "两张表由冻结后的三种设置重新跑出，只使用 development split；"
        "最终集未参与。\n\n"
        "## 命令\n\n"
        "```bash\n"
        "python scripts/build_tables.py\n"
        "```\n\n"
        f"总耗时约 {total:.1f} s，四次评测分别为："
        + "、".join(f"{name} {seconds:.2f} s" for name, seconds in timings.items())
        + "。\n\n"
        "## 环境\n\n"
        "| item | value |\n|---|---|\n"
        + "\n".join(f"| {key} | {value} |" for key, value in library_versions().items())
        + "\n\n## 文件\n\n"
        + "\n".join(f"- `{name}`" for name in files)
        + "\n\n## 口径\n\n"
        "- 排序表：Spearman rho 与 Kendall tau-b 均对负验证 PPL 计算；"
        "`score ties` 是评分并列的对数。\n"
        "- 预算表：每档预算取可行池中分数最高的 k 个，报记录 PPL；"
        "遗憾为 `mean_best_ppl - oracle_ppl`，是绝对差值。\n"
        "- 200 次重复只用于打散评分并列和构造均匀随机参照，"
        "不是训练种子标准差。\n"
        "- `beta-curve.csv` 是图 1（rho / tau-b 随 |beta| 变化）的绘图输入，"
        "取自 `results/beta-sweep/beta_sweep.csv`。\n"
        "- 图 1 本身由 `python scripts/make_figures.py` 从 `beta-curve.csv` 画出，"
        "需要 matplotlib；输出 `fig1-rho-vs-beta.png`。\n",
        encoding="utf-8",
    )

    print(f"\ntotal {total:.2f} s")
    print(f"wrote {output_dir.relative_to(REPO_ROOT)}/{{{','.join(files)}}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
