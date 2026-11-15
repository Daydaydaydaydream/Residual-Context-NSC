#!/usr/bin/env python3
"""Draw figure 1: rho and tau-b against |beta| for the three settings.

Input is ``results/tables/beta-curve.csv`` (exported by ``scripts/build_tables.py``
from the archived sweep). The negative branch is drawn solid and the positive
branch dashed, because the signed weight is not symmetric in beta: the selected
values all sit on the negative side.

matplotlib is not part of the scoring environment, so this script is optional and
prints a hint when the import fails.

Usage (repository root):

    python scripts/make_figures.py
    python scripts/make_figures.py --output results/tables/fig1-rho-vs-beta.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CURVE = REPO_ROOT / "results" / "tables" / "beta-curve.csv"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "tables" / "fig1-rho-vs-beta.png"

TICKS = ("0", "0.1", "1", "10", "100", "1000")
SETTINGS = {
    "main": {"color": "#1f77b4", "marker": "o", "label": "main (signed/ffn)"},
    "joint": {"color": "#d62728", "marker": "s", "label": "joint"},
    "exponential": {"color": "#2ca02c", "marker": "^", "label": "exponential"},
}
NEGATIVE = (-0.1, -1.0, -10.0, -100.0, -1000.0)
POSITIVE = (0.1, 1.0, 10.0, 100.0, 1000.0)


def load_curve(path: Path) -> dict[tuple[str, float], tuple[float, float]]:
    """Return {(setting, beta): (rho, tau_b)} from the exported curve."""
    series: dict[tuple[str, float], tuple[float, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            series[(row["setting"], float(row["beta"]))] = (
                float(row["spearman_rho"]),
                float(row["tau_b_all_pairs"]),
            )
    return series


def draw(series: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = series[("main", 0.0)]
    fig, (ax_rho, ax_tau) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    zero = TICKS.index("0")

    for setting, style in SETTINGS.items():
        for axis, column in ((ax_rho, 0), (ax_tau, 1)):
            axis.plot(
                zero,
                baseline[column],
                marker=style["marker"],
                color="0.25",
                markersize=6,
                zorder=5,
            )
            for betas, linestyle, width, size, alpha in (
                (NEGATIVE, "-", 1.5, 5, 1.0),
                (POSITIVE, "--", 1.0, 4, 0.75),
            ):
                xs = [zero]
                ys = [baseline[column]]
                for beta in betas:
                    xs.append(TICKS.index(f"{abs(beta):g}"))
                    ys.append(series[(setting, beta)][column])
                axis.plot(
                    xs,
                    ys,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=linestyle,
                    linewidth=width,
                    markersize=size,
                    alpha=alpha,
                    label=style["label"] if betas is NEGATIVE else None,
                )

    for axis, column, baseline_rho in ((ax_rho, 0, baseline[0]), (ax_tau, 1, baseline[1])):
        axis.axhline(baseline_rho, linestyle=":", linewidth=1.2, color="0.35")
        axis.annotate("original NSC", (zero + 0.05, baseline_rho), fontsize=7, color="0.35")
        axis.set_xticks(range(len(TICKS)))
        axis.set_xticklabels(TICKS)
        axis.set_xlabel("|beta|   (solid: beta < 0, dashed: beta > 0)", fontsize=8.5)
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.tick_params(labelsize=8)

    ax_rho.set_title("Spearman rho against negative validation PPL", fontsize=9.5)
    ax_tau.set_title("Kendall tau-b (all pairs)", fontsize=9.5)
    ax_rho.set_ylabel("Spearman rho", fontsize=9)
    ax_tau.set_ylabel("Kendall tau-b", fontsize=9)
    ax_rho.legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", type=Path, default=CURVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "matplotlib is not installed; the figure is optional and this script "
            "only needs results/tables/beta-curve.csv to redraw it."
        )
        return 1

    series = load_curve(args.curve)
    draw(series, args.output)
    print(f"wrote {args.output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
