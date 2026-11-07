"""Bridge between the course starter interface and the project's score.

The starter's evaluator calls ``score(config)`` and ``ablation(config)`` with a
single configuration dictionary. The project score needs per-layer specs and an
``alpha``, so this module does the conversion and reads ``alpha`` from the
environment at call time (``NSC_ALPHA``), which lets one file serve a whole
sweep without editing code between runs.

Run from ``course/nsc-starter``:

    NSC_ALPHA=1 python evaluate.py --split development \
        --student ../../scripts/starter_adapter.py

On a console without a POSIX shell, set the variable first:

    $env:NSC_ALPHA = "1"

``alpha = 0`` reproduces the original additive NSC, so ``ablation`` always uses
alpha = 0 regardless of the environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cs5491_nsc import (  # noqa: E402
    TransformerLayerSpec,
    original_nsc_score,
    score_transformer,
)

#: Matches the course archive's fixed initialization convention.
INIT_STD = 0.02
#: Used when NSC_ALPHA is unset; equals the project's default in `score_transformer`.
DEFAULT_ALPHA = 1.0


def _per_layer(value, n_layer, field):
    """Expand an int or validate a per-layer list into length n_layer"""
    if isinstance(value, int) and not isinstance(value, bool):
        return [value] * n_layer
    if isinstance(value, (list, tuple)):
        if len(value) != n_layer:
            raise ValueError(f"{field} must have length n_layer={n_layer}")
        return list(value)
    raise ValueError(f"{field} must be an int or a list of ints")


def to_layer_specs(config):
    """Convert one archived configuration into per-layer specs"""
    hidden = config["d_model"]
    n_layer = config["n_layer"]
    heads = _per_layer(config["n_head"], n_layer, "n_head")
    inner = _per_layer(config["d_inner"], n_layer, "d_inner")
    return [
        TransformerLayerSpec(hidden_width=hidden, heads=h, ffn_width=f)
        for h, f in zip(heads, inner)
    ]


def current_alpha():
    """alpha for this run: NSC_ALPHA if set, otherwise the project default."""
    raw = os.environ.get("NSC_ALPHA", "")
    if raw.strip() == "":
        return DEFAULT_ALPHA
    value = float(raw)
    if value < 0:
        raise ValueError("NSC_ALPHA must be non-negative")
    return value


def score(config):
    """Residual-context score; larger predicts better trained performance."""
    layers = to_layer_specs(config)
    return score_transformer(layers, init_std=INIT_STD, alpha=current_alpha())


def ablation(config):
    """Additive ablation: alpha = 0, i.e. the original NSC over the same matrices."""
    return original_nsc_score(to_layer_specs(config), init_std=INIT_STD)
