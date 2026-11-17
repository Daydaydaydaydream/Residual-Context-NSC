"""Starter-compatible score(config) and ablation(config) entry points.

The course evaluator imports a student module and calls these two functions with
one archived configuration dictionary. This module is the thin adapter between
that interface and the project scoring primitives.

Environment variables select the main setting and the two controls without
editing code between runs:

* NSC_BETA: signed beta (float). Defaults to DEFAULT_BETA.
* NSC_FORM: "signed" (main) or "exponential" (form control).
* NSC_SCOPE: "ffn" (main) or "joint" (scope control).

The ablation always uses beta = 0 over the same starter matrices, independent of
the environment, so it remains a clean mechanism ablation.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any

from .student_score import (
    DEFAULT_BETA,
    WEIGHT_FORMS,
    WEIGHT_SCOPES,
    ablation_config,
    score_config,
)


def _selected_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite float, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float, got {raw!r}")
    return value


def _selected_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {raw!r}")
    return raw


def selected_beta() -> float:
    """Return NSC_BETA or the provisional project default."""
    return _selected_float("NSC_BETA", DEFAULT_BETA)


def selected_form() -> str:
    """Return NSC_FORM or the main signed-rational form."""
    return _selected_choice("NSC_FORM", "signed", WEIGHT_FORMS)


def selected_scope() -> str:
    """Return NSC_SCOPE or the main FFN-only scope."""
    return _selected_choice("NSC_SCOPE", "ffn", WEIGHT_SCOPES)


def score(config: Mapping[str, Any]) -> float:
    """Signed layer-order score; larger values predict better quality."""
    return score_config(
        config,
        beta=selected_beta(),
        form=selected_form(),
        scope=selected_scope(),
    )


def ablation(config: Mapping[str, Any]) -> float:
    """Disable the mechanism: beta = 0, the original additive NSC."""
    return ablation_config(config)
