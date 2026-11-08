#!/usr/bin/env python3
"""Course-starter student module for the signed FFN layer-order score.

Run the archived evaluator from course/nsc-starter, for example:

    NSC_BETA=1 python3 evaluate.py --split development \
        --student ../../scripts/student_score.py

The evaluator calls score(config) and ablation(config). Set NSC_BETA, NSC_FORM
and NSC_SCOPE to sweep the signed main setting or the two controls; see
cs5491_nsc.submission for details.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cs5491_nsc.submission import ablation, score  # noqa: E402

__all__ = ["ablation", "score"]
