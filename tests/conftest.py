"""Shared fixtures for the CS5491 NSC test suite."""

import importlib.util
import json
from pathlib import Path

import pytest

from cs5491_nsc import TransformerLayerSpec

# The course evaluation convention fixes the initialization std at 0.02.
COURSE_INIT_STD = 0.02

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_ROOT = REPO_ROOT / "course" / "nsc-starter"


@pytest.fixture
def init_std() -> float:
    return COURSE_INIT_STD


@pytest.fixture
def ascending_widths() -> list[TransformerLayerSpec]:
    """Two layers with narrow-then-wide FFN."""
    return [
        TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=256),
        TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=512),
    ]


@pytest.fixture
def descending_widths() -> list[TransformerLayerSpec]:
    """The same widths as ascending_widths, FFN order reversed."""
    return [
        TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=512),
        TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=256),
    ]


@pytest.fixture
def reorder_configs() -> tuple[dict, dict]:
    """Equivalent unlabelled starter configs with reversed FFN widths."""
    forward = {
        "d_model": 128,
        "n_layer": 2,
        "n_head": 4,
        "d_inner": [256, 512],
    }
    reverse = {
        "d_model": 128,
        "n_layer": 2,
        "n_head": 4,
        "d_inner": [512, 256],
    }
    return forward, reverse


@pytest.fixture(scope="session")
def archived_configs() -> dict:
    """All 200 archived configurations, used only for interface checks."""
    path = STARTER_ROOT / "data" / "architectures.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def starter_nsc():
    """Load the archived starter nsc.py without modifying sys.path."""
    path = STARTER_ROOT / "nsc.py"
    spec = importlib.util.spec_from_file_location("course_starter_nsc", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
