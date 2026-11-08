"""Tests for the starter-compatible score(config)/ablation(config) adapter."""

import importlib.util
from pathlib import Path

import pytest

from cs5491_nsc import (
    DEFAULT_BETA,
    ablation_config,
    exponential_decay_score,
    joint_weighted_score,
    layer_capacities,
    layer_specs_from_config,
    score_config,
    signed_ffn_score,
    submission,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _clear_environment(monkeypatch) -> None:
    for name in ("NSC_BETA", "NSC_FORM", "NSC_SCOPE"):
        monkeypatch.delenv(name, raising=False)


def test_default_environment_uses_provisional_beta(
    monkeypatch, reorder_configs
) -> None:
    _clear_environment(monkeypatch)
    config = reorder_configs[0]
    assert submission.score(config) == pytest.approx(
        score_config(config, beta=DEFAULT_BETA)
    )


def test_ablation_ignores_environment_controls(monkeypatch, reorder_configs) -> None:
    config = reorder_configs[0]
    _clear_environment(monkeypatch)
    baseline = ablation_config(config)
    monkeypatch.setenv("NSC_BETA", "1000")
    monkeypatch.setenv("NSC_FORM", "exponential")
    monkeypatch.setenv("NSC_SCOPE", "joint")
    assert submission.ablation(config) == pytest.approx(baseline)


def test_score_reads_main_setting_from_environment(
    monkeypatch, reorder_configs
) -> None:
    config = reorder_configs[0]
    _clear_environment(monkeypatch)
    monkeypatch.setenv("NSC_BETA", "-1")
    assert submission.score(config) == pytest.approx(
        signed_ffn_score(*_capacities(config), beta=-1.0)
    )

    monkeypatch.setenv("NSC_SCOPE", "joint")
    assert submission.score(config) == pytest.approx(
        joint_weighted_score(*_capacities(config), beta=-1.0)
    )

    monkeypatch.setenv("NSC_SCOPE", "ffn")
    monkeypatch.setenv("NSC_FORM", "exponential")
    assert submission.score(config) == pytest.approx(
        exponential_decay_score(*_capacities(config), beta=-1.0)
    )


def test_beta_zero_score_matches_starter(
    monkeypatch, archived_configs, starter_nsc
) -> None:
    monkeypatch.setenv("NSC_BETA", "0")
    for key in list(archived_configs)[:20]:
        config = archived_configs[key]
        assert submission.score(config) == pytest.approx(
            starter_nsc.original_nsc(config), rel=1e-10, abs=1e-12
        ), key


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("NSC_BETA", "nan"),
        ("NSC_BETA", "inf"),
        ("NSC_BETA", "not-a-number"),
        ("NSC_FORM", "unknown"),
        ("NSC_SCOPE", "unknown"),
    ],
)
def test_invalid_environment_values_are_rejected(
    monkeypatch, reorder_configs, name: str, value: str
) -> None:
    _clear_environment(monkeypatch)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        submission.score(reorder_configs[0])


def test_student_score_script_is_evaluator_compatible(
    monkeypatch, archived_configs, starter_nsc
) -> None:
    path = REPO_ROOT / "scripts" / "student_score.py"
    spec = importlib.util.spec_from_file_location("student_score_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setenv("NSC_BETA", "0")
    config = archived_configs["config_0_j0"]
    assert module.ablation(config) == pytest.approx(
        starter_nsc.original_nsc(config), rel=1e-10, abs=1e-12
    )
    assert module.score(config) == pytest.approx(
        starter_nsc.original_nsc(config), rel=1e-10, abs=1e-12
    )


def _capacities(config: dict):
    capacities = [layer_capacities(layer) for layer in layer_specs_from_config(config)]
    attention, ffn, loads = zip(*capacities)
    return attention, ffn, loads
