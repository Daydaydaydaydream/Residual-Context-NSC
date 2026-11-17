"""Tests for the vendored upstream MP spectral capacity utilities."""

import numpy as np
import pytest
from scipy import integrate

from cs5491_nsc import mp_density, psi_mp, xavier_sigma


def test_mp_density_integrates_to_one() -> None:
    """The MP density must be a normalized probability distribution."""
    for gamma in (0.25, 0.5, 1.0):
        lower = (1 - np.sqrt(gamma)) ** 2
        upper = (1 + np.sqrt(gamma)) ** 2
        total = integrate.quad(mp_density, lower, upper, args=(gamma,), limit=200)[0]
        assert total == pytest.approx(1.0, abs=1e-6)


def test_mp_density_is_zero_outside_support() -> None:
    gamma = 0.5
    lower = (1 - np.sqrt(gamma)) ** 2
    upper = (1 + np.sqrt(gamma)) ** 2
    assert mp_density(lower - 1e-6, gamma) == 0.0
    assert mp_density(upper + 1e-6, gamma) == 0.0


def test_psi_mp_is_symmetric_in_shape() -> None:
    """psi_mp uses N=min(m, n) and M=max(m, n), so it must not depend on order."""
    assert psi_mp(64, 256, 0.02) == pytest.approx(psi_mp(256, 64, 0.02))


def test_psi_mp_is_positive_and_increases_with_sigma() -> None:
    small = psi_mp(128, 512, 0.01)
    large = psi_mp(128, 512, 0.02)
    assert 0.0 < small < large


def test_psi_mp_degenerate_width_is_zero() -> None:
    assert psi_mp(0, 128, 0.02) == 0.0
    assert psi_mp(128, 0, 0.02) == 0.0


def test_xavier_sigma_matches_glorot_formula() -> None:
    assert xavier_sigma(512, 2048) == pytest.approx(np.sqrt(2.0 / (512 + 2048)))


def test_psi_mp_cache_returns_consistent_values() -> None:
    """Repeated calls must hit the cache without changing the result."""
    first = psi_mp(96, 384, 0.02)
    second = psi_mp(96, 384, 0.02)
    assert first == second
