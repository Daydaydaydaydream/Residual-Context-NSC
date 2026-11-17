"""Tests for the course scoring rules.

Two properties define the project and are asserted here:

1. The original NSC score is additive, so it is invariant to FFN reordering.
2. The residual-context score depends on preceding residual load, so it is
   order-sensitive at ``alpha > 0`` and collapses to the additive baseline at
   ``alpha = 0``.
"""

import pytest

from cs5491_nsc import (
    TransformerLayerSpec,
    additive_ablation,
    ffn_residual_load,
    layer_capacities,
    original_nsc_score,
    residual_context_score,
    score_transformer,
)


def test_original_nsc_is_invariant_to_layer_order(
    ascending_widths: list[TransformerLayerSpec],
    descending_widths: list[TransformerLayerSpec],
) -> None:
    """The additive baseline cannot distinguish FFN orderings."""
    assert original_nsc_score(ascending_widths) == pytest.approx(
        original_nsc_score(descending_widths)
    )


def test_residual_context_score_is_order_sensitive(
    ascending_widths: list[TransformerLayerSpec],
    descending_widths: list[TransformerLayerSpec],
) -> None:
    """The proposed heuristic must respond to a controlled reordering."""
    assert score_transformer(ascending_widths, alpha=1.0) != pytest.approx(
        score_transformer(descending_widths, alpha=1.0)
    )


def test_zero_alpha_reduces_to_additive_ablation(
    ascending_widths: list[TransformerLayerSpec],
    descending_widths: list[TransformerLayerSpec],
) -> None:
    layers = ascending_widths + descending_widths
    assert score_transformer(layers, alpha=0.0) == pytest.approx(
        original_nsc_score(layers)
    )


def test_additive_ablation_matches_manual_sum(
    ascending_widths: list[TransformerLayerSpec],
) -> None:
    """The ablation must equal the plain sum of selected matrix capacities."""
    attention, ffn, loads = zip(
        *(layer_capacities(layer) for layer in ascending_widths)
    )
    expected = sum(a + f for a, f in zip(attention, ffn))
    assert additive_ablation(attention, ffn, loads) == pytest.approx(expected)


def test_residual_load_grows_with_width() -> None:
    assert ffn_residual_load(128, 512) > ffn_residual_load(128, 256)


def test_first_layer_is_unaffected_by_prior_load() -> None:
    """Layer 0 has empty history, so its contribution is A_0 + F_0."""
    attention, ffn, loads = ([1.0], [2.0], [0.5])
    assert residual_context_score(attention, ffn, loads, alpha=5.0) == pytest.approx(
        3.0
    )


def test_identical_layers_reduce_to_additive_score() -> None:
    """Uniform FFN widths make the residual load constant across layers."""
    layers = [TransformerLayerSpec(128, 4, 256) for _ in range(4)]
    attention, ffn, loads = zip(*(layer_capacities(layer) for layer in layers))
    assert residual_context_score(attention, ffn, loads, alpha=1.0) < sum(
        a + f for a, f in zip(attention, ffn)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_width": 0, "heads": 4, "ffn_width": 256},
        {"hidden_width": 128, "heads": 0, "ffn_width": 256},
        {"hidden_width": 128, "heads": 4, "ffn_width": 0},
        {"hidden_width": 130, "heads": 4, "ffn_width": 256},
    ],
)
def test_invalid_layer_specs_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        layer_capacities(TransformerLayerSpec(**kwargs))


@pytest.mark.parametrize("alpha", [-1.0, float("nan"), float("inf")])
def test_invalid_alpha_is_rejected(alpha: float) -> None:
    with pytest.raises(ValueError):
        residual_context_score([1.0], [1.0], [1.0], alpha=alpha)


def test_mismatched_sequence_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        residual_context_score([1.0, 1.0], [1.0], [1.0, 1.0])


def test_empty_model_scores_zero() -> None:
    assert original_nsc_score([]) == 0.0
    assert score_transformer([], alpha=1.0) == 0.0
