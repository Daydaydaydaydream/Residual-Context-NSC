"""Tests for the signed FFN layer-order score and its controls.

The mechanism claims checked here are deliberately label-free:

* beta = 0 reproduces the course's original additive NSC;
* layer 0 has weight 1 because it has no preceding load;
* positive/negative beta move the weight in the correct direction and reach
  the stated limits;
* both controls differ from the main form but collapse at beta = 0;
* a controlled FFN-width reordering changes the score even though the original
  NSC is invariant.
"""

import math

import pytest

from cs5491_nsc import (
    BETA_CANDIDATES,
    DEFAULT_BETA,
    TransformerLayerSpec,
    ablation,
    ablation_config,
    additive_ablation,
    exponential_context_weight,
    exponential_decay_score,
    ffn_residual_load,
    joint_weighted_score,
    layer_capacities,
    layer_order_weights,
    layer_specs_from_config,
    original_nsc_score,
    psi_mp,
    residual_context_score,
    score,
    score_config,
    score_transformer,
    signed_context_weight,
    signed_ffn_score,
    weighted_layer_score,
)

BETAS = (-1000.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 1000.0)
FORMS = ("signed", "exponential")
SCOPES = ("ffn", "joint")


def test_beta_candidates_match_the_frozen_grid() -> None:
    assert set(BETA_CANDIDATES) == {
        0.0,
        0.1,
        -0.1,
        1.0,
        -1.0,
        10.0,
        -10.0,
        100.0,
        -100.0,
        1000.0,
        -1000.0,
    }
    assert DEFAULT_BETA in BETA_CANDIDATES


@pytest.mark.parametrize("load", [0.0, 0.01, 0.25, 1.0])
@pytest.mark.parametrize("beta", BETAS)
def test_signed_weight_matches_closed_form(load: float, beta: float) -> None:
    expected = 1.0 + beta * load / (1.0 + abs(beta) * load)
    assert signed_context_weight(load, beta) == pytest.approx(expected)


@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("form", FORMS)
def test_first_layer_weight_is_one(beta: float, form: str) -> None:
    weights = layer_order_weights([0.01, 0.2, 0.3], beta=beta, form=form)
    assert weights[0] == 1.0

    attention, ffn, loads = [1.5], [2.5], [0.4]
    assert weighted_layer_score(
        attention, ffn, loads, beta=beta, form=form, scope="ffn"
    ) == pytest.approx(4.0)
    assert weighted_layer_score(
        attention, ffn, loads, beta=beta, form=form, scope="joint"
    ) == pytest.approx(4.0)


def test_positive_beta_increases_later_ffn_weight() -> None:
    load = 0.5
    for beta in (0.1, 1.0, 10.0, 1000.0):
        weight = signed_context_weight(load, beta)
        assert 1.0 < weight < 2.0
    assert signed_context_weight(load, 0.1) < signed_context_weight(load, 1.0)
    assert signed_context_weight(load, 1.0) < signed_context_weight(load, 10.0)
    # More preceding load means more amplification at a fixed positive beta.
    assert signed_context_weight(0.1, 1.0) < signed_context_weight(1.0, 1.0)


def test_negative_beta_decreases_later_ffn_weight() -> None:
    load = 0.5
    for beta in (-0.1, -1.0, -10.0, -1000.0):
        weight = signed_context_weight(load, beta)
        assert 0.0 < weight < 1.0
    assert signed_context_weight(load, -0.1) > signed_context_weight(load, -1.0)
    assert signed_context_weight(load, -1.0) > signed_context_weight(load, -10.0)
    # More preceding load means more damping at a fixed negative beta.
    assert signed_context_weight(0.1, -1.0) > signed_context_weight(1.0, -1.0)


def test_signed_weight_limits_and_zero() -> None:
    load = 0.5
    huge = 1e12
    assert signed_context_weight(load, 0.0) == 1.0
    assert signed_context_weight(load, huge) == pytest.approx(2.0, abs=1e-9)
    assert signed_context_weight(load, -huge) == pytest.approx(0.0, abs=1e-9)
    # The first layer has no history, so its weight is one at every beta.
    assert signed_context_weight(0.0, huge) == 1.0
    assert signed_context_weight(0.0, -huge) == 1.0


def test_exponential_control_direction_and_limits() -> None:
    load = 0.5
    huge = 1e12
    assert exponential_context_weight(load, 0.0) == 1.0
    assert 1.0 < exponential_context_weight(load, 1.0) < 2.0
    assert 0.0 < exponential_context_weight(load, -1.0) < 1.0
    assert exponential_context_weight(load, 0.1) < exponential_context_weight(load, 1.0)
    assert exponential_context_weight(load, -0.1) > exponential_context_weight(
        load, -1.0
    )
    assert exponential_context_weight(load, huge) == pytest.approx(2.0, abs=1e-9)
    assert exponential_context_weight(load, -huge) == pytest.approx(0.0, abs=1e-9)
    assert exponential_context_weight(0.0, huge) == 1.0
    assert exponential_context_weight(0.0, -huge) == 1.0


def test_main_score_uses_only_preceding_load() -> None:
    attention, ffn, loads = [1.0, 2.0], [3.0, 4.0], [0.5, 0.25]
    beta = 1.0
    weights = layer_order_weights(loads, beta=beta)
    expected = attention[0] + weights[0] * ffn[0]
    expected += attention[1] + weights[1] * ffn[1]
    assert signed_ffn_score(attention, ffn, loads, beta=beta) == pytest.approx(expected)
    # The second weight depends on the first layer's load, not its own.
    assert weights[1] == pytest.approx(signed_context_weight(loads[0], beta))


def test_joint_control_weights_attention_too() -> None:
    attention, ffn, loads = [1.0, 2.0], [3.0, 4.0], [0.5, 0.25]
    weights = layer_order_weights(loads, beta=1.0)
    expected = sum(w * (a + f) for w, a, f in zip(weights, attention, ffn))
    assert joint_weighted_score(attention, ffn, loads, beta=1.0) == pytest.approx(
        expected
    )
    assert joint_weighted_score(attention, ffn, loads, beta=1.0) != pytest.approx(
        signed_ffn_score(attention, ffn, loads, beta=1.0)
    )


def test_exponential_control_uses_signed_exponential_weights() -> None:
    attention, ffn, loads = [1.0, 2.0], [3.0, 4.0], [0.5, 0.25]
    weights = layer_order_weights(loads, beta=-1.0, form="exponential")
    expected = sum(a + w * f for w, a, f in zip(weights, attention, ffn))
    assert exponential_decay_score(attention, ffn, loads, beta=-1.0) == pytest.approx(
        expected
    )
    assert exponential_decay_score(attention, ffn, loads, beta=-1.0) != pytest.approx(
        signed_ffn_score(attention, ffn, loads, beta=-1.0)
    )


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("scope", SCOPES)
def test_zero_beta_collapses_every_variant_to_nsc(form: str, scope: str) -> None:
    attention, ffn, loads = [1.0, 2.0], [3.0, 4.0], [0.5, 0.25]
    expected = sum(a + f for a, f in zip(attention, ffn))
    assert weighted_layer_score(
        attention, ffn, loads, beta=0.0, form=form, scope=scope
    ) == pytest.approx(expected)
    assert additive_ablation(attention, ffn, loads) == pytest.approx(expected)


def test_factory_convenience_scores_match_weighted_layer_score() -> None:
    attention, ffn, loads = [1.0, 2.0], [3.0, 4.0], [0.5, 0.25]
    beta = -1.0
    assert signed_ffn_score(attention, ffn, loads, beta) == pytest.approx(
        weighted_layer_score(
            attention, ffn, loads, beta=beta, form="signed", scope="ffn"
        )
    )
    assert joint_weighted_score(attention, ffn, loads, beta) == pytest.approx(
        weighted_layer_score(
            attention, ffn, loads, beta=beta, form="signed", scope="joint"
        )
    )
    assert exponential_decay_score(attention, ffn, loads, beta) == pytest.approx(
        weighted_layer_score(
            attention, ffn, loads, beta=beta, form="exponential", scope="ffn"
        )
    )


def test_ffn_width_reordering_is_detected_without_labels(
    ascending_widths: list[TransformerLayerSpec],
    descending_widths: list[TransformerLayerSpec],
    reorder_configs: tuple[dict, dict],
) -> None:
    forward, reverse = reorder_configs
    assert original_nsc_score(ascending_widths) == pytest.approx(
        original_nsc_score(descending_widths)
    )
    assert ablation_config(forward) == pytest.approx(ablation_config(reverse))

    for beta in (-1.0, 1.0):
        for form, scope in (
            ("signed", "ffn"),
            ("signed", "joint"),
            ("exponential", "ffn"),
        ):
            forward_score = score_transformer(
                ascending_widths, beta=beta, form=form, scope=scope
            )
            reverse_score = score_transformer(
                descending_widths, beta=beta, form=form, scope=scope
            )
            assert forward_score != pytest.approx(reverse_score)
            assert score_config(
                forward, beta=beta, form=form, scope=scope
            ) == pytest.approx(forward_score)
            assert score_config(
                reverse, beta=beta, form=form, scope=scope
            ) == pytest.approx(reverse_score)


def test_zero_beta_matches_starter_nsc_for_all_archived_configs(
    archived_configs: dict, starter_nsc
) -> None:
    for key, config in archived_configs.items():
        expected = starter_nsc.original_nsc(config)
        assert ablation_config(config) == pytest.approx(
            expected, rel=1e-10, abs=1e-12
        ), key
        assert score_config(config, beta=0.0) == pytest.approx(
            expected, rel=1e-10, abs=1e-12
        ), key
        for form, scope in (("signed", "joint"), ("exponential", "ffn")):
            assert score_config(
                config, beta=0.0, form=form, scope=scope
            ) == pytest.approx(expected, rel=1e-10, abs=1e-12), key


def test_config_parser_matches_starter_layer_grouping(
    archived_configs: dict, starter_nsc
) -> None:
    for key, config in archived_configs.items():
        hidden, heads, inner = starter_nsc.layers(config)
        specs = layer_specs_from_config(config)
        assert [(spec.hidden_width, spec.heads, spec.ffn_width) for spec in specs] == [
            (hidden, head, ffn) for head, ffn in zip(heads, inner)
        ], key


def test_layer_capacities_keep_starter_grouping_and_init_std() -> None:
    layer = TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=256)
    attention, ffn, load = layer_capacities(layer)
    expected_attention = 3 * 4 * psi_mp(128, 32, 0.02) + psi_mp(128, 128, 0.02)
    assert attention == pytest.approx(expected_attention)
    assert ffn == pytest.approx(2 * psi_mp(128, 256, 0.02))
    assert load == pytest.approx(0.5 * 128 * 256 * 0.02**4)


def test_legacy_alpha_is_negative_beta() -> None:
    attention, ffn, loads = [1.0, 2.0], [3.0, 4.0], [0.5, 0.25]
    assert residual_context_score(attention, ffn, loads, alpha=1.5) == pytest.approx(
        signed_ffn_score(attention, ffn, loads, beta=-1.5)
    )

    layers = [
        TransformerLayerSpec(128, 4, 256),
        TransformerLayerSpec(128, 4, 512),
    ]
    assert score_transformer(layers, alpha=1.0) == pytest.approx(
        score_transformer(layers, beta=-1.0)
    )
    assert score_transformer(layers, alpha=0.0) == pytest.approx(
        original_nsc_score(layers)
    )
    with pytest.raises(ValueError):
        score_transformer(layers, beta=1.0, alpha=1.0)


@pytest.mark.parametrize("beta", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_beta_is_rejected(beta: float) -> None:
    with pytest.raises(ValueError):
        signed_context_weight(0.5, beta)
    with pytest.raises(ValueError):
        exponential_context_weight(0.5, beta)
    with pytest.raises(ValueError):
        layer_order_weights([0.5], beta=beta)


@pytest.mark.parametrize("load", [-1.0, float("nan"), float("inf")])
def test_invalid_prior_load_is_rejected(load: float) -> None:
    with pytest.raises(ValueError):
        signed_context_weight(load, 1.0)
    with pytest.raises(ValueError):
        exponential_context_weight(load, 1.0)


def test_invalid_form_and_scope_are_rejected() -> None:
    with pytest.raises(ValueError):
        layer_order_weights([0.5], form="unknown")
    with pytest.raises(ValueError):
        weighted_layer_score([1.0], [1.0], [0.5], scope="unknown")


def test_mismatched_sequence_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        weighted_layer_score([1.0, 2.0], [1.0], [0.5, 0.5])
    with pytest.raises(ValueError):
        weighted_layer_score([1.0], [1.0, 2.0], [0.5, 0.5])
    with pytest.raises(ValueError):
        weighted_layer_score([1.0], [1.0], [0.5, 0.5])


def test_negative_capacities_are_rejected() -> None:
    with pytest.raises(ValueError):
        weighted_layer_score([-1.0], [1.0], [0.5])
    with pytest.raises(ValueError):
        weighted_layer_score([1.0], [-1.0], [0.5])
    with pytest.raises(ValueError):
        weighted_layer_score([1.0], [1.0], [-0.5])


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"d_model": 0, "n_layer": 2, "n_head": 4, "d_inner": [256, 512]},
        {"d_model": 128, "n_layer": 0, "n_head": 4, "d_inner": []},
        {"d_model": 128, "n_layer": 2, "n_head": 3, "d_inner": [256, 512]},
        {"d_model": 128, "n_layer": 2, "n_head": [4], "d_inner": [256, 512]},
        {"d_model": 128, "n_layer": 2, "n_head": [4, 4], "d_inner": [256]},
        {"d_model": 128, "n_layer": 2, "n_head": [4, 0], "d_inner": [256, 512]},
        {"d_model": True, "n_layer": 2, "n_head": 4, "d_inner": [256, 512]},
    ],
)
def test_invalid_configs_are_rejected(config: dict) -> None:
    with pytest.raises(ValueError):
        layer_specs_from_config(config)


def test_empty_model_scores_zero() -> None:
    assert original_nsc_score([]) == 0.0
    assert score_transformer([], beta=1.0) == 0.0
    assert signed_ffn_score([], [], [], beta=1.0) == 0.0
    assert layer_order_weights([], beta=1.0) == []


def test_direct_config_aliases_match_explicit_functions(
    reorder_configs: tuple[dict, dict],
) -> None:
    forward, _ = reorder_configs
    assert score(forward) == pytest.approx(score_config(forward, beta=DEFAULT_BETA))
    assert ablation(forward) == pytest.approx(ablation_config(forward))
    assert score(forward, beta=0.0) == pytest.approx(ablation(forward))


def test_score_is_monotone_in_signed_beta(
    ascending_widths: list[TransformerLayerSpec],
) -> None:
    betas = (-1000.0, -10.0, -1.0, 0.0, 1.0, 10.0, 1000.0)
    values = [score_transformer(ascending_widths, beta=beta) for beta in betas]
    assert all(left <= right for left, right in zip(values, values[1:]))
    assert values[0] < values[-1]


def test_score_limits_are_attention_only_and_attention_plus_twice_ffn() -> None:
    attention, ffn, loads = [1.0, 2.0], [3.0, 4.0], [0.5, 0.25]
    huge = 1e12

    # Layer 0 has no preceding load, so its weight stays 1 in both limits.
    negative_limit_ffn = sum(attention) + ffn[0]
    positive_limit_ffn = sum(attention) + ffn[0] + 2 * sum(ffn[1:])
    negative_limit_joint = attention[0] + ffn[0]
    positive_limit_joint = (
        attention[0] + ffn[0] + 2 * sum(a + f for a, f in zip(attention[1:], ffn[1:]))
    )

    assert signed_ffn_score(attention, ffn, loads, beta=-huge) == pytest.approx(
        negative_limit_ffn, abs=1e-9
    )
    assert signed_ffn_score(attention, ffn, loads, beta=huge) == pytest.approx(
        positive_limit_ffn, abs=1e-9
    )
    assert exponential_decay_score(attention, ffn, loads, beta=-huge) == (
        pytest.approx(negative_limit_ffn, abs=1e-9)
    )
    assert exponential_decay_score(attention, ffn, loads, beta=huge) == (
        pytest.approx(positive_limit_ffn, abs=1e-9)
    )
    assert joint_weighted_score(attention, ffn, loads, beta=-huge) == pytest.approx(
        negative_limit_joint, abs=1e-9
    )
    assert joint_weighted_score(attention, ffn, loads, beta=huge) == pytest.approx(
        positive_limit_joint, abs=1e-9
    )


def test_scores_remain_finite_at_extreme_beta(
    archived_configs: dict,
) -> None:
    config = archived_configs["config_0_j0"]
    for beta in (-1000.0, 1000.0):
        for form in FORMS:
            for scope in SCOPES:
                value = score_config(config, beta=beta, form=form, scope=scope)
                assert math.isfinite(value)


@pytest.mark.parametrize("value", [None, "4", True])
def test_invalid_per_layer_value_types_are_rejected(value: object) -> None:
    config = {
        "d_model": 128,
        "n_layer": 2,
        "n_head": value,
        "d_inner": [256, 512],
    }
    with pytest.raises(ValueError):
        layer_specs_from_config(config)


def test_non_mapping_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        layer_specs_from_config([])


@pytest.mark.parametrize("init_std", [-0.1, float("nan"), float("inf"), "bad"])
def test_invalid_init_std_is_rejected(init_std: object) -> None:
    with pytest.raises(ValueError):
        ffn_residual_load(128, 256, init_std=init_std)


@pytest.mark.parametrize(("load", "beta"), [("bad", 1.0), (0.5, "bad")])
def test_weight_argument_conversion_errors_are_rejected(
    load: object, beta: object
) -> None:
    with pytest.raises(ValueError):
        signed_context_weight(load, beta)


def test_invalid_capacity_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        weighted_layer_score([None], [1.0], [0.5])
    with pytest.raises(ValueError):
        weighted_layer_score([1.0], [float("nan")], [0.5])


def test_non_finite_total_is_rejected() -> None:
    with pytest.raises(ValueError):
        weighted_layer_score([1e308], [1e308], [0.0])


def test_legacy_alpha_conversion_errors_are_rejected() -> None:
    with pytest.raises(ValueError):
        residual_context_score([1.0], [1.0], [1.0], alpha="bad")
    with pytest.raises(ValueError):
        score_transformer([], alpha="bad")
    with pytest.raises(ValueError):
        score_transformer([], alpha=-1.0)
