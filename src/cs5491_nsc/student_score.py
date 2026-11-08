"""Course-project scores built on the upstream NSC MP utility.

The course starter groups each layer's scored matrices as

    A_l = 3 H_l psi_MP(d, d / H_l, s) + psi_MP(d, d, s)
    F_l = 2 psi_MP(d, f_l, s)
    q_l = 0.5 d f_l s^4

with the fixed initialization convention s = 0.02. This module keeps that
grouping and adds a signed layer-order context weight. With

    Q_l = sum_{j < l} q_j
    w(Q; beta) = 1 + beta Q / (1 + |beta| Q)
    S(beta) = sum_l [A_l + w(Q_l; beta) F_l],

beta = 0 is the original additive NSC (the ablation), beta < 0 damps later FFN
capacity, and beta > 0 amplifies it. For Q >= 0 the weight is bounded by
0 < w < 2, satisfies w(0; beta) = 1, and therefore leaves the first layer
untouched. The limits for a layer with Q > 0 are w -> 0 as beta -> -inf and
w -> 2 as beta -> +inf.

Two controls isolate the weighting scope and the functional form:

* scope="joint" applies the same weight to A_l + F_l instead of only F_l
  (joint attention/FFN weighting).
* form="exponential" uses a bounded signed exponential: exp(-|beta| Q) for
  beta < 0 and its reflection 2 - exp(-|beta| Q) for beta > 0. It has the same
  first-order behaviour and (0, 2) range as the main form, but a different tail.

The load proxy is a structural feature. It is motivated by the second-moment
ratio of a bias-free, unnormalized ReLU branch, but it is not accumulated
activation variance in the archived Post-LN models.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .nsc_utils import psi_mp

#: Fixed initialization standard deviation used by the course starter.
COURSE_INIT_STD = 0.02

#: Provisional main-setting value; freeze the development-selected value later.
DEFAULT_BETA = 1.0

#: Signed beta values declared in the project plan.
#: Ordered by |beta|, with the negative sign first to match the tie rule.
BETA_CANDIDATES = (
    0.0,
    -0.1,
    0.1,
    -1.0,
    1.0,
    -10.0,
    10.0,
    -100.0,
    100.0,
    -1000.0,
    1000.0,
)

WEIGHT_FORMS = ("signed", "exponential")
WEIGHT_SCOPES = ("ffn", "joint")

_UNSET = object()


@dataclass(frozen=True)
class TransformerLayerSpec:
    """The dimensions needed by the course's dense decoder score."""

    hidden_width: int
    heads: int
    ffn_width: int


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _per_layer(value: Any, n_layer: int, name: str) -> list[int]:
    """Expand an int or validate a per-layer sequence into length n_layer."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an int or a list of positive ints")
    if isinstance(value, int):
        _validate_positive_int(value, name)
        return [value] * n_layer
    if isinstance(value, (list, tuple)):
        if len(value) != n_layer:
            raise ValueError(f"{name} must have length n_layer={n_layer}")
        return [_validate_positive_int(item, name) for item in value]
    raise ValueError(f"{name} must be an int or a list of positive ints")


def layer_specs_from_config(config: Mapping[str, Any]) -> list[TransformerLayerSpec]:
    """Convert one archived starter configuration into layer specifications.

    This mirrors the starter's nsc.layers schema checks: d_model and n_layer are
    positive integers, n_head and d_inner may be an int or a per-layer sequence
    of length n_layer, and d_model must be divisible by each head count.
    """
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    try:
        hidden = _validate_positive_int(config["d_model"], "d_model")
        n_layer = _validate_positive_int(config["n_layer"], "n_layer")
        heads = _per_layer(config["n_head"], n_layer, "n_head")
        inner = _per_layer(config["d_inner"], n_layer, "d_inner")
    except KeyError as exc:
        raise ValueError(f"missing config field: {exc.args[0]}") from exc

    for head_count in heads:
        if hidden % head_count:
            raise ValueError("d_model must be divisible by each head count")
    return [
        TransformerLayerSpec(hidden_width=hidden, heads=head_count, ffn_width=ffn)
        for head_count, ffn in zip(heads, inner)
    ]


def _validate_init_std(init_std: float) -> float:
    try:
        value = float(init_std)
    except (TypeError, ValueError) as exc:
        raise ValueError("init_std must be finite and non-negative") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("init_std must be finite and non-negative")
    return value


def _validate_beta(beta: float) -> float:
    try:
        value = float(beta)
    except (TypeError, ValueError) as exc:
        raise ValueError("beta must be finite") from exc
    if not math.isfinite(value):
        raise ValueError("beta must be finite")
    return value


def _validate_load(prior_load: float) -> float:
    try:
        value = float(prior_load)
    except (TypeError, ValueError) as exc:
        raise ValueError("prior load must be finite and non-negative") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("prior load must be finite and non-negative")
    return value


def _validate_form(form: str) -> str:
    if form not in WEIGHT_FORMS:
        raise ValueError(f"form must be one of {WEIGHT_FORMS}")
    return form


def _validate_scope(scope: str) -> str:
    if scope not in WEIGHT_SCOPES:
        raise ValueError(f"scope must be one of {WEIGHT_SCOPES}")
    return scope


def ffn_residual_load(
    hidden_width: int, ffn_width: int, init_std: float = COURSE_INIT_STD
) -> float:
    """Estimate the output/input variance ratio for W2 ReLU(W1 x).

    Assumes independent Gaussian matrices, isotropic inputs, and no normalization
    or learned scaling. This proxy is a project hypothesis, not part of original NSC.
    """
    _validate_positive_int(hidden_width, "hidden_width")
    _validate_positive_int(ffn_width, "ffn_width")
    std = _validate_init_std(init_std)
    return 0.5 * hidden_width * ffn_width * std**4


def layer_capacities(
    layer: TransformerLayerSpec, init_std: float = COURSE_INIT_STD
) -> tuple[float, float, float]:
    """Return (attention capacity, FFN capacity, FFN residual load)."""
    d, heads, ffn = layer.hidden_width, layer.heads, layer.ffn_width
    _validate_positive_int(d, "hidden_width")
    _validate_positive_int(heads, "heads")
    _validate_positive_int(ffn, "ffn_width")
    if d % heads:
        raise ValueError("hidden_width must be divisible by heads")
    std = _validate_init_std(init_std)

    head_width = d // heads
    attention = 3 * heads * psi_mp(d, head_width, std) + psi_mp(d, d, std)
    ffn_capacity = 2 * psi_mp(d, ffn, std)
    load = 0.5 * d * ffn * std**4
    return attention, ffn_capacity, load


def signed_context_weight(prior_load: float, beta: float) -> float:
    """Return the main signed weight 1 + beta Q / (1 + |beta| Q).

    beta = 0 and Q = 0 both give 1. For Q >= 0 the result lies in (0, 2):
    negative beta damps (reciprocal form) and positive beta amplifies.
    """
    load = _validate_load(prior_load)
    beta = _validate_beta(beta)
    if beta == 0.0 or load == 0.0:
        return 1.0

    scaled = abs(beta) * load
    if beta < 0.0:
        # 1 - x/(1+x) written directly to avoid cancellation for small x.
        return 1.0 / (1.0 + scaled) if math.isfinite(scaled) else 0.0
    fraction = scaled / (1.0 + scaled) if math.isfinite(scaled) else 1.0
    return 1.0 + fraction


def exponential_context_weight(prior_load: float, beta: float) -> float:
    """Return the bounded signed exponential control weight.

    For beta < 0 this is exp(-|beta| Q) (exponential damping). For beta > 0 its
    reflection 2 - exp(-|beta| Q) keeps the same (0, 2) range and limits as the
    main form while remaining finite for every beta.
    """
    load = _validate_load(prior_load)
    beta = _validate_beta(beta)
    if beta == 0.0 or load == 0.0:
        return 1.0

    scaled = abs(beta) * load
    if beta < 0.0:
        return math.exp(-scaled) if math.isfinite(scaled) else 0.0
    # 2 - exp(-x) = 1 - expm1(-x), which is stable for small x.
    return 1.0 - math.expm1(-scaled)


def _context_weight(prior_load: float, beta: float, form: str) -> float:
    if form == "signed":
        return signed_context_weight(prior_load, beta)
    return exponential_context_weight(prior_load, beta)


def layer_order_weights(
    ffn_residual_loads: Sequence[float],
    beta: float = DEFAULT_BETA,
    form: str = "signed",
) -> list[float]:
    """Return the weight applied to each FFN capacity.

    The first layer always receives weight 1 because its prior load is zero.
    """
    _validate_form(form)
    _validate_beta(beta)
    weights = []
    prior_load = 0.0
    for load in ffn_residual_loads:
        weights.append(_context_weight(prior_load, beta, form))
        prior_load += _validate_load(load)
    return weights


def weighted_layer_score(
    attention_capacities: Sequence[float],
    ffn_capacities: Sequence[float],
    ffn_residual_loads: Sequence[float],
    beta: float = DEFAULT_BETA,
    form: str = "signed",
    scope: str = "ffn",
) -> float:
    """Score layers with a signed weight based on preceding FFN residual load.

    scope="ffn" computes sum_l [A_l + w(Q_l) F_l]. scope="joint" applies the
    same weight to both terms: sum_l w(Q_l) (A_l + F_l). beta = 0 reduces
    either scope to the additive NSC sum.
    """
    n = len(attention_capacities)
    if len(ffn_capacities) != n or len(ffn_residual_loads) != n:
        raise ValueError("all per-layer sequences must have the same length")
    _validate_form(form)
    _validate_scope(scope)
    _validate_beta(beta)

    total = 0.0
    prior_load = 0.0
    for attention, ffn, load in zip(
        attention_capacities, ffn_capacities, ffn_residual_loads
    ):
        try:
            attention = float(attention)
            ffn = float(ffn)
            load = float(load)
        except (TypeError, ValueError) as exc:
            raise ValueError("capacities and residual loads must be finite") from exc
        if not all(math.isfinite(value) for value in (attention, ffn, load)):
            raise ValueError("capacities and residual loads must be finite")
        if attention < 0 or ffn < 0 or load < 0:
            raise ValueError("capacities and residual loads must be non-negative")

        weight = _context_weight(prior_load, beta, form)
        if scope == "ffn":
            total += attention + weight * ffn
        else:
            total += weight * (attention + ffn)
        prior_load += load

    if not math.isfinite(total):
        raise ValueError("score must be finite")
    return total


def signed_ffn_score(
    attention_capacities: Sequence[float],
    ffn_capacities: Sequence[float],
    ffn_residual_loads: Sequence[float],
    beta: float = DEFAULT_BETA,
) -> float:
    """Main form: signed FFN-only layer-order score."""
    return weighted_layer_score(
        attention_capacities,
        ffn_capacities,
        ffn_residual_loads,
        beta=beta,
        form="signed",
        scope="ffn",
    )


def joint_weighted_score(
    attention_capacities: Sequence[float],
    ffn_capacities: Sequence[float],
    ffn_residual_loads: Sequence[float],
    beta: float = DEFAULT_BETA,
) -> float:
    """Control: signed weight applied jointly to attention and FFN capacity."""
    return weighted_layer_score(
        attention_capacities,
        ffn_capacities,
        ffn_residual_loads,
        beta=beta,
        form="signed",
        scope="joint",
    )


def exponential_decay_score(
    attention_capacities: Sequence[float],
    ffn_capacities: Sequence[float],
    ffn_residual_loads: Sequence[float],
    beta: float = DEFAULT_BETA,
) -> float:
    """Control: bounded signed exponential weight on FFN capacity."""
    return weighted_layer_score(
        attention_capacities,
        ffn_capacities,
        ffn_residual_loads,
        beta=beta,
        form="exponential",
        scope="ffn",
    )


def residual_context_score(
    attention_capacities: Sequence[float],
    ffn_capacities: Sequence[float],
    ffn_residual_loads: Sequence[float],
    alpha: float = 1.0,
) -> float:
    """Legacy reciprocal-damping score with non-negative alpha.

    This is the earlier project prototype. It is exactly the signed main form
    at beta = -alpha: F_l / (1 + alpha Q_l). New code should use
    weighted_layer_score or signed_ffn_score.
    """
    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be finite and non-negative") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("alpha must be finite and non-negative")
    return weighted_layer_score(
        attention_capacities,
        ffn_capacities,
        ffn_residual_loads,
        beta=-value,
        form="signed",
        scope="ffn",
    )


def additive_ablation(
    attention_capacities: Sequence[float],
    ffn_capacities: Sequence[float],
    ffn_residual_loads: Sequence[float],
) -> float:
    """Return the additive NSC baseline over the selected matrices."""
    return weighted_layer_score(
        attention_capacities,
        ffn_capacities,
        ffn_residual_loads,
        beta=0.0,
        form="signed",
        scope="ffn",
    )


def score_transformer(
    layers: Sequence[TransformerLayerSpec],
    init_std: float = COURSE_INIT_STD,
    beta: float | object = _UNSET,
    *,
    alpha: float | None = None,
    form: str = "signed",
    scope: str = "ffn",
) -> float:
    """Compute a layer-order score from a sequence of layer specs.

    beta is the signed context parameter. alpha is accepted only for backward
    compatibility with the earlier non-negative reciprocal prototype and maps
    to beta = -alpha; do not pass both.
    """
    if alpha is not None:
        if beta is not _UNSET:
            raise ValueError("pass only one of beta and alpha")
        try:
            value = float(alpha)
        except (TypeError, ValueError) as exc:
            raise ValueError("alpha must be finite and non-negative") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError("alpha must be finite and non-negative")
        beta = -value
    elif beta is _UNSET:
        beta = DEFAULT_BETA

    capacities = [layer_capacities(layer, init_std) for layer in layers]
    attention, ffn, loads = zip(*capacities) if capacities else ((), (), ())
    return weighted_layer_score(
        attention,
        ffn,
        loads,
        beta=beta,
        form=form,
        scope=scope,
    )


def original_nsc_score(
    layers: Sequence[TransformerLayerSpec], init_std: float = COURSE_INIT_STD
) -> float:
    """Compute the course's original additive NSC over selected projections."""
    capacities = [layer_capacities(layer, init_std) for layer in layers]
    attention, ffn, loads = zip(*capacities) if capacities else ((), (), ())
    return weighted_layer_score(
        attention,
        ffn,
        loads,
        beta=0.0,
        form="signed",
        scope="ffn",
    )


def score_config(
    config: Mapping[str, Any],
    beta: float = DEFAULT_BETA,
    init_std: float = COURSE_INIT_STD,
    form: str = "signed",
    scope: str = "ffn",
) -> float:
    """Score one starter configuration dictionary with the signed context form."""
    return score_transformer(
        layer_specs_from_config(config),
        init_std=init_std,
        beta=beta,
        form=form,
        scope=scope,
    )


def ablation_config(
    config: Mapping[str, Any], init_std: float = COURSE_INIT_STD
) -> float:
    """Return the beta=0 additive ablation for one starter configuration."""
    return original_nsc_score(layer_specs_from_config(config), init_std=init_std)


def score(
    config: Mapping[str, Any],
    beta: float = DEFAULT_BETA,
    init_std: float = COURSE_INIT_STD,
    form: str = "signed",
    scope: str = "ffn",
) -> float:
    """Convenience alias for :func:`score_config` with a fixed call signature."""
    return score_config(config, beta=beta, init_std=init_std, form=form, scope=scope)


def ablation(config: Mapping[str, Any], init_std: float = COURSE_INIT_STD) -> float:
    """Convenience alias for :func:`ablation_config`."""
    return ablation_config(config, init_std=init_std)
