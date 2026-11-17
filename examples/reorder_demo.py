"""Controlled width-reordering example from the course topic brief.

The two configurations have the same matrices and the same reported resources;
only the FFN-width order changes. Original NSC cannot tell them apart, while
the signed layer-order score and both controls can. There are no trained labels
for this pair: it checks mechanism sensitivity, not predictive validity.
"""

from cs5491_nsc import TransformerLayerSpec, original_nsc_score, score_transformer

if __name__ == "__main__":
    forward = [
        TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=256),
        TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=512),
    ]
    reversed_widths = [
        TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=512),
        TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=256),
    ]

    rows = [
        ("Original additive NSC (beta=0 ablation)", original_nsc_score),
        ("Signed beta=-1 (FFN only)", lambda x: score_transformer(x, beta=-1.0)),
        ("Signed beta=+1 (FFN only)", lambda x: score_transformer(x, beta=1.0)),
        (
            "Control: signed beta=+1 (joint)",
            lambda x: score_transformer(x, beta=1.0, scope="joint"),
        ),
        (
            "Control: exponential beta=-1 (FFN only)",
            lambda x: score_transformer(x, beta=-1.0, form="exponential"),
        ),
    ]
    print(f"{'score':42s} {'[256, 512]':>12s} {'[512, 256]':>12s}")
    for label, function in rows:
        print(
            f"{label:42s} {function(forward):12.6f} {function(reversed_widths):12.6f}"
        )
