<!--
Canonical proposal source; replaces ProjectProposal-cn-v0.md.
Submit one PDF per group on Canvas by 2026-10-15 23:59 Beijing time.
A4; Times New Roman 12 pt; 2.5 cm margins; single spacing.
Do not reduce font size or margins to fit. No abstract or reference list.
Course: CS5491, 2026/27 Semester A. Verify registered English names before submission.
-->

# Signed FFN Layer-Order Context for Neural Spectral Capacity

Jingzhe Feng (72655564), Jin Xing (72655700), Zheng Zhang (72655392)
Contact: 72655564@cityu-dg.edu.cn

## Research question

Zhu, Zhao and Lu's additive Neural Spectral Capacity (NSC) cannot distinguish FFN-width reorderings at fixed hidden width. Within Section 3's core scope, we test whether preceding FFN context improves ranking and budgeted selection, and whether increasing or decreasing weights work better.

## Method and mechanism checks

Keep starter attention/FFN capacities $A_l,F_l$, matrix grouping and $s=0.02$ fixed. With hidden/FFN widths $d_l,f_l$, define our specification-only heuristic:

$$
q_l=\tfrac12 d_l f_l s^4,\quad Q_l=\sum_{j<l}q_j,\quad w(Q;\beta)=1+\frac{\beta Q}{1+|\beta|Q},
$$

$$
S(\beta)=\sum_l[A_l+w(Q_l;\beta)F_l].
$$

For $Q\ge0$, $\beta<0$ recovers reciprocal damping; $\beta>0$ gives increasing weights; $\beta=0$ recovers NSC (ablation). Weights satisfy $0<w<2$ and $w(0;\beta)=1$. The proxy $q_l$ is motivated by the second-moment ratio of a bias-free, unnormalized ReLU branch with independent Gaussian weights and isotropic inputs. In this Post-LN archive, $Q_l$ is structural context, not accumulated activation variance. Check unlabelled reorderings, zero and limiting cases; sensitivity alone does not establish prediction. Joint attention/FFN weighting and exponential damping test weighting scope and form.

## Data and evaluation

Use 200 archived Transformers from Javaheripi et al.'s LiteTransformerSearch: 100 development and 100 final architectures. Select $\beta\in\{0,\pm0.1,\pm1,\pm10,\pm100,\pm1000\}$ by development Spearman correlation with negative validation perplexity (PPL); break ties by smaller $|\beta|$, then negative sign. Report both signs' sensitivity. Freeze choices and code before final test-PPL evaluation. Baselines: NSC/ablation, positional example, reported total parameters, reference FLOPs and uniform random selection.

| Evaluation | Protocol and safeguards |
|---|---|
| Ranking | Spearman $\rho$, Kendall $\tau_b$ against negative PPL, and ties; same-hidden-width pair comparisons with total-parameter ratio min/max ≥ 0.90 and pair counts. Bootstrap architectures, rebuilding overlapping pairs. |
| Budgeted selection | Total-parameter caps: 10M/20M/30M/50M; $k=1,3$, identical pools. Report pool sizes, best selected PPL and regret (selected minimum minus feasible minimum); 200 fixed-seed selection/tie-breaking trials. |
| Embedding effects | NSC excludes embeddings. Compare identical embedding settings; report group sizes, with descriptive summaries for small groups. Retain all main budgets. |

## Compute and deliverables

Laptop CPU, Python 3.9+; no new training, GPU or LLM API. Reproduce the starter, implement/check the score and ablation, then develop/freeze and evaluate. Deliver reproducible code, configurations, commands, commit/tag, scoring cost and a report. One outcome per architecture precludes training-seed uncertainty estimates; this heterogeneous panel cannot establish causal FFN-order effects. Negative results remain informative.
