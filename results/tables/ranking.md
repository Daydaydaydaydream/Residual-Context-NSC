# 排序对比（对负验证 PPL）

数据：`course/nsc-starter/data/`，development split 100 个架构；选择使用 starter 固定种子 `202627 + budget + k`，200 次重复。
设置取自 `docs/selected-settings.md`：main -100、joint -1、exponential -100。

| method | Spearman rho | Kendall tau-b (all pairs) | tau-b (180 size-controlled pairs) | score ties | delta rho vs original NSC |
|---|---:|---:|---:|---:|---:|
| original NSC | 0.969745 | 0.861414 | 0.622222 | 0 | -- |
| ablation (beta=0) | 0.969745 | 0.861414 | 0.622222 | 0 | +0.000000 |
| position example | 0.970069 | 0.865051 | 0.622222 | 0 | +0.000324 |
| main beta=-100 | 0.972817 | 0.879596 | 0.666667 | 0 | +0.003072 |
| joint beta=-1 | 0.970153 | 0.863434 | 0.633333 | 0 | +0.000408 |
| exponential beta=-100 | 0.972241 | 0.881616 | 0.600000 | 0 | +0.002496 |
| reported parameters | 0.727873 | 0.575758 | 0.133333 | 0 | -0.241872 |
| reference FLOPs | 0.967948 | 0.858471 | 0.622222 | 1 | -0.001797 |
