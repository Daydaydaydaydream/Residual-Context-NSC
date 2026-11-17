# 最终集评估

2026-11-15，运行前锁定评分版本 `61402bca87760135f2b2407d3c6b27ddc30d85c7`。三组参数取自该版本的 docs/selected-settings.md。以下命令从仓库根目录执行。

```sh
mkdir -p results/final
NSC_FORM=signed NSC_SCOPE=ffn NSC_BETA=-100 .venv/bin/python course/nsc-starter/evaluate.py --split final --student scripts/student_score.py --output results/final/main.json
NSC_FORM=signed NSC_SCOPE=joint NSC_BETA=-1 .venv/bin/python course/nsc-starter/evaluate.py --split final --student scripts/student_score.py --output results/final/joint.json
NSC_FORM=exponential NSC_SCOPE=ffn NSC_BETA=-100 .venv/bin/python course/nsc-starter/evaluate.py --split final --student scripts/student_score.py --output results/final/exponential.json
```

## 最终集结果

以 100 个留出架构的负测试困惑度为目标，相关系数越高表示排序越准确。

| 方法 | Spearman ρ | Kendall τb | 相近规模配对 τb |
|---|---:|---:|---:|
| 原始 NSC | 0.965941 | 0.843232 | 0.329730 |
| 主方法（β = −100） | 0.973381 | 0.869899 | 0.372973 |
| 共同加权（β = −1） | 0.966205 | 0.844040 | 0.329730 |
| 指数形式（β = −100） | 0.973465 | 0.876364 | 0.383784 |

主方法相对原始 NSC 的排序提高，但 8 个预算设置中有 7 项选模表现持平。50M 预算选 1 个时，入选架构的测试困惑度从 27.6652 升至 28.8824，表现变差。各预算的候选数、困惑度及绝对遗憾见 [main.json](main.json)。

## 开发集补充实验

在最终评估后扩展开发集的主方法 β 扫描。−100 的 Spearman ρ 为 0.972817，−1000 和 −10000 分别回落至 0.971617 和 0.970189。−5、−10、−50、−100 的八项预算遗憾相同。[完整开发集结果](../beta-sweep/development-main-sensitivity.csv)
