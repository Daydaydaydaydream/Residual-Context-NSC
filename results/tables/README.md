# 开发集结果表（11/16 交付）

两张表由冻结后的三种设置重新跑出，只使用 development split；最终集未参与。

## 命令

```bash
python scripts/build_tables.py
```

本次运行总耗时 2.38 s，四次评测分别为：baseline 1.82 s、main 0.18 s、joint 0.19 s、exponential 0.19 s。

本机首次执行约 8.5 s（首次导入与字节码编译），此后稳定在 2.2 s 左右。

## 随机种子

预算选择用 starter 的固定调度 `seed = 202627 + budget + k`，200 次重复；重复只用于打散评分并列和构造均匀随机参照，不是训练种子标准差。`tables.json` 的 `meta.random_seed` 里同样记录。

## 环境

| item | value |
|---|---|
| python | 3.14.5 |
| platform | Windows-11-10.0.26200-SP0 |
| numpy | 2.5.4 |
| scipy | 1.18.1 |

## 文件

- `ranking.md`
- `budget.md`
- `tables.json`
- `README.md`
- `beta-curve.csv`
- `fig1-rho-vs-beta.png`

## 口径

- 排序表：Spearman rho 与 Kendall tau-b 均对负验证 PPL 计算；`score ties` 是评分并列的对数。
- 预算表：每档预算取可行池中分数最高的 k 个，报记录 PPL；遗憾为 `mean_best_ppl - oracle_ppl`，是绝对差值；第一行 `candidates in pool` 是该档预算的可行架构数。
- 200 次重复只用于打散评分并列和构造均匀随机参照，不是训练种子标准差。
- `beta-curve.csv` 是图 1（rho / tau-b 随 |beta| 变化）的绘图输入，取自 `results/beta-sweep/beta_sweep.csv`。
- 图 1 本身由 `python scripts/make_figures.py` 从 `beta-curve.csv` 画出，需要 matplotlib；输出 `fig1-rho-vs-beta.png`。
