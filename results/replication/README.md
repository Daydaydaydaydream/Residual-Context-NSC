# 独立复现记录（2026-11-17）

按根 `README.md` 的「复现开发集结果」在**干净环境**重跑，逐项核对报告中引用的数字。
执行人：张正；供冯璟哲定版参考。

## 代码版本

`dev` @ `9bc5dd7`（合并 PR #9 之后，含 §5.4 最终集验证）

## 环境

| 项 | 值 |
|---|---|
| OS | Windows 10.0.19045 |
| Python | 3.12.7（`py -3.12 -m venv .venv`，未复用 Anaconda base） |
| numpy / scipy | 2.5.5 / 1.18.1 |
| 依赖来源 | `pip install -e ".[dev]"` |
| 与 11/15 记录的差异 | 原记录为 Python 3.14.5 + numpy 2.5.4（Windows 11），本轮为 3.12.7 + 2.5.5 |

## 命令

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

cd course\nsc-starter
..\..\.venv\Scripts\python.exe check_starter.py
cd ..\..

.\.venv\Scripts\python.exe scripts\verify_baseline.py
.\.venv\Scripts\python.exe scripts\verify_adapter.py
.\.venv\Scripts\python.exe scripts\build_tables.py
```

## 结果

| 检查 | 期望 | 实测 |
|---|---|---|
| `check_starter.py` | PASS | PASS；玩具例 121.820169804（两种顺序相同）；位置示例 96.957958136 vs 85.772296570 |
| `verify_baseline.py` | 与 `expected_development.json` 0 差异 | baseline 0 处、adapter 0 处、两者互比 0 处 |
| `verify_adapter.py` | 200 配置上 β=0 等于原始 NSC | PASS（1e-10 容忍），最大相对差 3.100e-15 |
| `build_tables.py` | 与仓库内结果表一致 | `ranking.md`、`budget.md` **字节级一致**（无 diff） |

### 排序指标（对负验证 PPL，100 个开发架构）

| 方法 | 复现 ρ | 复现 τ_b | 复现配对 τ_b | 报告值（表 2） |
|---|---:|---:|---:|---:|
| 原始 NSC | 0.969744974 | 0.861414141 | 0.622222222 | 0.969745 / 0.861414 / 0.622222 |
| 主方法 β=−100 | 0.972817282 | 0.879595960 | 0.666666667 | 0.972817 / 0.879596 / 0.666667 |
| 共同加权 β=−1 | 0.970153015 | 0.863434343 | 0.633333333 | 0.970153 / 0.863434 / 0.633333 |
| 指数形式 β=−100 | 0.972241224 | 0.881616162 | 0.600000000 | 0.972241 / 0.881616 / 0.600000 |

预算表 8 组（10M/20M/30M/50M × k=1,3）的遗憾值未发生变化，`budget.md` 无差异。
受控配对 180 对、可行架构数 24/65/83/96 均与报告一致。

### 耗时

| 项 | 复现 | 11/15 记录 |
|---|---:|---:|
| 总耗时 | 2.63 s | 2.38 s |
| baseline | 1.85 s | 1.82 s |
| main / joint / exponential | 0.27 / 0.25 / 0.26 s | 0.18 / 0.19 / 0.19 s |

同量级，差异来自 Python 版本与机器；不影响数值结论。

## 差异说明

重跑只改写了 `tables.json` 与 `results/baseline/*.json` 里的 `meta.environment` 和 `timings_seconds` 字段，
其余字段与两张 Markdown 表完全一致。这些环境/耗时字段已在本地还原，未提交。

## 结论

**复现通过。** 报告表 2 的 8 行、表 3 的 8 组数字、以及机制检查引用的玩具例数值，均可在干净的
Python 3.12 环境中重现；`verify_baseline` 的 0 差异说明评分适配层没有偏离课程基线，
`verify_adapter` 的 3.1e-15 说明 β=0 与原始 NSC 在 200 个配置上逐位一致。
