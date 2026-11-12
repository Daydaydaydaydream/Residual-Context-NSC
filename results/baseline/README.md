# 开发集基线复现（2026-11-12）

本目录记录课程 starter 在 development split 上的基线复现，用于在任何后续比较之前
确认评分适配层没有偏离课程基线。

- 代码版本：`dev` @ `8a59461`（`Implement signed layer-order scoring and ablations`）
- 数据：`course/nsc-starter/data/`，development split 100 个架构
- 评分口径：starter 原模板，以及项目适配层 `NSC_BETA=0`
- 随机性：starter 固定种子（`202627 + budget + k`），200 次重复只用于打散评分并列

## 环境

| 项 | 值 |
|---|---|
| OS | Microsoft Windows 11 家庭版 中文版 10.0.26200 |
| CPU | 12th Gen Intel Core i7-12700H |
| Python | 3.14.5（仓库内 `.venv`，AMD64） |
| numpy / scipy | 2.5.4 / 1.18.1 |

## 命令

从仓库根目录运行，脚本会写出本目录下的两个 JSON：

```bash
python scripts/verify_baseline.py
```

它内部等价于下面两步（Windows 下环境变量写法为 `$env:NSC_BETA=0`）：

```bash
cd course/nsc-starter
python evaluate.py --split development --output ../../results/baseline/development_baseline.json
NSC_BETA=0 python evaluate.py --split development \
    --student ../../scripts/student_score.py \
    --output ../../results/baseline/development_adapter_beta0.json
```

## 结果

| 方法 | Spearman ρ | Kendall τb（全部对） | τb（180 受控配对） |
|---|---:|---:|---:|
| original_nsc | 0.969745 | 0.861414 | 0.622222 |
| position_example | 0.970069 | 0.865051 | 0.622222 |
| student（适配层 β=0） | 0.969745 | 0.861414 | 0.622222 |
| ablation | 0.969745 | 0.861414 | 0.622222 |
| reported_parameters | 0.727873 | 0.575758 | 0.133333 |
| reference_flops | 0.967948 | 0.858471 | 0.622222 |

- 受控配对 180 对，预算/k 组合 8 组（10M/20M/30M/50M × k=1,3）。
- 耗时：原模板口径 0.50 s，适配层口径 1.71 s（含首次 import），单机 CPU，非冷启动基准。

## 对照核验

`python scripts/verify_baseline.py` 以 1e-12 相对容忍对比 `expected_development.json`
（只忽略参考输出本身不包含的计时字段）：

- 原模板口径：0 处差异。
- 适配层 β=0 口径：0 处差异。
- 两个口径互比：0 处差异。

容忍范围内唯一的非严格相等是预算表里随机基线的 `sd_best_ppl`：
11.825936547801545 对 11.825936547801547，差 1 个 ULP（浮点求和顺序不同所致）。

`scripts/verify_adapter.py` 另行在 200 个配置上核对 `ablation` 与三种设置下的 β=0
都等于 starter 原始 NSC，最大相对差 3.1e-15。

## 环境说明

`core.autocrlf=true` 的机器上，`course/nsc-starter/data/*.json` 会被检出为 CRLF，
导致 `check_starter.py` 里的 SHA-256 断言失败。仓库根的 `.gitattributes` 已把该目录
固定为 `text eol=lf`；若本地仍是旧检出，重新检出一次即可：

```bash
git rm --cached -r --quiet course/nsc-starter/data
git checkout -- course/nsc-starter/data
```

## 文件

- `development_baseline.json`：starter 原模板口径的完整输出
- `development_adapter_beta0.json`：项目适配层 `NSC_BETA=0` 口径的完整输出
