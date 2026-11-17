# 选中配置与评估命令

## 状态与版本

2026-11-12（北京时间）整理，用于实验交接与正式冻结准备。三种设置已通过独立开发集命令验证；组内联调后再确定正式冻结 commit/tag。

- 已验证代码版本：`9652a2f6191f557d7e6415a1fe5b37c02001963b`。
- 选参依据：[`beta_sweep.json`](../results/beta-sweep/beta_sweep.json)，100 个开发架构，三种设置各 11 个 β。
- 选择规则：与负验证 PPL 的 Spearman ρ 最大；并列时先取较小 |β|，再优先负号。
- 数据：`course/nsc-starter/data/`；本页命令仅使用 `development` 标签。
- 评分入口：`scripts/student_score.py`；初始化标准差固定为 0.02；`ablation` 恒为 β=0。
- 本机验证环境：conda `cs5491-nsc`，Python 3.12.16，NumPy 2.5.4，SciPy 1.18.1，macOS arm64。

## 配置清单

| 设置 / 输出名 | NSC_FORM | NSC_SCOPE | NSC_BETA | 预期 student Spearman ρ |
|---|---|---|---:|---:|
| 主方法 / main | signed | ffn | -100 | 0.972817282 |
| 共同加权 / joint | signed | joint | -1 | 0.970153015 |
| 指数形式 / exponential | exponential | ffn | -100 | 0.972241224 |

每次运行显式设置全部三个环境变量；评分接口默认 β=1，不能用于复现本表的选中方法。

## 环境准备

在仓库根目录运行。已有环境直接激活；新机器可先创建并安装依赖：

```bash
conda create -n cs5491-nsc python=3.12 pip -y
conda activate cs5491-nsc
python -m pip install -e ".[dev]"
```

## macOS / Linux 命令

以下所有命令均从仓库根目录运行。输出放在被 Git 忽略的 `work/selected-settings/`，用于联调；正式实验归档由实验负责人整理到 `results/` 并附版本记录。

```bash
conda activate cs5491-nsc
mkdir -p work/selected-settings

NSC_FORM=signed NSC_SCOPE=ffn NSC_BETA=-100 python course/nsc-starter/evaluate.py --split development --student scripts/student_score.py --output work/selected-settings/main.json

NSC_FORM=signed NSC_SCOPE=joint NSC_BETA=-1 python course/nsc-starter/evaluate.py --split development --student scripts/student_score.py --output work/selected-settings/joint.json

NSC_FORM=exponential NSC_SCOPE=ffn NSC_BETA=-100 python course/nsc-starter/evaluate.py --split development --student scripts/student_score.py --output work/selected-settings/exponential.json
```

## Windows PowerShell 命令

从仓库根目录运行；每组均覆盖三个设置，避免沿用上一组参数。

```powershell
conda activate cs5491-nsc
New-Item -ItemType Directory -Force work/selected-settings | Out-Null

$env:NSC_FORM = "signed"
$env:NSC_SCOPE = "ffn"
$env:NSC_BETA = "-100"
python course/nsc-starter/evaluate.py --split development --student scripts/student_score.py --output work/selected-settings/main.json

$env:NSC_FORM = "signed"
$env:NSC_SCOPE = "joint"
$env:NSC_BETA = "-1"
python course/nsc-starter/evaluate.py --split development --student scripts/student_score.py --output work/selected-settings/joint.json

$env:NSC_FORM = "exponential"
$env:NSC_SCOPE = "ffn"
$env:NSC_BETA = "-100"
python course/nsc-starter/evaluate.py --split development --student scripts/student_score.py --output work/selected-settings/exponential.json

Remove-Item Env:NSC_FORM, Env:NSC_SCOPE, Env:NSC_BETA
```

PowerShell 写法供 Windows 交接使用，本次实测平台为 macOS。

## 输出验收与指标口径

每个 JSON 应包含 `split=development`、`label=valid_ppl`、`architectures=100`，以及 8 个 `selection` 条目（10M/20M/30M/50M × k=1/3）。

- `ranking.student`：本次选中方法的排序结果，应与上表一致（表中数值保留九位小数）。
- `selection[].methods.student`：本次选中方法的预算选择结果。
- 同一输出中的 `original_nsc`、`ablation`、`position_example`、`reported_parameters`、`reference_flops` 和 `random`：比较基线。
- `mean_best_ppl`：选择 k 个架构后最佳验证 PPL 的重复试验均值，越低越好。
- `oracle_ppl`：当前预算候选池中的最低验证 PPL。
- **starter 的 `mean_regret` 是绝对差值**：`mean_best_ppl - oracle_ppl`。若报告相对遗憾，另计算 `mean_regret / oracle_ppl`；百分比再乘 100%，不要把原字段直接标为相对遗憾。
- `sd_best_ppl`：重复选择的波动。starter 使用 200 次重复；评分方法重复用于打散并列，随机基线用于均匀随机选择。它不是训练种子的标准差。

2026-11-12 本机实测：三条命令均成功；三组 Spearman 与扫描记录一致；每组均生成完整 8 个预算/k 组合。

## 交接与正式冻结

1. Jingzhe Feng 提供本页配置、命令及已验证代码版本。
2. Jin Xing 按本页完成实验联调，归档预算结果并整理对比表。
3. 联调通过后，将方法、配置及文档纳入正式冻结版本，记录完整 commit/tag。若联调引起评分修改，重新验证并更新版本记录。
4. 正式冻结后再安排最终集评估；最终标签不参与重新选参。
