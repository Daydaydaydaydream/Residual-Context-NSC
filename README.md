# Residual-Context NSC

**面向 Transformer 架构排序的残差上下文神经谱容量研究**

CS5491 课程组项目：研究 Neural Spectral Capacity（NSC）能否通过引入 Transformer 残差路径中的层序信息，改进架构排序与固定资源预算下的架构选择。

本仓库以 Neural Spectral Capacity（NSC）为方法基线，核心实现及对应 MIT 许可证已随项目提供。课程 starter 数据与评测程序归档在 `course/nsc-starter/`。

本项目是 CS5491 课程研究工程，不是上游官方仓库。项目名为 **Residual-Context NSC**，发行包名为 `residual-context-nsc`；Python 导入路径继续使用 `cs5491_nsc`，兼容已有示例和实验脚本。

## 研究问题

标准 NSC 根据架构规格计算每个权重矩阵的 MP 谱容量，再对选定矩阵求和。由于求和不依赖层顺序，具有相同层配置但 FFN 宽度分配顺序不同的模型会得到相同分数。本项目研究一个残差上下文启发式：用前序 FFN 分支的估计残差负载，调整后续 FFN 容量的贡献。

我们分别检查两个问题：

1. 评分能否对受控的层顺序变化作出响应？
2. 这种敏感性是否改善课程归档架构上的排序或预算选择？

受控示例中的分数变化只说明评分对顺序敏感，不等同于预测性能提升。

## 方法

上游方法对矩阵容量采用

\[
\psi_{MP}(m,n,s)=N\int_{\lambda_-}^{\lambda_+}\log(1+Ms^2\lambda)f_{MP}(\lambda;\gamma)\,d\lambda,
\quad N=\min(m,n),\ M=\max(m,n),\ \gamma=N/M.
\]

课程基线按每层注意力头和 FFN 投影分组，固定初始化标准差 `s = 0.02`：

```text
A_l = 3 H_l psi_MP(d, d / H_l, s) + psi_MP(d, d, s)
F_l = 2 psi_MP(d, f_l, s)
```

项目评分加入 FFN 负载代理 `q_l = d f_l s^4 / 2`：

```text
S(alpha) = sum_l [A_l + F_l / (1 + alpha * sum_{j<l} q_j)]
```

`alpha = 0` 是加性消融，对应课程基线的原始 NSC 求和。负载代理假设两层独立高斯线性映射、中间 ReLU、各向同性输入，并忽略归一化和可学习缩放；它是项目中的待检验启发式，不是上游论文提出或证明的公式。

实验中将纳入 harmonic/min 聚合作为对照，以区分一般非加性聚合与**权重由前序残差负载决定的层序依赖**。

## 项目结构

```text
src/cs5491_nsc/          可安装的项目包：MP 工具与课程评分实现
tests/                   自动化测试（pytest）
examples/                受控 FFN 宽度换序示例
scripts/                 立项书 PDF 生成脚本及后续研究脚本
course/nsc-starter/      课程 starter 数据和原始评测程序（逐字保留，不参与格式化）
docs/                    课程主题、项目计划、立项书源文件及唯一正式 PDF
docs/archive/            旧版立项书与历史校验产物压缩归档
```

## 环境与运行

需要 Python 3.9 或更高版本（开发环境使用 3.12）。核心代码只依赖 NumPy 和 SciPy，无需 GPU 或 LLM API。

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

也可以使用 Makefile（`PYTHON` 默认为 `python`）：

```bash
make install
make check      # lint + 格式检查 + 测试，与 CI 一致
make demo       # 受控 FFN 换序示例
make starter    # 课程 starter 自检
```

### 受控换序示例

```bash
python -m examples.reorder_demo
```

预期输出（原始 NSC 对两种顺序完全相同，残差上下文评分则不同）：

```text
Original additive NSC:
  [256, 512]: 121.820170
  [512, 256]: 121.820170
Residual-context score (alpha=1):
  [256, 512]: 121.697831
  [512, 256]: 121.692809
```

### 课程 starter

starter 自带评测程序，只依赖 Python 标准库。运行自检与开发集基线：

```bash
cd course/nsc-starter
python3 check_starter.py
python3 evaluate.py --split development --output development_results.json
```

### 上游 NSC 通用接口

```python
from cs5491_nsc import psi_mp, xavier_sigma

sigma = xavier_sigma(512, 2048)
ffn_capacity = 2 * psi_mp(2048, 512, sigma)
```

课程比较固定使用 `s = 0.02`，评分接口已据此提供：

```python
from cs5491_nsc import TransformerLayerSpec, original_nsc_score, score_transformer

layers = [
    TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=256),
    TransformerLayerSpec(hidden_width=128, heads=4, ffn_width=512),
]
baseline = original_nsc_score(layers)
proposed = score_transformer(layers, alpha=1.0)
```

## 开发规范

- `make lint` 与 `make format` 使用 Ruff。`course/nsc-starter/` 为逐字归档的课程材料，已在配置中排除，不参与检查与格式化。
- 提交前安装 pre-commit 钩子：`python -m pip install pre-commit && pre-commit install`。
- `src/cs5491_nsc/nsc_utils.py` 为上游 vendored 实现，修改前先确认是否应在上游仓库进行。
- CI（`.github/workflows/ci.yml`）在 Python 3.10 与 3.12 上运行 lint、测试与 starter 自检。

## Git 协作规范

当前以 `dev` 为日常开发的集成分支，采用「一个任务一个分支」，不按成员维护长期个人分支。

### 分支用途与命名

| 分支 | 用途 |
|---|---|
| `main` | 保存经过验证的稳定版本与课程里程碑版本，由 `dev` 通过 PR 合并 |
| `dev` | 汇总日常开发成果；任务分支从这里创建，完成后通过 PR 合并回来 |
| `feat/<用户名>-<任务>` | 新功能，例如 `feat/zhang-evaluation-adapter` |
| `fix/<用户名>-<问题>` | 修复问题，例如 `fix/li-score-validation` |
| `docs/<用户名>-<内容>` | 文档更新，例如 `docs/wang-experiment-guide` |
| `chore/<用户名>-<任务>` | 依赖、工具或配置维护，例如 `chore/chen-ci-config` |

用户名使用 GitHub 用户名或姓名拼音，任务描述使用简短英文、小写字母和连字符。开始前先约定任务范围及主要修改文件，避免重复开发。

### 开始开发

先确认工作区没有未提交的改动，再同步 `dev` 并创建任务分支。以下示例中的用户名和任务名需替换为自己的：

```bash
git switch dev
git pull --ff-only origin dev
git switch -c feat/zhang-evaluation-adapter
```

首次克隆后，如果本地尚无 `dev` 分支，先运行 `git fetch origin`，再用 `git switch --track origin/dev` 创建本地跟踪分支。

### 提交与发起 PR

在已激活项目虚拟环境的终端中运行检查，再检查差异、按文件选择要提交的改动：

```bash
make check starter demo
git status --short
git diff
# 示例：替换为本次任务实际修改的文件
git add src/cs5491_nsc/student_score.py tests/test_student_score.py
git diff --cached
git commit -m "feat: add residual-context scoring control"
git push -u origin feat/zhang-evaluation-adapter
```

提交信息采用 `类型: 简短说明`，类型与任务一致，例如 `feat`、`fix`、`docs` 或 `chore`；每次提交尽量只处理一件事。

在 GitHub 创建 Pull Request（PR），**目标分支（base）选择 `dev`**，来源分支选择自己的任务分支。说明修改目的、主要变化、验证结果和未完成事项；尚未完成时可创建 Draft PR。

### 同步与合并约定

- 不直接向 `dev` 或 `main` 推送日常修改，不对共享分支强制推送。建议在 GitHub 设置分支保护，要求 PR 审查与 CI 通过；本节约定不代表已启用保护规则。
- PR 至少由另一名组员检查，CI 通过后再合并到 `dev`；合并完成后可删除对应的远端任务分支。
- 开发期间若需同步其他成员的成果，先提交或妥善保存当前改动，再在自己的任务分支执行以下命令。出现冲突时与相关成员确认，解决后重新运行检查并提交，不直接覆盖他人修改。

  ```bash
  git fetch origin
  git merge origin/dev
  ```

- 完成一个任务后，下一个任务重新从最新 `dev` 创建分支，不继续复用已合并的任务分支。
- 里程碑交付前，由组内负责人组织验证，并创建 `dev` → `main` 的 PR；记录用于课程提交的 commit hash 或 tag。
- 不提交虚拟环境、密钥和临时产物。课程 starter 保持原样，实验开发仅使用开发集，方法冻结后才评估最终集。

## 课程实验流程

1. 运行课程 starter 自检并复现原始评分、资源计数和开发集基线；将研究原型适配到 starter 接口。
2. 对比原始 NSC、位置加权教学示例、残差上下文评分、加性消融、参数量、参考 FLOPs 及 harmonic/min 聚合。
3. 报告 Spearman ρ、Kendall τb、并列数，以及同宽且参数量相差不超过 10% 的比较数量。
4. 在 10M、20M、30M、50M 参数上限下评估 `k = 1, 3` 的最佳困惑度和相对遗憾，并对比均匀随机选择。
5. 只用开发集选择 `alpha` 和方法设置；冻结方案后再运行最终集。保存精确命令、配置与提交标识。

课程 starter 和固定 200 架构面板已归档在 `course/nsc-starter/`。在运行 starter 自检、完成评分接口适配并复现开发集基线前，不报告正式课程评估结果。上游仓库列出的其他基准数据、模型权重和训练资源仍属外部资源，并未随本项目提供。课程归档每个架构只有一个结果，因此本项目不估计训练种子不确定性。

## 数据、结果与脚本约定

- 课程数据统一使用 `course/nsc-starter/data/`，来源、版本与限制以 starter README 和 `provenance.json` 为准。额外数据需注明来源与许可，并与课程固定面板分开保存；替代数据仅用于开发调试，不能替代正式评估。开发过程不得查看最终集结果。
- 上游另列的 GPT-2/LiteTransformerSearch、FlexiBERT、AutoFormer、NATS-Bench、MobileNetV3、Transformer-XL、LoNAS 资源不随项目提供，也不自动属于课程固定面板。
- 实际运行实验后再创建 `results/`，记录精确命令、输入输出、数据划分、评分版本和代码提交；指标与对照按上面的课程实验流程记录。不得预填实验结果。
- 研究脚本统一放入 `scripts/`，调用方式及输入输出随实验记录保存。
- 测试覆盖 MP 密度归一化与支撑集、容量的形状对称性与单调性、缓存一致性，以及评分的层序敏感性、加性退化、输入校验和边界情况。`tests/conftest.py` 提供固定初始化标准差与换序层规格；新增行为时补充相关测试。
- 本地临时预览与校验记录放入 `work/`，不提交版本库；正式立项书只保留 `docs/ProjectProposal-en.pdf`。历史材料与恢复方式见[文档索引](docs/README.md)。

## 参考资料

文档索引见 [`docs/README.md`](docs/README.md)。

- 课程项目主题说明：[`docs/topic-neural-spectral-capacity.pdf`](docs/topic-neural-spectral-capacity.pdf)
- 项目规划：[`docs/project-plan.md`](docs/project-plan.md)
- 立项书：[`英文源文件`](docs/ProjectProposal-en.md) · [`正式 PDF`](docs/ProjectProposal-en.pdf)
- NSC 方法论文与官方实现：[论文](https://arxiv.org/abs/2609.23087) · [代码仓库](https://github.com/Optima-CityU/neural-spectral-capacity)
