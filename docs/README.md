# 文档索引

本目录集中存放 Residual-Context NSC（CS5491 课程项目）的说明性材料。数据与代码约定另见仓库根目录的 `README.md`。

| 文件 | 内容 | 用途 |
|---|---|---|
| [`topic-neural-spectral-capacity.pdf`](topic-neural-spectral-capacity.pdf) | 课程主题说明（官方原件） | 核对研究问题、数据划分与提交要求 |
| [`project-plan.md`](project-plan.md) | 项目规划与时间表 | 里程碑安排、本周优先事项、风险应对 |
| [`ProjectProposal-en.md`](ProjectProposal-en.md) | 唯一立项书源文件（英文，含评估表格） | 2026-10-15 23:59 北京时间提交；按 A4 / Times New Roman 12 pt / 2.5 cm 页边距 / 单倍行距导出一页 PDF |

| [`ProjectProposal-en.pdf`](ProjectProposal-en.pdf) | 当前正式英文立项书 PDF | 与英文源文件配套，仓库内仅保留这一份 |
| [`CS5491-项目背景与研究动机.pdf`](CS5491-项目背景与研究动机.pdf) | 项目背景与研究动机 | 辅助研究材料 |
| [`archive/proposal-2026-10-13.zip`](archive/proposal-2026-10-13.zip) | 旧中文立项书、英文稿预览图和历史校验记录 | 历史参考，不作为当前提交版本 |

## 立项书生成与历史归档

生成脚本统一放在 [`scripts/build_proposal.py`](../scripts/build_proposal.py)。在 macOS 上安装 `reportlab`、`pypdf`、`pdfplumber`，并确保 `/System/Library/Fonts/Supplemental/` 中存在 Times New Roman 常规与粗体字体后，运行：

```bash
python scripts/build_proposal.py
```

脚本读取当前英文源文件，写入 `docs/ProjectProposal-en.pdf`，并将校验结果写入忽略提交的 `work/proposal/verification.json`。这些文档生成依赖不属于核心评分环境。

归档保留原始文件内容与原路径，并附 `SHA256.json` 校验清单。需要历史材料时可解压到独立临时目录：

```bash
python -m zipfile -e docs/archive/proposal-2026-10-13.zip /tmp/cs5491-proposal-archive
```

重复的 `output/pdf/ProjectProposal-en.pdf` 与正式 PDF 的 SHA-256 完全相同，已合并；原 `work/proposal/build_pdf.py` 已迁入 `scripts/`。数据、结果、脚本及测试的分散说明已合并到根 README。

## 阅读顺序建议

1. 先读课程主题说明，确认任务边界与评分口径。
2. 再读项目规划，确认当前阶段的交付物与实验约定。
3. 编写立项书时参考英文源文件，并核对课程提交要求。

## 约定

- 文档中的结论区分「已验证」与「待检验」。评分对层序敏感不等同于预测性能提升，两者分别报告。
- 正式实验结果不得在运行前填写。开发集用于方法选择，最终集只在冻结方案后评估。
- 上游实现的许可与来源记录见仓库根目录的 `THIRD_PARTY_NOTICES.md`。
