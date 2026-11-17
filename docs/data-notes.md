# 数据说明

> 课程归档面板（CS5491, 2026/27 Semester A），**路径 `course/nsc-starter/data/`**。
> 本文只记录"数据是什么、怎么用、哪里会踩坑"；字段与统计均于 2026-11-12 实测核对。
> 作者：张正；核对：行晋。

---

## 1. 来源与完整性

| 项 | 内容 |
|---|---|
| 数据包 | `course/nsc-starter/data/`，版本 `CS5491-202627A-v1`（`provenance.json`） |
| 上游 | LiteTransformerSearch（Javaheripi et al., NeurIPS 2022, arXiv:2203.02094）的 decoder-Transformer 搜索结果 |
| 完整性 | `provenance.json` 记录各文件 SHA-256；`check_starter.py` 会逐一校验 |
| 使用范围 | 仅用于本课程项目；引用时注明上游 |

---

## 2. 文件与字段

| 文件 | 内容 |
|---|---|
| `architectures.json` | 200 个 ID → 完整配置 |
| `resources.json` | 参数分解：`AdaEmb` / `Attn` / `FFN` / `nonembedding` / `total` |
| `splits.json` | `development` 100 / `final` 100 |
| `development_labels.json` | `valid_ppl` |
| `final_labels.json` | `test_ppl`（冻结前不得读） |
| `provenance.json` | 版本、来源、哈希 |

评分器只用 4 个字段，**逐层字段都是列表**：

| 键 | 类型 | 实测取值 |
|---|---|---|
| `d_model` | int | 64 (49) / 128 (51) / 256 (53) / 512 (47) |
| `n_layer` | int | 2–16 |
| `n_head` | list[int] | 逐层，200/200 全是列表；`d_model` 能被每个 `H_l` 整除 |
| `d_inner` | list[int] | 逐层，512–2048，共 1079 个不同值；长度恒等于 `n_layer` |

其余字段只用于理解架构假设，不进入评分：`adaptive=true`（全部）、`pre_lnorm=false`（全部，即 Post-LN）、
`mem_len=192`、`attn_type=0`、`weight_init_std=0.02`（全部）、`n_token=50264`、`div_val=4`、
`d_embed` 与 `d_model` 不同的有 **141/200**。

---

## 3. 资源口径（只读 `total`）

| 键 | 含义 |
|---|---|
| `total` | **预算选择的唯一口径** |
| `nonembedding` / `AdaEmb` / `Attn` / `FFN` | 分量，互相重叠 |

> ⚠️ **不能相加重建 `total`**。实测 `total − (AdaEmb + nonembedding)` 在 5.05 万–31.34 万之间（均值约 10.9 万），
> `nonembedding ≠ Attn + FFN`。一律直接读 `total`。

参考 FLOPs 由 `resource_estimate` 计算：`P_proj + 2·L·K·d`，键长 `K=384`（192 当前 + 192 缓存），
只计密集 QK/AV，FLOPs = 2 × MACs。**它不是实测延迟**。

---

## 4. 关键统计

| 项 | 值 |
|---|---|
| 架构数 / 唯一配置 | 200 / 200 |
| 深度范围 | 2–16 |
| FFN 宽度范围 | 512–2048 |
| `total` 范围 | 3.75M – 59.37M |
| `valid_ppl` | 24.35 – 74.38（中位 38.34） |
| `test_ppl` | 25.06 – 73.50（中位 38.11） |

**预算分档**

| 上限 | 全体 | 开发 | 最终 | 开发集 PPL 中位 |
|---|---:|---:|---:|---:|
| 10M | 49 | 24 | 25 | 47.35 |
| 20M | 122 | 65 | 57 | 42.30 |
| 30M | 162 | 83 | 79 | 39.64 |
| 50M | 192 | 96 | 96 | 38.55 |

**嵌入占比（AdaEmb / total）**

- 全体 200 个：最大 25,915,392，占比 7.8% – 97.9%
- 10M 桶 49 个：最大 8,377,280，占比 34.4% – 93.9%

> 立项书里的"8,377,280 / 34.4%–93.9%"只对 **10M 桶**成立；报告中引用必须写清范围。

**配对样本**：同一 `d_model` 且参数量比 ≥ 0.90，开发集 **180 对**（已实测）。这些配对共享架构，**不是独立样本**。

---

## 5. 审计结果（2026-11-12）

| # | 检查项 | 结果 |
|---|---|---|
| 1 | 200 条，dev/final 各 100 | ✅ |
| 2 | ID 唯一；两 split 交集为空、并集覆盖全部 | ✅ |
| 3 | 必填字段无缺失 | ✅ |
| 4 | 逐层列表长度 = `n_layer` | ✅ 0 例不符 |
| 5 | `d_model` 可被每个 `H_l` 整除 | ✅ 0 例不符 |
| 6 | PPL 为正困惑度 | ✅ |
| 7 | `total` 与分量和不相等（只能直读） | ✅ 已记录 |
| 8 | 预算桶计数 49/122/162/192 | ✅ |
| 9 | 完全相同配置 0 例 | ✅ |
| 10 | 异常值检验 | ⬜ 未做（报告需声明） |
| 11 | 数据哈希 | ✅ `check_starter.py` 通过 |

---

## 6. 使用规则

1. **只读 `course/nsc-starter/data/`**，不修改；派生结果写 `results/`。
2. **split 隔离**：开发阶段只读 `valid_ppl`；打 tag 前不读 `test_ppl`。
3. **预算只用 `total`**。
4. **并列处理**：比较前统一到 12 位有效数字；随机决策固定种子（`202627 + budget + k`）。
5. **可追溯**：结果文件记录命令、commit、split、种子。

---

## 7. 两个已知的坑

**① CRLF 会让哈希校验失败**

`core.autocrlf=true` 的机器上，`data/*.json` 会被检出为 CRLF，导致 `check_starter.py` 的 SHA-256 断言失败。
仓库根的 `.gitattributes` 已把该目录固定为 `text eol=lf`；若本地仍是旧检出：

```bash
git rm --cached -r --quiet course/nsc-starter/data
git checkout -- course/nsc-starter/data
```

**② 环境版本**

仓库 `.python-version` 为 **3.12**，CI 跑 3.10 / 3.12。基线复现记录用的是 Python 3.14.5 + numpy 2.5.4 + scipy 1.18.1
（见 `results/baseline/README.md`）。独立复现建议用干净的 3.12 环境，与 CI 对齐。

---

## 8. 解释边界

- **异构面板**：深度、宽度、嵌入策略同时变化，任何"评分更好"都可能混入规模效应。
- **Post-LN**：`pre_lnorm=false`，`Q_l` 只能当结构代理。
- **AdaEmb 主导**：最高占 97.9%，10M 档压缩了非嵌入组件的区分空间（该档 PPL 中位 47.35，明显高于其他档）。
- **单条标签**：无训练种子方差。
- **最终集不是独立基准**：同归档内的架构留出。
