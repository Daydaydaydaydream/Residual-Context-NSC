# 第三方代码与来源

`src/cs5491_nsc/nsc_utils.py` 复用了上游 NSC 项目的参考 MP 评分实现。来源仓库在本地参考目录中检出的提交为 `7f003cbfc650a69b4456d85602357f4d115c3d69`。

- 项目：Chenyu Zhu、Ruoyu Zhao、Zhichao Lu，*Neural Spectral Capacity: Measuring and Designing Architectures from Network Specification Alone*。
- 上游仓库：<https://github.com/Optima-CityU/neural-spectral-capacity>
- 上游核心文件：<https://github.com/Optima-CityU/neural-spectral-capacity/blob/7f003cbfc650a69b4456d85602357f4d115c3d69/nsc/nsc_utils.py>
- 许可证：MIT；随本地评分文件保存在 [`src/cs5491_nsc/LICENSE`](src/cs5491_nsc/LICENSE)。

课程研究改动位于 `src/cs5491_nsc/student_score.py`，包括固定 `s = 0.02` 的课程架构分组评分，以及层序依赖的残差上下文启发式；这些不是上游仓库的官方方法。
