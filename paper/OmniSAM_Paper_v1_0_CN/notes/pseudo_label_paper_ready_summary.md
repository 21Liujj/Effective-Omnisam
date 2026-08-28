# Pseudo-label Paper-ready Summary for OmniSAM Paper v1.0

## 1. Final Position in Paper v1.0

**主要定位选择：辅助贡献 + 消融实验 + Discussion，而不是主贡献。**

伪标签优化线不建议作为 OmniSAM 中文论文 v1.0 的主贡献。根据 0-3a 已确认结果，`R2_class_balanced` 相比 `R0_identity` 的 mIoU 仅从 60.02 到 60.07，提升约 +0.05；`R2R4_proto_gate` 的 mIoU 为 60.05，相比 R2 略低。因此当前证据不足以支持“显著提升最终性能”或“作为最强主结果”的表述。

但伪标签优化线仍然可以写入 Method 或 Discussion，原因是：

- 它对应 UDA 训练中的真实问题：目标域 pseudo label 噪声和可靠性不足；
- 诊断指标显示 R2/R2R4 确实改变了 pseudo label 的过滤强度、类别保留比例和 prototype similarity 统计；
- 它可以作为 memory-efficient 主线之外的目标域监督信号可靠性模块，用于解释框架的扩展方向；
- 当前更适合作为探索性消融和后续组合实验的铺垫。

与 memory-efficient 主线的关系：

- memory-efficient 线仍应是 v1.0 的主要贡献和主结果来源；
- pseudo-label refinement 线可作为辅助可靠性组件或附加消融出现；
- 目前不能把 PL 线和 MTC+SMR 合并成最终强结论，因为 0-3a 明确指出完整组合实验仍需确认。

## 2. Paper-safe Pseudo-label Results

| 实验 | 论文建议名称 | mIoU | 相比 R0 变化 | 相比上一配置变化 | 是否可写入论文 | 推荐写法 |
|---|---|---:|---:|---:|---|---|
| R0_identity | Identity PL Control | 60.02 | N/A | N/A | 可以 | 作为 fixed/reused PL 设置下的 identity 对照，不声称其为最终 baseline |
| R2_class_balanced | Class-balanced Pseudo-label Refinement (CBPR) | 60.07 | +0.05 | +0.05 | 可以，需谨慎 | “R2_class_balanced 带来轻微 mIoU 改善，但幅度较小，主要作为伪标签可靠性调节的初步证据” |
| R2R4_proto_gate | CBPR + Prototype-guided Reliability Gate (PRG) | 60.05 | +0.03 | -0.02 | 可以，需作为探索性结果 | “Prototype gate 在当前设置下未进一步提升最终 mIoU，但提供了基于特征原型建模伪标签可靠性的方向” |

说明：本节只使用 0-3a 已确认的 mIoU 数字。其他指标虽然在 0-3a facts note 中有记录，但本摘要表为避免主线混淆，仅保留 mIoU 和相对变化。

## 3. Diagnostic Evidence Summary

| 模块/实验 | 诊断现象 | 对论文的价值 | 是否适合进正文 |
|---|---|---|---|
| R2_class_balanced | 相比 R0，R2 对 pseudo label 进行了更强过滤，并显示类别保留/类别均衡调节现象 | 可用于说明该模块的目标不是直接追求大幅 mIoU 提升，而是调节目标域监督信号质量 | 可少量进入正文或 ablation 分析 |
| R2_class_balanced | rare/extreme-rare 类保留比例高于 common 类，说明 class-balanced 策略对类别覆盖有调节作用 | 可支持“类别均衡伪标签优化”的方法动机 | 适合正文简述，详细数值更适合 appendix/notes |
| R2R4_proto_gate | 相比 R2，R2R4 进一步增强过滤强度，但 mIoU 未进一步提升 | 可说明 prototype gate 当前更像可靠性建模工具，而不是已验证的性能增益模块 | 适合 Discussion 或 limitation |
| R2R4_proto_gate | prototype similarity 相关统计可用，且未出现 NaN 诊断问题 | 可说明 PRG 工程路径可运行，并提供后续调参/组合实验基础 | 更适合 appendix/notes，正文只保留结论 |
| R6_memory_plq | 仅有配置，状态为 NOT_IMPLEMENTED_YET | 可作为 future work：结合 memory consistency 的 pseudo-label quality refinement | 不适合进入实验正文，只适合 Discussion / Future Work |

总体判断：诊断证据适合解释“可靠性动机”和“为什么需要伪标签质量控制”，但不应被包装成最终性能主结论。

## 4. Recommended LaTeX Table Draft

```latex
\begin{table}[t]
  \centering
  \caption{Preliminary pseudo-label refinement ablation under the reused/fixed pseudo-label setting. The results are intended to analyze pseudo-label reliability modules rather than to claim a significant performance gain.}
  \label{tab:preliminary_pl_refinement}
  \begin{tabular}{lccccl}
    \toprule
    Configuration & Class-balanced & Prototype gate & mIoU & $\Delta$ vs. R0 & Remark \\
    \midrule
    R0 Identity PL & -- & -- & 60.02 & -- & identity control \\
    R2 CBPR & \checkmark & -- & 60.07 & +0.05 & slight improvement \\
    R2R4 CBPR + PRG & \checkmark & \checkmark & 60.05 & +0.03 & no further mIoU gain over R2 \\
    \bottomrule
  \end{tabular}
  \\
  \footnotesize{R6 memory-aware PLQ is not included because it is currently a TODO / not implemented result in v1.0.}
\end{table}
```

注意：caption 已明确写成 preliminary pseudo-label refinement ablation；表中不出现“significant improvement”等表述；R6_memory_plq 只作为脚注 TODO，不作为已完成结果。

## 5. Paper-ready Claims

1. 初步实验显示，class-balanced pseudo-label refinement 在 reused/fixed PL 设置下带来轻微 mIoU 改善：R2 为 60.07，R0 为 60.02。
2. R2_class_balanced 的主要价值更适合解释为调节目标域伪标签的类别保留和监督信号可靠性，而不是显著提升最终 mIoU。
3. Prototype gate 在当前设置下未进一步提升最终 mIoU：R2R4 为 60.05，略低于 R2 的 60.07；但它提供了利用特征原型建模伪标签可靠性的方向。
4. 伪标签优化模块当前更适合作为 memory-efficient 框架的辅助可靠性组件、探索性消融或 Discussion 内容，而不是 v1.0 的最强主贡献。
5. 后续需要补充与 MTC+SMR 的完整组合实验、多 seed 稳定性验证，以及 R6_memory_plq 的实现和评估。

## 6. Unsafe Claims

以下说法不能写入论文 v1.0：

- 不能声称 pseudo-label refinement 显著提升 mIoU。
- 不能声称 prototype gate 带来最终性能提升。
- 不能声称 R6_memory_plq 已完成或已有实验结果。
- 不能声称已完成与 memory-efficient 主线，尤其是 MTC+SMR 的完整组合实验。
- 不能把伪标签线作为当前论文 v1.0 的最强主贡献。
- 不能声称多 seed 稳定，因为 0-3a 未确认多 seed 结果。
- 不能声称已经全面解决目标域 pseudo-label noise。
- 不能把 pseudo-label 线的结果和 memory-efficient 主线结果混用成同一个最终主表强结论。

## 7. Suggested Placement in Paper

| 论文位置 | 是否建议写入 | 写法 |
|---|---|---|
| Abstract | 最多一句，且不突出为主结果 | 可写“we further explore pseudo-label reliability refinement”，但不要写具体 mIoU 提升，也不要作为核心贡献句 |
| Introduction | 建议写入 | 可作为第二个挑战：panoramic UDA 中目标域 pseudo-label noise 与类别不均衡会影响监督信号质量 |
| Method | 建议写入 | 可保留 Pseudo-label Quality Refinement / PQR 模块，说明包含 confidence filtering、class-balanced refinement 和 prototype-guided reliability modeling |
| Experiments | 谨慎写入 | 不放入主结果表；可在 supplementary-style paragraph 或小节中说明 preliminary reused/fixed PL 结果 |
| Ablation | 建议写入 | 放 preliminary ablation，展示 R0/R2/R2R4；结论写“轻微改善/基本持平/仍需验证”，不写显著提升 |
| Discussion / Future Work | 强烈建议写入 | 重点写 R6_memory_plq、MTC+SMR 组合、多 seed、full protocol 仍需补充 |
| Conclusion | 可以谨慎一句 | 可写“伪标签可靠性建模提供了后续提升目标域监督质量的方向”，不要写成最终已验证主贡献 |

## 8. Integration with Memory-efficient Main Line

| 问题 | 当前证据 | 论文 v1.0 处理 |
|---|---|---|
| 是否与 MTC+SMR 组合完成 | 0-3a 指出 summary 虽标记 memory-efficient 相关，但 metrics 中 memory token 记录与最终压缩配置关系仍需确认 | 不能声称完整组合完成；作为 TODO 或待确认实验 |
| 是否可以作为最终框架的一部分 | 方法上可以作为目标域监督信号可靠性模块；但结果上只有轻微 mIoU 改善和诊断证据 | 可以在框架图中以辅助模块形式出现，但图注需避免暗示它是最终性能主来源 |
| 是否应在方法图中出现 | 可以出现，但应弱化为 optional / auxiliary pseudo-label reliability branch | 建议标注为 “Pseudo-label Reliability Refinement (optional)” 或 “PQR branch” |
| 是否应作为主表结果 | 当前 R2/R2R4 mIoU 提升极小，且组合与 full protocol 仍待确认 | 不建议放入主结果表；更适合放入 ablation 或 appendix |
| 是否应作为 future experiment | R6_memory_plq 未完成，多 seed 与 MTC+SMR 组合缺失 | 应明确作为 future experiment / limitation |
| 是否能支持主贡献 | 目前不能。PL 线不是 v1.0 当前最强主贡献 | 主贡献仍应围绕 memory-efficient UDA；PL 线作为辅助可靠性探索 |

建议结论：当前 v1.0 可以在方法图中以辅助模块形式出现，但不要在主结果表中和 MTC+SMR 合并成最终强结论；组合实验应作为 TODO。

## 9. Suggested Naming

| 工程名称 | 论文建议名称 | 是否建议正文使用 | 备注 |
|---|---|---|---|
| R0_identity | Identity PL Control | 建议用于 ablation 表 | 作为不修改 pseudo label 的对照配置 |
| R2_class_balanced | Class-balanced Pseudo-label Refinement (CBPR) | 建议使用，但可作为 PQR 子模块 | 当前唯一带来轻微 mIoU 改善的 PL refinement 配置 |
| R2R4_proto_gate | CBPR + Prototype-guided Reliability Gate (PRG) | 谨慎使用 | 可用于探索性消融；不能声称进一步提升 mIoU |
| R6_memory_plq | Memory-aware Pseudo-label Quality Refinement / Memory-aware PQR | 不建议作为已完成正文结果 | 只作为 TODO / Future Work；NOT_IMPLEMENTED_YET |
| pseudo_conf_mask | Confidence-based Pseudo-label Masking / Confidence Filtering | 可在 Method 中简述 | 若 acronym 太多，不建议单独命名为新贡献 |
| prototype_gate | Prototype-guided Reliability Gate (PRG) | 可作为 PQR 子模块命名 | 当前更适合描述为可靠性建模方向 |
| prototype_bank | Prototype Bank | 暂不建议正文使用 | 0-3a 确认当前为 placeholder，不能作为已完成模块 |

命名建议：

- 若论文希望控制 acronym 数量，建议只保留 **Pseudo-label Quality Refinement (PQR)** 作为总称；
- **CBPR** 和 **PRG** 可以作为 PQR 的两个子模块名出现在方法或 ablation 中；
- `prototype_bank` 不建议作为正文模块名，因为当前没有已完成实验支撑。
