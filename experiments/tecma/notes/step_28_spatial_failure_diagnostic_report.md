# Step 28: Spatial Failure Diagnostic Report

**Status**: 脚本框架完成,完整推理执行pending(需~30分钟GPU时间)

## 目标

验证假设:"MTC+SMR压缩后的mIoU下降(60.26→60.02/-0.24)是否在全景图特定空间位置(接缝/畸变区)更严重"

## 方法局限性发现

在实施前发现DensePASS数据集特性与标准ERP假设不符:
- DensePASS val图像分辨率: 2048×400 (宽高比5.12:1)
- 标准ERP全景图: 宽高比2:1 (e.g. 2048×1024, 4096×2048)
- 结论: DensePASS **不是**标准等距柱状投影,可能是裁剪后的部分视场或非标准投影

因此原假设"x=0/x=2048是360°接缝,畸变集中在两极"可能不成立。需要先通过以下方式验证:
1. 检查DensePASS论文/数据集文档,确认投影类型和视场范围
2. 可视化几张样本,人工判断左右边界是否视觉连续(真接缝应该左右边缘内容能拼接)

## 已产出

1. **诊断脚本框架**: `experiments/tecma/diagnostics/spatial_failure_analysis.py` (需根据数据集实际特性调整分桶策略)
2. **推理脚本**: `scripts/run_spatial_diagnostic_inference.sh` (需先给validation.py添加`--save_preds_to`和plugin配置支持)

## 执行路径(如果数据集验证通过)

1. 扩展validation.py:添加`--plugin_config`参数读取YAML并调用`memory_bank.configure_*`
2. 运行`scripts/run_spatial_diagnostic_inference.sh` (~15-20分钟,生成baseline+compressed的100张预测)
3. 运行`experiments/tecma/diagnostics/spatial_failure_analysis.py`后处理,生成空间分桶mIoU对比图

## 替代方案

如果空间假设不成立,改为**语义难度分析**:
- 按类别频率分桶(rare vs common classes)
- 按对象尺寸分桶(small vs large instances)
- 这不需要额外推理,可直接用现有per-class IoU数据

## 时间估算

- validation.py扩展: 1-2小时开发+测试
- 推理执行: 20-30分钟(baseline+compressed各100张)
- 后处理脚本: 30分钟
- **总计: 2-3小时**

建议优先完成失效样本可视化(已有pred,只需rendering),空间诊断作为optional扩展。
