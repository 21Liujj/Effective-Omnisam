# Experiment Rules

- 每个实验必须记录：seed、GPU、commit、启动脚本、配置文件路径
- 不要覆盖 `data_pd_exp_final_2000/` 下的 checkpoint 和伪标签
- 实验结果必须写入 `docs/claude/experiment-conventions.md` 或 `CLAUDE.md` 第 5 节
- 速度实验必须包含 warm-up（使用 `base_first` 热身后的状态）
- 速度统计必须区分：Train / PL / Val / E2E
- 新实验的 `pd_root` 必须用新目录名（含时间戳或实验名），不要复用旧目录
- checkpoint 删除前必须确认是否有对应完整日志和 metrics
- 结论必须由 metrics.jsonl 或 .log 支撑，推测需标注"推测"

## 新增资产管理规则

- 新正式实验脚本放在 `scripts/exp/`（或保持根目录 scripts/ 的现有风格，保持一致性）
- 新数据准备脚本放在 `scripts/data/`
- 新评估脚本放在 `scripts/eval/`
- 新工具脚本放在 `scripts/utils/`
- 新实验日志放在 `logs/<EXPERIMENT_NAME>/`（不要散落在 logs/ 根目录）
- 新文档放在 `docs/claude/`（跨对话参考）或 `docs/`（项目文档）
- 不要创建命名不清晰的文件（`test.sh`, `new.sh`, `final2.sh`, `临时说明.txt`）
- 添加文件前先查阅 `docs/claude/script-index.md`、`docs/claude/document-index.md`、`docs/claude/log-index.md`
- legacy 资产移动前必须在 `docs/claude/legacy-assets.md` 中登记
- 不要删除未在 legacy-assets.md 中登记的资产
