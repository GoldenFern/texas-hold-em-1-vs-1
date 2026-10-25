# LLM 工具消融实验

## 目的

比较同一个 LLM 在五种工具配置下对 Project DQN 的表现，观察工具是否改变决策质量和工具使用行为。

五组使用同一批牌、同一个 Bot、同一套规则。每组只改变暴露给模型的辅助工具，以及系统提示词里对应的“强烈推荐但非强制”使用建议。

## 五组设计

| 组 | 工具 | 说明 |
|---|---|---|
| `none` | `apply_action` | 只提交动作，没有任何辅助分析工具 |
| `random` | `apply_action`、`get_random_number` | 随机数用于混合策略或打破僵局 |
| `distribution` | `apply_action`、`estimate_hand_probabilities` | 最终牌型分布，不是胜率 |
| `equity` | `apply_action`、`estimate_equity` | 当前摊牌 equity，默认对手随机手牌 |
| `ev` | `apply_action`、`call_ev` | 静态跟注 EV；省略 equity 时自动估算胜率 |

## 规则与口径

- 模型：`longcat / LongCat-2.5-Preview`，默认 thinking 开启。
- 对手：`models/rlcard/nlh_dqn_project.pth`，DQN，推理取 argmax。
- 1000 筹码，5/10 盲注，每手重置筹码。
- 每组 300 手，等于 150 副牌 × 2 个座位，同牌换座。
- 五组使用同一个 seed，牌和座位分配一致。
- 每组 10 并发，五组顺序执行。
- 2026-10-25 起默认重试改为 OpenCode 的 10 次预算；此前结果使用脚本内的 4/5 次包装重试，跨版本比较时需要留意。
- 每手新建 LLM 对话，不跨手记忆。
- Agent 语言默认 `zh-CN`，加 `--agent-locale en-US` 可让该组使用英文提示词、工具说明和工具返回。

## 提示词

默认提示词保持中立，只说明工具用途和可选性。实验通过 `build_system_prompt(..., extra_instructions=...)` 追加一段“工具使用建议（强烈推荐，非强制）”，按组只推荐本组工具。

运行时不设置 `require_estimate_first` 或 `require_estimate_each_turn`，模型不调用辅助工具不会收到错误，也不会被强制跟注或弃牌。

## 命令

```bash
# 先跑冒烟
python scripts/eval_llm_tool_matrix.py --hands 2 --workers 2 \
  --output tmp/llm_tool_matrix/smoke

# 全量
python scripts/eval_llm_tool_matrix.py --hands 300 --workers 10

# 只重算汇总
python scripts/eval_llm_tool_matrix.py --report-only \
  --output tmp/llm_tool_matrix/<timestamp>
```

## 输出

每组目录包含：

- `hands.csv`：每手一行，含净筹码、工具调用、token、自动行动等。
- `hands.jsonl`：完整动作、压缩事件、`turn_state` 和工具结果。
- `summary.json`：组内汇总。
- `report.md`：组内报告。
- `prompt.txt`：实际系统提示词。
- `run.log`：该组的运行日志。

根目录包含：

- `manifest.json`：模型、Bot 哈希、seed、参数和命令。
- `combined_summary.json`：五组汇总与相对无工具组的配对差异。
- `combined_report.md`：总表、工具使用、数据质量。

## 指标

- 净筹码、每手均值、bb/100、胜率。
- 同牌换座 95% 置信区间。
- 辅助工具调用次数、每回合调用率、使用工具的回合占比、工具错误。
- `call_ev` 的适用与不适用次数。
- API 调用、输入/输出/推理 token、耗时。
- 自动行动、非法动作自动、API 错误自动、重试手数。

## 解释注意事项

- EV 组的 `call_ev` 在省略 `equity` 时自动估算胜率，因此 EV 组也会间接获得胜率信息。
- 随机数工具本身不提供牌局信息，主要用于混合策略，不应期待单独带来系统性优势。
- 300 手/组的置信区间仍然较宽，组间差异要结合区间和工具行为一起看。
- 同牌换座降低牌运和位置偏差，但 LLM 的行动会改变后续牌局路径，不能当作逐动作对照。

## 2026-10-22 实跑记录（中断）

LongCat API 额度在评测中途耗尽，五组没有全部完成。终止后没有继续发起模型请求。

| 组 | 状态 | 结果 |
|---|---|---|
| `none` | 可用 | 300 手，+500.7 bb/100，同牌换座 95% CI ±351.5 |
| `random` | 部分可用 | 286 手干净子集，+337.1 bb/100，95% CI ±364.0；815 次随机数调用，工具回合占比 78.4% |
| `distribution` | 不可用 | 全部 API 错误，自动行动污染 |
| `equity` | 不可用 | 全部 API 错误，自动行动污染 |
| `ev` | 未完成 | 终止时没有 summary |

随机数组排除 14 手 API 错误后，与无工具组在 286 个相同牌局座位上配对，差值为 -147.0 bb/100，95% CI ±337.0，不能认为随机数工具产生了显著影响。

完整的中断记录和干净子集分析保留在实验输出目录，不随仓库发布。要完成五组对比，需要用新额度重跑 `distribution`、`equity`、`ev`，并重跑 `random` 的污染样本。
