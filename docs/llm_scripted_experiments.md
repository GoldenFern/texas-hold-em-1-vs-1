# 固定策略 Bot 与工具对照实验

## 目的

在单挑规则不变的前提下，观察 GPT-6 Luna 对三种极端固定策略的应对，以及
牌型分布、胜率、跟注 EV、随机数四种工具对同一随机对手的表现差异。

## 设计

- 每手独立，1000 筹码，盲注 5/10，筹码每手重置。
- 同牌换座：每副牌打两次，GPT 分别在按钮位和大盲位。
- 实验 A 只提供 `apply_action`，不提供分析工具和桌边喊话。
- 实验 B 每组只多开放一个辅助工具，桌边喊话关闭。
- 实验 B 的提示词对本组工具做“强烈推荐、非强制”的附加说明；空白组不加。
- 所有组使用同一模型、同一批牌、同一随机种子。

### 实验 A

| 对手 | 策略 |
|---|---|
| All-in Bot | 能全下就全下，否则跟注或过牌 |
| Min-Raise Bot | 能加注就最小加注，否则跟注或过牌，从不弃牌 |
| Call/Check Bot | 从不加注、从不弃牌，有跟注就跟，否则过牌 |

### 实验 B

| 组 | 工具 |
|---|---|
| 空白 | 只有 `apply_action` |
| 随机数 | `get_random_number` |
| 牌型分布 | `estimate_hand_probabilities` |
| 胜率 | `estimate_equity` |
| 跟注 EV | `call_ev` |

随机 Agent 在合法动作里等概率选择；下注或加注金额在允许的 raise-to 范围内
均匀取整数，按手牌种子固定。

## 运行

```bash
# 小样本校准
python scripts/run_llm_experiments.py --stage pilot --pilot-hands 10 \
    --workers 2 --reasoning-effort low --output tmp/llm_experiments/pilot

# 正式实验
python scripts/run_llm_experiments.py --stage full --full-hands 400 \
    --workers 2 --reasoning-effort low --output tmp/llm_experiments/full
```

`run_llm_experiments.py` 会顺序执行 8 个条件，记录累计估算费用，支持断点续跑，
并在单个条件出现 API 错误时重跑该条件。

## DXH 实验配置

`config/experiments/dxh_providers.json` 只给离线评测用，不接入前端和服务端 Provider 列表。
使用时显式传入：

```bash
python scripts/eval_llm_vs_bot.py --provider dxh --model gpt-6-luna \
    --provider-config config/experiments/dxh_providers.json \
    --llm-name "GPT-6 Luna (DXH)" --opponent-kind random --tools none --hands 1 --workers 1
```

DXH 中转的流式、工具调用和 usage 正常；小样本下 16 并发没有 429 或超时。
它的 prompt token 统计明显高于 4Router。DXH 实验单价按 4Router 标准档的 44% 计：
输入 ¥0.0198/M，输出 ¥0.099/M。

只跑五组工具对照：

```bash
python scripts/run_llm_experiments.py --stage full --experiment b \
    --provider dxh --provider-config config/experiments/dxh_providers.json \
    --model gpt-6-luna --llm-name "GPT-6 Luna (DXH)" \
    --workers 16 --reasoning-effort low --relay-host api.430123.xyz \
    --price-input 0.0198 --price-output 0.099 --budget-cny 10 \
    --output tmp/llm_experiments/dxh_b
```

## 2026-10-24 compact_slim 26 手校准

DXH 上新策略 `compact_slim` 跑了 8 个条件，每组 26 手（13 副牌 × 2 座位），16 并发：

- 全部完成，0 API 错误，费用 ¥0.0719
- B 组干净数据的每手费用约 ¥0.000285，旧 `legacy` 约 ¥0.000315，只低约 10%
- 输入 token 每组低 2% 到 16%，缓存命中 52% 到 92%，没有出现连续对局里 96% 以上的稳定效果
- 26 手的绝对 bb/100 全部为负，和之前 400/1000 手方向不一致，只能当流程校准，不能当胜负结论

原因是当前评测每手独立重置，跨手摘要压缩用不上，`compact_slim` 的主要收益只剩精简回合状态和工具结果。

## 2026-10-25 1000 手 compact_slim 结果

DXH、`compact_slim`、每组 1000 手（500 副牌 × 2 座位），随机 Agent 对手：

| 组 | bb/100 | 95% CI | 胜率 | 摊牌率 | 工具/回合 | 费用 |
|---|---:|---:|---:|---:|---:|---:|
| 空白 | +136.7 | ±145.2 | 37.4% | 5.1% | 0.000 | ¥0.186 |
| 随机数 | +170.0 | ±136.2 | 38.5% | 4.5% | 0.005 | ¥0.203 |
| 牌型分布 | +131.5 | ±135.1 | 37.6% | 5.2% | 0.279 | ¥0.259 |
| 胜率 | +438.2 | ±190.6 | 44.1% | 10.5% | 0.895 | ¥0.397 |
| 跟注 EV | +636.0 | ±213.2 | 49.4% | 19.1% | 0.447 | ¥0.322 |

相对空白组的同牌同座位配对：

| 组 | Δ bb/100 | 95% CI | p | 胜/负/平 |
|---|---:|---:|---:|---:|
| 随机数 | +33.4 | ±149.9 | 0.663 | 141/142/717 |
| 牌型分布 | -5.1 | ±134.5 | 0.940 | 133/152/715 |
| 胜率 | +301.6 | ±180.8 | 0.0011 | 189/158/653 |
| 跟注 EV | +499.3 | ±243.2 | 0.0001 | 228/141/631 |

这是第一次出现清楚的正向工具效果：胜率和跟注 EV 都显著优于空白组，且 Bonferroni
校正后仍然显著；牌型分布基本没有作用。两次显著结果都伴随更高的摊牌率和更大的底池。

注意：对手仍是均匀随机 Agent，不能把这里的 bb/100 当成对强 Bot 或人类的稳定优势；
工具可能只是减少了对随机对手的明显错误，换成强对手后效果可能变小。

## 4Router 运行注意事项

- `gpt-6-luna` 在当前网络下 2 个并发比较稳定；3 个以上并发容易出现连接超时或
  429，自动行动会污染结果。
- `reasoning_effort=low` 明显快于 `medium`；`medium` 在长手牌里可能每手超过
  一分钟，不适合大批量夜间运行。
- 读超时设 45 秒、连接超时 8 秒、单次请求 5 次指数退避，比长时间等待更容易
  从偶发连接失败里恢复。
- 如果 `4router.net:443` 整体不可达，先暂停并等待恢复，不要提高并发硬顶。

## 输出

每个条件保留：

- `hands.csv`：每手净筹码、动作、工具调用、token、费用、耗时。
- `hands.jsonl`：完整压缩事件，便于复核具体回合。
- `summary.json`：汇总指标和运行参数。
- `report.md`：单条件报告。

实验 B 另有 `full/b_tools/combined_report.md`，包含五组总表和相对空白组的
同牌配对比较。总目录的 `combined_report.md` 汇总实验 A 和实验 B。

补跑更多牌局时，用不同的 `--deal-offset` 避免和已有牌局重叠，然后合并：

```bash
python scripts/run_llm_experiments.py --stage full --experiment b \
    --full-hands 600 --deal-offset 200 --output tmp/llm_experiments/dxh_b_extra

python scripts/merge_llm_runs.py \
    --runs tmp/llm_experiments/dxh_b/full/b_tools \
           tmp/llm_experiments/dxh_b_extra/full/b_tools \
    --output tmp/llm_experiments/dxh_b_1000 \
    --expected-hands 1000 --price-input 0.0198 --price-output 0.099
```

`merge_llm_runs.py` 按 `(deal_index, assignment)` 去重，重新计算每组的
bb/100、置信区间和相对空白组的配对差异。

## 口径

- 费用按输入 ¥0.045/M、输出 ¥0.225/M 估算，不含缓存。
- 400 手一组的 95% 置信区间仍然较宽，结果只适合做方向性比较。
- 每手独立意味着 GPT 不能跨手学习对手风格；实验 A 测的是单次决策，不是长期适应。
