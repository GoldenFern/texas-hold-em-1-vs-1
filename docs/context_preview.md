# 上下文压缩预实验（2026-10-24）

## 目的

用较小的成本先验证三件事：

1. 当前 `legacy` 裁剪是否真的会造成缓存前缀失效。
2. 按手摘要压缩（`compact`）能否在长上下文下稳定缓存。
3. 精简回合状态与工具结果（`compact_slim`）能否进一步降低 token。

这是预实验，不是完整对照实验。每个策略只跑了 10 手，行动不同导致后续牌局状态不同，不能据此判断策略强弱。

## 设置

- 模型：`deepseek/deepseek-flash`
- 思考：开启，`reasoning_effort=low`
- 上下文：1M 窗口
- 对手：`Project DQN`（`models/rlcard/nlh_dqn_project.pth`）
- 牌局：1000 筹码，5/10 盲注，自动重购，固定 seed `2026`，每组 10 手
- 工具：`ALL_TOOL_NAMES` 五个基础工具
- 语言：`zh-CN`
- 输出：`tmp/context_preview/`

策略：

| 策略 | 说明 |
|---|---|
| `legacy` | 当前实现，超过硬上限后从最早的 turn 开始删 |
| `full` | 不裁剪，作为长上下文上限参考 |
| `compact` | 每手结束写紧凑摘要，保留最近 2 手详情，20k 高水位 / 8k 低水位 |
| `compact_slim` | `compact` + 精简回合状态 + 精简工具结果 |

## 结果

| 策略 | 有效上限 | 手数 | API 调用 | turn | 缓存命中 | 输入 token | 缓存 miss | 输出 token | 费用 USD | 费用/turn | miss/turn | 压缩次数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `legacy` | 20k | 10 | 37 | 32 | 50.1% | 663,680 | 331,136 | 10,484 | $0.0570 | $0.00178 | 10,348 | 0 |
| `compact` | 20k | 10 | 31 | 28 | 88.5% | 461,057 | 52,993 | 14,103 | $0.0176 | $0.00063 | 1,893 | 2 |
| `compact_slim` | 20k | 10 | 29 | 25 | 96.7% | 279,275 | 9,323 | 7,776 | $0.0069 | $0.00028 | 373 | 0 |
| `full` | 1M | 10 | 35 | 27 | 96.7% | 898,612 | 29,364 | 24,673 | $0.0218 | $0.00081 | 1,088 | 0 |
| `legacy` | 10k | 10 | 33 | 28 | 44.8% | 325,954 | 180,034 | 19,152 | $0.0389 | $0.00139 | 6,430 | 0 |
| `compact_slim` | 10k | 10 | 44 | 35 | 89.4% | 369,070 | 39,214 | 31,735 | $0.0259 | $0.00074 | 1,120 | 5 |

按请求位置拆分的缓存命中：

| 策略 | 全部请求 | 每手第一条 | 后续请求 |
|---|---:|---:|---:|
| `legacy` 20k | 50.1% | 37.2% | 55.0% |
| `compact` 20k | 88.5% | 87.0% | 89.3% |
| `compact_slim` 20k | 96.7% | 97.1% | 96.3% |
| `full` | 96.7% | 97.6% | 96.2% |
| `legacy` 10k | 44.8% | 44.4% | 44.9% |
| `compact_slim` 10k | 89.4% | 60.0% | 96.7% |

### 逐请求观察

`legacy` 20k 在第 6 手触发了一次裁剪，裁掉 9 条消息。裁剪前单次请求命中大多在 90% 以上；裁剪后大量请求直接掉到 7.2% 到 7.5%，只有偶尔几次重新命中 95% 左右。聚合后的 50.1% 其实已经是被裁剪前的高命中拉高后的结果。

`compact` 20k 在第 5 手和第 7 手各压缩一次。压缩后的第一条请求命中分别只有 14.2% 和 13.1%，下一条就恢复到 93% 以上。这说明按手压缩的代价是“每次压缩一次 miss”，而不是 legacy 那种持续低命中。

## 观察

1. **`legacy` 的缓存命中明显下降。** 20k 上限下只有 50.1%，10k 上限下 44.8%。原因是裁剪点紧跟在 system prompt 后面，之后的整个前缀都变了。每手第一条请求也只有 37% 到 44% 命中，说明裁剪发生在发请求之前，连当前手的第一条请求也失去了旧前缀。
2. **按手摘要压缩有效。** 同样 20k 上限，`compact` 的缓存命中从 50.1% 提升到 88.5%，miss/turn 从 10,348 降到 1,893，费用/turn 从 $0.00178 降到 $0.00063。压缩事件显示一次可以把 53 条消息压到 24 条，上下文从 20.4k 降到 10.5k。
3. **精简状态和工具结果的收益最大。** `compact_slim` 在 20k 上限下 10 手都没有触发压缩，上下文始终低于 20k；缓存命中 96.7%，费用/turn $0.00028，约为 `legacy` 的 1/6。
4. **压缩频率不能太高。** `compact_slim` 在 10k 上限下触发了 5 次压缩，整体命中仍有 89.4%，但每次压缩后的第一条请求只有 60% 命中。高水位设得太低会把缓存收益吃掉一部分。
5. **`full` 的缓存命中很高，但单次请求最大。** 平均每次请求 25.7k 输入 token，费用/turn 是 `compact_slim` 的约 2.9 倍。缓存命中率高不等于总成本低。
6. **API 和缓存字段正常。** 所有 arm 的 `estimated_usage_requests` 都是 0，说明拿到的是真实 usage；`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` 解析正常；没有 API 错误，也没有因为缺少 `reasoning_content` 产生 400。

### 20 手补充

为了看压缩在更长对局里的表现，单独跑了 `compact_slim` 20 手，20k 高水位 / 8k 低水位 / keep 2：

| 指标 | 值 |
|---|---:|
| 手数 | 20 |
| 压缩次数 | 8 |
| 缓存命中 | 91.8% |
| 每手第一条请求命中 | 82.6% |
| 后续请求命中 | 93.8% |
| 输入 token | 1,326,854 |
| 缓存 miss | 108,678 |
| 输出 token | 108,917 |
| 费用 | $0.0853 |
| 净筹码 | +535 |
| bb/100 | +267.5 ± 1041.6 |

20 手仍然太少，置信区间很宽，不能说明策略更强。但它说明：压缩触发 8 次后缓存仍能回到 90% 以上，没有出现 `legacy` 那种持续低命中。压缩后的第一条请求命中会掉到 82.6%，后续恢复到 93.8%。

### A. user_id 缓存隔离复核（20 手）

为了排除不同 run 之间共享 DeepSeek 缓存的影响，给两次运行分别设置了独立的 `user_id`：

- `legacy`: `ctx-a-legacy-20k-20261024`
- `compact_slim`: `ctx-a-slim-20k-20261024`

两边第一次请求的缓存命中都是 **0.0%**，说明缓存域已经隔离。结果：

| 策略 | 手数 | turn | API 调用 | 裁剪/压缩 | 缓存命中 | 首请求 | 稳态 | 输入 token | 缓存 miss | 输出 token | 费用 | 费用/turn | miss/turn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `legacy` 20k | 20 | 57 | 68 | 34 trim | 44.9% | 0.0% | 45.0% | 1,438,381 | 792,749 | 22,755 | $0.1345 | $0.00236 | 13,908 |
| `compact_slim` 20k | 20 | 48 | 50 | 1 compact | 96.9% | 0.0% | 97.2% | 623,274 | 19,498 | 11,790 | $0.0118 | $0.00025 | 406 |

同样 20 手、同一 seed、同一对手：

- `legacy` 触发了 34 次裁剪，缓存命中只有 44.9%；`compact_slim` 只压缩 1 次，缓存命中 96.9%。
- 每手第一条请求：`legacy` 37.6%，`compact_slim` 97.1%；同一手后续请求：`legacy` 48.1%，`compact_slim` 96.7%。
- 费用/turn 从 $0.00236 降到 $0.00025，约 **9.6 倍**。
- 缓存 miss/turn 从 13,908 降到 406，约 **34 倍**。
- 两边净筹码分别是 -50 和 -1305，但手数太少、行动不同，不能用来判断强弱。

这组数据基本确认了预实验的核心结论：`legacy` 的问题不是“上下文太大”本身，而是裁剪方式把缓存前缀反复打断。

### B. 50 手连续对局（seed 2027）

两组都用独立 `user_id`、20k 上限、同一 seed、同一 Project DQN 对手，各 50 手：

| 策略 | 手数 | turn | API 调用 | 裁剪/压缩 | 缓存命中 | 每手首请求 | 同手后续 | 输入 token | 缓存 miss | 输出 token | 费用 | 费用/turn | miss/turn | 净筹码 | bb/100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `legacy` 20k | 50 | 109 | 112 | 76 trim | 31.1% | 23.9% | 37.3% | 2,522,835 | 1,736,915 | 43,245 | $0.2888 | $0.00265 | 15,935 | -848 | -169.6 ± 318.8 |
| `compact_slim` 20k | 50 | 173 | 185 | 6 compact | 96.7% | 97.2% | 96.6% | 2,661,115 | 86,651 | 65,372 | $0.0599 | $0.00035 | 501 | +2,524 | +504.8 ± 836.2 |

补充观察：

- 总费用 `compact_slim` 低约 4.8 倍；按 turn 归一化低约 7.6 倍；缓存 miss/turn 低约 32 倍。
- `legacy` 在 50 手里触发 76 次裁剪，缓存命中从 20 手时的 44.9% 继续降到 31.1%。
- `compact_slim` 触发 6 次压缩（第 5、10、16、24、32、48 手），每次把约 20k 上下文压到 4k 到 9k；压缩后缓存仍保持 96% 以上。
- 行为分布明显不同：`legacy` 是 38 fold / 24 raise / 2 bet / 9 all-in，4 次摊牌；`compact_slim` 是 36 fold / 27 raise / 34 bet / 4 all-in，13 次摊牌。`compact_slim` 平均底池 292.6，`legacy` 130.1。
- 50 手结果方向性很强（+2,524 vs -848），但 95% CI 很宽，且行动不同导致后续状态不同。不能据此断定 `compact_slim` 更强，只能说明它没有出现明显退化，而且更愿意下注/跟注。
- 把 A/B 两组同策略合并（70 手）：`legacy` -128.3 ± 389.2 bb/100，`compact_slim` +174.1 ± 628.2 bb/100。方向仍然偏正，但置信区间大幅重叠，仍然不能当作胜率结论。

### P1：160k 高水位生产候选

B 组的 20k 是为了在 50 手内逼出压缩的压力设置。按新的自动预算和 160k 高水位，再用同 seed 跑 50 手 `compact_slim`：

| 指标 | 160k 高水位 | 20k 高水位（B 组） |
|---|---:|---:|
| 手数 | 50 | 50 |
| turn | 159 | 173 |
| 压缩次数 | 0 | 6 |
| 缓存命中 | 99.5% | 96.7% |
| 每手首请求命中 | 99.5% | 97.2% |
| 输入 token | 11,363,072 | 2,661,115 |
| 缓存 miss | 55,680 | 86,651 |
| 输出 token | 102,349 | 65,372 |
| 费用 | $0.1037 | $0.0599 |
| 费用/turn | $0.00065 | $0.00035 |
| miss/turn | 350 | 501 |
| 终局上下文 | 约 118k | 被压到 20k 以下 |
| 净筹码 | +1,634 | +2,524 |

- 160k 没有触发压缩，上下文自然长到 118k；缓存命中 99.5%，费用/turn 约为 20k 的 1.9 倍，但仍然明显低于 legacy。
- 20k 更便宜，但压缩 6 次，每次压缩后的第一条请求会掉缓存。
- 新默认选择 160k：给上下文留出空间，同时用自动预算避免把 1M 模型一刀切到 220k。

## 注意

- 这不是配对实验。LLM 行动不同，后续牌局状态、turn 数、手牌长度都不同，`bb/100` 和总 token 不能直接比较。上表主要看缓存命中、miss/turn 和费用/turn。
- 10 手太少，所有 arm 对 Project DQN 都是明显亏损，不能据此判断压缩是否影响胜率。
- B 组 50 手使用了独立 `user_id`，但两组行动不同、turn 数不同，净筹码和 bb/100 仍然不是严格配对比较。
- A 组之前的 10 手预实验可能共享了部分 DeepSeek 缓存；A/B/C 正式组已经用 `user_id` 隔离。
- `compact_slim` 20k 没有触发压缩，只验证了去重；`compact_slim` 10k 才验证了压缩 + 去重。
- 赛后喊话还没有优化，本轮没有覆盖。

## 下一步

1. 已完成：`LLMClient` 支持可选 `user_id`，每个 run 用独立 `--cache-scope`，20 手复核确认首请求命中 0.0%。
2. 已完成：50 手连续对局，`legacy` vs `compact_slim`，见下方 B 组结果。
3. 已完成：赛后喊话 full vs fresh，见 `docs/final_talk_preview.md`。热缓存约 5 倍差距，冷缓存约 100 倍。
4. 已完成：默认切到 `compact_slim`，`max_context_tokens_estimate=0` 自动预算，高水位 160k、低水位 50k；赛后喊话默认 `fresh`。`legacy` / `full` 只作为回退。
5. 用更大样本或固定 warm-up 的配对实验，确认 `compact_slim` 的净筹码差异是压缩效果还是 50 手方差。
6. 赛后喊话可以再试一版 `fresh + 最近 3 手摘要`，看能否在 token 仍然很低的情况下保留更多具体细节。
7. 4Router GPT-6 Luna 的实体请求在 45 秒读超时和 5 次重试下仍未完成，已手动停止；配置值已按官方文档更新，但中转连通性需要单独排查。

## 复现

```bash
python scripts/preview_context.py --policy legacy --hands 10 --seed 2026 --max-context 20000 --stream-usage
python scripts/preview_context.py --policy compact --hands 10 --seed 2026 --high-water 20000 --low-water 8000 --keep-hands 2 --stream-usage
python scripts/preview_context.py --policy compact_slim --hands 10 --seed 2026 --high-water 20000 --low-water 8000 --keep-hands 2 --stream-usage
python scripts/preview_context.py --policy full --hands 10 --seed 2026 --stream-usage
python scripts/preview_context.py --policy compact_slim --hands 20 --seed 2026 --high-water 20000 --low-water 8000 --keep-hands 2 --stream-usage
python scripts/preview_context.py --policy legacy --hands 20 --seed 2026 --max-context 20000 --stream-usage --cache-scope ctx-a-legacy-20k-20261024
python scripts/preview_context.py --policy compact_slim --hands 20 --seed 2026 --high-water 20000 --low-water 8000 --keep-hands 2 --stream-usage --cache-scope ctx-a-slim-20k-20261024
python scripts/preview_context.py --policy legacy --hands 50 --seed 2027 --max-context 20000 --stream-usage --cache-scope ctx-b-legacy-50-20261024
python scripts/preview_context.py --policy compact_slim --hands 50 --seed 2027 --high-water 20000 --low-water 8000 --keep-hands 2 --stream-usage --cache-scope ctx-b-slim-50-20261024
python scripts/preview_context.py --policy compact_slim --hands 50 --seed 2027 --high-water 160000 --low-water 50000 --keep-hands 2 --stream-usage --cache-scope p1-slim-160k
```

输出目录在 `tmp/context_preview/`，每组包含 `hands.csv`、`requests.csv`、`hands.jsonl`、`summary.json` 和 `report.md`。

## checkpoint 压缩（2026-10-25）

这一版新增了 `checkpoint` 策略，默认配置已切换过去。它和 `compact_slim` 的区别：

- 触发参考 OpenCode：`prompt_budget - max(15% × window, 16k)`；`context_high_water_tokens=0` 表示自动，1M 窗口约 856k；正数是成本上限。
- 压缩时由同一个模型在完整对话末尾写一份比赛 checkpoint，复用正常请求的 prompt 缓存。
- 摘要请求使用 `tool_choice="none"`，不执行动作工具；摘要 usage 单独记录，不混进当前 turn。
- 系统提示词、一条 checkpoint 和最近 `context_keep_recent_hands` 手原始消息保留；`context_keep_recent_tokens` 是初始保留上限。压缩后如果低于低水位，会把已经总结过的更早手牌原样加回来，直到接近 `context_low_water_tokens`，所以最终原始区可能超过 `keep_recent_tokens`，但不会超过低水位。
- 压缩时由代码自动调用历史统计并注入摘要提示词，但 checkpoint 只写结论、关键手牌和调整；统计数字留给 `research_history`，不要复制统计表或 JSON。
- 摘要失败先写确定性兜底 checkpoint；连续失败 2 次回退 `compact_slim`。
- 摘要请求被 `context_length` 拒绝时，先试完整前缀，再试保留尾巴之前的历史，最后扁平化并按 85/70/50/35% 缩小。

复现：

```bash
python scripts/preview_context.py --policy checkpoint --hands 10 --seed 2026 \
  --high-water 0 --low-water 50000 --keep-hands 3 --keep-tokens 20000 --stream-usage
```

输出仍在 `tmp/context_preview/`。重点看：

- `context_summary_end` 的摘要调用 input/output/cache token；
- `compactions` 和 `summary_calls`；
- 压缩后第一条请求的缓存命中；
- fallback 次数；
- 平均每手费用和终局上下文。

目前还没有 checkpoint 的长样本实测数据，不要把默认值和费用当成已验证结论。
