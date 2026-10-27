# 项目规则 RL Bot 训练

这套训练流程直接驱动 `src/engine/heads_up.py`，不再使用 RLCard 的 `no-limit-holdem` 环境。原因是 RLCard 的双人规则与本项目不同：

- RLCard 的 `dealer` 座位实际是大盲，小盲在翻前和翻后都先行动。
- RLCard 固定 1/2 盲注，`chips_for_each=1000` 实际是 500bb。
- RLCard 没有严格最小加注和短码全下不重开。

本项目的规则是标准单挑：按钮=小盲，翻前按钮先行动，翻后大盲先行动，按钮每手轮换，最小加注为 `current_bet + last_full_raise`，短码全下不重开。

## 观测

`src/training/obs.py` 输出 73 维向量：

- 0..51：52 张牌 multi-hot，包含自己底牌和公共牌。
- 52..59：底池、双方筹码、双方本街投入、跟注额、最小/最大加注，全部除以大盲。
- 60..63：街道 one-hot。
- 64：是否按钮位。
- 65..69：上一个动作类型 one-hot。
- 70：上一个动作本次投入的筹码 / 动作后底池（v2；旧 checkpoint 按 v1 使用 raise-to 总额）。
- 71..72：上一个行动者相对位置。

观测始终以当前行动者为 hero，因此一个网络可以同时学两个座位。合法动作掩码单独返回，不拼进观测。

## 动作

`src/training/actions.py` 使用 10 个 slot：

```text
0 fold
1 check
2 call
3 min-raise
4 1/3 pot
5 1/2 pot
6 3/4 pot
7 pot
8 1.5x pot
9 all-in
```

底池比例加注使用标准公式：

```text
called_target = 自己本街投入 + 跟注额
pot_after_call = 底池 + 跟注额
raise_to = called_target + round_half_up(比例 * pot_after_call)
```

低于最小加注的尺寸直接屏蔽，不做夹取。只有 all-in 允许不足最小加注。效果相同的尺寸会去重。

## 环境

`src/training/env.py` 的 `ProjectHeadsUpEnv`：

- 每手重置到固定筹码，默认 1000。
- 按钮自动轮换，也可以显式指定。
- `reset(seed, button)` 创建新牌局，便于同牌换座评估。
- `play_hand(agents, training)` 返回双方终端奖励和每个座位的 transition。
- `feed_transitions` 把终端净筹码作为最后一条 transition 的奖励，中间奖励为 0。
- training 模式下 transition 的 `next_state` 连接**同一玩家**的下一次决策，最后一次决策连接到 terminal state。引擎每步都会把行动权交给对手，所以不能直接把 `step()` 返回的下一行动者状态写进当前玩家的 transition。

奖励默认按 `起始筹码 / 5` 缩放，满筹码大约 ±5。

## 训练

DQN：

```bash
python scripts/train_project_bot.py \
  --algorithm dqn \
  --hands 100000 \
  --opponent mix \
  --random-prob 0.2 \
  --output models/rlcard/nlh_dqn_project.pth
```

NFSP：

```bash
python scripts/train_project_bot.py \
  --algorithm nfsp \
  --hands 100000 \
  --opponent mix \
  --random-prob 0.2 \
  --output models/rlcard/nlh_nfsp_project.pth
```

常用参数：

- `--eval-every`：每多少手评估一次，默认 10000。
- `--eval-hands`：每次评估多少手，必须是偶数。
- `--opponent random|self|mix`：对手类型。
- `--device cpu|cuda`：网络设备，环境模拟仍在 CPU。
- `--mlp 128 128`：隐藏层。

训练日志写到 `tmp/training/<algorithm>_<timestamp>/`，包含 `hands.csv`、`eval.csv`、`manifest.json` 和 `checkpoints/`。

## 模型格式

训练脚本使用 `save_project_checkpoint` 保存安全格式：

```python
{
    "format": "project-bot-v1",
    "algorithm": "dqn" | "nfsp",
    "obs_dim": 73,
    "obs_version": 2,
    "num_actions": 10,
    "mlp_layers": [...],
    "state_dict": {...},
    "metadata": {...},
}
```

推理用 `src/agents/project_bot.py` 的 `ProjectBot`，`torch.load(..., weights_only=True)`，不加载完整 Python 对象。NFSP checkpoint 默认 `sample=True`，与训练中的平均策略评估一致；DQN 使用 argmax。加载时如果 checkpoint 的算法与请求的 `dqn`/`nfsp` 不符会直接报错。旧 checkpoint 没有 `obs_version` 时按 v1 推理；新训练写入 v2，最后一个动作尺寸特征改用本次投入的筹码，跨版本指标不要直接比较。

旧的 `RLCardBot` 仍然支持 54 维 / 5 动作的 RLCard checkpoint，但不属于这套项目规则训练流程。

## 评估

离线矩阵：

```bash
python scripts/eval_project_bots.py --hands 2000
python scripts/eval_project_bots.py --hands 20000 --output tmp/project_eval/main
```

训练中间的 checkpoint 可以用更大的样本重新挑选：

```bash
python scripts/select_project_checkpoint.py \
  --algorithm dqn \
  --checkpoint-dir tmp/training/dqn_100k/checkpoints \
  --hands 10000
```

它会把最稳的 checkpoint 复制到 `models/rlcard/nlh_dqn_project_selected.pth`。

评估使用同牌换座，报告：

- `a_mean_bb100`：A 的每百手大盲收益。显式写 `dqn_vs_random` 时 A 是 DQN；写 `random_vs_dqn` 时 A 是 Random，符号会反过来。
- `a_ci95_bb100`：按 deal 配对的 95% 置信区间。
- 胜率、平局率、摊牌率、全下率、平均底池、平均动作数。
- `zero_sum_ok`：每手净筹码是否守恒。

2000 手适合粗筛。要分辨 10bb/100 以内的差距，需要 20k 到 200k 手，具体取决于实际每手方差。

## 本次实跑结果（2026-10-21）

- DQN：100k 训练约 55 分钟；10k 手重选后使用 40k checkpoint。
- NFSP：训练到约 72k 时命令超时；10k 手重选后使用 70k checkpoint。
- 最终 20k 手/组合，同牌换座：

| 组合 | A 的 bb/100 | 95% CI | A 胜率 |
|---|---:|---:|---:|
| DQN vs Random | +1395.6 | ±67.9 | 67.6% |
| NFSP vs Random | +1053.5 | ±68.1 | 65.3% |
| DQN vs NFSP | +550.6 | ±63.5 | 52.6% |
| DQN vs DQN | 0.0 | ±0.0 | 48.1% |
| NFSP vs NFSP | 0.0 | ±0.0 | 48.7% |
| Random vs Random | 0.0 | ±0.0 | 49.5% |

所有组合的筹码守恒检查通过。自对弈均值为 0，说明同牌换座和座位平衡正常。DQN 目前明显强于 NFSP，但两者都远强于 RandomAgent。
