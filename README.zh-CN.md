# Heads-Up Lab · 二人德州扑克实验室

[English](README.md) · **简体中文**

一个专门为 **单挑（Heads-Up）无限注德州扑克** 做的实验台：人类、LLM、RLCard 三者任选其二上桌。它把重点放在把 LLM 的**思考、工具调用、参数与结果**完整地摊开在牌桌旁边。

- 游戏引擎：严格双人规则，Treys 牌力评估
- Agent：每次行动发一条新的牌局消息，保留上下文；五个基础工具：牌型概率 + 胜率 + 随机数 + 跟注 EV + 执行动作
- 桌边喊话：LLM vs LLM 时双方每个行动回合都必须喊话一次，默认把话术写进 `apply_action.table_talk`，也可以切回旧的 `send_table_message` 两步模式；比赛结束后还可以根据结果再说一句，可以祝贺、复盘或嘲讽；对手下一次决策时看到不可信标记，人类对 LLM 暂不开放
- 对局控制：设置框可以随时关闭，运行中的对局可以从设置框“中止本局”；客户端请求有超时，不会因为请求挂住把弹窗锁死
- 可视化：实时思考流、工具卡片、Token/延迟、最终牌型分布、胜率/跟注 EV、双方对话
- 上帝视角：双方底牌、烧牌、完整牌堆
- LLM：DeepSeek / 火山引擎 / MiniMax / Qwen / 智谱 GLM / Kimi / LongCat / MiMo / 4Router
- Bot：RLCard（RandomAgent 开箱可用，支持加载 DQN/NFSP/DMC checkpoint）
- 语言：界面首次打开默认英文，可在顶栏或对局配置里切换中文 / EN，选择会记住；`?lang=zh-CN` 或 `?lang=en` 可以临时覆盖。牌局和 API 校验错误会跟随界面语言。Agent 语言在开局时单独记录，控制该局的系统提示词、工具说明、工具返回和桌边喊话。

---

## 界面预览

### 牌桌与 Agent 观测台

左侧是单挑牌桌、公共牌、底池和行动记录；右侧是 Agent 观测台，实时展开思考流、工具卡片、日志和桌边喊话。

[![牌桌与 Agent 观测台](docs/screenshots/table.png)](docs/screenshots/table.png)

### 对局配置

人类、LLM、RLCard 任选其二，按座位选择 Provider、模型和推理参数。

[![对局配置](docs/screenshots/setup.png)](docs/screenshots/setup.png)

### 思考与工具调用

每次行动都能看到模型思考、跟注 EV、胜率估算和工具返回，不需要打开调试器。

[![思考与工具调用](docs/screenshots/agent-console.png)](docs/screenshots/agent-console.png)

### 牌型分布

最终牌型分布按当前最大概率归一化，并标出样本数、耗时和估算方法。

[![牌型分布](docs/screenshots/analysis.png)](docs/screenshots/analysis.png)

---

## 快速开始

### 1. Python 依赖

```bash
pip install -r requirements.txt
```

需要 Python 3.11+。当前项目在 Python 3.13 + Windows 下开发。

### 2. 配置 Key

```bash
copy .env.example .env        # Windows
cp .env.example .env          # macOS / Linux
```

把对应厂商的 Key 填入 `.env`：

```dotenv
DEEPSEEK_API_KEY=
DOUBAO_API_KEY=
MINIMAX_API_KEY=
QWEN_API_KEY=
GLM_API_KEY=
KIMI_API_KEY=
LONGCAT_API_KEY=
MIMO_API_KEY=
FOURROUTER_API_KEY=
```

`.env` 已在 `.gitignore` 中，不会入库。页面上只显示“Key 已配置/缺失”，永远不会把 Key 发到浏览器。

### 3. 构建前端

```bash
cd frontend
npm install
npm run build
cd ..
```

构建产物在 `static/dist/`，由 Flask 直接托管。

### 4. 启动

```bash
python main.py
# 或双击 start.bat（会自动检测并构建前端）
```

打开 <http://127.0.0.1:5050>。

开发模式（前端热更新）：

```bash
python main.py                 # 终端 1，:5050
cd frontend && npm run dev     # 终端 2，:5173，代理到 :5050
```

---

## 功能说明

### 对局配置

- 两个座位，各自选择 **人类 / LLM / RLCard**
- 不允许两个人类同时上桌
- Bot 座位可设置行动延迟（0 到 10 秒，默认 3 秒），方便看清每次动作
- 可配置：起始筹码、小盲/大盲、自动重购、重购筹码、最大手数、随机种子、上帝视角
- 快速预设：人 vs DeepSeek、DeepSeek vs GLM、人 vs RLCard

### 上帝视角

开启后：

- 双方底牌可见
- 烧牌可见
- 剩余牌堆按“下一张在左”的顺序完整展示
- 可以对任意玩家计算牌型分布

上帝视角**只影响观战界面**。Agent 的输入里没有对手底牌、没有牌堆。

### LLM Agent

每次轮到 LLM：

1. 追加一条新的 `【牌局更新】` 用户消息，包含底牌、公共牌、底池、双方筹码、需跟注、最小/最大加注、合法动作、本手行动历史、最近几手结果。
2. Agent 保留之前的完整对话（跨手牌、跨街道）。上下文过长时按“回合边界”压缩最早的回合，保留系统提示与最近回合。
3. `estimate_hand_probabilities` 是可选工具，返回最终牌型分布。默认不强制；打开 `require_estimate_first` 后，每条街第一次 `apply_action` 前会软提醒一次（不计入失败）。
4. `estimate_equity` 返回当前摊牌 equity（`胜率 + 平局率/2`），默认假设对手随机手牌。河牌和转牌精确枚举，翻牌和翻前各 20000 次蒙特卡洛，并走 LRU 缓存。
5. `call_ev` 用当前底池、跟注额和筹码计算静态跟注 EV。`equity` 参数可以省略，省略时自动估算；短码全下会先扣掉对手匹配不上的退款。
6. 如果分析后仍难以在几个接近的方案中取舍，Agent 可以调用 `get_random_number` 抽一个 `[0, 1)` 随机数。它只是打破僵局的备选方案，不能替代分析。
7. LLM vs LLM 且开局启用「桌边喊话」时，Agent 每个行动回合都必须喊话一次。默认 `inline` 模式把话术写进 `apply_action.table_talk`，一次请求完成“说 + 做”；旧的 `tool` 模式先调用 `send_table_message`，再调用 `apply_action`，通常要两次请求。缺少 `table_talk` 会被拒绝。比赛结束后，每个 LLM 还可以根据结果说一句赛后话，可以祝贺、复盘、调侃或嘲讽。对手在下一次决策时看到，内容默认不可信。系统只限制长度、格式和频率，不核查真伪。
8. `apply_action` 合法就进入下一回合；不合法会把错误、最新约束和合法动作回传给模型，让它重试。
9. 连续三次无效动作：自动 **call**；无法 call 则 **fold**。网络/超时等非动作失败则优先 check/call，避免无意义弃牌。
10. 厂商要求回传 `reasoning_content` 的（DeepSeek / Qwen / MiniMax / Kimi / GLM 等）会原样回传；不支持的可选参数在 400 时会自动降级重试。

#### 工具一览

| 工具 | 参数 | 用途 |
|---|---|---|
| `estimate_hand_probabilities` | `iterations` 可选，2000 到 50000，默认 30000 | 估算最终最佳五张牌的牌型分布。翻牌起精确枚举，翻牌前蒙特卡洛。不是胜率。 |
| `estimate_equity` | 无 | 估算对随机手牌的摊牌 equity（`胜率 + 平局率/2`）。 |
| `get_random_number` | 无 | 返回 `[0, 1)` 均匀随机数，只用于打破接近的决策。 |
| `call_ev` | `equity` 可选，0 到 1 | 计算静态跟注 EV、盈亏平衡胜率和有效底池；省略时自动估算。 |
| `apply_action` | `action` 必填（`fold/check/call/bet/raise/all_in`）；`amount` 仅 bet/raise；inline 喊话模式还有 `table_talk` | 每回合恰好提交一次最终动作。`amount` 是本街总额（raise-to），不是增量。inline 模式下每回合第一次动作必须带 `table_talk`。 |
| `send_table_message` | `text` 必填 | 仅 LLM vs LLM 的桌边喊话；只在旧的 `tool` 模式暴露。inline 模式的喊话放在 `apply_action.table_talk`。 |

`send_table_message` 只在 `tool` 模式暴露；`inline` 模式把喊话放在 `apply_action` 里。其余五个是核心工具集。工具事件与返回字段见 [`docs/protocol.md`](docs/protocol.md)。

### 离线 LLM 对 LLM 评测

`scripts/eval_llm_vs_llm.py` 支持两个座位使用不同的工具集和系统提示：

```bash
python scripts/eval_llm_vs_llm.py --hands 100 --workers 5
```

默认对照组只有 `apply_action`，工具组按 `ALL_TOOL_NAMES` 提供 `estimate_hand_probabilities`、`estimate_equity`、`get_random_number`、`call_ev` 与 `apply_action`，并要求公共牌少于五张时每次行动前至少估算一次。每手独立重置筹码，同牌换座，结果写到 `tmp/llm_vs_llm/`。

### 数据分析

牌型分布工具显示**最终最佳五张牌的牌型分布**（皇家同花顺、同花顺、四条、葫芦、同花、顺子、三条、两对、一对、高牌）；胜率工具显示胜/平/负和采样误差，跟注 EV 工具显示 EV、盈亏平衡胜率和有效底池。

- 翻牌前：蒙特卡洛（默认 30,000 次）
- 翻牌：精确枚举 C(47,2)=1,081 种
- 转牌：46 种；河牌：1 种
- 结果带：当前最佳牌型、方法、样本数、耗时、绝对概率
- 这不是胜率；胜率还取决于对手范围与行动

### RLCard Bot

- 观察编码严格匹配 RLCard `no-limit-holdem` 的 54 维向量与 5 个动作 ID
- 支持 `random` / `dqn` / `nfsp` / `dmc`
- RLCard 双人局的翻后行动顺序与本项目引擎不同（RLCard 按钮/小盲先行动，本项目大盲先行动）。RLCard checkpoint 直接放进本项目会有位置偏差；比较算法强弱建议在 RLCard 原生环境里跑。
- 默认 RandomAgent 不需要模型文件；DQN 等需要在配置里选择模型
- 训练/加载 DQN 需要 PyTorch：`pip install -r requirements-optional.txt`
- 训练：

```bash
python scripts/train_rlcard.py --episodes 50000 --chips 1000 --reward-scale 0.01
```

输出 `models/rlcard/nlh_dqn.pth`，在 UI 中选 **RLCard → DQN → 模型文件** 即可。详见 [`models/rlcard/README.md`](models/rlcard/README.md)。

### 项目规则 RL Bot

- 使用本项目 `HeadsUpGame` 的规则训练，73 维观测，10 个动作槽（fold/check/call/min-raise/1/3、1/2、3/4、pot、1.5x pot/all-in）。
- DQN / NFSP：

```bash
python scripts/train_project_bot.py --algorithm dqn --hands 100000 --opponent mix
python scripts/train_project_bot.py --algorithm nfsp --hands 100000 --opponent mix
```

- 离线矩阵评估：

```bash
python scripts/eval_project_bots.py --hands 2000
```

模型输出到 `models/rlcard/nlh_dqn_project.pth` / `nlh_nfsp_project.pth`。UI 里选 **Project DQN** 或 **Project NFSP**。训练细节见 [`docs/training.md`](docs/training.md)。

> 说明：仓库里的强化学习模型是实验训练产物，我个人觉得目前表现比较糟糕，仅供参考，不代表强 Bot 水平。

---

## LLM 接入

所有厂商走 OpenAI 兼容的 `/chat/completions`，但参数按厂商分别适配。

| Provider | Base URL | 模型 | Key | 备注 |
|---|---|---|---|---|
| DeepSeek | `https://api.deepseek.com` | `deepseek-flash` | `DEEPSEEK_API_KEY` | 1M 上下文，`thinking` + `reasoning_effort`，工具回合需回传 reasoning_content |
| 火山引擎 | `https://ark.cn-beijing.volces.com/api/v3` | `doubao-seed-2-1-turbo-260628` | `DOUBAO_API_KEY` | `thinking.type` 开关，OpenAI 兼容 |
| MiniMax | `https://api.minimaxi.com/v1` | `MiniMax-M3.1-Flash` | `MINIMAX_API_KEY` | 始终思考，忽略 temperature/top_p，支持 `reasoning_effort` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen3.8-flash` | `QWEN_API_KEY` | `enable_thinking`，`preserve_thinking`，思考模式温度 ≥0.6 |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-5.3-flash` | `GLM_API_KEY` | 思考常开，`reasoning_effort` low/high/max，流式工具调用 |
| Kimi | `https://api.moonshot.cn/v1` | `kimi-k2.7-code` | `KIMI_API_KEY` | 强制思考，262K 上下文，工具调用 |
| LongCat | `https://api.longcat.chat/openai/v1` | `LongCat-2.5-Preview` | `LONGCAT_API_KEY` | `thinking.type` 开关，1M 上下文，原生工具 |
| MiMo | `https://api.xiaomimimo.com/v1` | `mimo-v2.6-flash` | `MIMO_API_KEY` | `thinking.type` 开关，1M 上下文，回传 reasoning_content |
| 4Router | OpenAI 兼容接口 | `gpt-6-luna` | `FOURROUTER_API_KEY` | 始终推理，用 `reasoning_effort` 调档；不回传思考正文 |

参数不是“越保守越好”：默认 `max_tokens` 为 **32,768**，各家允许的上限不同，UI 里会显示范围与建议值。默认温度按厂商推荐设置（如 DeepSeek/GLM 1.0、Qwen 0.7、MiniMax 忽略）。

每个模型旁有 **测试连接** 按钮，会发一个很短的请求并显示延迟。

---

## 项目结构

```
main.py                       # 启动入口
config/
  game.json                   # 规则默认值 / 限制 / LLM 运行时
  providers.json              # 9 家 Provider、模型、参数控件
src/
  engine/                     # Card / Deck / Treys 评估 / 双人状态机
  analysis/                   # 牌型分布（精确枚举 + 蒙特卡洛 + 缓存）
  agents/                     # AgentView、RLCardBot
  llm/                        # providers / streaming client / prompt / tool loop
  server/                     # Flask + Socket.IO + GameSession 线程
frontend/                     # Vue 3 + Vite + TypeScript
  src/components/             # 牌桌、Agent 控制台、分析、配置弹窗
  src/store.ts                # socket + 增量事件 reducer + 状态同步
static/dist/                  # 前端构建产物（gitignore）
tests/                        # 引擎 / 分析 / LLM 工具循环 / RLCard 适配
scripts/train_rlcard.py       # DQN 训练
scripts/eval_llm_vs_bot.py    # LLM vs Project Bot 离线评测
scripts/eval_llm_vs_llm.py    # LLM vs LLM 工具使用对照
models/rlcard/                # checkpoint 目录（文件 gitignore）
```

离线 LLM 评测默认使用中文 Agent 提示词。给 `eval_llm_vs_bot.py`、`eval_llm_vs_llm.py` 或 `eval_llm_tool_matrix.py` 加 `--agent-locale en-US`，即可使用英文提示词和英文工具层。

---

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
cd frontend && npm run typecheck && npm test && npm run build
```

当前测试覆盖：

- 盲注/位置、最小加注、短码全下不重开下注、对手全下后禁止加注、弃牌退款、摊牌平分、随机 60 手筹码守恒
- 牌型分布精确/蒙特卡洛、重复牌拒绝、缓存座位键与 LRU
- LLM 工具循环、SSE 分块、工具调用合并、非流式降级、断线保留、取消、错误脱敏
- 服务器建局原子性、严格规则校验、JSON 类型校验
- RLCard 动作合法性与 54 维 observation 编码一致

---

## 已知限制

- 单进程、内存态，只维护一桌；重启服务后对局丢失
- 没有登录/权限，默认只监听 `127.0.0.1`，不要直接暴露到公网
- Socket.IO 默认只接受本机 `5050` 与 `5173` 来源；需要其他端口或主机时设置 `HU_ALLOWED_ORIGINS`
- God view 是观战功能，不会进入 Agent 上下文
- LLM 的 `reasoning_content` 是厂商原始输出；不同模型可能用英文思考
- RLCard 短训练容易退化；默认用 RandomAgent 验证链路，训练模型请自行评估
- 模型文件是 PyTorch pickle，只加载可信来源；模型路径限制在 `models/rlcard/` 下
- LLM 调用会产生厂商费用；「测试连接」也会发一个很小的真实请求
- 不做对手范围推断；胜率和跟注 EV 默认假设对手是随机手牌

---

## 技术栈与致谢

- [Treys](https://github.com/ihendley/treys)：TwoPlusTwo 查表牌力评估
- [RLCard](https://github.com/datamllab/rlcard)：强化学习扑克环境与 agent
- [DeepSeek V4.1 Flash](https://www.deepseek.com)：本项目的 AI 开发助手，参与代码、调试与文档撰写
- [OpenCode](https://opencode.ai)：提供 AI Agent 运行环境与开发工具支持
- Flask / Flask-SocketIO / Vue 3 / Vite / TypeScript
- Provider 图标是各厂商的品牌标识，仅用于界面识别，版权归各自厂商所有

## 许可证

MIT，详见 [LICENSE](LICENSE)。第三方品牌图标和第三方模型不在 MIT 授权范围内。

人类牌桌快捷键：**F** 弃牌、**C** 过牌/跟注、**R** 聚焦加注、**A** 全下。
