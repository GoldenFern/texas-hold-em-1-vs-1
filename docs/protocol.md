# 通信协议

所有 REST 路径以 `/api` 开头；Socket.IO 路径为 `/socket.io`。服务端默认 `127.0.0.1:5050`。Socket.IO 默认只接受 `127.0.0.1/localhost` 的 `5050` 与 `5173` 来源，可用 `HU_ALLOWED_ORIGINS` 覆盖。厂商错误文本在返回前会脱敏，Key 值不会进入响应。

## REST

### `GET /api/bootstrap`

返回 Provider/模型元数据、Key 是否配置、牌局默认值与限制、RLCard 可用性与模型文件列表。

关键字段：

```json
{
  "providers": [
    {
      "id": "deepseek",
      "name": "DeepSeek",
      "icon": "deepseek.svg",
      "accent": "#5B7CFA",
      "base_url": "https://api.deepseek.com",
      "api_key_env": "DEEPSEEK_API_KEY",
      "models": [
        {
          "provider_id": "deepseek",
          "provider_name": "DeepSeek",
          "icon": "deepseek.svg",
          "accent": "#5B7CFA",
          "base_url": "https://api.deepseek.com",
          "api_key_env": "DEEPSEEK_API_KEY",
          "key_present": true,
          "docs_url": "https://api-docs.deepseek.com",
          "id": "deepseek-flash",
          "name": "DeepSeek V4.1 Flash",
          "tagline": "1M 上下文 · 原生工具调用",
          "context_window": 1000000,
          "max_output": 131072,
          "defaults": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 32768, "thinking": true, "reasoning_effort": "high"},
          "controls": {
            "temperature": {"show": true, "min": 0, "max": 2, "step": 0.1, "hint": "..."},
            "thinking": {"mode": "toggle", "label": "深度思考", "hint": "..."}
          }
        }
      ]
    }
  ],
  "game": {"defaults": {}, "limits": {}, "llm_runtime": {}},
  "rlcard": {"available": true, "agents": [], "models": [], "model_dir": "models/rlcard"}
}
```

Key 值永远不会返回；只有 `key_present`。

### `GET /api/game/state`

返回当前 `GameSnapshot`。没有对局时 `has_match=false, match=null`。

### `GET /api/game/transcripts`

```json
{
  "match_id": "ab12...",
  "seats": {"0": [/* AgentEvent */], "1": []}
}
```

每个座位最近最多 6000 条事件；超过后丢弃最早 1000 条。

### `POST /api/game/new`

```json
{
  "seats": [
    {
      "kind": "llm",
      "name": "DeepSeek",
      "provider_id": "deepseek",
      "model_id": "deepseek-flash",
      "params": {"temperature": 1.0, "max_tokens": 32768, "thinking": true, "reasoning_effort": "high"}
    },
    {
      "kind": "human",
      "name": "你"
    }
  ],
  "rules": {
    "starting_chips": 1000,
    "small_blind": 5,
    "big_blind": 10,
    "auto_rebuy": false,
    "rebuy_amount": 1000,
    "max_hands": 0,
    "blind_increase_every_hands": 0,
    "blind_increase_factor": 2.0,
    "seed": null,
    "god_view": true
  },
  "table_talk": true
}
```

`blind_increase_every_hands` 默认 0，表示关闭；大于 0 时，每完成 N 手，在下一手发牌前按 `blind_increase_factor` 提高大小盲，并保持原来的大小盲比例。`blind_level` 会在状态快照中递增。

`kind` 只能是 `human` / `llm` / `bot`；两个 `human` 返回 400。`table_talk` 必须是布尔值；只有两个座位都是 `llm` 时才真正生效，其他组合会静默关闭并在 `match.table_talk` 返回 `false`。喊话投递方式由服务端 `config/game.json` 的 `llm_runtime.table_talk_mode` 决定，默认 `inline`；快照里的 `match.table_talk_mode` 会返回实际值（`inline` / `tool` / `off`）。
Bot 座位：

```json
{"kind": "bot", "name": "RLCard", "bot": {"agent_type": "random", "model_path": null}}
```

`agent_type`: `random` / `dqn` / `nfsp` / `dmc`。非 `random` 需要 `model_path`。

返回 `{"ok": true, "state": GameSnapshot}` 或 `{"ok": false, "error": "..."}`（400/500）。

### `POST /api/game/action`

仅人类回合：

```json
{"action": "raise", "amount": 40}
```

`amount` 对 `bet` / `raise` 是**本街总额（raise-to）**。`fold/check/call/all_in` 忽略 `amount`。动作历史里的 `added` 始终表示该动作本次投入的筹码；`amount` 在跟注时是本次投入，在下注/加注时是本街总额。

成功返回 `{"ok": true, "state": ...}`；非法返回 400 与 `{"ok": false, "error": "..."}`，人类需要重新提交。

### `POST /api/game/next-hand`

当前手已结束且比赛未结束时开始下一手。返回新状态。

### `POST /api/game/god-view`

```json
{"enabled": true}
```

只改变观战快照，不影响 Agent 输入。

### `POST /api/game/analyze`

```json
{"seat": 1, "iterations": 30000}
```

返回：

```json
{
  "ok": true,
  "analysis": {
    "hero_seat": 1,
    "method": "exact",
    "iterations": 1081,
    "elapsed_ms": 11.7,
    "total": 1081,
    "distribution": [
      {"key": "one_pair", "name": "一对", "short": "一对", "color": "#B9A58A", "count": 528, "probability": 0.4884}
    ],
    "current_best": {"category": "high_card", "category_name": "高牌", "score": 6418, "best_five": ["Ah", "Qs", "9d", "8c", "3d"]},
    "known_cards": ["9d", "Qs", "8c", "3d", "Ah"],
    "needed_board_cards": 2,
    "hand_id": 1,
    "street": "flop",
    "seat": 1
  }
}
```

非人类座位需要 `god_view=true` 才能分析。

### `POST /api/llm/test`

```json
{"provider_id": "deepseek", "model_id": "deepseek-flash", "params": {"max_tokens": 2048}}
```

发一个很短的连通性请求，返回 `text`、`latency_ms`、`usage`。不写数据库、不影响当前对局。

### `GET /api/health`

`{"ok": true, "version": "0.1.0"}`

---

## GameSnapshot

顶层是当前手牌字段，`match` 是对局元信息。关键字段：

```json
{
  "has_match": true,
  "seq": 42,
  "hand_id": 3,
  "street": "flop",
  "pot": 160,
  "current_bet": 40,
  "to_call": 40,
  "min_raise_to": 80,
  "max_raise_to": 920,
  "button": 0,
  "acting_seat": 0,
  "acting_kind": "llm",
  "legal_actions": ["fold", "call", "raise", "all_in"],
  "constraints": {"to_call": 40, "call_amount": 40, "min_raise_to": 80, "max_raise_to": 920, "can_raise": true, "raise_allowed": true, "pot": 160},
  "community": [{"code": "Ah", "rank": "A", "suit": "h", "symbol": "♥", "red": true, "label": "A♥"}],
  "burned": [],
  "deck_remaining": [],
  "deck_remaining_count": 45,
  "players": [
    {
      "seat": 0,
      "name": "DeepSeek",
      "kind": "llm",
      "chips": 920,
      "status": "active",
      "street_bet": 0,
      "total_bet": 80,
      "is_button": true,
      "is_small_blind": true,
      "is_big_blind": false,
      "is_acting": true,
      "hole_cards": [{"code": "9c", "label": "9♣", "rank": "9", "suit": "c", "symbol": "♣", "red": false}],
      "hole_hidden": false,
      "assessment": {"category": "one_pair", "category_name": "一对", "score": 4656, "best_five": ["Ah", "Qh", "Jh", "8c", "8h"]},
      "last_action": {"type": "raise", "label": "加注", "amount": 40, "pot_after": 120, "source": "llm", "auto": false},
      "rebuy_count": 0
    }
  ],
  "actions": [{"hand_id": 3, "street": "flop", "seat": 0, "type": "raise", "label": "加注", "amount": 40, "added": 40, "pot_after": 120, "chips_after": 920, "source": "llm", "auto": false}],
  "result": null,
  "match_over": false,
  "table_talk": [
    {
      "id": "7f3a...",
      "seq": 1,
      "hand_id": 3,
      "street": "flop",
      "seat": 0,
      "name": "DeepSeek",
      "text": "我这次不会再让你便宜看牌。",
      "turn_id": "1d3e...",
      "created_at": 1760000000.0,
      "kind": "turn"
    }
  ],
  "match": {
    "id": "ab12...",
    "hands_played": 2,
    "max_hands": 0,
    "match_over": false,
    "aborted": false,
    "god_view": true,
    "table_talk": true,
    "table_talk_mode": "inline",
    "can_next_hand": false,
    "rules": {"starting_chips": 1000, "small_blind": 10, "big_blind": 20, "starting_small_blind": 5, "starting_big_blind": 10, "blind_level": 1, "auto_rebuy": false, "rebuy_amount": 1000, "max_hands": 0, "blind_increase_every_hands": 10, "blind_increase_factor": 2.0, "seed": 123},
    "seats": [{"seat": 0, "name": "DeepSeek", "kind": "llm", "provider_id": "deepseek", "model_id": "deepseek-flash", "model_name": "DeepSeek V4.1 Flash", "icon": "deepseek.svg", "agent": {}}],
    "history": [],
    "last_result": null
  }
}
```

- `hole_hidden=true` 时 `hole_cards` 为空数组。
- `deck_remaining` / `burned` 只在 `god_view=true` 时返回。
- `table_talk` 是最近 40 条喊话，可能来自之前几手；前端按 `id` 去重、按 `seq` 排序。
- `seq` 是状态版本；前端丢弃同 match 下更小的 seq。

---

## AgentEvent

Socket.IO 事件 `agent_event` 的 payload。所有事件带：

```json
{"match_id": "ab12...", "seat": 0, "event_id": 123, "type": "..."}
```

### `turn_start`

```json
{
  "type": "turn_start",
  "turn_id": "1d3e...",
  "hand_id": 3, "hand_number": 3,
  "street": "flop",
  "pot": 160, "my_chips": 920, "opponent_chips": 920,
  "to_call": 40, "legal_actions": ["fold", "call", "raise", "all_in"],
  "position": "button",
  "table_talk_received": 1
}
```

`table_talk_received` 是本次决策前投递给该座位的未读喊话条数。

### `turn_state`

发送给 LLM 的完整牌局输入（UI 可展开查看）：

```json
{"type": "turn_state", "turn_id": "...", "text": "【牌局更新 · 第 3 手 · 翻牌】\n..."}
```

### `message_start`

```json
{"type": "message_start", "turn_id": "...", "message_id": "...", "step": 1}
```

### `llm_request`

```json
{
  "type": "llm_request", "turn_id": "...", "message_id": "...", "step": 1,
  "message_count": 12, "estimated_context_tokens": 6420
}
```

`estimated_context_tokens` 是发送前的粗略估算，用于 UI 展示上下文规模。

### `reasoning_delta` / `content_delta`

```json
{"type": "reasoning_delta", "turn_id": "...", "message_id": "...", "delta": "..."}
{"type": "content_delta", "turn_id": "...", "message_id": "...", "delta": "..."}
```

### `tool_call_start` / `tool_call_args_delta`

```json
{"type": "tool_call_start", "turn_id": "...", "message_id": "...", "index": 0, "call_id": "call_0", "name": "apply_action"}
{"type": "tool_call_args_delta", "turn_id": "...", "message_id": "...", "index": 0, "call_id": "call_0", "delta": "{\"action\":\"raise\","}
```

### `tool_call`

完整参数（流结束后）：

```json
{
  "type": "tool_call", "turn_id": "...", "message_id": "...",
  "index": 0, "call_id": "call_0", "name": "apply_action",
  "arguments": "{\"action\":\"raise\",\"amount\":80}",
  "parsed": {"action": "raise", "amount": 80},
  "parse_error": false
}
```

### `tool_result`

```json
{
  "type": "tool_result", "turn_id": "...", "index": 0,
  "call_id": "call_0", "name": "apply_action",
  "ok": true, "ignored": false,
  "result": {"ok": true, "accepted": true, "action": "raise", "amount": 80},
  "summary": "", "duration_ms": 0.9
}
```

估算工具 `result` 会包含完整 `distribution` / `current_best` / `method` / `total` / `elapsed_ms`，供前端画图。

胜率工具的 `result`：

```json
{
  "ok": true,
  "street": "turn",
  "method": "exact",
  "samples": 45540,
  "equity": 0.6368,
  "win": 0.6123,
  "tie": 0.0491,
  "lose": 0.3386,
  "std_error": 0.0,
  "opponent": "random",
  "cached": false
}
```

`equity` 是 `win + tie/2`，所有概率都是 0 到 1 的小数。

跟注 EV 工具的 `result`：

```json
{
  "ok": true,
  "street": "river",
  "equity_source": "internal",
  "equity_method": "exact",
  "equity_samples": 990,
  "equity": 0.6368,
  "ev_call": 12.4,
  "ev_fold": 0.0,
  "breakeven_equity": 0.25,
  "pot": 150,
  "to_call": 50,
  "call_amount": 50,
  "effective_pot": 200,
  "refund": 0,
  "pot_to_call_ratio": 3.0,
  "static_positive": true
}
```

`pot_to_call_ratio` 是底池/跟注的倍数，不是百分比；临界胜率看 `breakeven_equity`。短码全下时 `effective_pot` 会扣除 `refund`。

随机数工具的 `result`：

```json
{"ok": true, "value": 0.7391}
```

`value` 是 `[0, 1)` 的均匀随机数，每次调用重新抽取，供前端显示数值和刻度。

非法动作的 `result`：

```json
{"ok": false, "error": "最小加注到 80", "legal_actions": ["fold", "call", "raise", "all_in"], "to_call": 40, "min_raise_to": 80, "max_raise_to": 920, "pot": 160}
```

喊话工具的 `result`：

```json
{
  "ok": true,
  "entry": {
    "id": "7f3a...", "seq": 1, "hand_id": 3, "street": "flop",
    "seat": 0, "name": "DeepSeek", "text": "我这次不会再让你便宜看牌。",
    "turn_id": "1d3e...", "created_at": 1760000000.0, "kind": "turn"
  },
  "remaining_turn": 0
}
```

默认 `table_talk_mode=inline`：每个行动回合的第一次 `apply_action` 必须在 `table_talk` 参数里带一到两句喊话。`table_talk` 在 JSON schema 里是可选字段，由 agent 在每回合第一次动作强制要求；这样动作被拒绝后的重试可以省略喊话，也避免只校验 schema 的厂商提前拒绝。缺少 `table_talk` 时返回工具错误 `{"ok": false, "error": "...", "required_tool": "table_talk"}`；喊话通过后执行动作。如果动作本身被拒绝，重试时可以不再重复喊话，重复的 `table_talk` 会被忽略。`table_talk_mode=tool` 保留旧协议：先成功调用一次 `send_table_message`，否则 `apply_action` 返回错误，要求先喊话。旧模式下喊话失败不计入连续三次非法动作；inline 模式下缺少 `table_talk` 属于无效动作，会正常计入失败次数。

### `table_talk`

发送成功后立即广播：

```json
{
  "type": "table_talk",
  "match_id": "ab12...", "seat": 0, "event_id": 124,
  "id": "7f3a...", "seq": 1, "hand_id": 3, "street": "flop",
  "name": "DeepSeek", "text": "我这次不会再让你便宜看牌。",
  "turn_id": "1d3e...", "created_at": 1760000000.0, "kind": "turn"
}
```

对手不会立刻收到；它会在下一次 `turn_state` 的`【牌局更新】`里看到未读喊话。服务端只做文本规范化、长度和频率校验，不扫描真实底牌或牌力，也不核查真伪；前端必须按普通文本渲染，不要使用 `v-html`。

比赛结束后，每个 LLM 还会收到一次只暴露 `send_table_message` 的赛后请求。赛后消息的 `kind` 为 `"final"`，`street` 为 `"finished"`，每个座位最多一条。

### `message_end`

```json
{
  "type": "message_end", "turn_id": "...", "message_id": "...",
  "content": "...", "reasoning": "...",
  "finish_reason": "tool_calls",
  "usage": {"input_tokens": 1848, "output_tokens": 173, "reasoning_tokens": 28},
  "latency_ms": 2444.1,
  "tool_calls": [{"index": 0, "call_id": "call_0", "name": "apply_action", "arguments": "..."}]
}
```

### `turn_end`

```json
{
  "type": "turn_end", "turn_id": "...",
  "action": {"seat": 0, "type": "raise", "label": "加注", "amount": 80, "source": "llm", "auto": false},
  "auto": false, "steps": 1,
  "usage": {"input_tokens": 1848, "output_tokens": 173, "reasoning_tokens": 28},
  "latency_ms": 2445.9
}
```

### `note` / `error`

```json
{"type": "note", "level": "warn", "text": "..."}
{"type": "error", "kind": "auth|rate_limit|timeout|network|param|server|engine|agent|unknown", "text": "..."}
```

### `bot_decision`

RLCard 的紧凑决策事件：

```json
{
  "type": "bot_decision", "hand_id": 2, "street": "preflop",
  "agent_type": "random",
  "action_id": 2, "legal_ids": [0, 1, 2, 3, 4],
  "probabilities": {"1": 0.208, "2": 0.208},
  "mapped": {"type": "raise", "label": "加注", "amount": 20, "auto": false}
}
```

`action_id` 是 RLCard 原始 ID（0 fold, 1 check/call, 2 half pot, 3 pot, 4 all-in）；`mapped` 是引擎实际动作。

### `hand_result`

```json
{
  "type": "hand_result", "hand_id": 3,
  "result": {
    "hand_id": 3, "reason": "showdown", "winner_seats": [0],
    "pot": 160, "awarded": {"0": 160}, "refunds": {"0": 0, "1": 0},
    "board": ["8c", "3d", "Ah", "Jh", "Qh"],
    "players": [{"seat": 0, "chips_after": 1080, "total_bet": 80, "net": 80}],
    "showdown": [{"seat": 0, "category": "one_pair", "category_name": "一对", "best_five": ["Ah", "Qh", "Qs", "Jh", "9d"], "score": 3777}]
  },
  "match_over": false
}
```

### `match_over`

```json
{"type": "match_over", "text": "比赛结束。"}
```

### Socket.IO 其他

- `connect`：服务端立即 `emit("state", snapshot)`
- `request_state`：客户端请求后 `emit("state")`
- `request_transcripts`：`emit("transcripts", {match_id, seats})`

---

## 事件顺序

一次 LLM 回合：

```
turn_start
turn_state
message_start
reasoning_delta* / content_delta*
tool_call_start
tool_call_args_delta*
message_end
tool_call
tool_result
(message_start ... 第二轮，如果只调用了估算工具)
turn_end
```

RLCard 回合：

```
bot_decision
```

一手的结束：

```
... turn_end / bot_decision
hand_result
state（match.can_next_hand=true）
```

`state` 与 `agent_event` 可能交错到达；前端按 `seq` 处理 state，按 `event_id` 去重 agent_event。
