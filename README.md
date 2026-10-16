# Claude Provider Support

MaiBot 插件。

用官方 `anthropic` Python SDK 原生调用 Claude Messages API（`POST /v1/messages`），为 MaiBot 注册客户端类型 `claude`。注册后，Claude 模型可以像内置模型一样放进 `replyer`、`planner`、`utils` 等任何任务。

## 为什么不用 OpenAI 兼容层

MaiBot WebUI 自带的 "Anthropic (Claude)" 模板走的是 `client_type = "openai"` + Anthropic 的 OpenAI SDK 兼容层。官方明确说明兼容层主要用于测试对比，不建议长期或生产使用，而且会丢掉以下能力：

| 能力 | OpenAI 兼容层 | 本插件 |
|---|---|---|
| 提示缓存 | 不支持 | system、聊天历史、整段请求三处缓存断点，命中部分按缓存读取价计费（Opus 5.5 为输入价的 0.05 倍） |
| 思考内容 | 不返回 | 返回思考摘要，MaiBot 中显示为 reasoning |
| `json_schema` 结构化输出 | `response_format` 被忽略 | 映射为 `output_config.format`，按 schema 强约束 |
| 推理强度 | `reasoning_effort` 被忽略 | 映射为 `output_config.effort` |
| 拒答 | 直接返回空/拒答 | 支持的模型由服务端按拒答类别自动换模型重答 |

## 安装

在 MaiBot WebUI 的插件中心搜索 "Claude Provider Support" 安装。

也可以手动安装：把本仓库克隆到 MaiBot 的 `plugins/` 下（例如 `plugins/claude_provider/`），或只复制下面四个文件进去，然后重启 MaiBot：

- `_manifest.json`
- `plugin.py`
- `claude_adapter.py`
- `config_models.py`

依赖 `anthropic>=1.14.0,<2.0.0` 已在 manifest 中声明，MaiBot 启动时的插件依赖同步会用 uv 自动安装，无需手动 `pip install`。

运行 MaiBot 的机器需要能访问 `api.anthropic.com`。SDK 会遵循 `HTTPS_PROXY` 等代理环境变量；MaiBot 的 Provider 配置本身没有代理字段。

## 模型配置

在 `config/model_config.toml` 中新增一个 Provider 和模型：

```toml
[[api_providers]]
name = "Claude"
base_url = "https://api.anthropic.com"   # 不要带 /v1，SDK 会自己拼 /v1/messages
api_key = "<你的 Claude API Key>"
client_type = "claude"
timeout = 300                            # 秒；长思考请求可能持续数分钟，超时即中止
max_retry = 3

[[models]]
model_type = "llm"
model_identifier = "claude-opus-5-5"
name = "claude-opus-5-5"
api_provider = "Claude"
send_temperature = false                 # 必须关闭，原因见下
max_tokens = 16000                       # 思考 token 也计入 max_tokens
visual = true

[models.extra_params]
effort = "medium"

[model_task_config.replyer]
model_list = ["claude-opus-5-5"]
```

几个必须注意的点：

- **`send_temperature = false`**：Claude Opus 4.7 及之后的模型收到任何 `temperature` 都返回 400，Sonnet 5.5 / Haiku 5.5 只接受默认值。MaiBot 任务默认温度是 0.7，不关掉就会每次请求都失败。插件不会替你吞掉这个参数：开着就原样发给 API。
- **`auth_type` 保持默认值 `bearer`**：插件固定通过 `x-api-key` 头鉴权，其他取值会直接报错。
- **workspace ID**：只有 Key 没有限定到单个工作区时才需要（个人 Key 或服务账号 Key 创建时选了多个工作区），缺了 API 会返回 400。ID 在 Console 的 Workspaces 页 ID 列，格式为 `wrkspc_...`，加在 Provider 下：

  ```toml
  [api_providers.default_headers]
  anthropic-workspace-id = "wrkspc_..."
  ```

  WebUI 的 Provider 编辑页里也可以直接加这个请求头。插件会校验它的值必须是 `wrkspc_` 开头、不含空格，格式不对时直接报错，不会把请求发出去换回一个看不出原因的 400。
- **`max_tokens`**：MaiBot 任务默认只有 4096。思考 token 也计入这个上限，耗尽时插件会报错并提示调大，正文被截断时只记警告并返回已生成的部分。
- **价格**：MaiBot 按"元 / M token"统计。Opus 5.5 官方价为输入 $4、输出 $20、缓存读取 $0.20（每百万 token），请按汇率换算后填入 `price_in` / `price_out` / `cache_price_in`。缓存写入的 1.25 倍（5m）/ 2 倍（1h）溢价在 MaiBot 统计里体现不出来。

### extra_params

| 键 | 取值 | 作用 |
|---|---|---|
| `effort` | `low` / `medium` / `high` / `xhigh` / `max` | 推理强度，写入 `output_config.effort`；Opus 5.5 不写时默认 `medium` |
| `reasoning_effort` | 同上 | `effort` 的别名，兼容 MaiBot 的习惯写法；两者同时出现且不一致会报错 |
| `thinking` | `adaptive`（默认）/ `omit` | `omit` 表示不发送 `thinking` 参数，只给不支持自适应思考的旧模型（如 Haiku 4.5）用 |
| `thinking_display` | `summarized` / `omitted` | 覆盖插件级的思考显示设置 |

`temperature`、`max_tokens` 这两个键由 MaiBot 自己读取，插件不重复处理。除此之外的键一律报错，避免拼错的配置静默失效。

## 插件配置

插件设置页（或插件目录下的 `config.toml`）中的 `[plugin]` 节：

| 字段 | 默认 | 说明 |
|---|---|---|
| `prompt_cache` | `true` | 是否打缓存断点 |
| `cache_ttl` | `5m` | 缓存存活时间；群聊消息间隔经常超过 5 分钟时改成 `1h` |
| `thinking_display` | `summarized` | `summarized` 返回思考摘要；`omitted` 不返回思考文本，计费不变 |
| `refusal_fallback` | `true` | 只对 `claude-opus-5-5`、`claude-opus-5`、`claude-fable-5-1`、`claude-sonnet-5-5` 生效 |

## 行为说明

- **缓存**：一个断点打在最后一个 system 块上；一个打在最后一条模型侧内容（assistant 回复或 tool_result）上，覆盖到聊天历史末尾；再加一个请求级自动断点覆盖整段请求。MaiBot 每轮都会在末尾拼接时间、回复要求等一次性内容，第二个断点让历史部分不受这段尾巴影响仍能命中。同一条消息里被表情切开的文字片段会拼回一个文本块，避免长消息把断点挤出 API 的 20 块回看范围。MaiBot 按窗口裁剪历史时，消息前缀会变，但 tools + system 仍能命中。命中情况看 MaiBot 统计页的缓存命中数；如果长期为 0，通常是 system 提示里含有每次都变的内容（时间、随机值等）。
- **思考不回传**：MaiBot 不会把 Provider 原生的回放数据传给插件，带签名的 thinking 块因此回传不了。官方允许不回传，代价是工具调用循环里推理的连续性会变差。这样做也更稳妥：MaiBot 会改写历史，而对 2026-08-31 及之后创建的账号，Opus 5.5 会校验回传的思考块所在的历史是否被改动过，改动过就返回 400。
- **system 消息**：开头的 system 消息合并成顶层 `system`；对话中途出现的 system 消息原样以 `role: "system"` 发送（Opus 4.8 / 5 / 5.5、Sonnet 5.5、Haiku 5.5、Fable 系列支持；Sonnet 5 及更早的模型会由 API 报错）。
- **拒答**：`stop_reason = "refusal"` 时报错，错误信息带拒答类别和说明。启用兜底且模型支持时，由服务端自动换模型重答，日志会记录由哪个模型兜底。MaiBot 的计价仍按原模型的价格统计。
- **图片**：MaiBot 内部的 `jpg` 会转成 `image/jpeg`；支持 jpeg / png / gif / webp。
- **响应格式**：`json_schema` 经 `anthropic.transform_schema` 规整后走结构化输出（会自动补 `additionalProperties: false`，不支持的约束挪进描述）。`json_object` 没有对应的 API 能力，以追加 system 指令的方式约束。
- **重试与超时**：SDK 自身不重试（`max_retries=0`），由 MaiBot 按 Provider 的 `max_retry` 统一重试，避免两层重试叠加超出 MaiBot 的等待时间。注意 MaiBot 会重试插件的所有失败，包括 400。整个请求的总时长受 Provider `timeout` 限制，超时后插件会主动断开流式连接。
- **不支持的能力**：Claude API 没有 embedding、音频转写、图片嵌入，这些任务请继续用其他 Provider。

## 开发与验证

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

宿主契约测试会用 MaiBot 自己的类型构造请求、加载插件、回收输出，需要在 MaiBot 的虚拟环境里运行：

```bash
MAIBOT_ROOT=/path/to/MaiBot /path/to/MaiBot/.venv/bin/python -m pytest tests
```

真实 API 冒烟测试共发 6 次小请求，验证参数组合被接受、缓存命中（含末尾每轮变化时的历史命中）和工具调用续接：

```bash
ANTHROPIC_API_KEY=<你的 Key> ANTHROPIC_WORKSPACE_ID=<仅多工作区 Key 需要> uv run python scripts/live_smoke.py --model claude-haiku-5-5
```

## 兼容版本

已验证 MaiBot 1.3.9、插件 SDK 2.12.0 ~ 2.14.0，manifest 只声明这个范围。MaiBot 1.3.x 的后续补丁版本会以兼容模式加载；更高的 MaiBot 次版本或 SDK 高于 2.14.0 时宿主会拒绝加载，需要等插件验证后发布新版本。

## 许可证

[GNU Affero General Public License v3.0](./LICENSE)（`AGPL-3.0-only`）。
