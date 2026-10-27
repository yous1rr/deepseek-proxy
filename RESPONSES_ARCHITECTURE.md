# `/v1/responses` 架构与渠道接入

`/v1/responses` 是按能力门控的 OpenAI Responses 子集。它由与上游无关的模块组装而成；DeepSeek web 只是第一个渠道。接入下一个上游只需实现一个 trait 并如实声明能力，HTTP 层、存储和事件编码都不需要修改。`/v1/chat/completions` 的代码和行为不受影响。

## 分层

| 位置 | 职责 | 约束 |
|---|---|---|
| `crates/responses-core` | 协议规则：请求解析为 `TurnPlan`、能力模型与 profile、中立 transcript、事件编码器、id 生成 | 只依赖 serde/serde_json，编译器保证与渠道、网络、存储无关 |
| `crates/proxy/src/responses/mod.rs` | HTTP 与状态层：认证与 owner、能力门控、`previous_response_id` 链、线程锁、SSE/collect、持久化 | 对所有渠道通用 |
| `crates/proxy/src/responses/channel.rs` | 渠道契约：`ResponsesChannel`、`TurnRequest`、`TurnEvent`、`TurnOutcome`、`TurnError` | 每个上游实现一次 |
| `crates/proxy/src/responses/deepseek.rs` | DeepSeek web 适配器 `deepseek_web` | 复用 crate 根部 Chat 的 turn 机制，只调用不修改 |
| `tools/upstream-capability-probe` | 对 OpenAI 风格上游做合成探测，并转换为能力 profile | 仅证明 HTTP 层能力 |

## 一次请求的生命周期

`POST /v1/responses`：

1. 校验 bearer key 并解析 owner。未注册 key 时所有调用方共享 `anonymous`；注册后 owner 为 key id（`key-<hash12>`），从不使用原始密钥。
2. 严格解析为 `TurnPlan`。未知字段返回 400 `unsupported_parameter`，少数无害提示字段除外（见“能力模型”）。
3. 能力门控：每个改变行为的参数都必须对应已启用的能力，否则在任何上游工作前返回 400。然后确定模型（缺省为配置的 `model`）并校验工具 schema（禁止远程 `$ref`）。
4. 读取 `previous_response_id` 指向的已存 transcript，按 owner 隔离；不存在或属于其他 owner 时返回 400 `previous_response_not_found`。replay 的内容也要通过当前能力门控，因此收窄 profile 后，旧的图片或文件上下文不能绕过限制。
5. 拼接 transcript（父 transcript + 本次 input），确定线程 key。有父响应时沿用父响应的线程；否则依次使用 `x-thread-id`、会话 header、`prompt_cache_key`、首条用户文本，与 owner 一起哈希为 `responses_<24 hex>`，都没有时生成新 key。key 冲突是安全的：历史指纹不匹配就会 replay。
6. 启动独立任务：持有线程锁，调用 `run_turn`，把 `TurnEvent` 交给编码器。
   - `TurnEvent::Started` 之前的错误作为普通 HTTP 错误返回，因此 SDK 重试和 `Retry-After` 仍然有效。
   - 之后的错误在流式请求中以 `response.failed` 结束，在 collect 请求中返回 HTTP 错误。
7. 成功时编码剩余 item，先持久化 completed 响应和渠道 checkpoint，再发送 `response.completed`（或 collect 结果）。客户端断开会取消渠道 future，不提交任何内容。

线程锁一直持有到提交完成，同一线程的第二个请求会等待，而不会读到即将被替换的 cursor。`GET`/`DELETE /v1/responses/{id}` 同样按 owner 隔离；`GET ?stream=true` 返回 400，因为不支持恢复流。

## 能力模型

能力 id 一共 30 个，见 `Capability::ALL`。某个能力启用需要同时满足四个条件：

```text
启用 = 渠道声明 supported（能力上限）
     ∧ 引擎当前版本能表示（engine_supported）
     ∧ 可选 profile 未收窄
     ∧ 其所有依赖也已启用
```

- 依赖关系：`tool_choice` → `function_tools`；`previous_response`、`background` → `stored_responses`；`stream_resume` → `streaming` + `stored_responses`。
- profile 只能收窄。`unsupported`、`unknown` 和缺省都会禁用；未知 id 产生警告后忽略；profile 把渠道未实现的能力标为 `supported` 时只会产生警告。格式、版本或 `channel` 不匹配都会导致启动失败，不会静默回退。
- 渠道声明了引擎尚不能表示的能力时，启动时会自动禁用并发出警告。
- 默认值不产生需求：`temperature`/`top_p` 为 1.0、`reasoning.effort` 为 `none`/`minimal`、单独的 `reasoning.summary`。
- 忽略的提示字段：`prompt_cache_retention`、`prompt_cache_options`、`safety_identifier`、`user`、`service_tier`、`stream_options`。
- `reasoning` 未启用时，HTTP 层要求渠道使用 effort `none`，并且不转发渠道产生的任何推理事件。
- `stored_responses` 未启用时，`store` 强制为 `false`。

引擎当前能表示的能力：`text`、`streaming`、`non_streaming`、`image_input`、`file_input`、`function_tools`、`tool_choice`、`web_search`、`image_generation`、`reasoning`、`verbosity`、`sampling_controls`、`max_output_tokens`、`stored_responses`、`previous_response`、`exact_usage`。代码解释器、自定义工具、结构化输出、logprobs、conversations、background、流恢复、自动截断、compaction、提示模板仍没有完整请求或事件表示，不能启用。

## 存储与续接

- `responses` 表每行保存一个已存响应和它完整的 transcript 副本，按 `owner_id` 隔离。删除父响应不会破坏子响应，`previous_response_id` 列只用于记录。
- `store=true`：响应记录与渠道 checkpoint（cursor、账号绑定、历史指纹）在同一事务中写入。
- `store=false`：只保存渠道 checkpoint，不保存对话内容，与 Chat Completions 相同。
- 渠道没有返回 checkpoint（例如上游没有给出新 cursor）时只保存响应记录，下一轮会 replay。
- 已存 transcript 不保存推理内容；客户端回传的 `reasoning` item 在输入时丢弃。

## DeepSeek web 渠道（`deepseek_web`）

适配器把 `TurnRequest` 映射为 Chat 形状的请求，复用 crate 根部经过验证的机制：历史指纹与固定账号、prompt 扁平化、分叉时 replay 到新 session、限流重试、cursor 恢复和带 schema 校验的工具修复。

- **transcript 映射**：assistant 消息与紧随其后的 `function_call` 合并为一条带 `tool_calls` 的 assistant 消息，与 `TurnHistory::completed` 记录答案的方式一致。因此已存答案 replay 后指纹相同，下一轮可以只发送增量。服务端 `call_` id 无妨，指纹按序号规范化 id。
- **流解码**：本地的 `drain` 把 DeepSeek SSE 转换为 `TurnEvent`，Chat 的 `pump` 不变。第一个内容 fragment 到达时发送 `Started`；非 SSE 响应体只映射为固定提示，不进入日志。
- **与 Chat 的差异**：图片上传失败会使本轮失败，而不是静默丢图；从不开启 web search（未声明托管工具，全局 `search` 配置不影响 Responses）；verbosity 以 prompt 指令追加到本轮。

| 能力 | 提供方 / 方式 |
|---|---|
| `text` | upstream / `flattened_prompt` |
| `streaming` | proxy / `upstream_sse_reencoded` |
| `non_streaming` | proxy / `collected_stream` |
| `image_input` | upstream / `single_image_per_turn`（每轮一张，多余的丢弃并通知模型） |
| `function_tools`、`tool_choice` | proxy / 提示词工具协议 + JSON Schema 校验与修复 |
| `reasoning` | upstream / `thinking_toggle`：effort `none`/`minimal` 关闭，其余开启，缺省使用配置 `thinking`；无摘要，`summary` 为空 |
| `verbosity` | proxy / `prompted_directive` |
| `stored_responses`、`previous_response` | proxy / 本地 transcript |

其余 20 个能力都以明确原因声明为 unsupported，例如 `no_upstream_sampling_parameters`、`hosted_tool_not_implemented`、`output_tokens_estimated_locally`。输入 token 来自上游，输出 token 为本地估算，因此 `exact_usage` 未启用。


## ChatGPT Web 渠道（`chatgpt_web`）

ChatGPT Web 通过 `ResponsesChannel` 接入，并按模型与 DeepSeek 渠道并行路由。`chatgpt2api` 的 Python 实现被拆成三个 Rust 模块：`chatgpt-client` 负责 HTTP、Sentinel/PoW、Turnstile 受限解释器、图片上传/下载和搜索轮询；`chatgpt-protocol` 负责 JSON Patch SSE、可见文本、搜索来源和图片指针提取；`responses/chatgpt.rs` 负责 `TurnRequest` 映射和 Responses 事件装配。

| 能力 | 提供方 / 方式 |
|---|---|
| `text`、`streaming`、`non_streaming` | ChatGPT Web conversation SSE / `response` patch re-encoded |
| `reasoning` | ChatGPT Web `thought`/隐藏 channel 过滤后的 reasoning delta |
| `image_input` | `/backend-api/files` 上传并注入 `file-service://` asset pointer |
| `web_search` | `system_hints: ["search"]`、搜索源轮询、`web_search_call` output item |
| `image_generation` | `async_task_type=image_gen`、安全文件轮询、`image_generation_call` Base64 result |
| `stored_responses`、`previous_response` | proxy 本地 transcript replay；上游不伪造 cursor |

ChatGPT access token 只从 `CHATGPT_TOKEN_<n>`/`CHATGPT_ACCESS_TOKEN_<n>` 环境变量读取。配置了有效 token 后，`GET /v1/models` 才广告 `auto`、`gpt-5*`、`gpt-image-2` 和 `codex-gpt-image-2`；没有 token 时不会把不可用模型放进目录。`code_interpreter` 仍由能力门控拒绝，直到有可靠的沙箱事件映射。

Rust 客户端使用 `reqwest`/rustls，并明确不声称复现 `curl_cffi` 的 Chrome TLS ClientHello 指纹；若上游要求浏览器指纹，需单独替换传输层。PoW 使用 SHA3-512 原生实现，Turnstile 只执行固定的 allowlisted bytecode 子集，不执行任意 JavaScript。
## 接入新渠道

1. 在 `crates/proxy/src/responses/` 下新建模块并实现 `ResponsesChannel`：
   - `id()` 是稳定的渠道 id，与 profile 的 `channel` 匹配。
   - `capabilities()` 返回完整能力上限，30 个 id 都要声明，不支持的写明原因。只声明真正实现的能力；HTTP 层会拒绝其余能力，保证请求不会被静默忽略。
   - `run_turn()` 按顺序发送 `TurnEvent`：文本中不得残留渠道特有的工具语法；能确认上游已接受时发送 `Started`。返回 `TurnOutcome`（调用、用量、可选 checkpoint）或 `TurnError`。`events` 关闭时停止上游工作并返回 `Cancelled`。
2. 在 `main.rs` 中把 `Runtime::load` 的渠道换成新实现（或按配置选择）。
3. 测试：共享契约已由 `responses/tests.rs` 的 `FakeChannel` 固定；新渠道只需测试自己的映射、流解码和能力上限，可参考 `deepseek.rs` 的测试。
4. 对 OpenAI 风格上游，用 `tools/upstream-capability-probe` 探测并通过 `--proxy-profile PATH --channel <id>` 生成 profile，再用 `RESPONSES_CAPABILITY_PROFILE=PATH` 收窄。`unknown` 视为不可用，应重新探测，不要手工改为 `supported`。

扩展引擎时：新能力 id 要同时加入 `capability.rs` 和探测工具的 `CAPABILITY_IDS`，两者顺序必须一致。只有请求解析、事件编码和测试都完成后，才能在 `engine_supported` 中开启。新的 Responses 字段只能解析为需求，或明确列为忽略的提示字段，不能静默透传。

## 配置

- `RESPONSES_CAPABILITY_PROFILE`：能力 profile 路径，可选。不设置时使用渠道内置能力上限。

## 真实账号验证

`tools/verify_responses.py` 对隔离实例做黑盒验收：release 构建，仅监听 `127.0.0.1`，服务数据库的一致性临时副本，提示全部是合成内容，输出不含提示、模型输出、id 或密钥。会话复用通过临时库中的 `threads.session_id` 判断。

```bash
python tools/verify_responses.py run --db /path/to/test-copy.db --state /tmp/responses-state.json
# 重启代理后
python tools/verify_responses.py restart-check --db /path/to/test-copy.db --state /tmp/responses-state.json
# 最后执行：会向临时库写入两个测试 key
python tools/verify_responses.py keys --confirm-temporary-db --db /path/to/test-copy.db --state /tmp/responses-state.json
```

2026-10-23 的结果（3 个真实账号，临时库来自旧版 schema，因此也验证了迁移）：

- 首轮 `run`：68 项检查通过 64 项。覆盖能力门控、Chat Completions 回归、流式与 collect、`previous_response_id` 链（续接复用远端 session，去掉 instructions 后 replay）、分支 fork 与隔离、函数工具往返（续接同一 session）、推理开关与 verbosity、`store=false` 无状态重放（回传 output item 后续接同一 session）、删除、客户端断开（不落库）和图片输入。
- 失败的 4 项来自 collect、推理、断开 3 个阶段。它们恰好都分配到同一个账号，上游对该账号持续约 8 分钟返回 `Messages too frequent`；代理重试 5 次后返回 HTTP 429 `rate_limit_exceeded`，没有开始 SSE，其余账号同期正常。重启后按 `--phases H,D,K` 重跑，14 项全部通过。
- 重启后续接 2 项、按 key 隔离 9 项全部通过。
- 验证前后原数据库的 SHA-256、大小和修改时间不变，临时数据已删除。

观察：账号被上游持续限流时，冷却期一过它又被视为健康，新线程仍可能分配给它，这样的 turn 要等完全部重试才失败。这是账号池的共享行为，Chat Completions 相同，可作为独立的后续改进。

## 已知限制

- 用量为估算值，推理摘要始终为空。
- 尚未实现 `input_items`、cancel、compact、列表等端点。
- 已存响应不会自动过期，只能按 id 删除。
- 未注册 API key 时所有调用方共享 `anonymous` owner，能访问端口的人都可以读取或删除已知 id 的响应。对外暴露端口前请先注册 key。
- 尚未用官方 SDK 或 Codex 等真实客户端做兼容测试；事件形状按 SDK 定义实现，并由测试和验收脚本校验。
