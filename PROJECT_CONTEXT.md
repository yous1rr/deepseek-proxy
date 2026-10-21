# 项目上下文概览

本文档记录当前仓库的结构、主要运行链路、配置方式和验证状态，内容以当前源码和测试结果为准。

## 项目定位

这是一个 Rust workspace，实现面向 DeepSeek 私有 Web API 的 OpenAI 流式 Chat Completions 代理。当前接口范围明确限定为 `stream: true`，非流式 Chat Completions 不在支持范围内。

主要入口和接口：

- `deepseek-rs serve`：启动长期运行的 HTTP 服务。
- `POST /v1/chat/completions`：仅支持 SSE 流式聊天补全。
- `GET /v1/models`：返回可用模型列表。
- `GET /healthz`、`GET /readyz`：健康和就绪检查。
- `GET /stats`：请求计数、线程、数据库和账号状态。

## Workspace 结构

依赖方向严格向下：

```text
proxy → client → protocol → pow
```

- `crates/proxy`：Axum HTTP 服务、请求校验、线程管理、工具调用、图片处理、账户池、SQLite、认证、重试和 CLI。
- `crates/client`：DeepSeek 私有 API 客户端，负责登录、session、PoW、completion SSE 和图片上传。
- `crates/protocol`：DeepSeek SSE 行解析和 fragment 路由。
- `crates/pow`：DeepSeekHashV1 proof-of-work 实现和 nonce 搜索。
- `corpus/`：从 DeepSeek completion 接口捕获的 SSE 测试样本，以及 Python 路由器生成的期望结果。
- `tools/`：生成 corpus 期望结果和设备 ID 的辅助脚本。

关键入口：

- [workspace Cargo.toml](Cargo.toml)
- [代理主程序](crates/proxy/src/main.rs)
- [DeepSeek 客户端](crates/client/src/lib.rs)
- [SSE fragment 路由器](crates/protocol/src/fragment.rs)
- [PoW 实现](crates/pow/src/lib.rs)

## 请求处理链路

一次 `POST /v1/chat/completions` 大致按以下顺序执行：

1. 验证 bearer API key。没有注册 API key 时，服务保持开放；注册第一个 key 后开始校验。
2. 校验消息、模型、工具 schema、`tool_choice`、图片内容和工具消息字段。
3. 按以下优先级确定线程 key：请求体 `thread_id`、`x-thread-id` header、会话相关 header、首条用户消息的 hash。
4. 从内存或 SQLite 恢复线程的 DeepSeek session 和 parent message id。
5. 根据是否已有线程构建 prompt：新线程发送完整 transcript，续接线程只发送最近一轮增量消息。
6. 从当前增量消息中读取图片。DeepSeek 每轮最多接收一张图片；多张图片保留第一张，并在 prompt 中通知模型其余图片已丢弃。图片可以来自 `data:` URL 或公共 HTTP(S) URL。
7. 为图片上传或 completion 获取 PoW challenge，解决 nonce 后生成 `X-DS-PoW-Response`。
8. 通过 `client` 发起 DeepSeek 流式 completion。
9. 通过 `protocol` 解析 SSE 和 fragment：
   - `THINK` → OpenAI `reasoning_content`
   - `RESPONSE` → OpenAI `content`
   - `SEARCH` 和结果元数据丢弃
10. 通过 `proxy` 的流式 pump 输出 OpenAI 风格 SSE chunk，并保存最新的 response message id。
11. 如果模型产生工具调用，解析 `<invoke name="tool">{...}</invoke>`，校验工具名和 JSON Schema；格式错误、未知工具、参数错误或缺少必需调用时进入有限次数的 correction turn。
12. 遇到限流、过期 cursor 或上下文超限时，按错误类型重试、重建 session 或 replay transcript。

同一线程的请求由独立的 Tokio mutex 串行化，避免并发请求读取同一个 parent message id 后形成分支会话。

## DeepSeek 客户端和 PoW

[client crate](crates/client/src/lib.rs) 使用固定的 iOS 客户端 header，支持：

- `/api/v0/users/login`
- `/api/v0/chat_session/create`
- `/api/v0/chat/create_pow_challenge`
- `/api/v0/chat/completion`
- `/api/v0/file/upload_file`
- `/api/v0/file/fetch_files`

[PoW crate](crates/pow/src/lib.rs) 实现 DeepSeekHashV1。它基于 SHA3-256，但 Keccak-f[1600] 跳过 round 0，只执行 round 1 到 23，因此不能直接替换成普通 SHA3 实现。nonce 搜索会按照可用 CPU 并行执行。

## 会话、账号和持久化

SQLite 由 [store.rs](crates/proxy/src/store.rs) 管理，启动时自动迁移以下数据表：

- `threads`：DeepSeek session 和 parent message id。
- `thread_bindings`：线程与账号的绑定关系。
- `accounts`：通过 CLI 添加的账号。
- `account_tokens`：登录后缓存的 bearer token。
- `api_keys`：API key 的 SHA-256 hash、名称和前缀。

账户池位于 [pool.rs](crates/proxy/src/pool.rs)：

- 每个账号有独立的请求 gate、队列和冷却状态。
- 新线程选择健康且占用较少的账号。
- 线程绑定会写入 SQLite，重启后继续使用原账号。
- 登录失效时账号进入冷却，线程尝试 failover 到其他健康账号。
- 显式配置的 token 优先于缓存 token；密码登录可以获得新的 token。

账号来源包括：

- `DEEPSEEK_EMAIL_<n>`、`DEEPSEEK_PASSWORD_<n>`、`DEEPSEEK_TOKEN_<n>`。
- 数据库中的 `deepseek-rs accounts add ...` 记录。
- 相邻项目的 `../deepseek/.env`，用于本地开发兼容。

环境账号在邮箱冲突时覆盖数据库账号。

## 配置

配置优先级为：

```text
environment variables > config.toml > built-in defaults
```

配置文件由 `CONFIG_PATH` 指定；没有配置文件时使用默认值。相对 `db_path` 相对于配置文件目录解析，而不是进程当前目录。

常用配置项：

- `[proxy] host`、`port`：监听地址，默认 `0.0.0.0:9174`。
- `[storage] db_path`：SQLite 路径。
- `[deepseek] model`、`thinking`、`search`：模型和默认请求行为。
- `[limits] request_delay_secs`：同一账号请求完成后的间隔。
- `queue_limit`、`queue_timeout_secs`：账号队列容量和等待时间。
- `rate_limit_max_retries`、`rate_limit_backoff_secs`：限流重试策略。
- `tool_repair_max_retries`：工具调用 correction turn 次数，设为 `0` 可关闭。
- `request_timeout_secs`、`heartbeat_interval_secs`：整轮超时和 SSE keep-alive。
- `image_max_bytes`：图片大小上限，默认 10 MiB。

当前仓库的 [config.toml](config.toml) 把 `request_delay_secs` 设置为 2 秒；如果配置文件不存在，代码内置默认值是 0 秒。

## CLI

CLI 实现位于 [cli.rs](crates/proxy/src/cli.rs)：

```text
serve
start [-b|--bg]
stop
restart [-b]
status
health
stats [--json]
logs [-n N] [-f] [--path P]
accounts
accounts add <email> <password>
accounts add <email> --token <token>
accounts rm <email>
keys
keys add <name> <key>
keys rm <name|id>
check
url
version
help
```

后台模式使用运行目录中的 pidfile 和 logfile。运行目录由 `DEEPSEEK_RUNTIME_DIR`、`XDG_STATE_HOME` 或用户状态目录决定。`start --bg` 会等待 `/healthz` 返回成功后再结束。

## 工具调用协议

工具定义来自 OpenAI 请求的 `tools` 字段，代理会把协议说明和 JSON Schema 注入首轮 prompt，并在每轮末尾追加格式提醒。模型需要输出：

```text
<invoke name="tool_name">{"key":"value"}</invoke>
```

代理会：

- 只接受已声明的工具。
- 要求参数是 JSON object。
- 使用 `jsonschema` 校验参数。
- 处理 `tool_choice: none|required` 和指定工具对象。
- 在 `parallel_tool_calls=false` 时拒绝多个调用。
- 对 malformed、未知工具、schema 不匹配和缺失 required call 执行有限 correction turn。
- 过滤流式工具块，使普通文本不会把工具协议泄露给客户端。

相关实现位于 [dialect.rs](crates/proxy/src/dialect.rs) 和 [main.rs](crates/proxy/src/main.rs)。

## 图片处理

图片处理位于 [proxy images.rs](crates/proxy/src/images.rs) 和 [client images.rs](crates/client/src/images.rs)：

- 只接受 PNG 和 JPEG。
- 同时限制 base64 文本长度和解码后的字节数。
- 远程 URL 只允许 HTTP(S)，拒绝 loopback、私网、link-local、CGNAT、保留地址和重定向。
- 远程下载使用超时和流式读取，避免因错误的 `Content-Length` 无界分配。
- 图片上传失败时会降级为纯文本请求，并记录 warning。

## 错误恢复和流式行为

- 请求开始前的上游限流会按线性 backoff 重试。
- 已经向客户端发送内容后，不再重试整轮，避免重复输出。
- `invalid message id` 会触发 session 重建和 transcript replay。
- `input_exceeds_limit` 或 `context_length_exceeded` 会重建 session 并 replay 一次。
- SSE 空闲期间发送 `: keep-alive`，整轮请求仍受 `request_timeout_secs` 限制。
- 客户端断开后停止读取上游，并释放账号 rate slot。
- OpenAI usage 中的 prompt token 来自 DeepSeek 的累计用量；completion token 使用 `cl100k_base` 本地估算。

## 测试和验证

仓库包含各 crate 单元测试、协议 corpus 集成测试和工具调用/线程恢复测试。最近一次本地验证：

```bash
cargo test --workspace
```

结果：116 个测试全部通过，0 个失败。其中包括：

- `client`：7 个测试。
- `pow`：5 个测试。
- `protocol`：12 个单元测试和 2 个 corpus 集成测试。
- `proxy`：90 个测试。

格式检查：

```bash
cargo fmt --all -- --check
```

当前未通过，已有多个文件存在 rustfmt 差异；本次上下文整理没有自动格式化这些文件。编译还报告两个 warning：`main.rs:2680` 有重复的 `#[test]` 属性，`main.rs:2728` 有不必要的 `mut`。

`client-smoke` 是联网 smoke test，需要 `../deepseek/.env` 中的外部 DeepSeek 凭据；本次没有执行。

## 当前接口范围与真实账号验证

- 聊天接口要求客户端发送 `"stream": true`；非流式响应明确不支持，`stream:false` 或缺省值会在校验阶段返回 400。
- 指定目录实际包含的数据库文件名是 `D:\deepseek-rs-v0.1.0-windows-amd64\deepseek-rs.db`，不是 `deepseek.db`。验证时通过 `DB_PATH` 使用该文件。
- 打包版 `deepseek-rs.exe check` 对该数据库完成账号登录和 PoW 检查，返回成功。
- 使用数据库副本启动临时服务后，`GET /v1/models` 返回 HTTP 200，模型为 `DEFAULT`、`EXPERT`、`VISION`。
- 使用同一临时服务发起真实 `stream: true` chat，返回 HTTP 200 和 `text/event-stream`，SSE 正常结束于 `[DONE]`，拼接后的回答符合测试提示。临时服务随后停止；原始数据库未用于聊天写入。

## 功能拓展路线

以下方向不包括非流式接口。前两项优先于新增接口，因为它们完善当前已公开的能力。

1. **模型路由正确性**：`/v1/models` 已公开 `DEFAULT`、`EXPERT`、`VISION`，但当前 `CompletionBody::new()` 将 `model_type` 固定为 `default`，请求模型名没有映射到上游类型。应明确模型别名映射，为每种类型增加请求体单元测试，并用真实账号分别验证；在此之前不要承诺 `EXPERT`、`VISION` 已实际生效。
2. **长会话生命周期与上下文管理**：增加显式 reset/delete/list 操作，协调清理 SQLite thread/binding 与进程内 thread/turn lock；为长会话增加可控的历史裁剪或摘要 replay，避免 context limit 后完整 transcript 仍超限。
3. **账号池运维**：增加逐账号的 login/PoW 检查和健康结果；支持安全地热加载账号，减少新增或撤销账号必须重启服务的运维成本。当前 `check` 会连接账号池，但 PoW challenge 只针对 `pool.any()`。
4. **按 API key 的配额与用量**：现有 key 表只负责 key 验证，统计是进程级计数。可扩展每 key 的请求/并发限额、令牌估算、错误率和周期用量，并提供可审计的 CLI 或受保护管理接口。
5. **明确流式参数契约**：为支持的 OpenAI 请求字段建立白名单和能力说明。对 DeepSeek 上游确实支持的采样、停止条件或输出长度参数做映射与测试；其余字段应明确拒绝或说明忽略，避免客户端误以为设置已生效。
6. **流式兼容性测试**：增加 Axum 路由级契约测试，覆盖 SSE headers、role/content/reasoning/tool_calls/usage chunk、错误事件、断开连接和 `[DONE]`；使用目标 SDK 做端到端兼容回归。
7. **多模态扩展探索**：当前只接受 PNG/JPEG，且每轮仅发送第一张图片。可先确认 DeepSeek 上传接口对多图、PDF/文档或音频的真实支持与限制，再决定扩展格式和请求模型，避免仅在 OpenAI 请求层接受但上游无法处理。

## 当前维护注意事项

`main.rs` 和 `dialect.rs` 顶部仍保留部分早期“尚未移植”的注释，但当前实现已经包含 SQLite 持久化、多账户池、工具 dialect、DSML 兼容和 correction turn。这些注释与实际代码不同步，后续维护时应以实现和测试为准，并可单独更新注释。
六个拓展方向的详细源码调研、优先级和验收建议见 [EXTENSION_RESEARCH.md](EXTENSION_RESEARCH.md)。
