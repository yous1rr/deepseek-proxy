# 拓展方向调研报告

## 范围和结论

本报告基于六个独立的只读源码调研方向：模型路由、流式契约、会话生命周期、账号池运维、API key 配额与可观测性、多模态扩展。

当前产品边界保持不变：只支持 `POST /v1/chat/completions` 的 `stream: true` SSE；非流式 JSON 接口不纳入拓展范围。

调研结论是，下一阶段应先修正现有流式 API、模型路由、账号诊断和会话生命周期中的确定性问题，再扩展多图、配额或管理接口。当前代码已经具备较完整的 DeepSeek 流式主链路，主要风险来自“公开能力和实际行为不一致”以及“运行状态缺少可管理性”。

## P0：先修正现有能力

### 1. 模型名称和上游模型类型统一

当前存在三套互不一致的模型定义：

- [`models`](crates/proxy/src/main.rs) 只公开 `DEFAULT`、`EXPERT`、`VISION`。
- [`is_supported_model`](crates/proxy/src/main.rs) 还接受 `deepseek-chat` 和 `deepseek-v3`，但不接受常见的 `deepseek-reasoner`。
- 配置和 README 默认使用 `deepseek-chat`。
- [`completion_body`](crates/proxy/src/main.rs) 调用 [`CompletionBody::new`](crates/client/src/lib.rs)，而 `model_type` 在构造函数中固定为 `default`。
- 图片上传的 `x-model-type` 又直接使用客户端传入的原始模型字符串。

因此当前请求即使选择 `EXPERT` 或 `VISION`，completion body 也会发送 `model_type=default`。`VISION` 选择目前只是 API 层面的名义支持，不能从代码确认它实际切换了上游模型。

建议：

1. 增加一个集中式模型解析函数，把外部模型名归一化为公开模型 ID和上游 `model_type`。
2. 同时用于 completion body、图片上传 header、模型回显和 `/v1/models`。
3. 决定是否支持 `deepseek-reasoner`，这需要结合上游真实行为验证；不能仅凭名称加入白名单。
4. 在没有验证上游取值大小写和语义前，不承诺 `EXPERT`、`VISION` 已经生效。

验收测试：模型别名和大小写归一化、completion JSON 的 `model_type`、图片上传 header、非法模型 400、`/v1/models` 与白名单一致。

### 2. 明确 stream-only 契约

`ChatRequest` 虽然解析了 `stream`，但 handler 只把它写入日志，随后无条件进入 `stream_response` 并返回 `Sse`。这意味着 `stream:false` 或缺省值不会返回 JSON，也不会显式报错。

建议显式拒绝非流式请求：

- 缺省或 `stream:false` 返回 400。
- 错误 payload 使用 OpenAI `error` envelope。
- 不实现非流式转换路径。
- README、源码模块说明和上下文文档统一写成 stream-only。

流式响应还存在几个确定性兼容问题：

- 未授权请求返回 `{"detail": ...}`，与其他 OpenAI 错误响应格式不同。
- `include_usage=true` 时，opening、tail、tool_calls 和 finish chunk 没有统一携带 `usage: null`。
- 工具调用在末尾一次性发送完整 `tool_calls`，没有增量 arguments chunk。
- 流中错误作为普通 `data` payload 发送，没有单独的 `event: error` 标记。
- 未设置 `X-Accel-Buffering: no`，经过 Nginx 等反向代理时可能被缓冲。
- 当前没有 pump、stream response 或 Axum SSE 路由级测试。

建议先补一套无网络契约测试，覆盖：响应头、opening chunk、普通文本、reasoning、usage、工具调用、heartbeat、建流前错误、流中错误、客户端断开和 `[DONE]`。第三方 SDK 对流中错误、一次性工具调用和 heartbeat 注释的兼容性仍需真实客户端验证。

### 3. 修复线程和账号池的确定性缺陷

账号池当前有两个需要优先修复的行为问题：

- `least_crowded()` 在所有账号都处于 cooldown 时仍返回索引 0，可能把新线程路由到不健康账号。
- 已有 SQLite 绑定会优先复用，即使该账号当前处于 cooldown；只有请求失败后才触发 failover。

图片远程 URL 还应增加只允许 80/443 的端口限制。现有 DNS 检查与后续 HTTP 连接之间存在 DNS rebinding 窗口，属于需要进一步加固的 SSRF 边界。

## P1：提升可管理性

### 4. 会话 reset、删除、列表和过期清理

当前线程状态分为两部分：

- 进程内 `threads: HashMap` 保存 session 和 cursor。
- 进程内 `turns: HashMap` 保存每个线程的 mutex。
- SQLite 的 `threads` 保存 resume cursor，`thread_bindings` 保存账号绑定。

目前三者没有统一清理路径：

- `threads` 和 `turns` 没有淘汰机制，长期运行会无限增长。
- SQLite 已有 `delete_thread`，但它不会删除 `thread_bindings`，且没有调用方。
- 没有线程列表、删除、重置或 prune CLI/API。
- context limit 恢复总是把完整请求 transcript replay；超长 transcript 会重复失败。
- 代理不保存历史消息，不能单独从数据库构造摘要，裁剪只能作用于当前请求携带的 messages。

最小方案：

1. 添加统一的 `forget_thread`，先取得线程锁，再在内存和 SQLite 事务中清理 `threads` 与 `thread_bindings`。
2. 处理 `turns` mutex 的回收，不能在仍有 waiter 时直接移除 Arc，否则可能为同一线程创建第二把锁。
3. 增加 `updated_at` 索引和 TTL/prune 配置。
4. 增加受保护的线程 list/delete/reset，返回线程 key、更新时间和账号信息，不返回 DeepSeek session id。
5. 使用现有 token 估算逻辑实现 replay budget trimming；摘要需要额外的上游请求，应放在同一线程锁内，作为后续阶段。

线程管理接口必须先解决认证和所有权问题。当前 `thread_id` 可以由客户端任意指定，系统没有 API key 到线程的绑定；在开放认证模式下，不能提供全局线程列表或删除。

### 5. 账号池诊断、逐账号检查和热加载

当前 `check` 会对所有账号执行登录，但 PoW challenge 只取 `pool.any()` 的第一个账号；它还没有真正调用本地 nonce 求解链路。因此 `ok: N account(s), login + PoW verified` 对多个账号的表述过强。

启动时 `Pool::connect` 对账号串行登录，任一账号失败就整体退出；账号登录没有明确 HTTP 超时。账号增删写入 SQLite 后必须重启，账号删除也没有提示待重启。

`/readyz` 是静态 200；`stats.accounts[].healthy` 只表示本地 cooldown 是否结束，不能表示 token 仍有效或最近一次上游调用成功。线程绑定在重启后不会立即重新加载到内存 owner map，因此 bound thread 统计可能暂时为零。

优先级建议：

1. 逐账号执行 challenge 和实际 PoW solve，输出脱敏的逐账号结果；检查真实数据库时继续使用副本。
2. 增加 `last_success`、`last_error`、冷却原因和 pending-restart 状态。
3. 修复全部账号不健康时的路由行为，返回可解释的 503。
4. 为客户端请求增加总超时，避免单账号卡住整个启动流程。
5. 最后再做账号热加载；热加载需要可替换的 Pool、账号绑定重解析和并发中的旧连接生命周期管理。

### 6. API key 配额和可观测性

当前 `api_keys` 表只保存 hash、前缀、名称和创建时间。`authorize()` 只返回 bool，并丢弃 key 身份；后续 handler、stream task、统计和 account route 都不知道请求属于哪个 key。

当前统计是进程级原子计数，缺少：

- per-key 请求和并发数
- token 用量
- 延迟和取消数
- 状态码/错误类型
- 周期性持久化用量
- 审计事件

公开 `/stats` 还会返回账号邮箱，且没有认证层。key hash 是无盐 SHA-256，弱 key 可被离线猜测。

建议按依赖顺序实现：

1. `verify_api_key` 返回 `KeyIdentity`，开放模式使用匿名身份。
2. 将身份贯通 handler、stream task 和统计路径。
3. 增加 per-key semaphore 和请求配额，超限返回 429 与 `Retry-After`。
4. 增加按小时/日/月的原子计数和 SQLite rollup；completion token 目前部分是本地估算，只能先做软配额。
5. `/metrics`、管理接口和详细 key usage 必须使用独立 admin token；公开 `/stats` 应脱敏账号信息。
6. 新增 key 时使用高熵随机值；hash 算法迁移需要版本化 schema。

不要把 prompt、图片、原始 thread id 或完整 key 写入 usage/audit 表；thread id 应使用不可逆引用或截断 hash。

## P2：经过上游验证后再扩展

### 7. 多图、WebP、文档和音频

当前图片链路已经支持：

- 用户消息中的 PNG/JPEG `data:` URL 和公共 HTTP(S) URL。
- 10 MiB 限制、流式下载、SSRF 地址过滤、无重定向和上传状态轮询。
- 每轮只发送第一张图片，其他图片丢弃并通知模型。
- 上传失败后降级为纯文本。

候选方向的判断：

| 方向 | 建议 | 主要前置验证 |
| --- | --- | --- |
| 多图 | 高优先候选 | 上游 completion 是否接受多个 `ref_file_ids`，上传 PoW 和 body 限制能否承受 |
| WebP | 中优先候选 | 上游是否接受 `image/webp`；否则本地转 JPEG/PNG |
| PDF/文档 | 后续探索 | 逆向 Web 端上传字段、解析状态和 completion 引用方式 |
| 音频 | 暂缓 | 当前只有 `audio_id` 预留字段，没有确认上游端点或模型能力 |

多图不能直接把返回类型从单个 bytes 改成无限列表。需要配置最大张数、有界上传并发、动态 body limit、PoW CPU 预算和每轮总字节上限。context replay 还存在历史图片丢失问题：重建 session 时旧图片只剩 `[image]` 文本，除非持久化 file id，否则无法恢复完整视觉上下文。

## 依赖关系和建议顺序

```text
模型标准化 ───────┐
流式契约测试 ──────┼─→ 多图/模型能力扩展
账号诊断修复 ──────┘

认证身份贯通 ─→ per-key 配额/审计/管理接口

线程 reset/清理 ─→ TTL/list/prune ─→ 摘要 replay
```

建议实施顺序：

1. 显式拒绝 `stream:false`，统一 OpenAI 流错误格式，补充 SSE headers/chunk 契约测试。
2. 修复模型归一化和 `model_type` 传递，真实验证 `DEFAULT`、`EXPERT`、`VISION`。
3. 修复全 cooldown 路由、逐账号 PoW 检查、readyz/stats 状态和客户端超时。
4. 实现线程 reset/delete 的一致性清理，再加入 TTL 和 replay trimming。
5. 贯通 API key 身份后再实现配额、用量和 admin/metrics 接口。
6. 通过上游探针确认多图能力，再考虑受控多图；WebP、PDF 和音频排在后面。

## 调研限制

本轮 subagent 只读检查了源码和测试布局，没有修改项目文件、读取真实数据库、访问网络或执行构建。此前真实账号流式验证已经单独完成，但本报告中的多图、模型类型大小写、第三方 SDK 兼容和 PDF/音频能力仍需要后续真实上游探针或客户端测试。
