# Pi Desktop 分支兼容性实测

## 结论

已通过未修改的 Pi Desktop 便携版捕获 provider HTTP 请求，确认可以只修改 deepseek-proxy 来支持 Pi Desktop 的重新生成、编辑重发、切换旧 revision 后继续对话，无需修改 Pi Desktop 客户端。

关键输入已经存在：同一会话内操作保持会话标识不变，并发送所选分支的有效消息历史。代理需要根据请求历史选择分支基点，不能继续无条件使用线程最新 cursor。

独立 fork 是另一种操作：Pi Desktop 创建新 session id，并携带选定位置之前的历史。代理应把它作为新线程建立 session/replay；它不需要与原线程共用可变的最新 cursor。

## 验证环境

- Pi Desktop 可执行文件：`D:\Packages\PI-Desktop\apps\desktop\release\PI-Desktop-Portable-0.21.0-beta.4.exe`。
- 应用版本和 host 版本均为 `0.21.0-beta.4`，host 协议版本为 `11`。
- 便携版 SHA-256：`CCED845732B7BB20306D1A4B8E147BC7B53E7CD20FF308988396DB784B089576`。
- 参考源码 checkout：`D:\Packages\PI-Desktop`，HEAD 为 `a6f4ba121e524a167c42466c7b4debbb3cfd0744`。这是源码参考版本，未据此断言便携版与该提交字节一致。
- 使用临时 `PI_DESKTOP_DATA_DIR` 和独立 `--user-data-dir`，未复制安装版 profile、数据库或模型凭据。
- provider 为仅绑定 loopback 的 OpenAI Chat Completions 模拟端点，无需真实凭据。端点捕获的是便携版实际发出的 HTTP 请求。
- 调用便携版现有 `agent/prompt`、`session/activateRevision`、`session/fork` IPC。未修改 Pi Desktop 源码或发布包。
- 测试提示与回复均为合成标记；记录只保留结构、长度、哈希和这些标记，不保存消息正文、认证 header 或工具参数。

## 实际请求序列

下表省略 system 消息，`Un` 表示用户输入，`An` 表示本地端点生成的回答。消息数量包含 system。

| 请求 | 操作 | HTTP messages 中的历史与当前输入 | 消息数 | 会话标识 |
| --- | --- | --- | ---: | --- |
| 1 | 首轮 A | `U_A` | 2 | S1 |
| 2 | 正常续接 B | `U_A, A1, U_B` | 4 | S1 |
| 3 | 正常续接 C | `U_A, A1, U_B, A2, U_C` | 6 | S1 |
| 4 | 从 B 截断并编辑重发 | `U_A, A1, U_B_EDITED` | 4 | S1 |
| 5 | 激活 B 的原 revision，再发送 D | `U_A, A1, U_B, A2, U_C, A3, U_D` | 8 | S1 |
| 6 | 激活 B 的编辑 revision，再发送 E | `U_A, A1, U_B_EDITED, A4, U_E` | 6 | S1 |
| 7 | 在 A1 位置 fork，再发送 F | `U_A, A1, U_F` | 4 | S2 |

实际观察：

- 请求 4 保留 A 的历史，移除了原 B/C 的回答尾部。
- 请求 5 恢复原 B/C 的历史，不包含编辑分支 A4。
- 请求 6 恢复编辑分支 A4，不包含原 C，也不包含原分支后续 D。
- 请求 1–6 的 `x-session-affinity` 相同，值等于 Pi session id。
- 请求 7 的 `x-session-affinity` 等于 fork 后的新 Pi session id，与前六次不同。
- 七次请求均有与自身 Pi session id 相同的 `prompt_cache_key`。
- OpenAI `messages` 不包含 Pi message `id`、`revisionRootId`、`activeRevision`、`beforeMessageId` 或 `afterMessageId`。

因此不能从 HTTP 请求读取 Pi 内部 revision 序号；可以通过稳定会话身份和消息内容历史确定分支。

## 验收结果

7 次 HTTP completion 请求，10 项检查全部通过：

1. 未修改的便携版在隔离 profile 中正常启动。
2. 正常续接包含之前的 user/assistant 历史。
3. 截断重发保留共同前缀并移除原后续历史。
4. Pi 保存了两个 revision。
5. 原 revision 的历史正确传到 provider，排除新 revision 的回答。
6. 新 revision 的历史正确传到 provider，排除原 revision 的后续消息。
7. regenerate/revision switching 保持同一会话 affinity。
8. fork 只保留选定位置的前缀。
9. fork 使用独立会话 affinity。
10. OpenAI 请求依靠内容历史，未传 Pi 内部 revision id。

便携版退出后，检查到便携版调试端口监听数为 0、测试便携版进程数为 0；安装版进程仍在运行。Pi Desktop 源码 checkout 没有产生改动。

## 代理实现策略

探针前的版本仅恢复一个最新 cursor，再通过 `delta_messages` 取最后一条 assistant 之后的新输入。这会把请求 4、5、6 的不同历史误当成同一条最新分支上的普通续接。

当前实现以入站 HTTP 消息历史为依据：

1. 历史规范化指纹包含角色、文本/图片内容、工具名和参数及工具结果，忽略 reasoning、provider 和 UI 元数据；等价文本表示、JSON 参数键顺序和工具调用 ID 改写保持稳定。
2. SQLite `thread_history` 只保存当前成功 head 的指纹、账号和消息数，不保存会话正文。cursor、账号绑定和可信检查点在同一事务中提交。
3. 在原有线程锁内匹配请求基点。账号、模型/工具范围、消息数和历史全部匹配时续接，否则新建 DeepSeek session 并 replay 当前选中的有效历史。
4. 回退、编辑重发、同请求重新生成、revision 切换和压缩都可以触发重建；fork 使用独立会话 key。
5. 本轮创建 session、上传、completion、重试和工具修复固定同一账号；切换账号后重放，避免使用其他账号的 session/cursor。
6. 上游工作前清除旧检查点，完整流和工具修复成功后才保存新检查点；失败、不完整流和已检测到的断开不留下可信 head。

当前采用重放实现分支语义，不保存旧 head 到任意上游 cursor 的可复用分支树。第二阶段若要复用原生分支，仍需单独验证旧 `parent_message_id` 的有效性及首次生成/重新生成语义。当前实现没有增加 Pi Desktop 自定义 header 或修改其 session 文件格式。

## 适用范围与未验证部分

- 本次验证确认的是上述便携版在短会话下的实际 provider 请求行为。长会话压缩后发送的是摘要/保留尾部构成的有效上下文，不一定是原始完整 transcript；代理需要将此类历史变化作为单独的重建情形处理。
- 目标历史会话 `d8b040a3-9411-477c-8267-8665aab18d9c` 的会话/revision 文件已读取结构，但原始 HTTP 请求没有被保存。本次是独立的可控复现，没有恢复该历史会话当时的网络数据。
- 该历史 UUID 与项目根目录运行数据库中 7 条线程的精确关联仍未查实。仅凭没有命中 hash，不能断言数据库错误或由某个特定 header 导致。
- 便携版探针只验证了 Pi Desktop 的请求形状；DeepSeek 上游的代理集成验证另见下节。

## 代理集成验收

使用 `D:\Packages\deepseek-proxy\deepseek-rs.db` 的一致性临时副本，启动只监听 `127.0.0.1:19176` 的隔离代理。没有修改 Pi Desktop 源码、安装版客户端或其 profile。

- 启动时账号登录、PoW 和 `/healthz` 成功。
- `tools/verify_branches.py` 的 7 个阶段、21 项检查全部通过，覆盖普通续接复用 session、回退后排除已撤回内容、两分支来回切换及同请求重新生成。
- 服务重启后继续同一合成会话，远端 session 复用、上下文保留和持久化检查点三项检查全部通过。
- 验证工具通过只读连接检查临时数据库，报告仅包含检查结果，不保存模型回复或凭据。
- 隔离服务已停止，临时数据库和便携版 profile 已删除，端口监听数为 0。原数据库的 SHA-256、大小和修改时间在验证前后完全一致。

这些验证确认了通过新 session/replay 重建选中历史的行为，没有验证任意旧 DeepSeek parent cursor 的原生分支复用。
