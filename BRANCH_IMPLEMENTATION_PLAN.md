# Pi Desktop branch compatibility implementation plan

## Outcome and boundaries

Support Pi Desktop regeneration, edit/resend, revision switching, and independent forks by changing deepseek-proxy only. Preserve the stream-only API. The portable-client probe in PI_DESKTOP_BRANCH_VALIDATION.md established that a stable session header and the selected effective message history already reach the proxy; Pi revision/message IDs do not.

This slice prioritizes context correctness: reuse a remote session only if the request base exactly matches the last successful canonical history and the account still matches. A divergent, compacted, legacy, failed, or unknown history starts a fresh remote session and replays the request history. Native reuse of arbitrary old remote parent cursors is deferred until its upstream behavior is verified. No new public branch endpoints or client protocol changes.

## Independent development units

| Unit | Owned files | Interface | Dependencies |
| --- | --- | --- | --- |
| History normalization | crates/proxy/src/history.rs (new) | fingerprint(&[serde_json::Value]) -> String | Existing sha2/serde_json/hex dependencies |
| Durable successful head | crates/proxy/src/store.rs | ThreadHistory; get_thread_history; clear_thread_history; save_thread_checkpoint | Existing ThreadResume and rusqlite connection |
| Black-box verification | tools/verify_branches.py (new) | --base-url; optional --db/--api-key-env/--output | Running proxy; synthetic conversation only |
| Lifecycle integration | crates/proxy/src/main.rs, README.md and context docs | Request preparation, fixed account, completion commit | Three interfaces above |

All writers have disjoint ownership. The parent integrates main.rs and validates the combined result. Subagents must not reformat unrelated files, edit owned paths concurrently, access real credentials, or run long foreground builds/tests. Temporary artifacts belong in session scratch.

## Fixed internal contracts

### History module

`pub fn fingerprint(messages: &[serde_json::Value]) -> String` returns a versioned SHA-256 digest. Normalize the message role/content/name, assistant tool names/arguments, and tool results. Parse and canonicalize valid JSON arguments. Preserve ordering, exact meaningful text, and image URLs; strings and equivalent text content parts compare equally. Ignore reasoning/provider/UI metadata. Canonicalize assistant tool-call IDs by occurrence and match result IDs to that occurrence, so client-side ID rewriting alone does not create a new branch; unknown result IDs remain distinct. No raw message content is persisted by this module.

The parent computes the incoming base from the prefix through the final assistant; the suffix is the current user/tool/system input. The expected successful head is the full incoming history plus exactly the prose and accepted calls sent to the client. Empty assistant content normalizes consistently with null/empty arrays. New system messages or changed message/tool semantics may conservatively trigger replay.

### Store module

Add `ThreadHistory { fingerprint: String, account_email: String, message_count: usize }` deriving Debug/Clone/PartialEq/Eq, without changing ThreadResume. Add:

- `get_thread_history(&self, thread_id: &str) -> Result<Option<ThreadHistory>, StoreError>`
- `clear_thread_history(&self, thread_id: &str) -> Result<(), StoreError>`
- `save_thread_checkpoint(&self, thread_id: &str, resume: &ThreadResume, history: &ThreadHistory) -> Result<(), StoreError>`

Use an additive `thread_history` table keyed by thread_id. Legacy DBs have no trusted history and safely replay once. save_thread_checkpoint must atomically update threads, account binding, and thread_history; clear removes the trust marker before upstream work. delete_thread clears thread_history in its existing transaction. Normal save_thread must invalidate a previous history marker so old callers cannot leave a stale trusted checkpoint behind. Do not store prompts or response bodies. Tests use memory/temp DBs only.

### Lifecycle integration

- Take the existing thread lock before reading history and preparing a remote session.
- Select one account and retain its Arc<Connection> for session creation, uploads, completion retries, context/cursor rebuilds, and correction turns.
- Continue only on a trusted matching base/account; otherwise create a new session and use the complete replay prompt.
- Clear history trust before any remote turn; commit a new checkpoint only after successful completion and accepted tool repair, with the final cursor.
- Failed, partial, or disconnected streams leave no trusted successful head. Next full-history request replays safely.
- Account failover may prepare routing for the next request, but must never send a session or cursor to another account during this turn.
- Preserve model type, thinking/search, per-account gates, tool protocol, usage/SSE behavior, and the request's effective history.
- Replay intake considers attachments from the replayed history within the existing one-image-per-prompt limit and reports dropped images.

## Acceptance

1. Ordinary full-history continuation retains remote session.
2. Edit/regenerate/rewind causes a new session without the discarded future.
3. Returning to either selected old branch reconstructs that branch and excludes the other branch's future.
4. Same prompt regeneration is safe even if message count is unchanged.
5. Checkpoints survive restart; legacy checkpoints replay safely once.
6. Tool call/result ID rewriting alone does not force replay; changed arguments/results do.
7. A different account never reuses the old session/cursor.
8. A failure or client disconnect never creates a trusted successful head.
9. Stream-only validation, corpus tests, and existing tool/cursor tests remain passing.

Run targeted module tests and cargo test --workspace after integration. Verify one real branching sequence with a temporary copy of the authorized account DB; start/requests/stop and long tests in background tasks. Preserve the installed Pi Desktop session and do not modify its profile. Record exact checks and residual limitations in the final report.
