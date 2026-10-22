//! SQLite persistence: thread resume state and API keys.
//!
//! Schema mirrors the Python proxy's `dispatch.db` closely enough that the two
//! can share a database during the parity phase:
//!
//! ```sql
//! CREATE TABLE threads (thread_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
//!                      parent_message_id TEXT, updated_at INTEGER);
//! CREATE TABLE api_keys (id TEXT PRIMARY KEY, key_hash TEXT UNIQUE NOT NULL,
//!                        key_prefix TEXT NOT NULL, name TEXT NOT NULL,
//!                        created_at INTEGER);
//! ```
//!
//! `threads.parent_message_id` is stored as TEXT because the values arrive as
//! strings in some responses.
//!
//! `thread_history` is an additive table holding the durable trust marker for
//! the last successfully completed canonical history of a thread
//! (`thread_id`, `fingerprint`, `account_email`, `message_count`,
//! `updated_at`). It never stores prompts or response bodies.

use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// Resume state for one conversation thread.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadResume {
    pub session_id: String,
    pub parent_message_id: Option<String>,
}

/// Durable trust marker for the last successfully completed canonical
/// history of a thread. Only a fingerprint is persisted; prompts and response
/// bodies never enter this table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadHistory {
    pub fingerprint: String,
    pub account_email: String,
    pub message_count: usize,
}

/// One account added through the CLI (`deepseek-rs accounts add`).
#[derive(Debug, Clone)]
pub struct AccountRow {
    pub email: String,
    pub password: String,
    pub token: Option<String>,
    pub created_at: i64,
}

/// One registered API key (only its hash is stored).
#[derive(Debug, Clone)]
pub struct ApiKeyRow {
    pub id: String,
    pub name: String,
    pub prefix: String,
    pub created_at: i64,
}

/// Row counts for `deepseek-rs stats` when the server is not running.
#[derive(Debug, Clone, Default)]
pub struct DbCounts {
    pub threads: i64,
    pub bindings: i64,
    pub api_keys: i64,
    pub accounts: i64,
    pub cached_tokens: i64,
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
}

/// Thread-safe handle to the proxy database.
#[derive(Clone)]
pub struct Store {
    conn: Arc<Mutex<Connection>>,
}

impl Store {
    #[allow(dead_code)] // test helper
    pub fn open_memory() -> Result<Self, StoreError> {
        Self::open_path(":memory:")
    }

    /// Open (and migrate) a database at `path`.
    pub fn open_path(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let conn = Connection::open(path)?;
        Self::migrate(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn migrate(conn: &Connection) -> Result<(), StoreError> {
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS threads (
                 thread_id         TEXT PRIMARY KEY,
                 session_id        TEXT NOT NULL,
                 parent_message_id TEXT,
                 updated_at        INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS api_keys (
                 id         TEXT PRIMARY KEY,
                 key_hash   TEXT UNIQUE NOT NULL,
                 key_prefix TEXT NOT NULL,
                 name       TEXT NOT NULL,
                 created_at INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS thread_bindings (
                 thread_id  TEXT PRIMARY KEY,
                 email      TEXT NOT NULL,
                 updated_at INTEGER NOT NULL
             );
             -- Cached login tokens so a restart skips the login round trip.
             CREATE TABLE IF NOT EXISTS account_tokens (
                 email      TEXT PRIMARY KEY,
                 token      TEXT NOT NULL,
                 updated_at INTEGER NOT NULL
             );
             -- Accounts added with the CLI. Environment credentials still work
             -- and are merged on top of these at startup.
             CREATE TABLE IF NOT EXISTS accounts (
                 email      TEXT PRIMARY KEY,
                 password   TEXT NOT NULL DEFAULT '',
                 token      TEXT,
                 created_at INTEGER NOT NULL
             );
             -- Durable trust marker for the last successfully completed
             -- canonical history of a thread. Additive: a legacy DB simply has
             -- no row and safely replays once. Only a fingerprint is stored.
             CREATE TABLE IF NOT EXISTS thread_history (
                 thread_id     TEXT PRIMARY KEY,
                 fingerprint   TEXT NOT NULL,
                 account_email TEXT NOT NULL,
                 message_count INTEGER NOT NULL,
                 updated_at    INTEGER NOT NULL
             );",
        )?;
        Ok(())
    }

    /// Account email a thread is bound to (survives restarts).
    pub fn get_binding(&self, thread_id: &str) -> Result<Option<String>, StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        let row = conn
            .query_row(
                "SELECT email FROM thread_bindings WHERE thread_id = ?1",
                params![thread_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(row)
    }

    pub fn bind(&self, thread_id: &str, email: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        conn.execute(
            "INSERT INTO thread_bindings (thread_id, email, updated_at)
             VALUES (?1, ?2, unixepoch())
             ON CONFLICT(thread_id) DO UPDATE SET
                 email = excluded.email, updated_at = excluded.updated_at",
            params![thread_id, email],
        )?;
        Ok(())
    }

    /// Cached bearer token for an account, if one was stored by a login.
    pub fn get_token(&self, email: &str) -> Result<Option<String>, StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        Ok(conn
            .query_row(
                "SELECT token FROM account_tokens WHERE email = ?1",
                params![email],
                |r| r.get(0),
            )
            .optional()?)
    }

    pub fn save_token(&self, email: &str, token: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        conn.execute(
            "INSERT INTO account_tokens (email, token, updated_at)
             VALUES (?1, ?2, unixepoch())
             ON CONFLICT(email) DO UPDATE SET
                 token = excluded.token, updated_at = excluded.updated_at",
            params![email, token],
        )?;
        Ok(())
    }

    /// Drop a cached token (it was rejected upstream).
    pub fn forget_token(&self, email: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        conn.execute(
            "DELETE FROM account_tokens WHERE email = ?1",
            params![email],
        )?;
        Ok(())
    }

    /// Accounts added through the CLI, oldest first.
    pub fn accounts(&self) -> Result<Vec<AccountRow>, StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        let mut stmt = conn.prepare(
            "SELECT email, password, token, created_at FROM accounts
             ORDER BY created_at, email",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(AccountRow {
                email: r.get(0)?,
                password: r.get(1)?,
                token: r.get(2)?,
                created_at: r.get(3)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Insert or update an account. An empty password keeps the stored one, and
    /// `None` keeps the stored token.
    pub fn upsert_account(
        &self,
        email: &str,
        password: &str,
        token: Option<&str>,
    ) -> Result<(), StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        conn.execute(
            "INSERT INTO accounts (email, password, token, created_at)
             VALUES (?1, ?2, ?3, unixepoch())
             ON CONFLICT(email) DO UPDATE SET
                 password = CASE WHEN excluded.password = ''
                                 THEN accounts.password ELSE excluded.password END,
                 token    = COALESCE(excluded.token, accounts.token)",
            params![email, password, token],
        )?;
        Ok(())
    }

    /// Remove an account and any token cached for it. True when a row was deleted.
    pub fn delete_account(&self, email: &str) -> Result<bool, StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        let n = conn.execute("DELETE FROM accounts WHERE email = ?1", params![email])?;
        conn.execute(
            "DELETE FROM account_tokens WHERE email = ?1",
            params![email],
        )?;
        Ok(n > 0)
    }

    pub fn api_keys(&self) -> Result<Vec<ApiKeyRow>, StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, name, key_prefix, created_at FROM api_keys
             ORDER BY created_at, name",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(ApiKeyRow {
                id: r.get(0)?,
                name: r.get(1)?,
                prefix: r.get(2)?,
                created_at: r.get(3)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Revoke by name or id. Returns how many rows were removed.
    pub fn delete_api_key(&self, name_or_id: &str) -> Result<usize, StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        Ok(conn.execute(
            "DELETE FROM api_keys WHERE name = ?1 OR id = ?1",
            params![name_or_id],
        )?)
    }

    /// Cheap row counts for `stats` when the server is not running.
    pub fn counts(&self) -> Result<DbCounts, StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        let one =
            |sql: &str| -> Result<i64, rusqlite::Error> { conn.query_row(sql, [], |r| r.get(0)) };
        Ok(DbCounts {
            threads: one("SELECT COUNT(*) FROM threads")?,
            bindings: one("SELECT COUNT(*) FROM thread_bindings")?,
            api_keys: one("SELECT COUNT(*) FROM api_keys")?,
            accounts: one("SELECT COUNT(*) FROM accounts")?,
            cached_tokens: one("SELECT COUNT(*) FROM account_tokens")?,
        })
    }

    pub fn get_thread(&self, thread_id: &str) -> Result<Option<ThreadResume>, StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        let row = conn
            .query_row(
                "SELECT session_id, parent_message_id FROM threads WHERE thread_id = ?1",
                params![thread_id],
                |r| {
                    Ok(ThreadResume {
                        session_id: r.get(0)?,
                        parent_message_id: r.get(1)?,
                    })
                },
            )
            .optional()?;
        Ok(row)
    }

    /// Save resume state without a trusted history checkpoint.
    ///
    /// Any previous trust marker is removed in the same transaction: callers
    /// that only move the cursor must not leave a stale fingerprint behind,
    /// or a later request could reuse a remote session whose history no longer
    /// matches.
    pub fn save_thread(&self, thread_id: &str, resume: &ThreadResume) -> Result<(), StoreError> {
        let mut conn = self.conn.lock().expect("store lock poisoned");
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO threads (thread_id, session_id, parent_message_id, updated_at)
             VALUES (?1, ?2, ?3, unixepoch())
             ON CONFLICT(thread_id) DO UPDATE SET
                 session_id = excluded.session_id,
                 parent_message_id = excluded.parent_message_id,
                 updated_at = excluded.updated_at",
            params![thread_id, resume.session_id, resume.parent_message_id],
        )?;
        tx.execute(
            "DELETE FROM thread_history WHERE thread_id = ?1",
            params![thread_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Trusted history marker for a thread, if its last turn completed.
    pub fn get_thread_history(&self, thread_id: &str) -> Result<Option<ThreadHistory>, StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        let row = conn
            .query_row(
                "SELECT fingerprint, account_email, message_count
                 FROM thread_history WHERE thread_id = ?1",
                params![thread_id],
                |r| {
                    Ok(ThreadHistory {
                        fingerprint: r.get(0)?,
                        account_email: r.get(1)?,
                        message_count: r.get(2)?,
                    })
                },
            )
            .optional()?;
        Ok(row)
    }

    /// Remove a thread's trust marker. Called before any upstream work so an
    /// interrupted or failed turn never leaves a trusted head behind.
    pub fn clear_thread_history(&self, thread_id: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        conn.execute(
            "DELETE FROM thread_history WHERE thread_id = ?1",
            params![thread_id],
        )?;
        Ok(())
    }

    /// Commit a successful turn: the resume cursor, the account binding and
    /// the trusted history marker move together in one transaction. Either
    /// all three are durable or none is.
    pub fn save_thread_checkpoint(
        &self,
        thread_id: &str,
        resume: &ThreadResume,
        history: &ThreadHistory,
    ) -> Result<(), StoreError> {
        let mut conn = self.conn.lock().expect("store lock poisoned");
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO threads (thread_id, session_id, parent_message_id, updated_at)
             VALUES (?1, ?2, ?3, unixepoch())
             ON CONFLICT(thread_id) DO UPDATE SET
                 session_id = excluded.session_id,
                 parent_message_id = excluded.parent_message_id,
                 updated_at = excluded.updated_at",
            params![thread_id, resume.session_id, resume.parent_message_id],
        )?;
        tx.execute(
            "INSERT INTO thread_bindings (thread_id, email, updated_at)
             VALUES (?1, ?2, unixepoch())
             ON CONFLICT(thread_id) DO UPDATE SET
                 email = excluded.email, updated_at = excluded.updated_at",
            params![thread_id, history.account_email],
        )?;
        tx.execute(
            "INSERT INTO thread_history
                 (thread_id, fingerprint, account_email, message_count, updated_at)
             VALUES (?1, ?2, ?3, ?4, unixepoch())
             ON CONFLICT(thread_id) DO UPDATE SET
                 fingerprint = excluded.fingerprint,
                 account_email = excluded.account_email,
                 message_count = excluded.message_count,
                 updated_at = excluded.updated_at",
            params![
                thread_id,
                history.fingerprint,
                history.account_email,
                history.message_count,
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Drop a thread, its account binding and its history marker in one
    /// transaction, so a deleted thread can never leave an orphan binding or
    /// trusted fingerprint behind. Deleting an unknown thread is a no-op.
    #[allow(dead_code)] // used by tests and future admin UI
    pub fn delete_thread(&self, thread_id: &str) -> Result<(), StoreError> {
        let mut conn = self.conn.lock().expect("store lock poisoned");
        let tx = conn.transaction()?;
        tx.execute(
            "DELETE FROM threads WHERE thread_id = ?1",
            params![thread_id],
        )?;
        tx.execute(
            "DELETE FROM thread_bindings WHERE thread_id = ?1",
            params![thread_id],
        )?;
        tx.execute(
            "DELETE FROM thread_history WHERE thread_id = ?1",
            params![thread_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Register an API key. Only the hash is stored.
    pub fn insert_api_key(&self, name: &str, key: &str) -> Result<(), StoreError> {
        let hash = hash_key(key);
        let prefix = key.chars().take(8).collect::<String>();
        let id = format!("key-{hash:.12}");
        let conn = self.conn.lock().expect("store lock poisoned");
        conn.execute(
            "INSERT INTO api_keys (id, key_hash, key_prefix, name, created_at)
             VALUES (?1, ?2, ?3, ?4, unixepoch())",
            params![id, hash, prefix, name],
        )?;
        Ok(())
    }

    /// True when `key` matches a stored hash. Always true when no keys exist,
    /// so a fresh install is not locked out.
    pub fn verify_api_key(&self, key: &str) -> Result<bool, StoreError> {
        let conn = self.conn.lock().expect("store lock poisoned");
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM api_keys", [], |r| r.get(0))?;
        if count == 0 {
            return Ok(true);
        }
        let hash = hash_key(key);
        let hit: Option<i64> = conn
            .query_row(
                "SELECT 1 FROM api_keys WHERE key_hash = ?1",
                params![hash],
                |r| r.get(0),
            )
            .optional()?;
        Ok(hit.is_some())
    }
}

pub fn hash_key(key: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(key.as_bytes());
    hex::encode(h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_cache_roundtrip() {
        let s = Store::open_memory().unwrap();
        assert_eq!(s.get_token("a@b").unwrap(), None);

        s.save_token("a@b", "tok-1").unwrap();
        assert_eq!(s.get_token("a@b").unwrap().as_deref(), Some("tok-1"));

        // Re-login overwrites rather than duplicating.
        s.save_token("a@b", "tok-2").unwrap();
        assert_eq!(s.get_token("a@b").unwrap().as_deref(), Some("tok-2"));

        // A rejected token is dropped so the next start logs in again.
        s.forget_token("a@b").unwrap();
        assert_eq!(s.get_token("a@b").unwrap(), None);
    }

    #[test]
    fn thread_roundtrip() {
        let s = Store::open_memory().unwrap();
        assert_eq!(s.get_thread("t1").unwrap(), None);

        s.save_thread(
            "t1",
            &ThreadResume {
                session_id: "sess".into(),
                parent_message_id: Some("42".into()),
            },
        )
        .unwrap();

        let got = s.get_thread("t1").unwrap().unwrap();
        assert_eq!(got.session_id, "sess");
        assert_eq!(got.parent_message_id.as_deref(), Some("42"));

        s.save_thread(
            "t1",
            &ThreadResume {
                session_id: "sess".into(),
                parent_message_id: Some("43".into()),
            },
        )
        .unwrap();
        assert_eq!(
            s.get_thread("t1")
                .unwrap()
                .unwrap()
                .parent_message_id
                .as_deref(),
            Some("43"),
            "upsert must overwrite"
        );
    }

    #[test]
    fn delete_thread_removes_thread_and_binding() {
        let s = Store::open_memory().unwrap();
        s.save_thread(
            "t1",
            &ThreadResume {
                session_id: "sess".into(),
                parent_message_id: Some("42".into()),
            },
        )
        .unwrap();
        s.bind("t1", "a@b").unwrap();
        assert_eq!(s.get_binding("t1").unwrap().as_deref(), Some("a@b"));

        s.delete_thread("t1").unwrap();

        assert_eq!(s.get_thread("t1").unwrap(), None);
        assert_eq!(s.get_binding("t1").unwrap(), None);
        assert_eq!(s.counts().unwrap().threads, 0);
        assert_eq!(s.counts().unwrap().bindings, 0);

        // Deleting again is a harmless no-op.
        s.delete_thread("t1").unwrap();
        assert_eq!(s.get_thread("t1").unwrap(), None);
        assert_eq!(s.get_binding("t1").unwrap(), None);
    }

    #[test]
    fn delete_thread_leaves_other_threads_untouched() {
        let s = Store::open_memory().unwrap();
        for id in ["t1", "t2"] {
            s.save_thread(
                id,
                &ThreadResume {
                    session_id: format!("sess-{id}"),
                    parent_message_id: None,
                },
            )
            .unwrap();
            s.bind(id, "a@b").unwrap();
        }

        s.delete_thread("t1").unwrap();

        assert_eq!(s.get_thread("t1").unwrap(), None);
        assert_eq!(s.get_binding("t1").unwrap(), None);
        assert_eq!(
            s.get_thread("t2").unwrap().unwrap().session_id,
            "sess-t2".to_string()
        );
        assert_eq!(s.get_binding("t2").unwrap().as_deref(), Some("a@b"));
    }

    #[test]
    fn api_keys_open_when_none_registered() {
        let s = Store::open_memory().unwrap();
        assert!(s.verify_api_key("anything").unwrap());
    }

    #[test]
    fn api_key_accepts_registered_and_rejects_others() {
        let s = Store::open_memory().unwrap();
        s.insert_api_key("test", "sk-secret-123").unwrap();
        assert!(s.verify_api_key("sk-secret-123").unwrap());
        assert!(!s.verify_api_key("sk-wrong").unwrap());
    }

    #[test]
    fn hash_is_stable_and_hides_input() {
        let h = hash_key("sk-abc");
        assert_eq!(h, hash_key("sk-abc"));
        assert_ne!(h, hash_key("sk-abd"));
        assert!(!h.contains("sk-abc"));
    }

    fn resume(session: &str, parent: Option<&str>) -> ThreadResume {
        ThreadResume {
            session_id: session.into(),
            parent_message_id: parent.map(str::to_string),
        }
    }

    fn history(fingerprint: &str, email: &str, count: usize) -> ThreadHistory {
        ThreadHistory {
            fingerprint: fingerprint.into(),
            account_email: email.into(),
            message_count: count,
        }
    }

    #[test]
    fn checkpoint_roundtrip_saves_resume_binding_and_history() {
        let s = Store::open_memory().unwrap();
        let first = history("fp-1", "a@b", 3);
        s.save_thread_checkpoint("t1", &resume("sess-1", Some("10")), &first)
            .unwrap();

        assert_eq!(s.get_thread_history("t1").unwrap(), Some(first.clone()));
        assert_eq!(
            s.get_thread("t1").unwrap(),
            Some(resume("sess-1", Some("10")))
        );
        assert_eq!(s.get_binding("t1").unwrap().as_deref(), Some("a@b"));

        // A new checkpoint moves all three parts together.
        let second = history("fp-2", "c@d", 5);
        s.save_thread_checkpoint("t1", &resume("sess-2", None), &second)
            .unwrap();
        assert_eq!(s.get_thread_history("t1").unwrap(), Some(second));
        assert_eq!(s.get_thread("t1").unwrap(), Some(resume("sess-2", None)));
        assert_eq!(s.get_binding("t1").unwrap().as_deref(), Some("c@d"));
    }

    #[test]
    fn checkpoint_rolls_back_resume_and_binding_when_history_write_fails() {
        let s = Store::open_memory().unwrap();
        let before = history("fp-old", "a@b", 2);
        s.save_thread_checkpoint("t1", &resume("sess-old", Some("10")), &before)
            .unwrap();
        {
            let conn = s.conn.lock().unwrap();
            conn.execute_batch(
                "CREATE TRIGGER fail_history_insert BEFORE INSERT ON thread_history
                 BEGIN SELECT RAISE(ABORT, 'injected'); END;
                 CREATE TRIGGER fail_history_update BEFORE UPDATE ON thread_history
                 BEGIN SELECT RAISE(ABORT, 'injected'); END;",
            )
            .unwrap();
        }

        let failed = s.save_thread_checkpoint(
            "t1",
            &resume("sess-new", Some("99")),
            &history("fp-new", "c@d", 9),
        );
        assert!(failed.is_err(), "injected history failure must surface");

        // The threads and thread_bindings writes were rolled back with it.
        assert_eq!(
            s.get_thread("t1").unwrap(),
            Some(resume("sess-old", Some("10")))
        );
        assert_eq!(s.get_binding("t1").unwrap().as_deref(), Some("a@b"));
        assert_eq!(s.get_thread_history("t1").unwrap(), Some(before));
    }

    #[test]
    fn clear_history_preserves_existing_resume_and_binding() {
        let s = Store::open_memory().unwrap();
        s.save_thread_checkpoint(
            "t1",
            &resume("sess-1", Some("10")),
            &history("fp-1", "a@b", 3),
        )
        .unwrap();

        s.clear_thread_history("t1").unwrap();

        assert_eq!(s.get_thread_history("t1").unwrap(), None);
        assert_eq!(
            s.get_thread("t1").unwrap(),
            Some(resume("sess-1", Some("10")))
        );
        assert_eq!(s.get_binding("t1").unwrap().as_deref(), Some("a@b"));

        // Clearing an unknown thread is a harmless no-op.
        s.clear_thread_history("missing").unwrap();
    }

    #[test]
    fn save_thread_invalidates_history_without_touching_other_threads() {
        let s = Store::open_memory().unwrap();
        for id in ["t1", "t2"] {
            s.save_thread_checkpoint(
                id,
                &resume(&format!("sess-{id}"), Some("10")),
                &history(&format!("fp-{id}"), "a@b", 3),
            )
            .unwrap();
        }

        s.save_thread("t1", &resume("sess-moved", Some("77")))
            .unwrap();

        assert_eq!(s.get_thread_history("t1").unwrap(), None);
        assert_eq!(
            s.get_thread("t1").unwrap(),
            Some(resume("sess-moved", Some("77")))
        );
        assert_eq!(
            s.get_thread_history("t2").unwrap(),
            Some(history("fp-t2", "a@b", 3))
        );
    }

    #[test]
    fn delete_thread_removes_history_and_leaves_other_threads_untouched() {
        let s = Store::open_memory().unwrap();
        for id in ["t1", "t2"] {
            s.save_thread_checkpoint(
                id,
                &resume(&format!("sess-{id}"), Some("10")),
                &history(&format!("fp-{id}"), "a@b", 3),
            )
            .unwrap();
        }

        s.delete_thread("t1").unwrap();

        assert_eq!(s.get_thread("t1").unwrap(), None);
        assert_eq!(s.get_binding("t1").unwrap(), None);
        assert_eq!(s.get_thread_history("t1").unwrap(), None);

        assert_eq!(
            s.get_thread("t2").unwrap(),
            Some(resume("sess-t2", Some("10")))
        );
        assert_eq!(s.get_binding("t2").unwrap().as_deref(), Some("a@b"));
        assert_eq!(
            s.get_thread_history("t2").unwrap(),
            Some(history("fp-t2", "a@b", 3))
        );
    }

    #[test]
    fn migration_keeps_legacy_rows_and_persists_new_checkpoints() {
        let path = std::env::temp_dir().join(format!(
            "deepseek-proxy-store-migration-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        {
            // A pre-thread_history database, exactly as older builds left it.
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE threads (
                     thread_id         TEXT PRIMARY KEY,
                     session_id        TEXT NOT NULL,
                     parent_message_id TEXT,
                     updated_at        INTEGER NOT NULL
                 );
                 CREATE TABLE api_keys (
                     id         TEXT PRIMARY KEY,
                     key_hash   TEXT UNIQUE NOT NULL,
                     key_prefix TEXT NOT NULL,
                     name       TEXT NOT NULL,
                     created_at INTEGER NOT NULL
                 );
                 CREATE TABLE thread_bindings (
                     thread_id  TEXT PRIMARY KEY,
                     email      TEXT NOT NULL,
                     updated_at INTEGER NOT NULL
                 );
                 CREATE TABLE accounts (
                     email      TEXT PRIMARY KEY,
                     password   TEXT NOT NULL DEFAULT '',
                     token      TEXT,
                     created_at INTEGER NOT NULL
                 );
                 INSERT INTO threads VALUES ('legacy', 'sess-legacy', '7', 1);
                 INSERT INTO api_keys VALUES ('key-legacy', '3548f766def78742acc7e53dfda55eebcb2c7f1ede2f4f235d8828936af7b440', 'sk-legacy', 'legacy', 1);
                 INSERT INTO thread_bindings VALUES ('legacy', 'old@example', 1);
                 INSERT INTO accounts VALUES ('old@example', 'pw', NULL, 1);",
            )
            .unwrap();
        }

        let s = Store::open_path(&path).unwrap();
        assert_eq!(
            s.get_thread("legacy").unwrap(),
            Some(resume("sess-legacy", Some("7")))
        );
        assert_eq!(
            s.get_binding("legacy").unwrap().as_deref(),
            Some("old@example")
        );
        assert!(s.verify_api_key("sk-legacy").unwrap());
        assert_eq!(
            s.accounts()
                .unwrap()
                .into_iter()
                .map(|a| a.email)
                .collect::<Vec<_>>(),
            vec!["old@example".to_string()]
        );
        assert_eq!(
            s.get_thread_history("legacy").unwrap(),
            None,
            "legacy rows must start with no trusted history"
        );

        let checkpoint = history("fp-migrated", "new@example", 4);
        s.save_thread_checkpoint("legacy", &resume("sess-new", Some("8")), &checkpoint)
            .unwrap();
        drop(s);

        // Reopening migrates again (no-op) and keeps the new checkpoint.
        let reopened = Store::open_path(&path).unwrap();
        assert_eq!(
            reopened.get_thread_history("legacy").unwrap(),
            Some(checkpoint)
        );
        assert_eq!(
            reopened.get_thread("legacy").unwrap(),
            Some(resume("sess-new", Some("8")))
        );
        assert_eq!(
            reopened.get_binding("legacy").unwrap().as_deref(),
            Some("new@example")
        );
        drop(reopened);

        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
        }
    }

    #[test]
    fn thread_history_schema_stores_only_the_trust_marker() {
        let s = Store::open_memory().unwrap();
        let conn = s.conn.lock().unwrap();
        let mut stmt = conn.prepare("PRAGMA table_info(thread_history)").unwrap();
        let columns: Vec<String> = stmt
            .query_map([], |r| r.get::<_, String>(1))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(
            columns,
            vec![
                "thread_id",
                "fingerprint",
                "account_email",
                "message_count",
                "updated_at",
            ]
        );
    }
}
