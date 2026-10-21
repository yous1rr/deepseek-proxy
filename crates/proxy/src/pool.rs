//! Multi-account pool with per-account rate gating and failover.
//!
//! Port of `ConnectionPool` / `Connection` from the Python proxy:
//!
//! - Every account has its own gate. Only one request per account runs at a
//!   time, and the next one waits `request_delay` after the previous
//!   **finishes** (completion-to-start, not start-to-start).
//! - A thread binds to an account for its lifetime. New threads go to the
//!   least-crowded healthy account; the binding survives restarts via SQLite.
//! - A login-dead account enters cooldown and its threads fail over to the
//!   least-crowded remaining healthy account.

use client::Client;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, OwnedMutexGuard, RwLock};
use tokio::time::Instant;

#[derive(Debug)]
struct GateState {
    last_fire: Option<Instant>,
    cooldown_until: Option<Instant>,
}

/// One DeepSeek account and its rate gate.
pub struct Connection {
    pub email: String,
    password: String,
    /// Explicitly configured token (`DEEPSEEK_TOKEN_<i>`); never overwritten.
    token: Option<String>,
    client: RwLock<Client>,
    gate: Arc<Mutex<GateState>>,
    request_delay: Duration,
    waiting: AtomicUsize,
    queue_limit: usize,
    queue_timeout: Duration,
    store: crate::store::Store,
}

impl Connection {
    fn new(
        email: String,
        password: String,
        token: Option<String>,
        client: Client,
        request_delay: Duration,
        queue_limit: usize,
        queue_timeout: Duration,
        store: crate::store::Store,
    ) -> Self {
        Self {
            email,
            password,
            token,
            client: RwLock::new(client),
            gate: Arc::new(Mutex::new(GateState {
                last_fire: None,
                cooldown_until: None,
            })),
            request_delay,
            waiting: AtomicUsize::new(0),
            queue_limit,
            queue_timeout,
            store,
        }
    }

    pub async fn client(&self) -> Client {
        self.client.read().await.clone()
    }

    /// Wait for this account's rate slot. The returned ticket holds the slot;
    /// dropping it records the completion time and lets the next request in.
    pub async fn acquire(self: &Arc<Self>) -> Result<RateTicket, String> {
        if self.waiting.load(Ordering::Relaxed) >= self.queue_limit {
            return Err("account request queue is full".into());
        }
        self.waiting.fetch_add(1, Ordering::Relaxed);

        let guard =
            match tokio::time::timeout(self.queue_timeout, self.gate.clone().lock_owned()).await {
                Ok(g) => g,
                Err(_) => {
                    self.waiting.fetch_sub(1, Ordering::Relaxed);
                    return Err(format!(
                        "account queue timeout after {}s",
                        self.queue_timeout.as_secs_f64()
                    ));
                }
            };
        self.waiting.fetch_sub(1, Ordering::Relaxed);

        let now = Instant::now();
        let deadline = [
            guard.last_fire.map(|t| t + self.request_delay),
            guard.cooldown_until,
        ]
        .into_iter()
        .flatten()
        .max();

        if let Some(deadline) = deadline {
            let wait = deadline.saturating_duration_since(now);
            if !wait.is_zero() {
                tokio::time::sleep(wait).await;
            }
        }

        Ok(RateTicket { guard: Some(guard) })
    }

    /// Hold this account out of service for `cooldown` without moving threads.
    ///
    /// The rate-limit backoff uses this so the gate itself sleeps: a second
    /// timer would race `acquire()` and let the retry fire too early.
    pub async fn cool_down(&self, cooldown: Duration) {
        self.penalize(cooldown).await;
    }

    /// Mark this account unhealthy for `cooldown`.
    async fn penalize(&self, cooldown: Duration) {
        let mut g = self.gate.lock().await;
        let until = Instant::now() + cooldown;
        g.cooldown_until = Some(match g.cooldown_until {
            Some(existing) if existing > until => existing,
            _ => until,
        });
    }

    /// Point-in-time view of this account for `stats`.
    pub async fn snapshot(&self, bound_threads: usize) -> ConnectionSnapshot {
        let g = self.gate.lock().await;
        let now = Instant::now();
        let cooldown_ms = g
            .cooldown_until
            .filter(|u| *u > now)
            .map(|u| u.saturating_duration_since(now).as_millis() as u64);
        ConnectionSnapshot {
            email: self.email.clone(),
            healthy: cooldown_ms.is_none(),
            bound_threads,
            waiting: self.waiting.load(Ordering::Relaxed),
            cooldown_ms,
        }
    }

    async fn healthy(&self) -> bool {
        let g = self.gate.lock().await;
        g.cooldown_until
            .map(|u| u <= Instant::now())
            .unwrap_or(true)
    }

    /// Milliseconds before this account may serve traffic again; `0` when it
    /// is already healthy. Used to build a `Retry-After` hint.
    #[allow(dead_code)] // reached through `route_checked`, pending main.rs
    async fn cooldown_remaining_ms(&self) -> u64 {
        let g = self.gate.lock().await;
        match g.cooldown_until {
            Some(until) => until.saturating_duration_since(Instant::now()).as_millis() as u64,
            None => 0,
        }
    }

    /// Re-login, replacing the client. Resume ids stay valid.
    ///
    /// Any cached token is dropped first: this path is only reached after the
    /// server rejected a token, so reusing it would loop. A configured token
    /// is honoured when no password is available, but a password login wins
    /// because it yields a fresh token.
    async fn reconnect(&self) -> Result<(), String> {
        let _ = self.store.forget_token(&self.email);
        let fresh = self.login(None).await.map_err(|e| e.to_string())?;
        *self.client.write().await = fresh;
        Ok(())
    }

    /// Build a client from `cached`, or log in and persist the new token.
    async fn login(&self, cached: Option<String>) -> Result<Client, String> {
        if let Some(cached) = cached {
            tracing::debug!("account {}: reusing cached token", self.email);
            return Ok(Client::from_token(cached));
        }
        if !self.password.is_empty() {
            let client = Client::login(&self.email, &self.password, client::get_device_id())
                .await
                .map_err(|e| e.to_string())?;
            if let Err(e) = self.store.save_token(&self.email, client.token()) {
                tracing::warn!("account {}: could not cache token: {e}", self.email);
            }
            return Ok(client);
        }
        match &self.token {
            Some(token) => Ok(Client::from_token(token.clone())),
            None => Err(format!("account {}: no password or token", self.email)),
        }
    }

    /// Initial login: configured token > cached token > password login.
    async fn initial_login(&self) -> Result<Client, String> {
        if let Some(token) = &self.token {
            return Ok(Client::from_token(token.clone()));
        }
        let cached = self
            .store
            .get_token(&self.email)
            .ok()
            .flatten()
            .filter(|t| !t.is_empty());
        self.login(cached).await
    }
}

/// One account's state, as reported by `GET /stats`.
#[derive(Debug, Clone)]
pub struct ConnectionSnapshot {
    pub email: String,
    pub healthy: bool,
    pub bound_threads: usize,
    pub waiting: usize,
    pub cooldown_ms: Option<u64>,
}

/// Holds an account's rate slot; release is implicit on drop.
pub struct RateTicket {
    guard: Option<OwnedMutexGuard<GateState>>,
}

impl Drop for RateTicket {
    fn drop(&mut self) {
        if let Some(mut g) = self.guard.take() {
            g.last_fire = Some(Instant::now());
        }
    }
}

/// Why [`Pool::route_checked`] could not hand out an account.
///
/// Deliberately not an error string: callers (the HTTP layer) need to tell "no
/// capacity right now" apart from real failures so they can answer 503 with a
/// retry hint instead of hammering a cooling-down account.
#[allow(dead_code)] // returned by `route_checked`; main.rs maps it to a 503
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteError {
    /// Every configured account is cooling down, so nothing can be assigned.
    NoHealthyAccount {
        /// Remaining cooldown of the account that recovers first, in ms.
        /// `None` when the pool has no accounts at all.
        retry_after_ms: Option<u64>,
    },
}

impl fmt::Display for RouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RouteError::NoHealthyAccount { retry_after_ms } => {
                f.write_str("no healthy account available")?;
                if let Some(ms) = retry_after_ms {
                    write!(f, ": soonest cooldown ends in {ms} ms")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for RouteError {}

pub struct PoolConfig {
    pub request_delay: Duration,
    pub queue_limit: usize,
    /// Max time to wait for a busy account's rate slot. Python: `QUEUE_TIMEOUT`.
    pub queue_timeout: Duration,
    pub login_cooldown: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            request_delay: Duration::ZERO,
            queue_limit: 64,
            queue_timeout: Duration::from_secs(120),
            login_cooldown: Duration::from_secs(300),
        }
    }
}

pub struct Pool {
    conns: Vec<Arc<Connection>>,
    owner: Mutex<HashMap<String, usize>>,
    config: PoolConfig,
    store: crate::store::Store,
}

/// One configured account before login.
pub struct AccountSpec {
    pub email: String,
    pub password: String,
    pub token: Option<String>,
}

impl Pool {
    pub async fn connect(
        accounts: Vec<AccountSpec>,
        config: PoolConfig,
        store: crate::store::Store,
    ) -> Result<Self, String> {
        if accounts.is_empty() {
            return Err("no DeepSeek accounts configured".into());
        }
        let mut conns = Vec::new();
        for a in accounts {
            let conn = Connection::new(
                a.email,
                a.password,
                a.token,
                // Placeholder; replaced by `initial_login` below, which needs
                // the connection's own store handle for the token cache.
                Client::from_token(String::new()),
                config.request_delay,
                config.queue_limit,
                config.queue_timeout,
                store.clone(),
            );
            let client = conn
                .initial_login()
                .await
                .map_err(|e| format!("login {}: {e}", conn.email))?;
            *conn.client.write().await = client;
            conns.push(Arc::new(conn));
        }
        Ok(Self {
            conns,
            owner: Mutex::new(HashMap::new()),
            config,
            store,
        })
    }

    pub fn len(&self) -> usize {
        self.conns.len()
    }

    /// First account, without binding a thread to it. Used for health probes
    /// that must not leave a fake thread in the database.
    pub fn any(&self) -> Arc<Connection> {
        self.conns[0].clone()
    }

    #[allow(dead_code)] // used by tests and future admin UI
    pub fn is_empty(&self) -> bool {
        self.conns.is_empty()
    }

    /// Bind a thread to an account: honour an existing/persisted binding, else
    /// assign the least-crowded healthy account.
    ///
    /// Unchecked: this cannot report "no capacity", so when every account is
    /// cooling down it still hands out its stale binding (or account 0). New
    /// call sites should use [`Pool::route_checked`] and answer 503 on
    /// [`RouteError::NoHealthyAccount`].
    #[allow(dead_code)] // compatibility API; production uses route_checked
    pub async fn route(&self, thread_id: &str) -> Arc<Connection> {
        {
            let owner = self.owner.lock().await;
            if let Some(&idx) = owner.get(thread_id) {
                return self.conns[idx].clone();
            }
        }

        // Persisted binding (survives restart) — unknown emails fall through.
        let bound = self
            .store
            .get_binding(thread_id)
            .ok()
            .flatten()
            .and_then(|email| self.conns.iter().position(|c| c.email == email));

        let idx = bound.unwrap_or(0);
        let chosen = if bound.is_some() {
            idx
        } else {
            // `None` means every account is cooling down; the legacy path keeps
            // working by falling back to account 0.
            self.least_crowded_healthy().await.unwrap_or(0)
        };

        self.owner
            .lock()
            .await
            .insert(thread_id.to_string(), chosen);
        if bound.is_none() {
            let _ = self.store.bind(thread_id, &self.conns[chosen].email);
        }
        self.conns[chosen].clone()
    }

    /// Health-aware [`Pool::route`]: assign the thread an account that can
    /// actually serve it, or say why it cannot.
    ///
    /// Unlike `route()`, this
    ///
    /// - never returns an account that is inside its cooldown window;
    /// - trusts the in-memory binding only while the bound account is healthy,
    ///   and drops it otherwise so the thread can move;
    /// - honours a persisted SQLite binding only while that account is healthy:
    ///   a binding to a cooling-down account is re-pointed at the least-crowded
    ///   healthy account and persisted, like [`Pool::failover`];
    /// - returns `Err(RouteError::NoHealthyAccount)` when every account is
    ///   cooling down instead of silently picking account 0.
    ///
    /// Bindings are left untouched when nothing is healthy, so the thread
    /// returns to its account once the cooldown expires.
    #[allow(dead_code)] // main.rs will call this instead of `route()`
    pub async fn route_checked(&self, thread_id: &str) -> Result<Arc<Connection>, RouteError> {
        // 1. In-memory binding, only while its account can serve traffic.
        let owner_bound = {
            let owner = self.owner.lock().await;
            owner.get(thread_id).copied()
        };
        if let Some(idx) = owner_bound {
            if self.conns[idx].healthy().await {
                return Ok(self.conns[idx].clone());
            }
            // Cooling down: forget the binding so the thread can move.
            self.owner.lock().await.remove(thread_id);
        }

        // 2. Persisted binding (survives restart), same health check.
        let bound = self
            .store
            .get_binding(thread_id)
            .ok()
            .flatten()
            .and_then(|email| self.conns.iter().position(|c| c.email == email));
        if let Some(idx) = bound {
            if self.conns[idx].healthy().await {
                self.owner.lock().await.insert(thread_id.to_string(), idx);
                return Ok(self.conns[idx].clone());
            }
        }

        // 3. Fresh assignment among healthy accounts, or an explicit failure.
        let Some(chosen) = self.least_crowded_healthy().await else {
            return Err(self.no_healthy_account().await);
        };

        self.owner
            .lock()
            .await
            .insert(thread_id.to_string(), chosen);
        if bound != Some(chosen) {
            let _ = self.store.bind(thread_id, &self.conns[chosen].email);
        }
        Ok(self.conns[chosen].clone())
    }

    /// Least-crowded healthy account, or `None` when every account is cooling
    /// down.
    ///
    /// The old `least_crowded` returned index 0 in that case, which silently
    /// routed new work to a known-bad account.
    async fn least_crowded_healthy(&self) -> Option<usize> {
        let counts = self.occupancy().await;
        let mut best: Option<usize> = None;
        let mut best_score = usize::MAX;
        for (i, c) in self.conns.iter().enumerate() {
            if !c.healthy().await {
                continue;
            }
            let score = counts.get(&c.email).copied().unwrap_or(0);
            if best.is_none() || score < best_score {
                best_score = score;
                best = Some(i);
            }
        }
        best
    }

    /// Build the "everything is cooling down" error, carrying the shortest
    /// remaining cooldown as a retry hint.
    async fn no_healthy_account(&self) -> RouteError {
        let mut soonest: Option<u64> = None;
        for c in &self.conns {
            let ms = c.cooldown_remaining_ms().await;
            soonest = Some(match soonest {
                Some(prev) => prev.min(ms),
                None => ms,
            });
        }
        RouteError::NoHealthyAccount {
            retry_after_ms: soonest,
        }
    }

    async fn occupancy(&self) -> HashMap<String, usize> {
        let owner = self.owner.lock().await;
        let mut counts: HashMap<String, usize> = HashMap::new();
        for &idx in owner.values() {
            *counts.entry(self.conns[idx].email.clone()).or_default() += 1;
        }
        counts
    }

    /// Move a thread off a login-dead account. Returns the replacement.
    pub async fn failover(&self, thread_id: &str) -> Option<Arc<Connection>> {
        let current = {
            let owner = self.owner.lock().await;
            owner.get(thread_id).copied()
        };
        if let Some(idx) = current {
            self.conns[idx].penalize(self.config.login_cooldown).await;
        }
        self.owner.lock().await.remove(thread_id);

        let healthy: Vec<usize> = {
            let mut v = Vec::new();
            for (i, c) in self.conns.iter().enumerate() {
                if Some(i) != current && c.healthy().await {
                    v.push(i);
                }
            }
            v
        };
        if healthy.is_empty() {
            return None;
        }

        let counts = self.occupancy().await;
        let chosen = *healthy
            .iter()
            .min_by_key(|&&i| counts.get(&self.conns[i].email).copied().unwrap_or(0))?;

        self.owner
            .lock()
            .await
            .insert(thread_id.to_string(), chosen);
        let _ = self.store.bind(thread_id, &self.conns[chosen].email);
        Some(self.conns[chosen].clone())
    }

    /// Per-account state plus how many threads each one owns.
    pub async fn snapshot(&self) -> Vec<ConnectionSnapshot> {
        let counts = self.occupancy().await;
        let mut out = Vec::with_capacity(self.conns.len());
        for c in &self.conns {
            let bound = counts.get(&c.email).copied().unwrap_or(0);
            out.push(c.snapshot(bound).await);
        }
        out
    }

    /// Try to re-login an account (invalid/expired token recovery).
    pub async fn recover(&self, conn: &Arc<Connection>) -> bool {
        conn.reconnect().await.is_ok()
    }
}

/// True when the error looks like a dead login rather than a transient blip.
pub fn is_login_dead(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    [
        "login failed",
        "risk_device",
        "risk device",
        "biz_code=11",
        "unauthorized",
        "auth failed",
        "authentication failed",
    ]
    .iter()
    .any(|k| m.contains(k))
}

pub fn is_invalid_token(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    m.contains("invalidtoken") || m.contains("invalid token") || m.contains("40003")
}

/// Markers that mean "throttled, come back later" rather than "broken".
///
/// Verbatim from `_RATE_LIMIT_MARKERS` in `deepseek_proxy/backend.py`, including
/// the deliberately broad `busy`/`too many` entries the Python proxy shipped.
const RATE_LIMIT_MARKERS: [&str; 10] = [
    "rate_limit",
    "rate limit",
    "too frequent",
    "too many",
    "429",
    "throttl",
    "overloaded",
    "busy",
    "generation_err",
    "server is temporarily unavailable",
];

/// True when the error is a throttle, i.e. retrying with backoff can work.
pub fn is_rate_limit(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    RATE_LIMIT_MARKERS.iter().any(|k| m.contains(k))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_throttles_not_auth() {
        assert!(is_rate_limit(
            "upstream hint error: Messages too frequent. Try again later."
        ));
        assert!(is_rate_limit("rate_limit_reached"));
        assert!(!is_rate_limit("invalid token"));
        assert!(!is_rate_limit("login failed: risk_device"));
    }

    #[test]
    fn rate_limit_markers_do_not_claim_login_errors() {
        for msg in ["login failed", "risk_device", "authentication failed"] {
            assert!(!is_rate_limit(msg), "{msg} must be a login error");
            assert!(is_login_dead(msg));
        }
    }

    /// Pool backed by an in-memory store and token-only clients: no login, no
    /// network, no database file. Accounts keep their configured order, so
    /// "account 0" in these tests is `emails[0]`.
    fn test_pool(emails: &[&str]) -> Pool {
        let store = crate::store::Store::open_memory().expect("in-memory store");
        let config = PoolConfig::default();
        let conns = emails
            .iter()
            .map(|email| {
                Arc::new(Connection::new(
                    (*email).to_string(),
                    String::new(),
                    None,
                    Client::from_token("test-token".into()),
                    config.request_delay,
                    config.queue_limit,
                    config.queue_timeout,
                    store.clone(),
                ))
            })
            .collect();
        Pool {
            conns,
            owner: Mutex::new(HashMap::new()),
            config,
            store,
        }
    }

    #[tokio::test]
    async fn checked_route_skips_an_account_in_cooldown() {
        let pool = test_pool(&["a@x", "b@x"]);
        pool.conns[0].cool_down(Duration::from_secs(60)).await;

        let conn = pool.route_checked("t1").await.expect("healthy account");
        assert_eq!(conn.email, "b@x", "must not pick the cooling account 0");
    }

    #[tokio::test]
    async fn checked_route_moves_a_thread_off_a_cooling_in_memory_binding() {
        let pool = test_pool(&["a@x", "b@x"]);
        let first = pool.route_checked("t1").await.expect("healthy account");
        assert_eq!(first.email, "a@x");

        first.cool_down(Duration::from_secs(60)).await;

        let second = pool.route_checked("t1").await.expect("healthy account");
        assert_eq!(second.email, "b@x");
    }

    #[tokio::test]
    async fn checked_route_rebinds_a_persisted_binding_to_a_cooling_account() {
        let pool = test_pool(&["a@x", "b@x"]);
        pool.store.bind("t1", "a@x").unwrap();
        pool.conns[0].cool_down(Duration::from_secs(60)).await;

        let conn = pool.route_checked("t1").await.expect("healthy account");
        assert_eq!(conn.email, "b@x");
        assert_eq!(
            pool.store.get_binding("t1").unwrap().as_deref(),
            Some("b@x"),
            "the cooling binding must be replaced, not reused"
        );
    }

    #[tokio::test]
    async fn checked_route_honours_a_persisted_binding_while_healthy() {
        let pool = test_pool(&["a@x", "b@x"]);
        pool.store.bind("t1", "b@x").unwrap();

        let conn = pool.route_checked("t1").await.expect("healthy account");
        assert_eq!(conn.email, "b@x");
        assert_eq!(
            pool.store.get_binding("t1").unwrap().as_deref(),
            Some("b@x")
        );
    }

    #[tokio::test]
    async fn checked_route_reports_no_account_when_all_are_cooling_down() {
        let pool = test_pool(&["a@x", "b@x"]);
        for conn in &pool.conns {
            conn.cool_down(Duration::from_secs(60)).await;
        }

        let err = pool
            .route_checked("t1")
            .await
            .err()
            .expect("no account can serve t1");
        match err {
            RouteError::NoHealthyAccount { retry_after_ms } => {
                let ms = retry_after_ms.expect("both accounts are cooling down");
                assert!(ms > 0 && ms <= 60_000, "unexpected retry hint: {ms}");
            }
        }
        assert!(
            err.to_string().contains("no healthy account available"),
            "{err}"
        );

        // A failed route must not bind the thread to a bad account.
        assert!(!pool.owner.lock().await.contains_key("t1"));
        assert_eq!(pool.store.get_binding("t1").unwrap(), None);
    }

    #[tokio::test]
    async fn legacy_route_still_returns_an_account_when_all_are_cooling_down() {
        let pool = test_pool(&["a@x", "b@x"]);
        for conn in &pool.conns {
            conn.cool_down(Duration::from_secs(60)).await;
        }

        // API compatibility: `route()` cannot fail; the checked path is what
        // reports "nothing healthy".
        let conn = pool.route("t1").await;
        assert_eq!(conn.email, "a@x");
        assert!(pool.owner.lock().await.contains_key("t1"));
    }
}
