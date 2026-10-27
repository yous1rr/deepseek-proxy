//! Multi-account pool with per-account rate scheduling, bounded concurrency,
//! and failover.
//!
//! Port of `ConnectionPool` / `Connection` from the Python proxy:
//!
//! - Every account has a bounded number of in-flight requests. The default is
//!   one, preserving the old account-wide serialization behavior.
//! - A start reservation spaces concurrent requests with `request_delay`; when
//!   the limit is one, completion-to-start delay is preserved exactly.
//! - A thread binds to an account for its lifetime. New threads go to the
//!   least-crowded healthy account; the binding survives restarts via SQLite.
//! - A login-dead account enters cooldown and its threads fail over to the
//!   least-crowded remaining healthy account.

use client::Client;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;
use tokio::sync::{Mutex, OwnedSemaphorePermit, RwLock, Semaphore};
use tokio::time::Instant;

#[derive(Debug)]
struct GateState {
    /// Completion time of the most recently finished request.
    last_fire: Option<Instant>,
    /// Start time of the most recently started request.
    last_start: Option<Instant>,
    cooldown_until: Option<Instant>,
    next_reservation_id: u64,
    reservations: BTreeMap<u64, Instant>,
}

/// One DeepSeek account and its rate gate.
pub struct Connection {
    pub email: String,
    password: String,
    /// Explicitly configured token (`DEEPSEEK_TOKEN_<i>`); never overwritten.
    token: Option<String>,
    client: RwLock<Client>,
    /// Scheduler state is held only while reserving/checking a start time; it
    /// is never held across upstream I/O.
    gate: Arc<StdMutex<GateState>>,
    inflight: Arc<Semaphore>,
    max_inflight_per_account: usize,
    reservation_updates: tokio::sync::watch::Sender<()>,
    request_delay: Duration,
    waiting: AtomicUsize,
    queue_limit: usize,
    queue_timeout: Duration,
    store: crate::store::Store,
}

struct WaitingGuard<'a>(&'a AtomicUsize);

impl<'a> WaitingGuard<'a> {
    fn enter(waiting: &'a AtomicUsize, queue_limit: usize) -> Result<Self, String> {
        let mut current = waiting.load(Ordering::Relaxed);
        loop {
            if current >= queue_limit {
                return Err("account request queue is full".into());
            }
            match waiting.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(Self(waiting)),
                Err(observed) => current = observed,
            }
        }
    }
}

impl Drop for WaitingGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

struct ReservationGuard {
    gate: Arc<StdMutex<GateState>>,
    id: u64,
    request_delay: Duration,
    updates: tokio::sync::watch::Sender<()>,
    active: bool,
}

fn rebase_reservations(g: &mut GateState, request_delay: Duration) {
    let now = Instant::now();
    let mut next = now;
    if let Some(last_start) = g.last_start {
        next = next.max(last_start + request_delay);
    }
    if let Some(cooldown) = g.cooldown_until {
        next = next.max(cooldown);
    }
    for start in g.reservations.values_mut() {
        *start = next;
        next += request_delay;
    }
}

impl ReservationGuard {
    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for ReservationGuard {
    fn drop(&mut self) {
        if self.active {
            let mut g = self
                .gate
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            g.reservations.remove(&self.id);
            rebase_reservations(&mut g, self.request_delay);
            let _ = self.updates.send(());
        }
    }
}

impl Connection {
    fn new(
        email: String,
        password: String,
        token: Option<String>,
        client: Client,
        max_inflight_per_account: usize,
        request_delay: Duration,
        queue_limit: usize,
        queue_timeout: Duration,
        store: crate::store::Store,
    ) -> Self {
        let max_inflight_per_account = max_inflight_per_account.max(1);
        let (reservation_updates, _) = tokio::sync::watch::channel(());
        Self {
            email,
            password,
            token,
            client: RwLock::new(client),
            gate: Arc::new(StdMutex::new(GateState {
                last_fire: None,
                last_start: None,
                cooldown_until: None,
                next_reservation_id: 0,
                reservations: BTreeMap::new(),
            })),
            reservation_updates,
            inflight: Arc::new(Semaphore::new(max_inflight_per_account)),
            max_inflight_per_account,
            request_delay,
            waiting: AtomicUsize::new(0),
            queue_limit,
            queue_timeout,
            store,
        }
    }

    fn gate_state(&self) -> std::sync::MutexGuard<'_, GateState> {
        self.gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub async fn client(&self) -> Client {
        self.client.read().await.clone()
    }

    /// Wait for an in-flight permit and a reserved rate start. The returned
    /// ticket keeps the permit until the whole upstream body is finished.
    pub async fn acquire(self: &Arc<Self>) -> Result<RateTicket, String> {
        let _waiting = WaitingGuard::enter(&self.waiting, self.queue_limit)?;
        match tokio::time::timeout(self.queue_timeout, self.acquire_inner()).await {
            Ok(ticket_res) => ticket_res,
            Err(_) => Err(format!(
                "account queue timeout after {}s",
                self.queue_timeout.as_secs_f64()
            )),
        }
    }

    async fn acquire_inner(self: &Arc<Self>) -> Result<RateTicket, String> {
        let permit = self
            .inflight
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| "account request semaphore closed".to_string())?;
        self.wait_for_start().await;
        Ok(RateTicket {
            _permit: permit,
            gate: self.gate.clone(),
            max_inflight_per_account: self.max_inflight_per_account,
        })
    }

    async fn wait_for_start(&self) {
        if self.max_inflight_per_account == 1 {
            self.wait_for_start_serialized().await;
        } else {
            self.wait_for_start_concurrent().await;
        }
    }

    /// Serialized gate (max=1): strictly preserves completion-to-start delay.
    /// Since the caller holds the only in-flight permit, no other request can
    /// start until this one completes. Cancellation does not pollute state.
    async fn wait_for_start_serialized(&self) {
        loop {
            let now = Instant::now();
            let target = {
                let g = self.gate_state();
                let mut t = now;
                if let Some(last) = g.last_fire {
                    t = t.max(last + self.request_delay);
                }
                if let Some(cooldown) = g.cooldown_until {
                    t = t.max(cooldown);
                }
                t
            };

            let wait = target.saturating_duration_since(now);
            if wait.is_zero() {
                let mut g = self.gate_state();
                g.last_start = Some(now);
                return;
            }
            tokio::time::sleep(wait).await;
        }
    }

    /// Concurrent gate (max>1): provides start-to-start spacing.
    /// Start times are reserved with unique ticket IDs. If a waiting request is
    /// cancelled or times out, its reservation is cleanly removed on drop,
    /// preventing future requests from being delayed to a fake future.
    async fn wait_for_start_concurrent(&self) {
        let mut updates = self.reservation_updates.subscribe();
        let id = {
            let mut g = self.gate_state();
            let id = g.next_reservation_id;
            g.next_reservation_id += 1;
            g.reservations.insert(id, Instant::now());
            rebase_reservations(&mut g, self.request_delay);
            let _ = self.reservation_updates.send(());
            id
        };

        let mut guard = ReservationGuard {
            gate: self.gate.clone(),
            id,
            request_delay: self.request_delay,
            updates: self.reservation_updates.clone(),
            active: true,
        };

        loop {
            let now = Instant::now();
            let wait = {
                let mut g = self.gate_state();
                let target = g.reservations.get(&id).copied().unwrap_or(now);

                if target <= now {
                    g.reservations.remove(&id);
                    g.last_start = Some(now);
                    rebase_reservations(&mut g, self.request_delay);
                    guard.disarm();
                    let _ = self.reservation_updates.send(());
                    Duration::ZERO
                } else {
                    target - now
                }
            };

            if wait.is_zero() {
                return;
            }

            tokio::select! {
                _ = tokio::time::sleep(wait) => {}
                _ = updates.changed() => {}
            }
        }
    }

    /// Hold this account out of service for `cooldown` without moving threads.
    ///
    /// The rate-limit backoff uses this so the scheduler itself sleeps: a
    /// second timer would race `acquire()` and let the retry fire too early.
    pub async fn cool_down(&self, cooldown: Duration) {
        self.penalize(cooldown).await;
    }

    /// Mark this account unhealthy for `cooldown`.
    async fn penalize(&self, cooldown: Duration) {
        let mut g = self.gate_state();
        let until = Instant::now() + cooldown;
        g.cooldown_until = Some(match g.cooldown_until {
            Some(existing) if existing > until => existing,
            _ => until,
        });
        rebase_reservations(&mut g, self.request_delay);
        let _ = self.reservation_updates.send(());
    }

    /// Point-in-time view of this account for `stats`.
    pub async fn snapshot(&self, bound_threads: usize) -> ConnectionSnapshot {
        let g = self.gate_state();
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
        let g = self.gate_state();
        g.cooldown_until
            .map(|u| u <= Instant::now())
            .unwrap_or(true)
    }

    /// Milliseconds before this account may serve traffic again; `0` when it
    /// is already healthy. Used to build a `Retry-After` hint.
    #[allow(dead_code)] // reached through `route_checked`, pending main.rs
    async fn cooldown_remaining_ms(&self) -> u64 {
        let g = self.gate_state();
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

/// Holds an account's in-flight permit; release is implicit on drop.
pub struct RateTicket {
    _permit: OwnedSemaphorePermit,
    gate: Arc<StdMutex<GateState>>,
    #[allow(dead_code)]
    max_inflight_per_account: usize,
}

impl Drop for RateTicket {
    fn drop(&mut self) {
        let now = Instant::now();
        let mut g = self
            .gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        g.last_fire = Some(now);
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
        retry_after_ms: Option<u64>,
    },
    /// No DeepSeek account is configured at all: the channel is disabled, not
    /// busy, so a caller should answer `channel_unavailable` rather than a
    /// retry hint.
    NotConfigured,
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
            RouteError::NotConfigured => f.write_str("no DeepSeek account is configured"),
        }
    }
}

impl std::error::Error for RouteError {}

pub struct PoolConfig {
    pub request_delay: Duration,
    /// Maximum simultaneous upstream requests for one account. Values below
    /// one are normalized to one by `Pool::connect`/`Connection::new`.
    pub max_inflight_per_account: usize,
    pub queue_limit: usize,
    /// Max time to wait for a busy account's rate slot. Python: `QUEUE_TIMEOUT`.
    pub queue_timeout: Duration,
    pub login_cooldown: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            request_delay: Duration::ZERO,
            max_inflight_per_account: 1,
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
        let mut config = config;
        config.max_inflight_per_account = config.max_inflight_per_account.max(1);
        // Zero DeepSeek accounts is a supported deployment when another channel
        // (ChatGPT Web) serves the traffic, so build an empty pool instead of
        // refusing to start.
        if accounts.is_empty() {
            return Ok(Self::unconfigured(config, store));
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
                config.max_inflight_per_account,
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

    /// A pool with no accounts.
    ///
    /// Traffic from another channel can still be served; every DeepSeek route
    /// then fails with [`RouteError::NotConfigured`] so the caller can answer a
    /// clean 503 instead of opening a session nobody can serve.
    pub fn unconfigured(config: PoolConfig, store: crate::store::Store) -> Self {
        let mut config = config;
        config.max_inflight_per_account = config.max_inflight_per_account.max(1);
        Self {
            conns: Vec::new(),
            owner: Mutex::new(HashMap::new()),
            config,
            store,
        }
    }

    /// Whether any DeepSeek account is configured.
    pub fn is_configured(&self) -> bool {
        !self.conns.is_empty()
    }

    pub fn len(&self) -> usize {
        self.conns.len()
    }

    /// First account, without binding a thread to it. Used for health probes
    /// that must not leave a fake thread in the database.
    ///
    /// `None` when the pool has no accounts: a deployment that only serves
    /// ChatGPT Web is valid, so a health probe must not panic.
    pub fn any(&self) -> Option<Arc<Connection>> {
        self.conns.first().cloned()
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
        // A deployment with no DeepSeek account still serves other channels;
        // its DeepSeek routes are disabled rather than busy.
        if self.conns.is_empty() {
            return Err(RouteError::NotConfigured);
        }
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
        test_pool_with_config(emails, PoolConfig::default())
    }

    fn test_pool_with_config(emails: &[&str], config: PoolConfig) -> Pool {
        let store = crate::store::Store::open_memory().expect("in-memory store");
        let mut config = config;
        config.max_inflight_per_account = config.max_inflight_per_account.max(1);
        let conns = emails
            .iter()
            .map(|email| {
                Arc::new(Connection::new(
                    (*email).to_string(),
                    String::new(),
                    None,
                    Client::from_token("test-token".into()),
                    config.max_inflight_per_account,
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
            RouteError::NotConfigured => panic!("configured pool cannot return NotConfigured"),
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

    #[tokio::test]
    async fn fresh_threads_are_distributed_across_healthy_accounts() {
        let pool = test_pool(&["a@x", "b@x"]);
        let first = pool.route_checked("thread-a").await.unwrap();
        let second = pool.route_checked("thread-b").await.unwrap();
        assert_eq!(first.email, "a@x");
        assert_eq!(second.email, "b@x");
        assert_eq!(
            pool.store.get_binding("thread-a").unwrap().as_deref(),
            Some("a@x")
        );
        assert_eq!(
            pool.store.get_binding("thread-b").unwrap().as_deref(),
            Some("b@x")
        );
    }

    #[tokio::test]
    async fn max_1_acquire_serializes_and_preserves_request_delay() {
        let pool = test_pool_with_config(
            &["a@x"],
            PoolConfig {
                max_inflight_per_account: 1,
                request_delay: Duration::from_millis(60),
                ..Default::default()
            },
        );
        let conn = pool.any().unwrap();

        let ticket1 = conn.acquire().await.expect("first acquire succeeds");
        let acquired_second = Arc::new(tokio::sync::Notify::new());
        let acquired_second_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let conn_clone = conn.clone();
        let notify_clone = acquired_second.clone();
        let flag_clone = acquired_second_flag.clone();
        let handle = tokio::spawn(async move {
            let _ticket2 = conn_clone.acquire().await.expect("second acquire succeeds");
            flag_clone.store(true, Ordering::Release);
            notify_clone.notify_one();
        });

        // While ticket1 is held, ticket2 cannot be acquired.
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(
            !acquired_second_flag.load(Ordering::Acquire),
            "second acquire must wait while ticket1 is held"
        );

        let dropped_at = Instant::now();
        drop(ticket1);

        acquired_second.notified().await;
        let elapsed = dropped_at.elapsed();
        assert!(
            elapsed >= Duration::from_millis(45),
            "second acquire must wait for request_delay (60ms) after drop, elapsed: {:?}",
            elapsed
        );
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn max_2_acquires_overlap_and_third_waits() {
        let pool = test_pool_with_config(
            &["a@x"],
            PoolConfig {
                max_inflight_per_account: 2,
                request_delay: Duration::ZERO,
                ..Default::default()
            },
        );
        let conn = pool.any().unwrap();

        let ticket1 = conn.acquire().await.expect("first acquire succeeds");
        let ticket2 = conn
            .acquire()
            .await
            .expect("second acquire succeeds concurrently");

        let acquired_third_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let notify_third = Arc::new(tokio::sync::Notify::new());

        let conn_clone = conn.clone();
        let flag_clone = acquired_third_flag.clone();
        let notify_clone = notify_third.clone();
        let handle = tokio::spawn(async move {
            let _ticket3 = conn_clone.acquire().await.expect("third acquire succeeds");
            flag_clone.store(true, Ordering::Release);
            notify_clone.notify_one();
        });

        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(
            !acquired_third_flag.load(Ordering::Acquire),
            "third acquire must wait when max_inflight=2 is saturated"
        );

        // Drop one ticket; third acquire should proceed
        drop(ticket1);
        notify_third.notified().await;
        assert!(
            acquired_third_flag.load(Ordering::Acquire),
            "third acquire must succeed after ticket1 dropped"
        );

        drop(ticket2);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn waiting_gauge_lifecycle_no_leaks() {
        let pool = test_pool_with_config(
            &["a@x"],
            PoolConfig {
                max_inflight_per_account: 1,
                request_delay: Duration::ZERO,
                queue_limit: 1,
                queue_timeout: Duration::from_millis(50),
                ..Default::default()
            },
        );
        let conn = pool.any().unwrap();

        // 1. Success path: waiting returns to 0
        {
            let t1 = conn.acquire().await.unwrap();
            let conn_clone = conn.clone();
            let handle = tokio::spawn(async move {
                let _t2 = conn_clone.acquire().await.unwrap();
            });
            tokio::time::sleep(Duration::from_millis(15)).await;
            assert_eq!(conn.waiting.load(Ordering::Relaxed), 1);
            drop(t1);
            handle.await.unwrap();
            assert_eq!(
                conn.waiting.load(Ordering::Relaxed),
                0,
                "waiting must be 0 after successful acquire"
            );
        }

        // 2. Queue full path: waiting not leaked
        {
            let t1 = conn.acquire().await.unwrap();
            let conn_clone = conn.clone();
            let handle = tokio::spawn(async move {
                let _t2 = conn_clone.acquire().await;
            });
            tokio::time::sleep(Duration::from_millis(15)).await;
            assert_eq!(conn.waiting.load(Ordering::Relaxed), 1);

            // queue_limit is 1, so another acquire immediately fails
            let err = conn.acquire().await.err().expect("queue should be full");
            assert!(err.contains("queue is full"));
            assert_eq!(conn.waiting.load(Ordering::Relaxed), 1);

            drop(t1);
            handle.await.unwrap();
            assert_eq!(
                conn.waiting.load(Ordering::Relaxed),
                0,
                "waiting must be 0 after queue full"
            );
        }

        // 3. Queue timeout path: waiting returns to 0
        {
            let t1 = conn.acquire().await.unwrap();
            let err = conn.acquire().await.err().expect("queue should time out");
            assert!(err.contains("timeout after"));
            assert_eq!(
                conn.waiting.load(Ordering::Relaxed),
                0,
                "waiting must be 0 after queue timeout"
            );
            drop(t1);
        }

        // 4. Cancelled future path: waiting returns to 0
        {
            let t1 = conn.acquire().await.unwrap();
            let conn_clone = conn.clone();
            let handle = tokio::spawn(async move {
                let _ = conn_clone.acquire().await;
            });
            tokio::time::sleep(Duration::from_millis(15)).await;
            assert_eq!(conn.waiting.load(Ordering::Relaxed), 1);
            handle.abort();
            let _ = handle.await;
            assert_eq!(
                conn.waiting.load(Ordering::Relaxed),
                0,
                "waiting must be 0 after cancellation"
            );
            drop(t1);
        }
    }

    #[tokio::test]
    async fn concurrent_acquires_respect_start_spacing() {
        let pool = test_pool_with_config(
            &["a@x"],
            PoolConfig {
                max_inflight_per_account: 3,
                request_delay: Duration::from_millis(80),
                ..Default::default()
            },
        );
        let conn = pool.any().unwrap();
        let barrier = Arc::new(tokio::sync::Barrier::new(4));
        let started = Instant::now();
        let mut handles = Vec::new();

        for _ in 0..3 {
            let conn = conn.clone();
            let barrier = barrier.clone();
            handles.push(tokio::spawn(async move {
                barrier.wait().await;
                let ticket = conn.acquire().await.expect("acquire succeeds");
                let acquired_at = started.elapsed();
                drop(ticket);
                acquired_at
            }));
        }
        barrier.wait().await;

        let mut acquired_at = Vec::new();
        for handle in handles {
            acquired_at.push(handle.await.unwrap());
        }
        acquired_at.sort();
        assert!(
            acquired_at[1] - acquired_at[0] >= Duration::from_millis(50),
            "second request started too soon: {acquired_at:?}"
        );
        assert!(
            acquired_at[2] - acquired_at[1] >= Duration::from_millis(50),
            "third request started too soon: {acquired_at:?}"
        );
    }

    #[tokio::test]
    async fn cancelled_reservation_does_not_pollute_future_requests() {
        let pool = test_pool_with_config(
            &["a@x"],
            PoolConfig {
                max_inflight_per_account: 5,
                request_delay: Duration::from_millis(120),
                ..Default::default()
            },
        );
        let conn = pool.any().unwrap();

        // Start first request
        let t1 = conn.acquire().await.expect("first acquire succeeds");

        // Spawn 4 requests that reserve start times +120ms, +240ms, +360ms, +480ms
        let mut handles = Vec::new();
        for _ in 0..4 {
            let c = conn.clone();
            handles.push(tokio::spawn(async move {
                let _ = c.acquire().await;
            }));
        }

        // Let them reserve slots
        tokio::time::sleep(Duration::from_millis(25)).await;

        // Cancel all of them
        for h in handles {
            h.abort();
        }
        tokio::time::sleep(Duration::from_millis(25)).await;

        // Now a fresh request arrives. If reservations leaked to a fake future,
        // it would be delayed by ~600ms. With cancellation safety, it only waits
        // at most 120ms from t1's start.
        let fresh_start = Instant::now();
        let fresh_ticket = conn.acquire().await.expect("fresh acquire succeeds");
        let elapsed = fresh_start.elapsed();
        assert!(
            elapsed < Duration::from_millis(300),
            "fresh acquire took {:?}, should not be delayed by cancelled reservations",
            elapsed
        );
        drop(t1);
        drop(fresh_ticket);
    }

    #[tokio::test]
    async fn cooldown_blocks_new_acquire_existing_ticket_releases_permit() {
        let pool = test_pool_with_config(
            &["a@x"],
            PoolConfig {
                max_inflight_per_account: 1,
                request_delay: Duration::ZERO,
                queue_timeout: Duration::from_secs(5),
                ..Default::default()
            },
        );
        let conn = pool.any().unwrap();

        // 1. Acquire ticket1
        let ticket1 = conn.acquire().await.expect("acquire t1");

        // 2. Impose cooldown
        conn.cool_down(Duration::from_millis(150)).await;
        assert!(
            !conn.healthy().await,
            "account must be unhealthy during cooldown"
        );

        // 3. ticket1 is not cancelled; caller still owns it
        let acquired_t2 = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let notify_t2 = Arc::new(tokio::sync::Notify::new());

        let conn_clone = conn.clone();
        let flag_clone = acquired_t2.clone();
        let notify_clone = notify_t2.clone();
        let notify_wait = notify_t2.clone();
        let handle = tokio::spawn(async move {
            let _t2 = conn_clone
                .acquire()
                .await
                .expect("acquire t2 after cooldown");
            flag_clone.store(true, Ordering::Release);
            notify_clone.notify_one();
        });

        // 4. Drop ticket1 while still cooling down
        tokio::time::sleep(Duration::from_millis(30)).await;
        drop(ticket1);

        // Even though ticket1 released its permit, t2 cannot acquire until cooldown ends
        tokio::time::sleep(Duration::from_millis(40)).await;
        assert!(
            !acquired_t2.load(Ordering::Acquire),
            "new acquire must not proceed before cooldown expires even if permit returned"
        );

        notify_wait.notified().await;
        assert!(
            acquired_t2.load(Ordering::Acquire),
            "acquire succeeds once cooldown expires"
        );
        handle.await.unwrap();
        assert!(
            conn.healthy().await,
            "account is healthy after cooldown expires"
        );
    }
    #[tokio::test]
    async fn cancelling_earlier_reservation_wakes_later_waiter() {
        let pool = test_pool_with_config(
            &["a@x"],
            PoolConfig {
                max_inflight_per_account: 3,
                request_delay: Duration::from_millis(120),
                queue_timeout: Duration::from_secs(5),
                ..Default::default()
            },
        );
        let conn = pool.any().unwrap();
        let first_ticket = conn.acquire().await.expect("first acquire succeeds");

        let conn_first = conn.clone();
        let first = tokio::spawn(async move {
            let _ticket = conn_first.acquire().await.expect("first waiter succeeds");
        });
        for _ in 0..100 {
            if conn.gate_state().reservations.len() == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(conn.gate_state().reservations.len(), 1);

        let later_acquired = Arc::new(tokio::sync::Notify::new());
        let conn_later = conn.clone();
        let later_notify = later_acquired.clone();
        let later = tokio::spawn(async move {
            let _ticket = conn_later.acquire().await.expect("later waiter succeeds");
            later_notify.notify_one();
        });
        for _ in 0..100 {
            if conn.gate_state().reservations.len() == 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(conn.gate_state().reservations.len(), 2);

        first.abort();
        let _ = first.await;
        tokio::time::timeout(Duration::from_millis(180), later_acquired.notified())
            .await
            .expect("cancellation must wake and rebase the later reservation");
        later.await.unwrap();
        drop(first_ticket);
    }
}
