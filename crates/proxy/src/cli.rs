//! Command-line interface: account/key management, server lifecycle, logs,
//! stats.
//!
//! `serve` is the only long-running command and lives in `main.rs`; everything
//! else is short-lived and prints human-readable output. The background
//! lifecycle (`start`/`stop`/`status`) is driven by a pidfile plus a logfile in
//! the runtime directory (`$DEEPSEEK_RUNTIME_DIR`, else
//! `$XDG_STATE_HOME/deepseek-rs`, else `~/.local/state/deepseek-rs`).

use crate::{config, pool, store};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const DEFAULT_LOG_LINES: usize = 50;
const START_TIMEOUT: Duration = Duration::from_secs(30);

// Signal numbers passed to `send_signal`. Only the values matter on Unix; on
// Windows they are ignored and the request maps onto `TerminateProcess`.
#[cfg(unix)]
const SIG_TERM: i32 = libc::SIGTERM;
#[cfg(unix)]
const SIG_KILL: i32 = libc::SIGKILL;
#[cfg(windows)]
const SIG_TERM: i32 = 15;
#[cfg(windows)]
const SIG_KILL: i32 = 9;

// ---------------------------------------------------------------------------
// runtime dir / pidfile
// ---------------------------------------------------------------------------

pub struct Runtime {
    pub dir: PathBuf,
}

impl Runtime {
    pub fn resolve() -> Self {
        if let Ok(dir) = std::env::var("DEEPSEEK_RUNTIME_DIR") {
            let dir = dir.trim();
            if !dir.is_empty() {
                return Self {
                    dir: PathBuf::from(dir),
                };
            }
        }
        let base = std::env::var("XDG_STATE_HOME")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var("HOME")
                    .ok()
                    .map(|h| PathBuf::from(h).join(".local/state"))
            })
            .unwrap_or_else(|| PathBuf::from("."));
        Self {
            dir: base.join("deepseek-rs"),
        }
    }

    pub fn pidfile(&self) -> PathBuf {
        self.dir.join("deepseek-rs.pid")
    }

    pub fn logfile(&self) -> PathBuf {
        self.dir.join("deepseek-rs.log")
    }

    fn ensure(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.dir)
    }

    /// Running server pid, or `None` when the pidfile is missing/stale.
    pub fn pid(&self) -> Option<i32> {
        let text = std::fs::read_to_string(self.pidfile()).ok()?;
        let pid: i32 = text.trim().parse().ok()?;
        if pid > 0 && pid_alive(pid) {
            Some(pid)
        } else {
            None
        }
    }

    fn write_pid(&self) -> std::io::Result<()> {
        self.ensure()?;
        let mut f = std::fs::File::create(self.pidfile())?;
        writeln!(f, "{}", std::process::id())
    }

    fn clear_pid(&self) {
        let _ = std::fs::remove_file(self.pidfile());
    }
}

/// Removes the pidfile on drop, but only if it still names this process.
pub struct PidGuard {
    rt: Runtime,
}

impl PidGuard {
    /// Claim the pidfile. Fails when another live instance owns it.
    pub fn claim() -> Result<Self, String> {
        let rt = Runtime::resolve();
        if let Some(pid) = rt.pid() {
            return Err(format!(
                "another deepseek-rs is already running (pid {pid}); use `deepseek-rs stop` or `deepseek-rs restart`"
            ));
        }
        rt.write_pid()
            .map_err(|e| format!("write {}: {e}", rt.pidfile().display()))?;
        Ok(Self { rt })
    }
}

impl Drop for PidGuard {
    fn drop(&mut self) {
        let ours = std::fs::read_to_string(self.rt.pidfile())
            .ok()
            .and_then(|t| t.trim().parse::<i32>().ok())
            == Some(std::process::id() as i32);
        if ours {
            self.rt.clear_pid();
        }
    }
}

#[cfg(unix)]
fn pid_alive(pid: i32) -> bool {
    // Signal 0 only probes for existence; ESRCH means gone, EPERM means alive
    // but owned by someone else (still alive as far as we care).
    unsafe {
        libc::kill(pid, 0) == 0
            || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
}

#[cfg(windows)]
fn pid_alive(pid: i32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid as u32);
        if handle.is_null() {
            return false;
        }
        let mut code: u32 = 0;
        let ok = GetExitCodeProcess(handle, &mut code);
        CloseHandle(handle);
        ok != 0 && code as i32 == STILL_ACTIVE
    }
}

#[cfg(unix)]
fn send_signal(pid: i32, sig: i32) -> std::io::Result<()> {
    if unsafe { libc::kill(pid, sig) } == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// Terminate a background server. Windows has no Unix signals; map both the
/// graceful (SIGTERM) and forced (SIGKILL) requests onto `TerminateProcess`.
#[cfg(windows)]
fn send_signal(pid: i32, _sig: i32) -> std::io::Result<()> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{OpenProcess, TerminateProcess, PROCESS_TERMINATE};
    unsafe {
        let handle = OpenProcess(PROCESS_TERMINATE, 0, pid as u32);
        if handle.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        let ok = TerminateProcess(handle, 1);
        let err = std::io::Error::last_os_error();
        CloseHandle(handle);
        if ok != 0 {
            Ok(())
        } else {
            Err(err)
        }
    }
}

// ---------------------------------------------------------------------------
// accounts: env + database
// ---------------------------------------------------------------------------

fn env_from_dotenv(key: &str) -> Option<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../deepseek/.env");
    let text = std::fs::read_to_string(path).ok()?;
    text.lines()
        .filter_map(|l| l.split_once('='))
        .find(|(k, _)| k.trim() == key)
        .map(|(_, v)| v.trim().trim_matches('"').to_string())
}

fn env_val(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .or_else(|| env_from_dotenv(key).filter(|v| !v.trim().is_empty()))
}

/// Accounts from `DEEPSEEK_EMAIL_<i>`/`PASSWORD_<i>`/`TOKEN_<i>`, falling back to
/// unindexed names and then `../deepseek/.env`.
pub fn env_accounts() -> Vec<pool::AccountSpec> {
    let mut out = Vec::new();
    for i in 0.. {
        let email = env_val(&format!("DEEPSEEK_EMAIL_{i}"));
        let password = env_val(&format!("DEEPSEEK_PASSWORD_{i}"));
        let token = env_val(&format!("DEEPSEEK_TOKEN_{i}"));
        if token.is_none() && !(email.is_some() && password.is_some()) {
            break;
        }
        out.push(pool::AccountSpec {
            email: email.unwrap_or_else(|| format!("token-account-{i}")),
            password: password.unwrap_or_default(),
            token,
        });
    }

    if out.is_empty() {
        let email = env_val("DEEPSEEK_EMAIL");
        let password = env_val("DEEPSEEK_PASSWORD");
        let token = env_val("DEEPSEEK_TOKEN");
        if token.is_some() || (email.is_some() && password.is_some()) {
            out.push(pool::AccountSpec {
                email: email.unwrap_or_else(|| "token-account".into()),
                password: password.unwrap_or_default(),
                token,
            });
        }
    }
    out
}

/// Every usable account: environment first (explicit wins), then accounts
/// added with `deepseek-rs accounts add`, de-duplicated by email.
pub fn load_accounts(store: &store::Store) -> Vec<pool::AccountSpec> {
    let mut out = env_accounts();
    let known: Vec<String> = out.iter().map(|a| a.email.clone()).collect();
    for row in store.accounts().unwrap_or_default() {
        if known.contains(&row.email) {
            continue;
        }
        out.push(pool::AccountSpec {
            email: row.email,
            password: row.password,
            token: row.token,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

/// Base URL clients should use, resolving wildcard binds to loopback.
pub fn base_url(cfg: &config::Config) -> String {
    let host = match cfg.host.as_str() {
        "0.0.0.0" | "::" | "[::]" | "" => "127.0.0.1",
        h => h,
    };
    format!("http://{host}:{}/v1", cfg.port)
}

fn root_url(cfg: &config::Config) -> String {
    base_url(cfg).trim_end_matches("/v1").to_string()
}

fn http() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| e.to_string())
}

/// Plain GET: success only cares about the status code (used for `/healthz`,
/// which answers with the text `ok`, not JSON).
async fn get_status(url: &str) -> Result<(), String> {
    let resp = http()?.get(url).send().await.map_err(|e| e.to_string())?;
    if resp.status().is_success() {
        Ok(())
    } else {
        Err(format!("HTTP {}", resp.status().as_u16()))
    }
}

async fn get_json(url: &str) -> Result<serde_json::Value, String> {
    let resp = http()?.get(url).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status().as_u16()));
    }
    resp.json::<serde_json::Value>()
        .await
        .map_err(|e| e.to_string())
}

fn fmt_unix(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Howard Hinnant's days-to-civil conversion.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as i64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn human_age(unix_secs: i64) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let age = (now - unix_secs).max(0);
    match age {
        a if a < 60 => format!("{a}s ago"),
        a if a < 3600 => format!("{}m ago", a / 60),
        a if a < 86_400 => format!("{}h ago", a / 3600),
        a => format!("{}d ago", a / 86_400),
    }
}

// ---------------------------------------------------------------------------
// tail / follow
// ---------------------------------------------------------------------------

fn tail_lines(path: &Path, n: usize) -> std::io::Result<Vec<String>> {
    let mut f = std::fs::File::open(path)?;
    let len = f.metadata()?.len();
    const WINDOW: u64 = 256 * 1024;
    let mut buf = String::new();
    if len > WINDOW {
        f.seek(SeekFrom::Start(len - WINDOW))?;
        f.read_to_string(&mut buf)?;
        // Drop the first partial line.
        if let Some(i) = buf.find('\n') {
            buf.drain(..=i);
        }
    } else {
        f.read_to_string(&mut buf)?;
    }
    let mut lines: Vec<String> = buf.lines().map(str::to_string).collect();
    if lines.len() > n {
        lines.drain(..lines.len() - n);
    }
    Ok(lines)
}

fn follow(path: &Path) -> std::io::Result<()> {
    let mut pos = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let mut buf = String::new();
    loop {
        let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        if len < pos {
            // Truncated / rotated: start over from the beginning.
            pos = 0;
        }
        if len > pos {
            let mut f = std::fs::File::open(path)?;
            f.seek(SeekFrom::Start(pos))?;
            buf.clear();
            f.read_to_string(&mut buf)?;
            print!("{buf}");
            std::io::stdout().flush().ok();
            pos = len;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

fn cfg_or_exit() -> config::Config {
    match config::Config::load() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("config: {e}");
            std::process::exit(1);
        }
    }
}

fn open_store(cfg: &config::Config) -> Result<store::Store, String> {
    store::Store::open_path(&cfg.db_path).map_err(|e| format!("open {}: {e}", cfg.db_path))
}

/// Run a non-server command. Returns the process exit code.
pub async fn run(args: &[String]) -> Result<i32, Box<dyn std::error::Error>> {
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    let sub = args.get(2).map(String::as_str).unwrap_or("");

    match cmd {
        "help" | "-h" | "--help" => {
            usage();
            Ok(0)
        }
        "version" | "-V" | "--version" => {
            println!("deepseek-rs {}", env!("CARGO_PKG_VERSION"));
            Ok(0)
        }
        "url" => {
            println!("{}", base_url(&cfg_or_exit()));
            Ok(0)
        }
        "health" => health().await,
        "status" => status().await,
        "stats" => stats(args).await,
        "start" => start(args).await,
        "stop" => stop().await,
        "restart" => {
            let bg = start_flags(args).0;
            stop().await?;
            start_with(args, bg).await
        }
        "logs" | "log" => logs(args),
        "check" => check().await,
        "accounts" | "account" => accounts(sub, args),
        "keys" | "key" => keys(sub, args),
        "chatgpt" | "cg" => chatgpt(sub, args),
        other => {
            eprintln!("unknown command '{other}'\n");
            usage();
            Ok(2)
        }
    }
}

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

async fn health() -> Result<i32, Box<dyn std::error::Error>> {
    let cfg = cfg_or_exit();
    let url = format!("{}/healthz", root_url(&cfg));
    match http()?.get(&url).send().await {
        Ok(r) if r.status().is_success() => {
            println!("ok — {}", base_url(&cfg));
            Ok(0)
        }
        Ok(r) => {
            println!("unhealthy — HTTP {} at {url}", r.status().as_u16());
            Ok(1)
        }
        Err(e) => {
            println!("down — {url}: {}", short_err(&e.to_string()));
            Ok(1)
        }
    }
}

async fn status() -> Result<i32, Box<dyn std::error::Error>> {
    let cfg = cfg_or_exit();
    let rt = Runtime::resolve();
    match rt.pid() {
        Some(pid) => {
            let live = get_json(&format!("{}/stats", root_url(&cfg))).await.ok();
            println!("running   pid {pid}");
            println!("url       {}", base_url(&cfg));
            println!("pidfile   {}", rt.pidfile().display());
            println!("log       {}", rt.logfile().display());
            println!("db        {}", cfg.db_path);
            match live {
                Some(s) => {
                    let up = s.get("uptime_secs").and_then(|v| v.as_u64()).unwrap_or(0);
                    let accts = s
                        .get("accounts")
                        .and_then(|v| v.as_array())
                        .map(|a| a.len())
                        .unwrap_or(0);
                    println!("uptime    {}", fmt_duration(up));
                    println!("accounts  {accts}");
                }
                None => println!("health    no response yet (still starting?)"),
            }
            Ok(0)
        }
        None => {
            println!("stopped");
            println!("url       {} (not listening)", base_url(&cfg));
            println!("log       {}", rt.logfile().display());
            println!("db        {}", cfg.db_path);
            Ok(3)
        }
    }
}

/// Parse the flags accepted by `start` / `restart`.
///
/// Returns `(background, unknown)`. Foreground is the default, so `start`
/// prints the pretty logs in the terminal exactly like `deepseek start`.
/// `-b` / `--bg` detaches into a daemon instead.
fn start_flags(args: &[String]) -> (bool, Option<String>) {
    let mut bg = false;
    let mut unknown = None;
    // Skip `start` (index 1) and any sub-word; scan everything after it.
    for arg in args.iter().skip(2) {
        match arg.as_str() {
            "-b" | "--bg" | "--background" => bg = true,
            "-f" | "--foreground" | "--fg" => bg = false,
            other if other.starts_with('-') => unknown = Some(other.to_string()),
            _ => {}
        }
    }
    (bg, unknown)
}

async fn start(args: &[String]) -> Result<i32, Box<dyn std::error::Error>> {
    let (bg, unknown) = start_flags(args);
    start_with(args, bg).await.map(|code| {
        if let Some(flag) = unknown {
            eprintln!("warning: ignoring unknown start flag {flag}");
        }
        code
    })
}

/// `start` body. `bg == false` hands the terminal to the server.
async fn start_with(args: &[String], bg: bool) -> Result<i32, Box<dyn std::error::Error>> {
    let _ = args;
    let cfg = cfg_or_exit();
    let rt = Runtime::resolve();
    rt.ensure()?;

    if let Some(pid) = rt.pid() {
        println!("already running (pid {pid}) — {}", base_url(&cfg));
        return Ok(0);
    }

    if !bg {
        // Foreground: replace this process with the server so Ctrl-C and the
        // pidfile behave exactly as if the user had run `serve`.
        println!("running deepseek-rs in foreground (Ctrl-C to stop)...");
        println!("url       {}", base_url(&cfg));
        let exe = std::env::current_exe()?;
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            let err = std::process::Command::new(exe).arg("serve").exec();
            return Err(format!("exec serve: {err}").into());
        }
        #[cfg(not(unix))]
        {
            // Windows has no `exec`: spawn the server and wait for it to exit.
            let status = std::process::Command::new(exe).arg("serve").status()?;
            return Ok(status.code().unwrap_or(1));
        }
    }

    let exe = std::env::current_exe()?;
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(rt.logfile())?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("serve")
        .stdin(std::process::Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    // Detach into its own session so it survives the shell exiting.
    #[cfg(unix)]
    unsafe {
        use std::os::unix::process::CommandExt;
        cmd.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let child = cmd.spawn()?;
    let pid = child.id();

    // Wait for the pidfile, then for /healthz, so `start` means "ready".
    let deadline = std::time::Instant::now() + START_TIMEOUT;
    let mut ready = false;
    while std::time::Instant::now() < deadline {
        if rt.pid() == Some(pid as i32)
            && get_status(&format!("{}/healthz", root_url(&cfg)))
                .await
                .is_ok()
        {
            ready = true;
            break;
        }
        if !pid_alive(pid as i32) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    if ready {
        println!("started   pid {pid}");
        println!("url       {}", base_url(&cfg));
        println!("log       {}", rt.logfile().display());
        Ok(0)
    } else {
        eprintln!(
            "failed to start; last log lines from {}:",
            rt.logfile().display()
        );
        for line in tail_lines(&rt.logfile(), 20).unwrap_or_default() {
            eprintln!("  {line}");
        }
        Ok(1)
    }
}

async fn stop() -> Result<i32, Box<dyn std::error::Error>> {
    let rt = Runtime::resolve();
    let Some(pid) = rt.pid() else {
        println!("not running");
        rt.clear_pid();
        return Ok(0);
    };

    if let Err(e) = send_signal(pid, SIG_TERM) {
        eprintln!("signal {pid}: {e}");
        return Ok(1);
    }

    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    while std::time::Instant::now() < deadline {
        if !pid_alive(pid) {
            rt.clear_pid();
            println!("stopped   pid {pid}");
            return Ok(0);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    eprintln!("pid {pid} ignored SIGTERM; sending SIGKILL");
    let _ = send_signal(pid, SIG_KILL);
    tokio::time::sleep(Duration::from_millis(300)).await;
    rt.clear_pid();
    println!("killed    pid {pid}");
    Ok(0)
}

fn logs(args: &[String]) -> Result<i32, Box<dyn std::error::Error>> {
    let rt = Runtime::resolve();
    let mut n = DEFAULT_LOG_LINES;
    let mut follow_mode = false;
    let mut path = rt.logfile();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "-f" | "--follow" => follow_mode = true,
            "-n" | "--lines" => {
                i += 1;
                n = args
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(DEFAULT_LOG_LINES);
            }
            "--path" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    path = PathBuf::from(p);
                }
            }
            other if other.starts_with('-') => {
                eprintln!("unknown flag {other}");
                return Ok(2);
            }
            other => path = PathBuf::from(other),
        }
        i += 1;
    }

    if !path.exists() {
        eprintln!(
            "no log file at {} (server has not been started in the background?)",
            path.display()
        );
        return Ok(3);
    }

    for line in tail_lines(&path, n)? {
        println!("{line}");
    }
    if follow_mode {
        eprintln!("--- following {} (Ctrl-C to stop) ---", path.display());
        follow(&path)?;
    }
    Ok(0)
}

async fn stats(args: &[String]) -> Result<i32, Box<dyn std::error::Error>> {
    let cfg = cfg_or_exit();
    let as_json = args.iter().any(|a| a == "--json");

    match get_json(&format!("{}/stats", root_url(&cfg))).await {
        Ok(v) => {
            if as_json {
                println!("{}", serde_json::to_string_pretty(&v)?);
                return Ok(0);
            }
            print_stats_human(&cfg, &v);
            Ok(0)
        }
        Err(_) => {
            // Server down: report what the database alone can tell us.
            eprintln!("note: server is not running; showing database counts only\n");
            let store = open_store(&cfg)?;
            let counts = store.counts()?;
            if as_json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "server": "stopped",
                        "db": {
                            "threads": counts.threads,
                            "bindings": counts.bindings,
                            "api_keys": counts.api_keys,
                            "accounts": counts.accounts,
                            "cached_tokens": counts.cached_tokens,
                        }
                    }))?
                );
                return Ok(0);
            }
            println!("server    stopped");
            println!("db        {}", cfg.db_path);
            println!("threads   {}", counts.threads);
            println!("bindings  {}", counts.bindings);
            println!("api keys  {}", counts.api_keys);
            println!("accounts  {} (db)", counts.accounts);
            println!("tokens    {} cached", counts.cached_tokens);
            Ok(0)
        }
    }
}

fn print_stats_human(cfg: &config::Config, v: &serde_json::Value) {
    let u = |p: &str| v.pointer(p).and_then(|x| x.as_u64()).unwrap_or(0);
    let s = |p: &str| {
        v.pointer(p)
            .and_then(|x| x.as_str())
            .unwrap_or("-")
            .to_string()
    };

    println!(
        "server    running, pid {}",
        v.get("pid").and_then(|x| x.as_u64()).unwrap_or(0)
    );
    println!("url       {}", base_url(cfg));
    println!("uptime    {}", fmt_duration(u("/uptime_secs")));
    println!("db        {}", cfg.db_path);
    println!(
        "model     {} (thinking {}, search {})",
        s("/model"),
        v.pointer("/thinking")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        v.pointer("/search")
            .and_then(|x| x.as_bool())
            .unwrap_or(false)
    );
    println!();
    println!(
        "requests  {} total | {} ok | {} failed | {} auth-failed",
        u("/requests/total"),
        u("/requests/ok"),
        u("/requests/errors"),
        u("/requests/auth_failed")
    );
    println!(
        "upstream  {} rate-limit retries | {} tool calls",
        u("/requests/retries"),
        u("/requests/tool_calls")
    );
    println!(
        "threads   {} in memory | {} persisted",
        u("/threads/in_memory"),
        u("/db/threads")
    );
    println!("keys      {} registered", u("/db/api_keys"));
    println!();
    if let Some(accts) = v.get("accounts").and_then(|x| x.as_array()) {
        println!(
            "{:<32} {:<9} {:>7} {:>7}  {}",
            "account", "state", "threads", "queued", "cooldown"
        );
        for a in accts {
            let email = a.get("email").and_then(|x| x.as_str()).unwrap_or("-");
            let healthy = a.get("healthy").and_then(|x| x.as_bool()).unwrap_or(false);
            let bound = a.get("bound_threads").and_then(|x| x.as_u64()).unwrap_or(0);
            let waiting = a.get("waiting").and_then(|x| x.as_u64()).unwrap_or(0);
            let cooldown = a.get("cooldown_ms").and_then(|x| x.as_u64());
            let cd = cooldown
                .map(|ms| format!("{:.0}s", ms as f64 / 1000.0))
                .unwrap_or_else(|| "-".into());
            println!(
                "{:<32} {:<9} {:>7} {:>7}  {}",
                email,
                if healthy { "healthy" } else { "cooling" },
                bound,
                waiting,
                cd
            );
        }
    }
}

fn fmt_duration(secs: u64) -> String {
    let d = secs / 86_400;
    let h = (secs % 86_400) / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if d > 0 {
        format!("{d}d {h}h {m}m")
    } else if h > 0 {
        format!("{h}h {m}m {s}s")
    } else if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    }
}

async fn check() -> Result<i32, Box<dyn std::error::Error>> {
    let cfg = cfg_or_exit();
    let store = open_store(&cfg)?;
    let accounts = load_accounts(&store);
    if accounts.is_empty() {
        eprintln!(
            "no accounts configured — add one with `deepseek-rs accounts add <email> <password>`"
        );
        return Ok(1);
    }
    let n = accounts.len();
    let pool = crate::connect_pool(&cfg, &store, accounts).await?;
    if let Some(conn) = pool.any() {
        conn.client()
            .await
            .pow_challenge(client::COMPLETION_PATH)
            .await?;
    } else {
        eprintln!("no DeepSeek account configured");
        return Ok(1);
    }
    println!("ok: {n} account(s), login + PoW verified");
    Ok(0)
}

// ---------------------------------------------------------------------------
// accounts / keys
// ---------------------------------------------------------------------------

fn accounts(sub: &str, args: &[String]) -> Result<i32, Box<dyn std::error::Error>> {
    let cfg = cfg_or_exit();
    let store = open_store(&cfg)?;

    match sub {
        "" | "list" | "ls" => {
            let env = env_accounts();
            let db = store.accounts()?;
            if env.is_empty() && db.is_empty() {
                println!("no accounts configured");
                println!("add one:  deepseek-rs accounts add <email> <password>");
                return Ok(0);
            }
            println!("{:<32} {:<8} {:<9}  {}", "email", "source", "auth", "added");
            for a in &env {
                let auth = if a.token.is_some() {
                    "token"
                } else {
                    "password"
                };
                println!("{:<32} {:<8} {:<9}  {}", a.email, "env", auth, "-");
            }
            for a in &db {
                let auth = if a.token.is_some() {
                    "token"
                } else if a.password.is_empty() {
                    "none"
                } else {
                    "password"
                };
                println!(
                    "{:<32} {:<8} {:<9}  {}",
                    a.email,
                    "db",
                    auth,
                    human_age(a.created_at)
                );
            }
            if !db.is_empty() {
                println!(
                    "\n{} from the database, {} from the environment",
                    db.len(),
                    env.len()
                );
            }
            Ok(0)
        }
        "add" => {
            // accounts add <email> <password> | accounts add <email> --token <token>
            let mut email = None;
            let mut password = String::new();
            let mut token = None;
            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--token" | "-t" => {
                        i += 1;
                        token = args.get(i).cloned();
                        if token.is_none() {
                            eprintln!("--token needs a value");
                            return Ok(2);
                        }
                    }
                    "--password" | "-p" => {
                        i += 1;
                        password = args.get(i).cloned().unwrap_or_default();
                    }
                    other if email.is_none() => email = Some(other.to_string()),
                    other => password = other.to_string(),
                }
                i += 1;
            }
            let Some(email) = email else {
                eprintln!("usage: deepseek-rs accounts add <email> [password] [--token <token>]");
                return Ok(2);
            };
            if password.is_empty() && token.is_none() {
                eprintln!("need a password or --token");
                return Ok(2);
            }
            store.upsert_account(&email, &password, token.as_deref())?;
            println!("added {email}");
            println!("restart to apply: deepseek-rs restart");
            Ok(0)
        }
        "rm" | "remove" | "delete" => {
            let Some(email) = args.get(3) else {
                eprintln!("usage: deepseek-rs accounts rm <email>");
                return Ok(2);
            };
            let removed = store.delete_account(email)?;
            let in_env = env_accounts().iter().any(|a| &a.email == email);
            if removed {
                println!("removed {email} (and its cached token)");
            } else {
                println!("no database account {email}");
            }
            if in_env {
                println!("note: {email} also comes from the environment; remove DEEPSEEK_*_<i> to drop it fully");
            }
            Ok(0)
        }
        other => {
            eprintln!("unknown accounts subcommand '{other}'");
            eprintln!("usage: deepseek-rs accounts [list|add|rm]");
            Ok(2)
        }
    }
}

/// `deepseek-rs chatgpt <list|add|rm|check>`.
///
/// Secrets are only ever read from stdin (`-`) or a file, never from argv, so
/// they cannot leak through shell history or a process listing.
fn chatgpt(sub: &str, args: &[String]) -> Result<i32, Box<dyn std::error::Error>> {
    use crate::chatgpt_accounts as accounts_mod;
    use crate::store::ChatGptAccountRow;

    let cfg = cfg_or_exit();
    let store = open_store(&cfg)?;
    let now = accounts_mod::now();

    match sub {
        "" | "list" | "ls" => {
            let rows = store.chatgpt_accounts()?;
            if rows.is_empty() {
                println!("no ChatGPT accounts configured");
                println!("add one:  deepseek-rs chatgpt add --access-token -");
                return Ok(0);
            }
            println!(
                "{:<16} {:<18} {:<8} {:<9} {:<16}  {}",
                "id", "label", "plan", "alert", "expires in", "added"
            );
            for r in rows {
                println!(
                    "{:<16} {:<18} {:<8} {:<9} {:<16}  {}",
                    r.id,
                    r.label.clone().or(r.email.clone()).unwrap_or_default(),
                    r.plan.clone().unwrap_or_else(|| "-".into()),
                    r.notified_stage.clone().unwrap_or_else(|| "-".into()),
                    accounts_mod::describe_expiry(r.expires_at, now),
                    fmt_unix(r.created_at)
                );
            }
            println!();
            println!("Set BARK_URL or BARK_KEY to receive expiry pushes.");
            Ok(0)
        }
        "add" => {
            let mut token = None;
            let mut label = None;
            let mut proxy = None;
            // args: [exe, chatgpt, add, ...options]
            let mut i = 3;
            while i < args.len() {
                match args[i].as_str() {
                    "--access-token" | "-t" => {
                        i += 1;
                        token = args.get(i).cloned();
                        if token.is_none() {
                            eprintln!("--access-token needs a value (use - for stdin)");
                            return Ok(2);
                        }
                    }
                    "--label" | "-l" => {
                        i += 1;
                        label = args.get(i).cloned();
                    }
                    "--proxy" => {
                        i += 1;
                        proxy = args.get(i).cloned();
                    }
                    other => {
                        eprintln!("unexpected argument '{other}'");
                        return Ok(2);
                    }
                }
                i += 1;
            }
            let Some(token) = token else {
                eprintln!("usage: deepseek-rs chatgpt add --access-token <token|-> [--label L] [--proxy URL]");
                return Ok(2);
            };
            let token = if token == "-" {
                let mut buffer = String::new();
                std::io::Read::read_to_string(&mut std::io::stdin(), &mut buffer)?;
                buffer
            } else {
                token
            };
            let token = token.trim().to_string();
            if token.is_empty() || token.chars().any(char::is_control) {
                eprintln!("empty or invalid token");
                return Ok(2);
            }
            let times = accounts_mod::token_times(&token);
            let id = accounts_mod::account_id(&token);
            if let Some(expires) = times.expires_at {
                if expires <= now {
                    eprintln!(
                        "warning: this token expired {} ago",
                        accounts_mod::describe_expiry(times.expires_at, now)
                    );
                }
            } else {
                eprintln!("warning: the token carries no exp claim, so expiry cannot be tracked");
            }
            let created = store
                .chatgpt_accounts()?
                .into_iter()
                .find(|row| row.id == id)
                .map(|row| row.created_at)
                .unwrap_or(now);
            store.upsert_chatgpt_account(&ChatGptAccountRow {
                id: id.clone(),
                label,
                email: accounts_mod::token_email(&token),
                plan: None,
                access_token: token,
                expires_at: times.expires_at,
                proxy_url: proxy,
                notified_stage: None,
                notified_at: None,
                created_at: created,
                updated_at: now,
            })?;
            println!("added {id}");
            println!(
                "expires in {}",
                accounts_mod::describe_expiry(times.expires_at, now)
            );
            println!("restart to apply: deepseek-rs restart");
            Ok(0)
        }
        "rm" | "remove" | "delete" => {
            let Some(target) = args.get(3) else {
                eprintln!("usage: deepseek-rs chatgpt rm <id|label>");
                return Ok(2);
            };
            let rows = store.chatgpt_accounts()?;
            let found = rows.into_iter().find(|row| {
                &row.id == target
                    || row.label.as_deref() == Some(target.as_str())
                    || row.email.as_deref() == Some(target.as_str())
            });
            let Some(row) = found else {
                eprintln!("no ChatGPT account matches '{target}'");
                return Ok(3);
            };
            store.delete_chatgpt_account(&row.id)?;
            println!("removed {}", row.id);
            println!("restart to apply: deepseek-rs restart");
            Ok(0)
        }
        "check" => {
            // Report expiry only: there is no refresh path, so a probe cannot
            // extend an account, and calling upstream would only cost a turn.
            let rows = store.chatgpt_accounts()?;
            if rows.is_empty() {
                eprintln!("no ChatGPT accounts configured");
                return Ok(1);
            }
            let bark = accounts_mod::Bark::from_env();
            for row in rows {
                let stage = accounts_mod::due_stage(&row, now);
                println!(
                    "{:<16} {:<18} expires in {}{}",
                    row.id,
                    row.label.clone().or(row.email.clone()).unwrap_or_default(),
                    accounts_mod::describe_expiry(row.expires_at, now),
                    match stage {
                        Some(stage) => format!("  (alert due: {stage})"),
                        None => String::new(),
                    }
                );
            }
            println!();
            println!(
                "bark: {}",
                if bark.is_some() {
                    "configured"
                } else {
                    "not configured (set BARK_URL or BARK_KEY)"
                }
            );
            println!(
                "the server sweeps expiries periodically; no refresh path exists for these tokens"
            );
            Ok(0)
        }
        other => {
            eprintln!("unknown chatgpt subcommand '{other}'");
            eprintln!("usage: deepseek-rs chatgpt [list|add|rm|check]");
            Ok(2)
        }
    }
}

fn keys(sub: &str, args: &[String]) -> Result<i32, Box<dyn std::error::Error>> {
    let cfg = cfg_or_exit();
    let store = open_store(&cfg)?;

    // `key <name> <key>` is the original spelling of `keys add`.
    let (sub, offset) = match sub {
        "" | "list" | "ls" | "add" | "rm" | "remove" | "delete" => (sub, 3),
        _ => ("add", 2),
    };

    match sub {
        "" | "list" | "ls" => {
            let rows = store.api_keys()?;
            if rows.is_empty() {
                println!("no api keys registered (the proxy accepts any key while none exist)");
                return Ok(0);
            }
            println!(
                "{:<20} {:<12} {:<10}  {}",
                "name", "prefix", "id", "created"
            );
            for r in rows {
                println!(
                    "{:<20} {:<12} {:<10}  {}",
                    r.name,
                    format!("{}…", r.prefix),
                    r.id,
                    fmt_unix(r.created_at)
                );
            }
            Ok(0)
        }
        "add" => {
            let name = args.get(offset).map(String::as_str).unwrap_or("default");
            let Some(key) = args.get(offset + 1) else {
                eprintln!("usage: deepseek-rs keys add <name> <api-key>");
                return Ok(2);
            };
            store.insert_api_key(name, key)?;
            println!("registered key '{name}' ({}…)", &key[..key.len().min(8)]);
            Ok(0)
        }
        "rm" | "remove" | "delete" => {
            let Some(target) = args.get(offset) else {
                eprintln!("usage: deepseek-rs keys rm <name|id>");
                return Ok(2);
            };
            let n = store.delete_api_key(target)?;
            if n == 0 {
                println!("no key named {target}");
                return Ok(3);
            }
            println!("revoked {n} key(s) matching {target}");
            Ok(0)
        }
        _ => Ok(2),
    }
}

fn short_err(msg: &str) -> String {
    msg.lines().next().unwrap_or(msg).to_string()
}

fn usage() {
    println!(
        "deepseek-rs {version} — OpenAI-compatible proxy for the DeepSeek private API

USAGE
  deepseek-rs <command> [options]

SERVER
  serve                       run in the foreground (Ctrl-C to stop)
  start [-b|--bg]             foreground by default (pretty logs, Ctrl-C stops);
                              -b/--bg detaches and waits until healthy
  stop                        SIGTERM the server (SIGKILL after 15s)
  restart [-b]                stop, then start (same flags as start)
  status                      pid, url, uptime, account count
  health                      one-shot health check (exit 1 when down)
  stats [--json]              live counters, per-account state, db counts

LOGS
  logs [-n N] [-f] [--path P] last N lines (default {lines}); -f follows

ACCOUNTS  (stored in {db}; env vars still work and win)
  accounts                    list accounts and where they come from
  accounts add <email> <password>
  accounts add <email> --token <token>
  accounts rm <email>         remove account + its cached token

CHATGPT WEB ACCOUNTS  (stored in {db}; env CHATGPT_TOKEN_<n> still works)
  chatgpt                     list accounts, expiry and last alert
  chatgpt add --access-token <token|->   read from stdin with '-'
      [--label L] [--proxy URL]
  chatgpt rm <id|label>       remove an account
  chatgpt check               show remaining life and whether Bark is set

  No refresh path exists for these tokens: set BARK_URL or BARK_KEY and the
  server pushes a warning at 72h, 24h and 2h before expiry.

API KEYS
  keys                        list registered keys
  keys add <name> <key>       register a key (alias: key <name> <key>)
  keys rm <name|id>           revoke a key
MISC
  check                       verify login + PoW for every account, exit
  url                         print the OpenAI base URL
  version                     print the version
  help                        this message

ENVIRONMENT
  DEEPSEEK_LOG              prod (default) | debug  — pretty logs either way
  RUST_LOG                  full filter override (blank is ignored)
  DEEPSEEK_LOG_ANSI         1/0 to force/disable colour
  NO_COLOR                  disables colour
  PROXY_HOST / PROXY_PORT   bind address (default 0.0.0.0:9174)
  DB_PATH                   sqlite file (default deepseek-rs.db)
  CONFIG_PATH               alternate config.toml
  DEEPSEEK_RUNTIME_DIR      pidfile + log location
  REQUEST_DELAY             seconds between requests per account (default 0)
  QUEUE_LIMIT               max queued requests per account (default 64)
  THINKING_ENABLED / SEARCH_ENABLED
  DEEPSEEK_EMAIL_<i> / DEEPSEEK_PASSWORD_<i> / DEEPSEEK_TOKEN_<i>
                            accounts from the environment, checked in order
  CHATGPT_TOKEN_<i>         ChatGPT Web access token (cannot be refreshed)
  BARK_URL / BARK_KEY       push endpoint for token-expiry warnings",
        version = env!("CARGO_PKG_VERSION"),
        lines = DEFAULT_LOG_LINES,
        db = cfg_or_exit().db_path,
    );
}
