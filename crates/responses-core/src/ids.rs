//! Opaque object identifiers and timestamps.
//!
//! Identifiers only need to be unique; access control is enforced by owner
//! checks in the storage layer, so they are not secrets. Each id mixes the
//! wall clock, a process-wide counter and a randomly keyed hash, which keeps
//! ids unique even when the Windows clock resolution repeats a timestamp.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A new identifier such as `resp_…`, `msg_…`, `rs_…`, `fc_…` or `call_…`.
pub fn new_id(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u64(nanos);
    hasher.write_u64(count);
    format!(
        "{prefix}_{nanos:016x}{:016x}{:08x}",
        hasher.finish(),
        count as u32
    )
}

/// Current Unix time in whole seconds.
pub fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn ids_are_prefixed_and_unique_in_a_tight_loop() {
        let mut seen = HashSet::new();
        for _ in 0..10_000 {
            let id = new_id("call");
            assert!(id.starts_with("call_"));
            assert!(id[5..].chars().all(|c| c.is_ascii_hexdigit()));
            assert!(seen.insert(id), "duplicate id generated");
        }
    }
}
