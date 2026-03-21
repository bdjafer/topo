//! Content-addressed cache for topo parse results.
//!
//! Pure library — no filesystem I/O. Storage is injected via the
//! [`CacheStore`] trait. Consumers provide their own backends
//! (filesystem for CLI, in-memory for WASM).

use serde::{Deserialize, Serialize};

/// Cache envelope format version. Bump when the envelope structure changes.
const CACHE_FORMAT_VERSION: u32 = 1;

// ── Storage trait ────────────────────────────────────────────────────────

/// Backend for reading/writing cache entries.
///
/// Keys are namespace strings (e.g. `"graph"`, `"embeddings"`).
/// Values are serialized JSON strings.
pub trait CacheStore {
    /// Load a cached entry by key. Returns `None` on miss or read error.
    fn get(&self, key: &str) -> Option<String>;

    /// Store an entry.
    fn put(&mut self, key: &str, value: &str) -> Result<(), CacheError>;

    /// Remove an entry. Missing keys are not an error.
    fn delete(&mut self, key: &str) -> Result<(), CacheError>;
}

/// Cache operation error.
#[derive(Debug)]
pub struct CacheError(pub String);

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cache error: {}", self.0)
    }
}

impl std::error::Error for CacheError {}

impl From<std::io::Error> for CacheError {
    fn from(e: std::io::Error) -> Self {
        CacheError(e.to_string())
    }
}

impl From<serde_json::Error> for CacheError {
    fn from(e: serde_json::Error) -> Self {
        CacheError(e.to_string())
    }
}

// ── Content manifest ─────────────────────────────────────────────────────

/// A deterministic fingerprint of a set of source files.
///
/// Built from `(relative_path, content)` pairs. Sorted by path for
/// determinism regardless of filesystem walk order.
pub struct ContentManifest {
    /// Per-file hashes, sorted by path.
    file_hashes: Vec<(String, blake3::Hash)>,
}

impl ContentManifest {
    /// Build a manifest from an iterator of `(relative_path, file_content)`.
    ///
    /// The caller walks the filesystem (or provides file contents from any
    /// source). This function never does I/O.
    pub fn from_files(files: impl Iterator<Item = (String, Vec<u8>)>) -> Self {
        let mut file_hashes: Vec<(String, blake3::Hash)> = files
            .map(|(path, content)| {
                let mut hasher = blake3::Hasher::new();
                hasher.update(path.as_bytes());
                hasher.update(&content);
                (path, hasher.finalize())
            })
            .collect();

        file_hashes.sort_by(|a, b| a.0.cmp(&b.0));

        ContentManifest { file_hashes }
    }

    /// Compute the composite hash of the entire manifest.
    ///
    /// Combines all per-file hashes with the cache format version
    /// into a single blake3 hash. Returns hex-encoded string.
    pub fn hash(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        for (path, hash) in &self.file_hashes {
            hasher.update(path.as_bytes());
            hasher.update(hash.as_bytes());
        }
        hasher.update(&CACHE_FORMAT_VERSION.to_le_bytes());
        hasher.finalize().to_hex().to_string()
    }

    /// Number of files in the manifest.
    pub fn file_count(&self) -> usize {
        self.file_hashes.len()
    }
}

// ── Cache entry ──────────────────────────────────────────────────────────

/// Serialized cache envelope stored by a [`CacheStore`] backend.
#[derive(Serialize, Deserialize)]
pub struct CacheEntry {
    /// Envelope format version.
    pub format_version: u32,
    /// Version of the tool that produced this cache (e.g. topo CLI version).
    pub producer_version: String,
    /// Blake3 hash of the source content manifest.
    pub manifest_hash: String,
    /// When the cache was written (seconds since epoch).
    pub created_at: u64,
    /// The cached payload — an opaque JSON string.
    pub payload: String,
}

// ── Public API ───────────────────────────────────────────────────────────

/// Try to load a valid cache entry.
///
/// Returns `Some(payload)` on hit, `None` on miss/mismatch/corruption.
pub fn load_cached(
    store: &dyn CacheStore,
    key: &str,
    current_manifest_hash: &str,
    current_producer_version: &str,
) -> Option<String> {
    let raw = store.get(key)?;
    let entry: CacheEntry = serde_json::from_str(&raw).ok()?;

    if entry.format_version != CACHE_FORMAT_VERSION {
        return None;
    }
    if entry.producer_version != current_producer_version {
        return None;
    }
    if entry.manifest_hash != current_manifest_hash {
        return None;
    }

    Some(entry.payload)
}

/// Write a cache entry to the store.
///
/// Errors are returned but callers should treat cache write failures
/// as non-fatal — the parse result is still valid.
pub fn write_cache(
    store: &mut dyn CacheStore,
    key: &str,
    manifest_hash: &str,
    producer_version: &str,
    payload: &str,
) -> Result<(), CacheError> {
    let entry = CacheEntry {
        format_version: CACHE_FORMAT_VERSION,
        producer_version: producer_version.to_string(),
        manifest_hash: manifest_hash.to_string(),
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        payload: payload.to_string(),
    };

    let json = serde_json::to_string(&entry)?;
    store.put(key, &json)
}

// ── In-memory store (for WASM / testing) ─────────────────────────────────

/// In-memory cache store. Session-scoped, no persistence.
///
/// Useful for WASM builds (persists within a browser tab) and tests.
#[derive(Default)]
pub struct MemoryStore {
    entries: std::collections::HashMap<String, String>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl CacheStore for MemoryStore {
    fn get(&self, key: &str) -> Option<String> {
        self.entries.get(key).cloned()
    }

    fn put(&mut self, key: &str, value: &str) -> Result<(), CacheError> {
        self.entries.insert(key.to_string(), value.to_string());
        Ok(())
    }

    fn delete(&mut self, key: &str) -> Result<(), CacheError> {
        self.entries.remove(key);
        Ok(())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_deterministic_regardless_of_order() {
        let files_a = vec![
            ("b.rs".to_string(), b"fn b() {}".to_vec()),
            ("a.rs".to_string(), b"fn a() {}".to_vec()),
        ];
        let files_b = vec![
            ("a.rs".to_string(), b"fn a() {}".to_vec()),
            ("b.rs".to_string(), b"fn b() {}".to_vec()),
        ];

        let hash_a = ContentManifest::from_files(files_a.into_iter()).hash();
        let hash_b = ContentManifest::from_files(files_b.into_iter()).hash();
        assert_eq!(hash_a, hash_b);
    }

    #[test]
    fn manifest_changes_on_content_change() {
        let files_v1 = vec![("a.rs".to_string(), b"fn a() {}".to_vec())];
        let files_v2 = vec![("a.rs".to_string(), b"fn a() { 1 }".to_vec())];

        let hash_v1 = ContentManifest::from_files(files_v1.into_iter()).hash();
        let hash_v2 = ContentManifest::from_files(files_v2.into_iter()).hash();
        assert_ne!(hash_v1, hash_v2);
    }

    #[test]
    fn manifest_changes_on_path_change() {
        let files_v1 = vec![("a.rs".to_string(), b"fn a() {}".to_vec())];
        let files_v2 = vec![("b.rs".to_string(), b"fn a() {}".to_vec())];

        let hash_v1 = ContentManifest::from_files(files_v1.into_iter()).hash();
        let hash_v2 = ContentManifest::from_files(files_v2.into_iter()).hash();
        assert_ne!(hash_v1, hash_v2);
    }

    #[test]
    fn round_trip_through_memory_store() {
        let mut store = MemoryStore::new();
        let manifest = ContentManifest::from_files(
            vec![("a.rs".to_string(), b"hello".to_vec())].into_iter(),
        );
        let hash = manifest.hash();

        // Miss on empty store.
        assert!(load_cached(&store, "graph", &hash, "0.1.0").is_none());

        // Write.
        write_cache(&mut store, "graph", &hash, "0.1.0", "{\"nodes\":[]}").unwrap();

        // Hit.
        let payload = load_cached(&store, "graph", &hash, "0.1.0");
        assert_eq!(payload.as_deref(), Some("{\"nodes\":[]}"));

        // Miss on version change.
        assert!(load_cached(&store, "graph", &hash, "0.2.0").is_none());

        // Miss on hash change.
        assert!(load_cached(&store, "graph", "different_hash", "0.1.0").is_none());
    }
}
