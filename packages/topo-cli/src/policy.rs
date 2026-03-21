//! Policy loading from topo.toml / .topo.toml.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

/// Repository-level analysis policy.
#[derive(Debug, Default, Deserialize)]
pub struct Policy {
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub level: Option<String>,
    #[serde(default)]
    pub ignores: HashMap<String, String>,
}

/// Load policy from `topo.toml` or `.topo.toml` in the given directory.
/// Returns None if no policy file exists.
pub fn load_policy(dir: &Path) -> Result<Option<Policy>> {
    for name in &["topo.toml", ".topo.toml"] {
        let path = dir.join(name);
        if path.is_file() {
            let content = std::fs::read_to_string(&path)?;
            let policy: Policy = toml::from_str(&content)?;
            return Ok(Some(policy));
        }
    }
    Ok(None)
}
