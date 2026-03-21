# ripgrep — Structural Analysis

[ripgrep](https://github.com/BurntSushi/ripgrep) is a line-oriented search tool by Andrew Galloway (BurntSushi). It's a 9-crate Rust workspace with a clean layered architecture centered on the `Matcher` trait abstraction.

**Version analyzed:** 14.1.1 (tag `14.1.1`)

## Files

| File | Description |
|------|-------------|
| `graph.json` | Parsed code graph (2502 nodes, 4865 edges) |
| `analysis.json` | Full structural analysis (spectral modules, roles, issues) |
| `analysis.txt` | Human-readable formatted output |

## How to reproduce

```bash
git clone --depth 1 --branch 14.1.1 https://github.com/BurntSushi/ripgrep.git /tmp/ripgrep
cargo run -p topo-cli -- /tmp/ripgrep --verbose          # formatted text
cargo run -p topo-cli -- /tmp/ripgrep --json              # JSON output
cargo run -p topo-cli -- parse /tmp/ripgrep -o graph.json # graph only
```

## Graph summary

- **2502 nodes**: 98 modules, 360 structs/enums/traits, 2044 functions
- **4865 edges**: 2028 calls, 2482 defines, 235 imports, 120 inherits
- **17 crates** detected (9 core + test/bench/example crates)

## Analysis results

| Metric | Value |
|--------|-------|
| Spectral modules | 12 |
| Modularity Q | 0.84 |
| Silhouette | 0.60 |
| Structural issues | 228 |

### Detected modules

| Module | Size | Interpretation |
|--------|------|----------------|
| grep_printer + grep_searcher | 406 | Search output pipeline |
| rg (+ grep_printer) | 405 | CLI orchestration + output config |
| rg (+ ignore) | 335 | CLI + file traversal |
| rg + grep_printer | 307 | Search execution |
| ignore + rg | 296 | File ignore subsystem |
| rg + grep_regex (+ grep_matcher) | 157 | Regex/matching engine |
| rg (+ globset) | 140 | CLI + glob patterns |
| rg + grep_regex (+ grep_pcre2) | 133 | PCRE2 regex path |
| rg (+ integration) | 119 | CLI core + tests |
| rg + globset (+ ignore) | 114 | Glob + ignore patterns |
| rg + globset (+ integration) | 81 | Glob testing |

### Key structural findings

- **`grep_matcher`** correctly identified as a fragile hub (degree 84, high betweenness) — it's the central `Matcher` trait abstraction that all regex engines implement
- **Cross-package coupling** revealed between `grep_printer`, `grep_searcher`, and `grep_matcher` — these form a single functional unit despite being separate crates
- **`rg.flags`** identified as a fragile hub (degree 44) — the massive flag definition module that orchestrates all other crates
- **Cyclic dependencies** detected in `rg.flags` submodules (8-node cycle) and `ignore` internals (7-node cycle)
