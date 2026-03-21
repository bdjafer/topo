# tantivy — Structural Analysis

[Tantivy](https://github.com/quickwit-oss/tantivy) is a full-text search engine library inspired by Apache Lucene. It features segment-based indexing, BM25 scoring, and a composable collector/query architecture. Maintained by the Quickwit team.

**Version analyzed:** latest main (2025)

## Files

| File | Description |
|------|-------------|
| `graph.json` | Parsed code graph (5400 nodes, 14549 edges) |
| `analysis.json` | Full structural analysis |
| `analysis.txt` | Human-readable formatted output |

## How to reproduce

```bash
git clone --depth 1 https://github.com/quickwit-oss/tantivy.git /tmp/tantivy
cargo run -p topo-cli -- /tmp/tantivy --verbose          # formatted text
cargo run -p topo-cli -- /tmp/tantivy --json              # JSON output
cargo run -p topo-cli -- parse /tmp/tantivy -o graph.json # graph only
```

## Graph summary

- **5400 nodes**: 410 modules, 879 structs/enums/traits, 4111 functions
- **14549 edges**: 6095 calls, 5172 defines, 2958 imports, 324 inherits
- **56 crates** detected (9 workspace members + test/bench/example crates)

## Analysis results

| Metric | Value | Notes |
|--------|-------|-------|
| Spectral modules | 3 | Under-partitioned (eigengap chose k=3 for 56 crates) |
| Modularity Q | 0.005 | Near-zero — spectral clustering found no meaningful module structure |
| Silhouette | 0.73 | Misleading — high only because 2 large clusters have low internal variance |
| NMI vs packages | 0.06 | Very low agreement with crate boundaries |
| Structural issues | 356 | |

### What this result means

Tantivy is a heavily interconnected library where most components depend on shared core types (Schema, Term, Document, Segment). The spectral analysis finds that the coupling graph has **no clear community structure** — the codebase is one tightly-connected unit, not a collection of separable modules.

This contrasts sharply with ripgrep (Modularity Q = 0.84), which has clean crate boundaries that spectral analysis recovers well. The difference reflects a genuine architectural distinction: ripgrep is a layered tool with separable concerns; tantivy is a monolithic library with pervasive cross-cutting types.

### Key structural findings

- **356 issues detected** — mostly cross-module coupling and spectral outliers
- Role classification still works: identifies hubs (core types), bridges (query↔index coupling), and utilities
- The result honestly reflects that spectral module detection has limits on highly interconnected graphs — this is documented in CLAUDE.md as an open question
