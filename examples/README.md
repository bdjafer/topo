# examples/ — Dataset Registry & Cache

Single source of truth for all codebases topo analyzes.

## Structure

```
examples/
  registry.toml          # manifest: 50 repos, pinned commits
  scripts/
    fetch_and_analyze.sh  # full pipeline: clone → parse → analyze
    harvest_corpus.py     # lightweight: clone → parse only
    validate_examples.py  # CI: check committed artifacts
    collect_metadata.py   # fetch GitHub API metadata
  ripgrep/
    graph.json           # parsed dependency graph (from parse step)
    analysis.json        # full topo analysis output (from analyze step)
    analysis.txt         # human-readable analysis
    embeddings.json      # semantic embeddings (optional)
    metadata.json        # GitHub metadata (optional)
  flask/
    graph.json           # parse only (no analysis cached)
  ...
```

## What produces what

| Step | Script | Input | Output | Stored in |
|---|---|---|---|---|
| **Parse** | `harvest_corpus.py` | source code (cloned) | `graph.json` | `examples/<name>/` |
| **Parse + Analyze** | `fetch_and_analyze.sh` | source code (cloned) | `graph.json` + `analysis.json` + `analysis.txt` | `examples/<name>/` |
| **Benchmark** | `evaluate_mutations.py` | `graph.json` | scorecard (not cached per-repo) | `benchmark/results/` |

The benchmark re-uses `analysis.json` for the clean baseline if it exists. Mutated graphs are always analyzed fresh (they're generated in memory).

## registry.toml

Every repo is an `[[example]]` entry with:
- `name`, `repo` (GitHub URL), `commit` (pinned SHA), `language`, `entrypoint`
- Optional: `ref` (git tag), `description`, `[example.tags]` (size/quality/pattern), `[example.cli_overrides]` (exclude dirs)

## Commands

```bash
# List all registered repos
make examples-list

# Parse specific repos (clone + parse → graph.json)
make harvest REPOS="flask click requests"

# Parse all 50 repos
make harvest-all

# Full pipeline for a single repo (parse + analyze → graph.json + analysis.json)
./examples/scripts/fetch_and_analyze.sh ripgrep

# Run mutation benchmark on all repos that have graph.json
make benchmark
```

## Adding a new repo

1. Add an `[[example]]` entry to `registry.toml` with pinned commit
2. Run `make harvest REPOS=<name>` to parse it
3. Optionally run `./examples/scripts/fetch_and_analyze.sh <name>` for full analysis

## What gets committed

Only curated examples with full analysis (ripgrep, tantivy) have artifacts committed. Bulk-harvested `graph.json` files are local cache — regenerate with `make harvest`.
