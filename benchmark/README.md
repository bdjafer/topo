# benchmark/ — Mutation Sensitivity Evaluation

Tests whether topo's issue detectors fire on structurally degraded codebases.

## How it works

```
examples/<name>/graph.json          (pre-parsed, from harvest or fetch_and_analyze)
        │
        ├── clean analysis          (reuses analysis.json if cached, otherwise runs fresh)
        │
        ├── mutate (topo mutate)    → mutated graph (in memory)
        │
        ├── analyze mutated graph   → always runs fresh (mutated graph doesn't exist on disk)
        │
        └── compare: did expected diagnostic fire?
```

## Commands

```bash
# Prerequisite: parse some repos first
make harvest REPOS="flask click requests ripgrep"

# Run mutation evaluation on all repos that have graph.json
make benchmark

# Run on specific repos or mutations
python3 benchmark/scripts/evaluate_mutations.py --repos=ripgrep,flask
python3 benchmark/scripts/evaluate_mutations.py --mutations=inject_cycle,wide_interface
```

Requires: `target/release/topo` binary (built by `make harvest` or `cargo build --release -p topo-cli`).

## Scripts

| Script | Purpose | Reads from | Writes to |
|---|---|---|---|
| `harvest_corpus.py` | Clone + parse repos | `examples/registry.toml` | `examples/<name>/graph.json` |
| `evaluate_mutations.py` | Mutation sensitivity scorecard | `examples/*/graph.json` | `benchmark/results/` |
| `evaluate_distribution.py` | Cross-codebase diagnostic stats | `examples/*/graph.json` | `benchmark/results/` |

## Output

```
results/
  mutation_sensitivity/
    scorecard.json        # per-mutation sensitivity, specificity, collateral
```

## Mutation operators

5 implemented in `packages/topo-benchmark/src/mutations/`. Full status and scorecard in [MUTATIONS.md](../packages/topo-benchmark/src/MUTATIONS.md).
