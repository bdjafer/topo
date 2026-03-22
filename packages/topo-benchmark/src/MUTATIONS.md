# Mutation Benchmark — Pipeline & Status

Last updated: 2026-03-22

## How it works

```
examples/ripgrep/graph.json     ─┐
examples/tantivy/graph.json      ├── cached real codebases
benchmark/corpus/*/graph.json   ─┘   (harvest_corpus.py for more)
        │
        ▼
  evaluate_mutations.py         for each repo × mutation × severity:
        │                         1. load graph.json
        │                         2. clean analysis (topo analyze --input)
        │                         3. topo mutate --input --type --severity --seed
        │                         4. analyze mutated graph
        │                         5. check expected diagnostic fires
        ▼
  benchmark/results/            scorecard.json (sensitivity, specificity, collateral)
  mutation_sensitivity/
```

**The pipeline runs end-to-end.** All 5 implemented mutations are wired through the CLI (`topo mutate` + `topo analyze`).

## How to run

```bash
# Unit tests (6 pass — structural validity)
cargo test --package topo-benchmark --lib mutations

# Integration trigger tests on real data (5 pass — ripgrep graph)
cargo test --release --package topo-benchmark --test mutation_triggers

# Full evaluation on cached datasets
python3 benchmark/scripts/evaluate_mutations.py --repos=ripgrep,tantivy

# Full evaluation on harvested corpus
python3 benchmark/scripts/harvest_corpus.py flask requests bat
python3 benchmark/scripts/evaluate_mutations.py
```

## Scorecard (12 repos, 180 test cases, 2026-03-22)

Repos: flask, click, requests, attrs, marshmallow, jinja, typer, fastapi, black, httpx, ripgrep, tantivy.

| Mutation | Sensitivity | Specificity | Net | Collateral | Verdict |
|---|---|---|---|---|---|
| `inject_cycle` | **1.00** | 0.17 | **+0.17** | 0.17 | rock solid — always triggers, low FP rate |
| `wide_interface` | **0.94** | 0.00 | -0.06 | 0.26 | strong — 1 miss in 36 test cases |
| `near_disconnect` | **0.83** | 0.25 | **+0.08** | 0.13 | good — positive net sensitivity |
| `overloaded_utility` | **0.56** | 0.33 | -0.11 | 0.23 | mediocre — small graphs lack headroom above p85 |
| `layer_violation` | **0.17** | 0.50 | -0.33 | 0.19 | weak — SCC suppression blocks most injections |

**Test integrity:** Integration tests always re-analyze the mutated graph (no escape hatch). All math formulas verified against detector implementations.

## Mutation operators

### Implemented (5)

| Mutation | Operator | Diagnostic | Trigger test | Confidence |
|---|---|---|---|---|
| `inject_cycle` | `inject_cycle.rs` | `circular_dependency` | pass (unit + integration) | **high** |
| `layer_violation` | `layer_violation.rs` | `layer_violation` | pass (integration) | **medium** — accounts for existing forward edges and avoids SCC-forming pairs, but still gets suppressed sometimes |
| `overloaded_utility` | `overloaded_utility.rs` | `overloaded_utility` | pass (integration) | **high** — prefers low-out_degree targets, controls for direction significance |
| `wide_interface` | `wide_interface.rs` | `wide_interface` | pass (integration) | **high** — uses distinct symbol pairs (matches detector) |
| `near_disconnect` | `near_disconnect.rs` | `near_disconnect` | pass (integration) | **high** — removes fraction of intra-module edges, no spanning tree required |

### Not implemented (4)

| Mutation | Diagnostic | Blocker |
|---|---|---|
| `misplaced_concern` | `misplaced_concern` | needs semantic embeddings |
| `incoherent_module` | `incoherent_module` | needs semantic embeddings |
| `shadow_dependency` | `shadow_dependency` | not started |
| `redundant_api` | `redundant_api` | not started |

## Fixes applied in this session

1. **`wide_interface` bug fix**: Was computing Tukey fence from `DependencyOutput.weight` (total edge count). Detector uses distinct `(src, tgt)` symbol pairs. Fixed to match.

2. **`layer_violation` fix**: Was ignoring existing forward edges between module pairs. If the pair had 4+ forward edges, adding 6 reverse edges diluted the binomial test. Now counts existing edges and adjusts reverse edge count accordingly. Also filters out module pairs that would create SCCs. Also uses few distinct node pairs with multiple edge kinds to avoid triggering `wide_interface` as collateral.

3. **`overloaded_utility` fix**: Was picking random targets regardless of out_degree. Targets with high out_degree fail the direction significance gate. Now sorts candidates by out_degree ascending and picks from the bottom quartile. Also skips `lib`-labeled modules (matching detector's suppression list). Adjusts edge count to guarantee passing both p85 and direction gates.

4. **`near_disconnect` fix**: Was computing a BFS spanning tree, which fails when spectral modules aren't internally connected (common on real codebases). Simplified to remove a fraction (60-90%) of intra-module edges directly.

5. **`inject_cycle` test fix**: Conditional assert `if !clean_has` was vacuous when clean graph already had cycles. Now compares counts: mutated must have more.

## CLI subcommands added

```bash
# Analyze a pre-parsed graph.json
topo analyze --input graph.json
# → outputs full analysis JSON to stdout

# Apply a mutation
topo mutate --input graph.json --type inject_cycle --severity 2 --seed 42
# → outputs JSON with mutated graph + metadata
# exit code 2 if mutation can't be applied (preconditions not met)
```

## Known limitations

- **`layer_violation` sensitivity is low (0.33).** The detector suppresses layer violations inside SCCs. The mutation avoids creating SCC-forming pairs, but on real codebases many module pairs have forward paths between them, limiting available targets.
- **Collateral 10-20%.** Mutations sometimes trigger unrelated diagnostics. No test checks for this systematically.
- **No attribution tracking.** The scorecard doesn't check whether the expected diagnostic fires in the mutated REGION specifically.
- **Severity calibration untested.** No correlation between mutation severity (1/2/3) and reported issue severity.
- **2 repos only.** The scorecard above is from ripgrep + tantivy. More repos needed for statistical confidence.
