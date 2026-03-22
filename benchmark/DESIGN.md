# Benchmark V2 — Perturbation Sensitivity

## Core Idea

We don't need ground-truth labels. We need to know that when we MAKE something
structurally wrong, the right diagnostic fires — and doesn't fire on the clean
version.

This gives us precision/recall on synthetic ground truth, fully automated.

## Three Self-Supervised Signals

### Signal 1: Mutation Sensitivity at Scale

Parse 200+ real repos from the corpus. For each:

1. Run clean analysis. Record all issues.
2. Apply each of 9 mutation types at 3 severity levels.
3. Run analysis on each mutated graph.
4. Check: did the expected diagnostic fire? Did it NOT fire on clean?

This gives 200 × 9 × 3 = 5,400 test cases with known ground truth.

**Metrics per mutation type:**
- **Sensitivity** (true positive rate): fraction of mutated graphs where expected diagnostic fires
- **Specificity**: fraction of clean graphs where diagnostic does NOT already fire
- **Net sensitivity**: TP_rate - FP_rate (must be > 0)
- **Severity calibration**: correlation between mutation level (1/2/3) and output severity
- **Collateral ratio**: how many unrelated diagnostics change
- **Attribution@3**: whether mutated region appears in top-3 issues

### Signal 2: Temporal Retrodiction

For repos with git history, find commits where developers:
- Moved a function across modules (`git log --diff-filter=R`)
- Broke a cycle (removed an import that was part of an SCC)
- Split a module (new `__init__.py`/`mod.rs` with moved functions)

Run topo on the pre-fix commit. If the relevant diagnostic flagged the issue
before the developer fixed it, we retrodicted correctly.

The developer's refactoring decision IS the ground truth.

### Signal 3: Cross-Codebase Distributional Sanity

Run on 500+ repos, collect aggregate statistics:
- Issue rate per diagnostic type (should be 5-60%, not 0% or 95%)
- Severity distribution (should be right-skewed, not all 1.0)
- Diagnostic concordance (do related diagnostics co-occur?)
- Issue-per-node ratio (should be O(1), not scaling linearly)

## Mutation Types

| Mutation | Expected Diagnostic | Severity 1 (mild) | Severity 2 (medium) | Severity 3 (severe) |
|---|---|---|---|---|
| inject_cycle | circular_dependency | 2-node within module | 3-node across 2 modules | 5+ node across 3+ modules |
| wide_interface | wide_interface | 5 cross-module calls | 12 cross-module calls | 20 cross-module calls |
| misplaced_concern | misplaced_concern | Swap 1 embedding | Swap 2 embeddings | Swap 3+ to distant modules |
| incoherent_module | incoherent_module | 2 random embeddings | 30% random | Merge 2 modules |
| shadow_dependency | shadow_dependency | Clone 1 function | Clone 2 + refs | Clone subgraph |
| layer_violation | layer_violation | 1 upward edge | 3 upward edges | 5+ upward edges |
| near_disconnect | near_disconnect | Remove to 2 bridges | Remove to 1 bridge | Remove to 1 edge |
| overloaded_utility | overloaded_utility | Route 50% through 1 | Route 70% | Route 80%+ |
| redundant_api | redundant_api | Duplicate 1 entry | Duplicate 2 | Duplicate 3+ |

## Architecture

```
Mutation operators (Rust)          Orchestration (Python)
┌──────────────────────┐          ┌─────────────────────────┐
│ mutations/           │          │ evaluation.py           │
│   inject_cycle.rs    │──PyO3──▶│   signal 1: sensitivity │
│   wide_interface.rs  │          │ retrodiction.py         │
│   misplaced_concern  │          │   signal 2: git history │
│   ...                │          │ distribution.py         │
│                      │          │   signal 3: sanity      │
│ types.rs             │          │ corpus.py               │
│   MutationResult     │          │   harvesting + caching  │
│   MutationType       │          └─────────────────────────┘
│   SeverityConfig     │
└──────────────────────┘
```

**Why Rust operators:** They need graph topology awareness (find bridges,
compute degree, identify modules). These algorithms exist in topo-analyzer.
Reimplementing in Python is duplication.

**Why Python orchestration:** Iteration speed for data analysis. The evaluation
loop calls Rust operators + analyzer via PyO3, collects results, computes
aggregate metrics.

## Corpus

`corpus_manifest.json` lists 200+ repos with pinned SHAs. Cached parsed graphs
in `corpus/` (gitignored). The manifest is committed for reproducibility.

Selection criteria:
- 20+ nodes, <50,000 nodes
- Mix of Python and Rust
- Mix of sizes and architectural quality
- Must parse successfully with current topo-parser

## Evaluation Output

```
benchmark/results/
  mutation_sensitivity/
    scorecard.json        # Per-type sensitivity, specificity, calibration
    per_repo.jsonl        # Per-repo per-mutation results
  retrodiction/
    scorecard.json        # Pre-fix detection rate
    per_commit.jsonl      # Per-commit results
  distribution/
    summary.json          # Aggregate statistics
    histograms.json       # Per-diagnostic distributions
```

## Implementation Sequence

Phase A: Mutation operators (Rust) — inject_cycle, layer_violation first
Phase B: Corpus harvesting (Python) — 50 repos initial
Phase C: Signal 1 evaluation — run on initial corpus
Phase D: Signal 3 distribution — quick win
Phase E: Signal 2 retrodiction — hardest, requires git analysis
