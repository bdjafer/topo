# Benchmark V2

This document defines how `topo-analyzer` should be evaluated.

The goal is strict: if the benchmark score goes up, the analyzer is meaningfully
better at recovering real structure, ranking worse structure below better
structure, staying stable under irrelevant changes, and detecting known
structural antipatterns.

## Core Constraint

The analyzer's core bet — that spectral analysis of code graphs produces
architecturally meaningful results — is still being validated. This benchmark is
sized accordingly. It is an honest evaluation tool for an unproven approach, not
a publication-grade framework for a mature system.

## What "Better" Means

A candidate analyzer is better than a reference only if:

1. All mutation cases that passed before still pass.
2. At least one previously-failing case now passes, or per-dimension scores
   improve.
3. No dimension score regresses by more than `0.03`.
4. No guardrail fails.
5. The candidate still matches or beats the simple baselines on primary tasks.

No statistical machinery at this scale. With fewer than 30 cases per dimension,
pass/fail on individual cases is more honest than bootstrap confidence intervals.
Statistical testing enters when case count exceeds 30.

## Principles

1. Measure claims, not vibes. Every score corresponds to a concrete claim.
2. Prefer objective truth over subjective judgment. Controlled mutations and
   pairwise ordering over free-form ratings.
3. Exclude ambiguity instead of averaging it in.
4. Separate analyzer-only evaluation from end-to-end evaluation.
5. A result that does not beat simple baselines is not a result.
6. Use fixed dataset versions and fixed seeds.
7. Never let one strong metric hide one weak metric.
8. Do not dress up small samples in large-sample statistics.

## Benchmark Tiers

### Analyzer Tier (V1 — build this first)

Input is a frozen serialized `CodeGraph`. This isolates analyzer quality from
parser noise. All V1 evaluation happens here.

### End-to-End Tier (V2 — deferred)

Input is source code. Tests parser + analyzer integration, projection handling,
and CLI pipeline regressions. Build this after the analyzer tier proves useful.

## Scope

### In Scope for V1

- `module`-level analysis only
- analyzer-only benchmark on frozen graphs
- graph-level mutations with known ordering
- stability under structure-preserving transformations
- seeded anomaly detection (not general anomaly discovery)
- one architecture recovery dataset (topo itself)

### Explicitly Out of Scope for V1

- end-to-end source-level cases
- domain-model extraction
- temporal drift analysis
- cross-layer disagreement quality measurement
- calibration metrics (ECE, Brier) — deferred until real anomaly ground truth
  exists
- hidden split — deferred until case count exceeds 30
- symbol-level analysis
- LLM usefulness of summaries

## Four Benchmark Dimensions

### 1. Architecture Recovery

Question: does the analyzer recover real architectural boundaries?

#### Known Limitation

For most Python codebases, high-confidence module labels approximate directory
structure. This means V1 architecture recovery primarily validates that spectral
clustering is consistent with directory structure, not that it improves on it.
V2 should include repositories with documented directory-architecture mismatches
(monorepos with shared utilities, historical package splits that should be
merged, codebases where maintainers have documented that directory structure
diverges from architectural intent).

#### Dataset

Each dataset item contains:

- a frozen `CodeGraph`
- a gold module label for each included node
- an explicit `excluded_nodes` list
- metadata describing label provenance

Only nodes with high-confidence labels are included. Ambiguous nodes are
excluded, not weakly labeled.

V1 uses one repository: topo itself (known intimately, monorepo with clear
package boundaries). Additional repositories enter in later versions after the
labeling protocol is validated.

#### Labeling Rules

1. Label only coarse architectural ownership, not fine-grained intent.
2. Generated code is excluded by default.
3. Glue code, adapters, and mixed-responsibility nodes are excluded unless
   ownership is obvious.
4. A node enters the gold set only if:
   - documentation strongly implies its module, or
   - maintainers agree on its module, or
   - the label can be justified from architecture docs and code structure without
     guesswork
5. Any disagreement or uncertainty moves the node to `excluded_nodes`.
6. Labels must be documented with a one-line rationale per module boundary
   decision.

#### Primary Metrics

- `NMI`: Normalized Mutual Information between predicted modules and gold
  modules on the included node set. NMI uses entropy rather than pair-counting,
  so it correctly measures clustering quality even when the predicted partition
  is finer-grained than the gold partition — which is expected behavior for
  spectral clustering. ARI is not used here because it penalizes refinement as
  disagreement (see design decision in CLAUDE.md).
- `Boundary F1`: for every observed edge between included nodes, classify it as
  intra-module or cross-module according to the gold partition. Compute F1 on
  the cross-module class (binary F1, positive class = cross-module). When the
  cross-module class is the minority, this is a conservative metric — high
  scores require both precision and recall on boundary edges.
- `Coverage`: fraction of included gold nodes that receive a non-unassigned
  predicted module.

#### Concrete Defaults

- Coverage floor: `0.80`. Below this, the candidate fails the guardrail
  regardless of other scores.

#### Dimension Score

```text
architecture_recovery =
  geometric_mean(NMI, BoundaryF1) * min(1.0, Coverage / 0.80)
```

The coverage term is `1.0` when coverage meets or exceeds the floor, and scales
linearly below it. This is less harsh than raw multiplication while still
penalizing abstention.

#### Guardrails

- Coverage must be at least `0.80`.
- The candidate must match or beat directory/package grouping on NMI.
- "Beat" means: candidate NMI >= baseline NMI. Exact ties are acceptable.

### 2. Mutation Ranking

Question: does the analyzer rank structurally worse variants below better ones?

This is the most important dimension. Build it first.

#### Dataset

Each item is an ordered pair or ordered chain of graphs with known structural
ordering. Mutations are graph-level rewrites on frozen `CodeGraph` fixtures.
Source-level rewrites are deferred to V2.

#### Required Mutation Families

- reverse dependency across module boundaries
- strongly connected cycle across module boundaries
- injected bridge/hub node connecting separate regions
- oversized merged module from natural clusters
- misplaced utility in a high-betweenness position
- repaired versions of the same defects

#### Derived Signals

The runner extracts these from structured analyzer output. These are the
analyzer's own signals — they measure internal consistency, not ground truth
correctness. This is a known limitation: a systematically wrong analyzer could
pass by being consistently wrong. The architecture recovery and anomaly
dimensions provide the external correctness check.

- `partition_similarity_to_clean`: ARI between predicted module partitions for
  the current variant and the `clean` variant on their shared nodes. ARI is
  appropriate here (unlike in architecture recovery) because both partitions
  come from the same analyzer and have comparable granularity.
- `largest_module_ratio`: `health.largest_module_ratio`
- `module_count`: `clustering.module_count`
- `max_cross_module_severity`: maximum anomaly severity where
  `kind == "cross_module"`, default `0`
- `max_cycle_severity`: maximum anomaly severity where
  `kind == "cycle_member"`, default `0`
- `target_role(node_id)`: predicted role for the declared target node
- `target_has_spectral_outlier(node_id)`: whether any
  `kind == "spectral_outlier"` anomaly contains the declared target node
- `attribution_at_3(mutated_region)`: whether one of the top 3 anomalies by
  severity overlaps the declared mutated region (IoU >= `0.3` on node sets)

#### Case Expectations

Each mutation case declares which signals must move and in which direction. A
case passes only if all of its required inequalities hold by a minimum margin
`epsilon = 0.05`. Exact ties count as failure.

#### V1 Mutation Suite

12 analyzer-only cases on the public split. Rules:

- keep graphs small and hand-auditable
- use graph-level rewrites only
- use `module` projection by default
- allow `symbol` projection only when the case is explicitly about a single
  utility, bridge, or spectral outlier
- include a `repaired` variant for at least half the cases

##### Input Design Rules

- each case has one dominant structural change
- the `clean` variant must already be slightly imperfect; no toy graphs with
  perfectly separated cliques
- the `mutated` variant must not collapse into global structural chaos
- the main signal should be recoverable without requiring the analyzer to solve
  several unrelated problems at once
- a mutation case may produce one primary signal and at most one plausible
  secondary side effect

##### Case Table

| ID | Level | Variants | Mutation | Required expectations |
| --- | --- | --- | --- | --- |
| `revdep_light` | `module` | `clean`, `mutated`, `repaired` | Add one reverse `calls` edge across a directional module boundary. | `max_cross_module_severity(mutated) > clean`; `attribution_at_3(mutated_region) = 1`; `partition_similarity_to_clean(mutated)` stays high; `repaired` closer to `clean` than `mutated`. |
| `revdep_balanced` | `module` | `clean`, `mutated` | Add enough reverse edges to make the boundary nearly bidirectional. | `max_cross_module_severity(mutated)` higher than `revdep_light`; `attribution_at_3(mutated_region) = 1`; `partition_similarity_to_clean(mutated)` stays high. |
| `revdep_multilayer` | `module` | `clean`, `mutated`, `repaired` | Add reverse `calls` and `imports` edges across the same boundary. | Combined-mode `max_cross_module_severity(mutated) > clean`; `attribution_at_3(mutated_region) = 1`; `repaired` moves back toward `clean`. |
| `cycle_two_module` | `module` | `clean`, `mutated`, `repaired` | Add one back edge that closes a 2-module SCC. | `max_cycle_severity(mutated) > clean`; `attribution_at_3(mutated_region) = 1`; `partition_similarity_to_clean(mutated)` stays high; `repaired` reduces cycle severity. |
| `cycle_three_module_ring` | `module` | `clean`, `mutated` | Add `C -> A` to an `A -> B -> C` chain. | `max_cycle_severity(mutated) > clean`; `attribution_at_3(mutated_region) = 1`; `module_count(mutated)` stays stable. |
| `cycle_multilayer` | `module` | `clean`, `mutated`, `repaired` | Create a cycle visible only when `calls` and `imports` are combined. | Combined-mode `max_cycle_severity(mutated) > clean`; `attribution_at_3(mutated_region) = 1`; `repaired` removes cycle signal. |
| `bridge_connector` | `module` | `clean`, `mutated`, `repaired` | Insert one connector node between two separate regions. | `target_role(connector)` becomes `bridge` or `hub`; `largest_module_ratio(mutated) > clean`; `attribution_at_3(mutated_region) = 1`; `repaired` weakens bridge signal. |
| `bridge_hub_escalation` | `module` | `clean`, `mutated` | Expand connector to touch three regions. | `target_role(connector)` becomes `hub` or `bridge`; `largest_module_ratio(mutated)` higher than `bridge_connector`; `partition_similarity_to_clean(mutated)` decreases. |
| `bridge_multilayer` | `module` | `clean`, `mutated`, `repaired` | Connect regions through mixed `calls` and `imports` edges. | Combined-mode `target_role(connector)` becomes `bridge` or `hub`; `attribution_at_3(mutated_region) = 1`; `repaired` moves back toward `clean`. |
| `boundary_erosion_light` | `module` | `clean`, `mutated`, `repaired` | Add a sparse band of cross edges between two natural clusters. | `partition_similarity_to_clean(mutated)` decreases slightly; `largest_module_ratio(mutated) > clean`; `module_count(mutated) <= clean`; `repaired` restores separation. |
| `boundary_merge_full` | `module` | `clean`, `mutated` | Add enough cross edges to collapse two clusters into one. | `partition_similarity_to_clean(mutated)` decreases strongly; `largest_module_ratio(mutated)` increases strongly; `module_count(mutated) < clean`. |
| `utility_misplaced` | `symbol` | `clean`, `mutated`, `repaired` | Move a leaf utility into a cross-module flow path. | `target_role(clean) = utility`; `target_role(mutated) != utility`; `target_has_spectral_outlier(target)` or `attribution_at_3(mutated_region) = 1`; `repaired` restores role. |

#### Primary Metrics

- `Pairwise Accuracy`: fraction of ordered cases whose declared expectations are
  all satisfied (with margin `epsilon = 0.05`).
- `Repair Accuracy`: fraction of repaired variants ranked closer to `clean` than
  their broken counterparts on the case's primary signals.
- `Attribution@3`: fraction of cases where the mutated region appears in the top
  3 anomalies (IoU >= `0.3` on node sets).

#### Dimension Score

```text
mutation_ranking =
  geometric_mean(
    PairwiseAccuracy,
    RepairAccuracy,
    AttributionAt3
  )
```

### 3. Stability

Question: does the analyzer stay stable when the architecture did not change?

#### Dataset

Each item is a base graph plus perturbations that preserve architectural
structure.

Perturbation families:

- symbol renaming
- definition reordering
- file splitting or merging without changing inter-module dependencies
- extraction or inlining of trivial wrappers
- formatting, comments, and docstring-only edits
- addition of dead leaf helpers

V1 uses perturbations that do not rename nodes (no `node_mapping` required).
Rename-based perturbations enter in V2 when the mapping infrastructure is
validated.

#### Primary Metrics

- `Partition Stability`: ARI between predicted module partitions before and
  after perturbation. ARI is appropriate here because both partitions come from
  the same analyzer on structurally-equivalent graphs with identical node sets.
- `Role Stability`: macro F1 of structural roles on the same node set before
  and after perturbation.

Top-K Stability (anomaly list overlap) is deferred to V2. It is only meaningful
after anomaly detection quality is validated independently.

#### Dimension Score

```text
stability =
  geometric_mean(
    PartitionStability,
    RoleStability
  )
```

#### Guardrails

- Stability is evaluated only on nodes present in both variants.
- Cases that change true structure do not enter this dimension. "Changes true
  structure" means: the perturbation adds or removes inter-module edges, or
  changes the degree distribution by more than 10%.

### 4. Seeded Anomaly Detection

Question: when the analyzer flags a known structural antipattern, does it find
it?

This dimension is explicitly limited to seeded anomalies — structural defects
injected by the benchmark authors. It measures whether the analyzer detects
known antipatterns (reverse dependencies, cycles, bridge overload), not whether
it discovers unknown problems in real codebases. General anomaly discovery
quality requires evaluation on documented real-world anomalies, which is
deferred to V2.

#### Dataset

Each item contains:

- a graph with one or more seeded anomalous regions
- gold anomalous regions as node sets and/or edge sets
- anomaly kind labels
- optional non-anomalous distractor regions

#### Matching Rule

A predicted anomaly matches a gold anomaly if:

- its `kind` matches the gold kind (when specified), and
- its predicted region overlaps the gold region with IoU >= `0.3` on node sets

#### Primary Metrics

- `Average Precision`: ranking quality over anomaly predictions.
- `Precision@3`: precision at cutoff 3.

#### Dimension Score

```text
seeded_anomaly_detection =
  geometric_mean(
    AveragePrecision,
    PrecisionAt3
  )
```

#### Guardrails

- The analyzer must not improve by flooding the anomaly list. A candidate that
  produces more than `3x` the anomaly count of the reference on the same graph
  fails this guardrail, even if precision improves.
- False positive rate on clean graphs: the runner includes at least one
  well-structured graph with no seeded anomalies. The analyzer must produce
  zero high-confidence (severity > `0.7`) findings on this graph.

## Dataset Format

### Split Structure

V1 uses a single `public` split. All cases are visible and safe for iteration.

A `hidden` split for promotion gating enters when total case count per dimension
exceeds 30. A `smoke` split (3-5 fast cases for pre-commit) can be tagged within
the public split.

### Layout

```text
benchmark/
  datasets/
    architecture/
      repo_id/
        graph.json
        labels.json
        metadata.json
    mutations/
      case_id/
        base_graph.json
        variants/
          clean.json
          mutated.json
          repaired.json        # when applicable
        expectations.json
        metadata.json
    stability/
      case_id/
        base_graph.json
        perturbations/
          reorder.json
          split_file.json
          add_dead_leaf.json
        metadata.json
    anomalies/
      case_id/
        graph.json
        gold.json
        metadata.json
    anomalies/
      _clean/
        graph.json             # no gold.json — false positive test
        metadata.json
```

### `graph.json`

Frozen serialization of `CodeGraph`. Must use the same schema as the analyzer's
actual input format — not a parallel spec.

```json
{
  "nodes": [
    {
      "id": "pkg.mod.fn",
      "kind": "function",
      "file": "pkg/mod.py",
      "line": 12,
      "name": "fn"
    }
  ],
  "edges": [
    {
      "source": "pkg.mod.fn",
      "target": "pkg.other.helper",
      "kind": "calls"
    }
  ]
}
```

### `labels.json`

```json
{
  "analysis_level": "module",
  "included_nodes": {
    "pkg.auth.login": "auth",
    "pkg.auth.token": "auth",
    "pkg.billing.invoice": "billing"
  },
  "excluded_nodes": [
    "pkg.bootstrap.main",
    "pkg.generated.schema"
  ]
}
```

### `expectations.json`

```json
{
  "ordering": [
    ["clean", "mutated"],
    ["repaired", "mutated"]
  ],
  "required_expectations": [
    {
      "variants": ["clean", "mutated"],
      "signal": "max_cross_module_severity",
      "direction": "higher_in_second",
      "margin": 0.05
    },
    {
      "variants": ["clean", "mutated"],
      "signal": "attribution_at_3",
      "expect": true
    }
  ],
  "mutated_region": {
    "nodes": ["pkg.a.core", "pkg.b.util"]
  }
}
```

### `gold.json`

```json
{
  "anomalies": [
    {
      "kind": "cross_module",
      "region": {
        "nodes": ["pkg.a.core", "pkg.b.util"]
      }
    }
  ]
}
```

## Baselines

Every benchmark run must include simple baselines.

V1 baselines:

- **Directory grouping**: assign each node to its top-level package. Trivial,
  free, often good. This is the floor.
- **Louvain**: community detection on the same projected graph. The
  sophisticated cheap baseline.
- **Degree heuristics**: degree and betweenness centrality for role
  classification. SCC membership for cycle detection.

"Beating a baseline" means: candidate score >= baseline score on that
dimension's primary metric. Exact ties are acceptable. This is checked per
dimension, not per case. If the candidate loses to directory grouping on
architecture recovery NMI, it fails the guardrail regardless of mutation ranking
performance.

## Aggregation

### Dimension Score

Each dimension produces a primary score in `[0, 1]`.

### Overall Score

```text
overall_primary =
  geometric_mean(
    architecture_recovery,
    mutation_ranking,
    stability,
    seeded_anomaly_detection
  )
```

Geometric mean prevents any single strong dimension from hiding a weak one.
Known tradeoff: it also suppresses excellence in a single dimension. Per-
dimension scores are the primary development signal. `overall_primary` is the
promotion gate.

### Within-Dimension Aggregation

- Each repository contributes equal weight within architecture recovery.
- Each mutation case contributes equal weight within mutation ranking.
- Each perturbation case contributes equal weight within stability.
- Each anomaly case contributes equal weight within anomaly detection.

## Promotion Rule

A candidate passes if:

1. All previously-passing cases still pass.
2. `overall_primary(candidate) >= overall_primary(reference)`.
3. No dimension regresses by more than `0.03` (absolute).
4. No guardrail fails (coverage floor, baseline wins, false positive check,
   anomaly flood check).

When case count exceeds 30 per dimension, add:

5. 95% bootstrap CI for the `overall_primary` delta excludes zero (10,000
   samples, fixed seed per benchmark version).

## CLI

Target V1 interface:

```bash
uv run topo-benchmark run --dimension mutations --split public
uv run topo-benchmark run --dimension all --split public
uv run topo-benchmark compare HEAD main
uv run topo-benchmark report .benchmark/runs/latest
```

Build incrementally. Start with `--dimension mutations`, add others as
dimensions are implemented.

### Run Artifacts

```text
.benchmark/runs/<timestamp>/
  scorecard.json
  per_case.jsonl
  baselines/
    directory.json
    louvain.json
    heuristics.json
  summary.md
```

### `scorecard.json`

```json
{
  "overall_primary": 0.78,
  "dimensions": {
    "architecture_recovery": 0.81,
    "mutation_ranking": 0.84,
    "stability": 0.77,
    "seeded_anomaly_detection": 0.70
  },
  "guardrails": {
    "coverage_ok": true,
    "baseline_ok": true,
    "no_regressions": true,
    "false_positive_ok": true,
    "no_anomaly_flood": true
  },
  "cases_passed": 11,
  "cases_total": 12,
  "failing_cases": ["bridge_hub_escalation"],
  "promotion_decision": "fail"
}
```

## Versioning

Benchmark data is checked into the repository. Changes to gold labels require a
rationale in the commit message. Splits are not reshuffled without a new
directory.

Runner version is tracked as a constant in the runner code. When metric
computation changes, the runner version increments and old runs are not directly
comparable.

## Rollout

### Phase 1 (V1 — build this)

1. Mutation ranking on 12 graph-level cases (public split).
2. Architecture recovery on topo (1 repository, module level).
3. Stability on 3-4 non-renaming perturbations.
4. Seeded anomaly detection on injected defects + 1 clean graph.
5. Baselines: directory grouping, Louvain, degree heuristics.
6. CLI: `run`, `compare`, `report` commands.

### Phase 2

- End-to-end tier (source code input, parser + analyzer).
- Hidden split when case count > 30.
- Rename-based stability perturbations with node mapping.
- Architecture recovery on 2-3 additional repositories, including at least one
  with documented directory-architecture mismatch.
- Real-world anomaly ground truth from documented refactoring decisions.
- Calibration metrics (ECE, Brier) on real anomalies only.

### Phase 3

- Cross-layer disagreement quality measurement.
- Temporal drift analysis (spectral fingerprints across git snapshots).
- Multilayer-specific benchmark cases.
- Statistical promotion rule (bootstrap CIs) with sufficient case count.

## What This Benchmark Does Not Claim

This benchmark validates that the analyzer behaves sensibly on controlled
structural inputs. It does not validate that the analyzer produces useful
insights on arbitrary real codebases. That validation requires developer
feedback on real analysis output — which is a different process (empirical
iteration, not automated benchmarking).

The mutation ranking and seeded anomaly dimensions test internal consistency and
pattern detection. The architecture recovery dimension tests alignment with
human-labeled structure. None of these prove the tool is useful. They prove it
is not broken.

Usefulness is validated by developers who know their codebases saying "this is
right" or "this helped." That is the ultimate metric, and no benchmark replaces
it.
