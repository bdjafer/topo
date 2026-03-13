# Benchmark

This document defines how `topo-analyzer` should be evaluated.

The goal is strict: if the benchmark score goes up, the analyzer should be
meaningfully better at recovering real structure, ranking worse structure below
better structure, staying stable under irrelevant changes, and surfacing true
anomalies with honest confidence.

This benchmark is for the analyzer first. The parser matters, but analyzer
quality and parser quality must not be conflated.

## Desired Workflow

Target CLI for local development:

```bash
uv run topo-benchmark run --tier analyzer --split public
uv run topo-benchmark compare --candidate HEAD --reference main
uv run topo-benchmark report --input .benchmark/runs/latest
```

Target CI gate:

```bash
uv run topo-benchmark compare \
  --candidate HEAD \
  --reference origin/main \
  --split hidden \
  --fail-on-regression
```

Expected run artifacts:

```text
.benchmark/runs/2026-03-12T18-30-00Z/
  summary.md
  scorecard.json
  dimensions.json
  per_case.jsonl
  failures.json
  baselines/
    directory.json
    louvain.json
    heuristics.json
```

Minimum `scorecard.json` shape:

```json
{
  "tier": "analyzer",
  "split": "hidden",
  "overall_primary": 0.78,
  "dimensions": {
    "architecture_recovery": 0.81,
    "mutation_ranking": 0.84,
    "stability": 0.77,
    "anomaly_precision_calibration": 0.70
  },
  "guardrails": {
    "coverage_ok": true,
    "calibration_ok": true,
    "baseline_ok": true,
    "no_material_regressions": true
  },
  "promotion_decision": "pass"
}
```

## What "Better" Means

A candidate analyzer is better than a reference only if all of the following are
true on the benchmark split used for promotion:

1. Its `overall_primary` score is higher.
2. The 95% bootstrap confidence interval for the delta excludes zero on the
   positive side.
3. No dimension primary score regresses beyond tolerance.
4. No guardrail metric fails.
5. It still beats or matches the simple baselines on the primary tasks.

This is intentionally strict. A candidate does not count as better if it gets
slightly smarter on one dimension while becoming unstable, overconfident, or
easier to game.

## Principles

1. Measure claims, not vibes. Every score must correspond to a concrete claim
   the analyzer makes about structure.
2. Prefer objective truth over subjective judgment. Controlled mutations and
   pairwise ordering are preferred to free-form ratings.
3. Exclude ambiguity instead of averaging it in. If a label is uncertain, it
   does not enter the gold set.
4. Separate analyzer-only evaluation from end-to-end evaluation.
5. A result that does not beat simple baselines is not a result.
6. Use fixed dataset versions, fixed seeds, and immutable splits.
7. Never let one strong metric hide one weak metric.

## Benchmark Tiers

### Analyzer Tier

Input is a frozen serialized `CodeGraph`.

Use this tier to evaluate:

- structural module recovery
- structural mutation ranking
- stability under graph-preserving or architecture-preserving perturbations
- anomaly ranking and confidence calibration

This tier isolates analyzer quality from parser noise.

### End-to-End Tier

Input is source code.

Use this tier to evaluate:

- parser + analyzer integration
- projection and scope handling
- multilayer analysis on real repositories
- regressions in the CLI-facing pipeline

The analyzer tier is the primary research benchmark. The end-to-end tier is a
release guardrail.

## Scope

### In Scope for V1

- `module`-level architecture recovery
- analyzer-only benchmark on frozen graphs
- end-to-end smoke coverage on real repositories
- graph mutations with known ordering
- stability under structure-preserving transformations
- anomaly ranking and confidence calibration for seeded defects

### Explicitly Out of Scope for V1

- domain-model extraction
- LLM usefulness of summaries
- prose quality of findings
- symbol-level full-repository gold labeling
- free-form "does this feel insightful?" human ratings

## Four Benchmark Dimensions

## 1. Architecture Recovery

Question: does the analyzer recover real architectural boundaries?

### Dataset

Use real repositories with coarse, high-confidence gold partitions at
`module` level.

Each dataset item contains:

- a frozen `CodeGraph`
- a gold module label for each included node
- an explicit `excluded_nodes` list
- metadata describing label provenance

Only nodes with high-confidence labels are included. Ambiguous nodes are
excluded, not weakly labeled.

### Labeling Rules

1. Label only coarse architectural ownership, not fine-grained intent.
2. Generated code is excluded by default.
3. Glue code, adapters, and mixed-responsibility nodes are excluded unless
   ownership is obvious.
4. A node enters the gold set only if:
   - documentation strongly implies its module, or
   - maintainers agree on its module, or
   - a benchmark curator can justify the label from architecture docs and code
     structure without guesswork
5. Any disagreement or uncertainty moves the node to `excluded_nodes`.

### Primary Metrics

- `ARI`: Adjusted Rand Index between predicted modules and gold modules on the
  included node set. Negative values are clamped to `0`.
- `Boundary F1`: for every observed edge between included nodes, predict whether
  it is intra-module or cross-module. Compute F1 on the cross-module class.
- `Coverage`: fraction of included gold nodes that receive a non-unassigned
  predicted module.

### Dimension Score

```text
architecture_recovery =
  geometric_mean(
    max(0, ARI),
    BoundaryF1
  ) * Coverage
```

### Guardrails

- `Coverage` must remain above a floor. The analyzer does not get to win by
  abstaining on hard nodes.
- The candidate must beat:
  - directory/package grouping
  - Louvain or Leiden on the same projected graph

## 2. Mutation Ranking

Question: does the analyzer rank structurally worse variants below better ones?

This is the most pragmatic part of the benchmark.

### Dataset

Each item is an ordered pair or ordered chain of graphs with known structural
ordering.

Examples:

- `clean > reverse_dependency`
- `clean > cross_module_cycle`
- `clean > god_bridge`
- `merged > separated` is false, so use `separated > merged`
- `broken < repaired`

Mutations should be produced in two ways:

- graph-level rewrites on frozen `CodeGraph` fixtures
- source-level rewrites for end-to-end cases

### Required Mutation Families

- reverse dependency across two otherwise separate modules
- strongly connected cycle across module boundaries
- injected bridge node connecting previously separate regions
- oversized merged module created from two natural clusters
- misplaced utility moved into a high-betweenness position
- repaired versions of the same defects

### Benchmark-Owned Signals

The runner should not depend on free-form prose. It should derive a stable set
of signals from `StructuralAnalysis`.

Required signals:

- module partition
- `largest_module_ratio`
- cross-module anomaly severity
- cycle anomaly severity
- role assignments
- anomaly anchors and node ids

### Case Expectations

Each mutation case declares which signals must move and in which direction.

Examples:

- reverse dependency:
  - cross-module anomaly severity must increase
  - boundary integrity must decrease
- cross-module cycle:
  - cycle anomaly severity must increase
  - top anomaly must overlap the mutated region
- god bridge:
  - injected node must become a `bridge` or `hub`
  - module separation must worsen

### Primary Metric

`Pairwise Accuracy`: fraction of ordered cases whose declared expectations are
satisfied.

A case passes only if all of its required inequalities pass by a minimum margin
`epsilon`. Exact ties count as failure.

### Secondary Metrics

- `Attribution@k`: whether the mutated region appears in the top `k` anomalies
  or affected nodes.
- `Repair Accuracy`: fraction of repaired variants ranked above their broken
  counterparts.

### Dimension Score

```text
mutation_ranking =
  geometric_mean(
    PairwiseAccuracy,
    RepairAccuracy,
    AttributionAtK
  )
```

### Initial V1 Mutation Suite

Start with a small public suite of 12 analyzer-only cases.

Rules for this suite:

- keep graphs small and hand-auditable
- prefer graph-level rewrites over source-level rewrites
- use `module` projection by default
- allow `symbol` projection only when the case is explicitly about a single
  utility, bridge, or spectral outlier
- include a `repaired` variant whenever the defect has a clean local fix

### Input Design Rules

Mutation inputs must be balanced. They should be realistic enough to contain
some background structural noise, but not so noisy that the intended signal is
ambiguous.

Required rules:

- each case has one dominant structural change
- the `clean` variant must already be slightly imperfect; do not use toy graphs
  with perfectly separated cliques unless the case is explicitly synthetic
- the `mutated` variant must not collapse into global structural chaos
- the main signal should be recoverable without requiring the analyzer to solve
  several unrelated problems at once
- if a case produces many unrelated findings, redesign the input rather than
  loosening the assertions
- prefer small parsed source fixtures over raw synthetic graphs when the
  synthetic graph becomes either too perfect or too ambiguous

Practical target:

- a mutation case may produce one primary signal and at most one plausible
  secondary side effect
- examples:
  - reverse dependency case: one `reverse_dependency` finding, maybe one related
    boundary anomaly
  - cycle case: one `cycle_member` signal, maybe one reverse-flow side effect
  - boundary erosion case: weaker package alignment and weaker separation,
    without turning into a reverse-dependency case

This balance matters. The benchmark should reward analyzers that detect the
intended structural change, not analyzers that happen to produce the longest
list of warnings.

### Derived Signals for V1 Cases

The runner should derive these signals from structured analyzer output rather
than from free-form summaries:

- `partition_similarity_to_clean`: ARI between predicted module partitions for
  the current variant and the `clean` variant on their shared nodes
- `largest_module_ratio`: `health.largest_module_ratio`
- `module_count`: `clustering.module_count`
- `max_cross_module_severity`: maximum anomaly severity where
  `kind == "cross_module"`, default `0`
- `max_cycle_severity`: maximum anomaly severity where
  `kind == "cycle_member"`, default `0`
- `target_role(node_id)`: predicted role for the declared target node
- `target_has_spectral_outlier(node_id)`: whether any
  `kind == "spectral_outlier"` anomaly contains the declared target node
- `attribution_at_3(mutated_region)`: whether one of the top 3 anomalies
  overlaps the declared mutated region

At least half of the public V1 cases should include a `repaired` variant. For
any case with `repaired`, the runner should assert that `repaired` is closer to
`clean` than `mutated` on that case's primary signals.

| ID | Level | Variants | Mutation | Required expectations |
| --- | --- | --- | --- | --- |
| `revdep_light` | `module` | `clean`, `mutated`, `repaired` | Add one reverse `calls` edge across an otherwise directional module boundary. | `max_cross_module_severity(mutated) > clean`; `attribution_at_3(mutated_region) = 1`; `partition_similarity_to_clean(mutated)` stays high; `repaired` is closer to `clean` than `mutated`. |
| `revdep_balanced` | `module` | `clean`, `mutated` | Add enough reverse edges to make the boundary nearly bidirectional. | `max_cross_module_severity(mutated)` is higher than in `revdep_light`; `attribution_at_3(mutated_region) = 1`; `partition_similarity_to_clean(mutated)` stays high. |
| `revdep_multilayer` | `module` | `clean`, `mutated`, `repaired` | Add reverse `calls` and `imports` edges across the same boundary. | Combined-mode `max_cross_module_severity(mutated) > clean`; `attribution_at_3(mutated_region) = 1`; `repaired` moves back toward `clean`. |
| `cycle_two_module` | `module` | `clean`, `mutated`, `repaired` | Add one back edge that closes a 2-module strongly connected component. | `max_cycle_severity(mutated) > clean`; `attribution_at_3(mutated_region) = 1`; `partition_similarity_to_clean(mutated)` stays high; `repaired` reduces cycle severity. |
| `cycle_three_module_ring` | `module` | `clean`, `mutated` | Add `C -> A` to an `A -> B -> C` dependency chain. | `max_cycle_severity(mutated) > clean`; `attribution_at_3(mutated_region) = 1`; `module_count(mutated)` stays stable. |
| `cycle_multilayer` | `module` | `clean`, `mutated`, `repaired` | Create a cycle that only appears when `calls` and `imports` are considered together. | Combined-mode `max_cycle_severity(mutated) > clean`; `attribution_at_3(mutated_region) = 1`; `repaired` removes the cycle signal. |
| `bridge_connector` | `module` | `clean`, `mutated`, `repaired` | Insert one connector node between two otherwise separate regions. | `target_role(connector)` becomes `bridge` or `hub`; `largest_module_ratio(mutated) > clean`; `attribution_at_3(mutated_region) = 1`; `repaired` weakens the bridge signal. |
| `bridge_hub_escalation` | `module` | `clean`, `mutated` | Expand the connector so it touches three regions and becomes a local bottleneck. | `target_role(connector)` becomes `hub` or `bridge`; `largest_module_ratio(mutated)` is higher than in `bridge_connector`; `partition_similarity_to_clean(mutated)` decreases. |
| `bridge_multilayer` | `module` | `clean`, `mutated`, `repaired` | Connect regions through mixed `calls` and `imports` edges that only become obvious in combined mode. | Combined-mode `target_role(connector)` becomes `bridge` or `hub`; `attribution_at_3(mutated_region) = 1`; `repaired` moves back toward `clean`. |
| `boundary_erosion_light` | `module` | `clean`, `mutated`, `repaired` | Add a sparse band of cross edges between two natural clusters. | `partition_similarity_to_clean(mutated)` decreases slightly; `largest_module_ratio(mutated) > clean`; `module_count(mutated) <= clean`; `repaired` restores separation. |
| `boundary_merge_full` | `module` | `clean`, `mutated` | Add enough cross edges to collapse two natural clusters into one weakly separated region. | `partition_similarity_to_clean(mutated)` decreases strongly; `largest_module_ratio(mutated)` increases strongly; `module_count(mutated) < clean`. |
| `utility_misplaced` | `symbol` | `clean`, `mutated`, `repaired` | Move a leaf-like utility into a cross-module flow path so it stops behaving like a utility. | `target_role(clean) = utility`; `target_role(mutated) != utility`; `target_has_spectral_outlier(target)` or `attribution_at_3(mutated_region) = 1`; `repaired` restores the original role profile. |

These 12 cases are enough to validate the first benchmark runner. They cover the
core mutation families, they are small enough to debug by hand, and they can be
mirrored later in the hidden split with different topology and node names.

## 3. Stability

Question: does the analyzer stay stable when the architecture did not change?

### Dataset

Each item is a base graph or repository plus one or more perturbations that
should preserve architectural structure.

Required perturbation families:

- symbol renaming
- definition reordering
- file splitting or merging without changing inter-module dependencies
- extraction or inlining of trivial wrappers
- formatting, comments, and docstring-only edits
- addition of dead leaf helpers

### Matching Rules

When perturbations rename nodes, stability is evaluated on a provided
`node_mapping` from original ids to transformed ids. Nodes with no stable
identity mapping are excluded.

### Primary Metrics

- `Partition Stability`: ARI between predicted module partitions on matched
  nodes before and after perturbation.
- `Role Stability`: macro F1 of structural roles on matched nodes.
- `Top-K Stability`: overlap of top `k` anomalies after excluding perturbations
  that intentionally add or remove true anomalies. This is only computed for
  perturbation families that should preserve anomaly sets.

### Dimension Score

```text
stability =
  geometric_mean(
    PartitionStability,
    RoleStability,
    TopKStability
  )
```

### Guardrails

- Stability is evaluated only on unchanged or identity-mapped nodes.
- Cases that change true structure do not enter this dimension.
- Confidence should not fluctuate wildly under benign perturbations.

## 4. Anomaly Precision and Calibration

Question: when the analyzer flags a structural problem, is it right, and is its
confidence honest?

### Dataset

Use seeded anomaly cases and a smaller set of real, documented anomalies.

Each item contains:

- a graph or repository
- one or more gold anomalous regions
- anomaly kind labels when known
- optional non-anomalous distractor regions

Gold anomalous regions should be explicit:

- node sets
- edge sets
- or both

### Evaluation Unit

The runner evaluates anomalies as ranked predictions. A predicted anomaly is a
match if:

- its `kind` matches the gold kind, when a kind is specified, and
- its predicted region overlaps the gold region above an IoU threshold

If the dataset item does not specify a kind, only region overlap is required.

### Primary Metrics

- `Average Precision`: ranking quality over anomaly predictions
- `Precision@k`: for small `k` such as `1`, `3`, and `5`
- `Recall@k`: same cutoffs, mainly as diagnostic output

### Calibration Metrics

- `ECE`: Expected Calibration Error over anomaly confidence bins
- `Brier`: Brier score on matched vs unmatched anomaly predictions

Convert calibration into a score where higher is better:

```text
calibration_score = 1 - min(1, normalized_calibration_error)
```

### Dimension Score

```text
anomaly_precision_calibration =
  geometric_mean(
    AveragePrecision,
    PrecisionAtK,
    calibration_score
  )
```

### Guardrails

- Confidence inflation is a hard failure even if ranking improves.
- Precision matters more than raw anomaly volume. The analyzer must not improve
  by flooding the list.

## Dataset Splits

Use three immutable splits:

- `public`: visible to contributors and safe for day-to-day iteration
- `hidden`: used for promotion and release decisions
- `smoke`: tiny, fast subset used in local pre-commit or rapid CI

Rules:

1. Public and hidden splits must both contain real and synthetic cases.
2. Hidden cases should differ in repository, mutation placement, and anomaly
   shape from public ones.
3. Split membership is fixed per dataset version.

## Dataset Format

Proposed layout:

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
          repaired.json
        expectations.json
        metadata.json
    stability/
      case_id/
        base_graph.json
        perturbations/
          rename.json
          reorder.json
          split_file.json
        node_mapping.json
        metadata.json
    anomalies/
      case_id/
        graph.json
        gold.json
        metadata.json
```

### `graph.json`

Frozen serialization of `CodeGraph`.

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

Example for architecture recovery:

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

Example for mutation ranking:

```json
{
  "ordering": [
    ["clean", "reverse_dependency"],
    ["repaired", "reverse_dependency"]
  ],
  "required_expectations": [
    {
      "case": ["clean", "reverse_dependency"],
      "signal": "cross_module_anomaly_severity",
      "direction": "higher_in_second",
      "margin": 0.05
    },
    {
      "case": ["clean", "reverse_dependency"],
      "signal": "boundary_integrity",
      "direction": "lower_in_second",
      "margin": 0.05
    }
  ],
  "mutated_region": {
    "nodes": ["pkg.a.core", "pkg.b.util"]
  }
}
```

## Baselines

Every benchmark run must include simple baselines.

Required baselines:

- directory or package grouping
- Louvain or Leiden community detection on the same projected graph
- degree and betweenness heuristics for role classification
- anomaly heuristics based on:
  - SCC size
  - cross-module edge count
  - bridge centrality

The benchmark should report both absolute score and win rate over baselines.

## Final Scoring

Each dimension produces a primary score in `[0, 1]`.

The final primary score is:

```text
overall_primary =
  geometric_mean(
    architecture_recovery,
    mutation_ranking,
    stability,
    anomaly_precision_calibration
  )
```

Do not use a plain weighted sum. A weighted sum hides failures too easily.

### Promotion Rule

A candidate passes only if:

1. `overall_primary(candidate) > overall_primary(reference)`
2. The 95% bootstrap confidence interval of the delta is strictly positive
3. No dimension regresses by more than `0.02`
4. No guardrail fails
5. Baseline win rate does not regress
6. End-to-end tier does not materially regress, even if analyzer tier improves

This gives a usable definition of "better" with minimal ambiguity.

## Aggregation Rules

To avoid large repositories dominating the benchmark:

- each repository contributes equal weight within its dataset family
- each mutation case contributes equal weight within its family
- each benchmark dimension contributes equal weight to `overall_primary`

Within one repository, metrics may be averaged across cases, but repository
weights stay uniform.

## Statistics

Use bootstrap resampling over top-level benchmark items:

- repositories for architecture recovery
- mutation cases for mutation ranking
- perturbation cases for stability
- anomaly cases for anomaly ranking and calibration

Recommended defaults:

- 10,000 bootstrap samples
- 95% confidence interval
- fixed random seed per benchmark version

The benchmark must publish:

- point estimate
- confidence interval
- per-dimension deltas
- failing cases

## Anti-Gaming Rules

1. Unassigned nodes count against coverage.
2. Confidence inflation is penalized through calibration.
3. Free-form summaries are ignored by the runner.
4. Hidden split results decide promotion.
5. A new metric cannot be added to the promotion rule without versioning the
   benchmark.

## Benchmark Governance

Benchmark data is versioned.

Rules:

- No silent dataset edits
- No split reshuffling without a new dataset version
- No relabeling without a short rationale in the dataset metadata
- Baselines must be rerun when the dataset version changes

Suggested versioning:

```text
benchmark version = <dataset version> + <metric version> + <runner version>
```

## Rollout Plan

### Phase 1

Build the analyzer-only benchmark first:

- frozen `CodeGraph` serialization
- architecture recovery on a small labeled set
- mutation ranking on graph rewrites
- stability on graph-preserving perturbations
- anomaly ranking on seeded cases

### Phase 2

Add end-to-end source-level cases:

- parser + analyzer smoke coverage
- source rewrites for mutations and stability
- hidden split for release gating

### Phase 3

Expand breadth:

- more repositories
- more mutation families
- multilayer-specific benchmark cases
- documented real-world anomaly sets

## Recommended V1 Constraints

To keep the benchmark honest and shippable:

- start at `module` level, not `symbol` level
- use a small number of well-labeled repositories
- rely heavily on mutation pairs
- prefer seeded anomalies over subjective anomaly reviews
- treat calibration as a first-class requirement

## Summary

This benchmark treats analyzer quality as four independent claims:

1. It recovers real architecture.
2. It ranks worse structure below better structure.
3. It stays stable when the structure did not change.
4. It surfaces true anomalies with honest confidence.

The score only matters if all four claims improve together.

That is the core benchmark contract for `topo-analyzer`.
