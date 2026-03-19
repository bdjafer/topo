# Self-Analysis Issue Classification — `uv run topo .`

**Date:** 2026-03-19
**Version:** Post-Phase 8 enrichment (dual-level analysis)
**Total findings:** 48

## Classification Taxonomy

Four irreducible categories. Each finding goes into exactly one.

| Category | Definition | Action |
|---|---|---|
| **TRUE-ISSUE** | Real structural concern worth surfacing to a developer. | Keep as-is. |
| **TRUE-EXPECTED** | Real structural property, but intentional or inherent to the design. Not a problem. | Lower severity, add context, or suppress conditionally. |
| **ARTIFACT-FIX** | False/misleading finding due to a fixable flaw in detector logic. | Fix the detector. |
| **ARTIFACT-INHERENT** | False/misleading finding due to a fundamental limitation at this scale/resolution. | Add guard clause, suppress, or document limitation. |

---

## Classifications by Finding Type

### 1. `fragile_hub` (1 finding)

**Finding:** `topo_analyzer.analysis is both a structural hub (degree 34) and a high-betweenness bottleneck — single point of failure.`

**Classification: TRUE-EXPECTED**

**Justification:** `analysis.py` is the central orchestrator of the entire analyzer pipeline. Degree 34 in a 32-node graph means it touches nearly every other module. This is structurally accurate — `analysis.py` imports from `spectral`, `modules`, `anomalies`, `projection`, `roles`, and `parser`, and is imported by `topo_cli` and `topo_benchmark`. For a ~900-line orchestrator in a 274-symbol codebase, this is a deliberate design choice, not an accident. The finding is correct but not actionable at this project scale.

**Improvement opportunity:** Severity 1.0 is too high for a deliberate hub. Consider scaling severity by project size or adding a "central orchestrator" acknowledgment heuristic.

---

### 2. `spectral_outlier` (12 findings)

**Findings:** Various symbol-level nodes flagged as N-sigma from module 0 centroid, with "nearest alternative: topo_benchmark" for ALL 12.

| Node | sigma | Classification |
|---|---|---|
| `topo_benchmark.dimensions.mutations._eval_signal` | 5.5 | ARTIFACT-INHERENT |
| `topo_analyzer.analysis._compute_coverage` | 5.1 | ARTIFACT-INHERENT |
| `topo_analyzer.modules._cluster_component` | 3.9 | ARTIFACT-INHERENT |
| `topo_analyzer.analysis.StructuralAnalysis.summary` | 3.6 | ARTIFACT-INHERENT |
| `topo_analyzer.modules._estimate_k` | 3.3 | ARTIFACT-INHERENT |
| `topo_benchmark.experiments.exp1_architecture.run_experiment_1` | 2.5 | ARTIFACT-INHERENT |
| `topo_analyzer.modules._kmeans` | 2.3 | ARTIFACT-INHERENT |
| `topo_analyzer.modules._silhouette_score` | 2.3 | ARTIFACT-INHERENT |
| `topo_benchmark.experiments.run.main` | 2.2 | ARTIFACT-INHERENT |
| `topo_analyzer.analysis.Style._wrap` | 2.2 | ARTIFACT-INHERENT |
| `topo_benchmark.runner.run_benchmark` | 2.0 | ARTIFACT-INHERENT |
| `topo_benchmark.experiments.exp1_architecture._print_codebase_report` | 2.0 | ARTIFACT-INHERENT |

**Classification: ARTIFACT-INHERENT (all 12)**

**Justification:** Two compounding problems make these findings uninformative:

1. **Weak clustering precondition.** Module 0 contains 30/32 nodes (93.8%). With virtually all nodes in one module, outlier detection within that module measures deviation from a very broad centroid. The outliers are structurally peripheral nodes (internal helpers, entry points) that naturally sit far from the centroid of a catch-all module. This is a tautology: "node in huge undifferentiated module is not average" — true but uninformative.

2. **Degenerate nearest alternative.** Every single outlier reports "nearest alternative: topo_benchmark" — even nodes from `topo_analyzer`. This happens because the only non-trivial module IS the mega-module. The alternative modules (modules 1-30) are mostly singletons from symbol-level aggregation, and the label heuristic picks "topo_benchmark" because benchmark nodes outnumber others in module 0 (18/30 are topo_benchmark). The "nearest alternative" is effectively the module they're already in, just with a misleading label.

**Fix:** Suppress spectral outlier findings when `largest_module_ratio > 0.8`. The precondition for meaningful outlier detection (clear module separation) is not met. The "nearest alternative" label should also never match the node's own module label.

---

### 3. `god_module` (1 finding)

**Finding:** `Module topo_benchmark has 30 nodes (100% of edges). Consider splitting it into smaller, focused modules.`

**Classification: ARTIFACT-FIX**

**Justification:** Two distinct bugs:

1. **Wrong label.** The module is labeled "topo_benchmark" but contains nodes from ALL 4 packages: topo_benchmark (18), topo_analyzer (7), topo_parser (4), topo_cli (1). The label heuristic (`max(set(prefixes), key=prefixes.count)`) picks the plurality prefix, which is misleading. A developer reading "Module topo_benchmark has 100% of edges" would think the benchmark package is the problem, when really it's that the clustering found no meaningful separation.

2. **Redundant with module-separation.** When the only non-trivial module contains everything, "god module" and "weak module separation" are saying the same thing. The god-module finding implies the *code* should be split, but the real issue is that the *analyzer* couldn't find existing boundaries. This misdirects developer attention.

**Fix:**
- Suppress god-module when it's the only non-trivial module (it adds no information beyond module-separation:weak).
- Fix the label heuristic: when a module spans multiple top-level packages, use a label like "main cluster" or list all packages.

---

### 4. `module_separation` (1 finding)

**Finding:** `The largest structural module still covers 93.8% of the analysis graph.`

**Classification: TRUE-EXPECTED**

**Justification:** This finding is structurally accurate. The spectral clustering genuinely cannot find well-separated sub-communities in this codebase at symbol level (242 nodes, 1 clusterable component). The topo monorepo is a tight pipeline: parser -> analyzer -> cli, with benchmark coupling to all three. Cross-package edges are dense relative to within-package structure.

This is a genuine property: the codebase is more of a "single cohesive system" than a "collection of independent modules." For a small monorepo with 3 dependent packages, this is expected rather than problematic.

**Note:** If spectral clustering were run on a larger codebase with similar structure, it would likely find more separation. The 32-node MODULE-level graph (or 242-node SYMBOL graph) may simply be too small for spectral methods to distinguish sub-communities that a human would recognize.

---

### 5. `layer_discrepancy` (30 findings)

**Findings:** 27 follow the pattern `X is imports-central (p9X%) but calls-peripheral (p22%)`, and 3 follow `X is imports-central (p8X-9X%) but calls-peripheral (p52%)`.

**Classification: ARTIFACT-FIX (all 30)**

**Justification:** Three compounding flaws make these findings systematically wrong:

1. **Uniform calls-peripheral percentile (p22%).** 27 of 30 findings show exactly p22% for calls. This means ~22% of nodes have calls degree <= these nodes' calls degree — i.e., they all share the same (likely zero) calls degree. When most nodes in a layer have zero degree, the percentile is dominated by ties. The detector is flagging "has zero calls" as "calls-peripheral," which is true but trivially so — it's the NORMAL state for module-level nodes in Python.

2. **Module-kind nodes shouldn't be compared across import/calls layers.** In Python's symbol graph, MODULE nodes (file-level) naturally have high import degree (they're the entities doing the importing) but zero or near-zero calls degree (FUNCTIONS do the calling, not modules). This is a language-level structural property, not an architectural anomaly. Flagging every MODULE node for having more imports than calls is equivalent to flagging "Python uses imports" — noise, not signal.

3. **Volume overwhelms signal.** 30/48 findings (63%) are the same type with the same pattern. Even if some were individually valid, the sheer count makes the detector useless. A developer seeing 30 "high severity" findings of the same kind stops reading after the third.

**Evidence of systematic artifact:** The two distinct calls percentile clusters (p22% for 27 nodes, p52% for 3 nodes) correspond to MODULE-kind vs CLASS-kind nodes. Classes have slightly higher calls degree (their methods get called), confirming the node-kind dependency.

**Fix (choose one or combine):**
- **Filter by node kind:** Only flag FUNCTION/CLASS nodes, not MODULE nodes. Layer discrepancy between import and call layers is only meaningful for entities that could plausibly participate in both layers.
- **Minimum degree threshold:** Require degree >= 2 in BOTH layers before flagging. Nodes with degree 0 in a layer are "absent," not "peripheral."
- **Cap per finding type:** Maximum 5-10 findings of the same type. After that, emit a summary: "30 nodes show the same import-vs-call discrepancy pattern — this appears to be a systemic property rather than individual anomalies."

---

### 6. `phantom_import` (1 finding)

**Finding:** `3 import(s) between topo_analyzer and topo_benchmark with no corresponding calls — possibly unused coupling or type-only imports.`

**Classification: TRUE-EXPECTED**

**Justification:** The phantom import detector operates on **spectral modules**, not Python packages. At symbol level:
- Module 0 (labeled "topo_analyzer"): 237 nodes spanning all 4 packages
- Module 1 (labeled "topo_benchmark"): 5 nodes (`bootstrap_delta_ci`, `compare`, `compare_runs`, `_load_dimension_case_scores`, `Scorecard.load`)

The "3 imports" are import edges crossing the boundary between spectral modules 0 and 1 with no corresponding call edges crossing that same boundary. The labels "topo_analyzer" and "topo_benchmark" refer to spectral module labels, not Python packages — a source of confusion.

This is expected in Python: modules import types (classes, enums, dataclasses) for type annotations, isinstance checks, and function signatures without necessarily calling functions from those modules. The finding correctly identifies "coupling via type reference without runtime coupling."

**Note:** The description "possibly unused coupling or type-only imports" is appropriately hedged. Confidence (0.5) is correctly low. However, the use of spectral module labels that coincide with package names is confusing — the finding reads as if it's about package-level coupling when it's actually about spectral module boundaries.

---

### 7. `orphan` (2 findings)

**Finding:** `topo_benchmark.bootstrap` and `topo_benchmark.main` — "No inbound or outbound edges — may be dead code"

**Classification: ARTIFACT-FIX (both) — verified by subagent**

The root cause is NOT edge-set inconsistency (as originally hypothesized). It is a **role aggregation priority bug** in `_aggregate_roles_to_report_level()`.

#### Mechanism (verified):
1. Role classification runs at SYMBOL level using ALL edge kinds (calls + imports + inherits).
2. At symbol level, individual functions inside these modules may have degree 0 (e.g., `bootstrap_ci` is never called; `main.py`'s `cmd_run`/`cmd_compare`/`cmd_report` use lazy imports the parser can't detect).
3. These zero-degree child symbols get classified as ORPHAN.
4. `_aggregate_roles_to_report_level()` uses `_ROLE_PRIORITY` where ORPHAN (priority 4) beats REGULAR (priority 5). So if ANY child symbol is ORPHAN, the entire report-level module inherits the ORPHAN label — even when OTHER children are well-connected.
5. Result: `topo_benchmark.bootstrap` (which contains both `bootstrap_ci` [degree 0] and `bootstrap_delta_ci` [degree 2]) gets labeled ORPHAN because of ONE zero-degree child.

#### `topo_benchmark.main` — ARTIFACT-FIX
`main.py` is the CLI entry point with lazy imports (`from topo_benchmark.runner import run_benchmark` inside function bodies). The parser can't detect these as edges, so the child functions get degree 0 and ORPHAN classification. At MODULE level, the module itself has degree 1 (one outbound edge). The "no edges" description is wrong.

#### `topo_benchmark.bootstrap` — ARTIFACT-FIX
`bootstrap.py` is NOT dead code. `bootstrap_delta_ci` has in_degree=2 at symbol level. The ORPHAN label propagates from `bootstrap_ci` (degree 0) through the aggregation priority bug.

**Fix:** The `_aggregate_roles_to_report_level()` function should not let a single ORPHAN child override well-connected siblings. Options:
- Only propagate ORPHAN to parent if ALL children are ORPHAN.
- Use a weighted aggregation: if the majority of a module's symbols are non-ORPHAN, the module is not ORPHAN.
- Separate "some dead symbols inside this module" from "this entire module is dead code" as distinct finding types.

---

## Summary

### Distribution

| Category | Count | % |
|---|---|---|
| TRUE-ISSUE | 0 | 0% |
| TRUE-EXPECTED | 3 | 6.3% |
| ARTIFACT-FIX | 33 | 68.7% |
| ARTIFACT-INHERENT | 12 | 25.0% |
| **Total** | **48** | **100%** |

**0 of 48 findings are genuine, actionable issues.** Every finding is either an expected structural property or an artifact of the analyzer.

### Fixes needed (ranked by impact)

| # | Fix | Findings addressed |
|---|---|---|
| 1 | Layer discrepancy: require min degree >= 2 in both layers; filter MODULE-kind nodes | 27-30 |
| 2 | Spectral outlier: suppress when `largest_module_ratio > 0.8` | 12 |
| 3 | Orphan: fix role aggregation priority (ORPHAN should not propagate from single child) | 2 |
| 4 | God-module: suppress when it's the only non-trivial module | 1 |
| 5 | God-module: fix label heuristic for multi-package modules | 1 |
| 6 | Per-type finding cap (max 5-10 per type, summary after) | reduces noise on any future flood |

### If all fixes were applied: 48 -> ~3 findings

The remaining ~3 would be:
1. fragile-hub:topo_analyzer.analysis (TRUE-EXPECTED, lower severity)
2. module-separation:weak (TRUE-EXPECTED)
3. phantom-import (TRUE-EXPECTED)

---

## Root Cause Analysis

The 48 findings collapse into 3 systemic issues:

1. **Clustering failure cascade** (affects 25 findings: 12 spectral outliers + 1 god-module + 1 module-separation + scattered severity inflation). The spectral clustering produces 1 mega-module containing 93.8% of nodes. Every detector that depends on module quality (spectral outliers, god-module) inherits this failure and generates noise. Root cause: the codebase is too small/dense for spectral methods to find structure.

2. **Node-kind blindness in layer discrepancy** (affects 30 findings). The detector treats all nodes equally, but MODULE-kind nodes inherently have different import-vs-calls profiles than FUNCTION-kind nodes. This is a language property, not a code property.

3. **Role aggregation priority bug in orphan detection** (affects 2 findings). `_aggregate_roles_to_report_level()` gives ORPHAN higher priority than REGULAR in `_ROLE_PRIORITY`, so a single zero-degree child symbol marks an entire module as ORPHAN — even when other children are well-connected. Compounded by the parser's inability to detect lazy imports (function-body imports), which makes CLI entry-point functions appear as orphans.

---

## Verification

All classifications independently verified by subagent (agent ID: a99b352f5e0fed3df). Three corrections applied:
1. Layer discrepancy count corrected from 28 to 30.
2. Phantom import justification corrected: operates on spectral modules, not Python packages.
3. Orphan root cause corrected: role aggregation priority bug, not edge-set inconsistency.
