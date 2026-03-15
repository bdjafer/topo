# Proving the Bet: Experiment Methodology

**Date:** 2026-03-14
**Status:** Draft — pre-registration before execution

## The Bet

> "Spectral analysis of code graphs produces structurally meaningful results — that the mathematical decomposition aligns with architectural reality as understood by developers who know the codebase."

This decomposes into two falsifiable claims:

1. **Accuracy**: Spectral clustering recovers real architectural modules.
2. **Marginal value**: Spectral clustering recovers architectural structure that cheaper methods (directory grouping, Louvain) miss.

Claim 1 without Claim 2 = "it works, but so does `tree`."
Claim 2 without Claim 1 = "it's different from baselines, but wrong."
Both must hold for the bet to pay off.

---

## Why Current Validation Is Insufficient

| Problem | Detail |
|---------|--------|
| **Post-hoc evaluation** | "These clusters look reasonable" is unfalsifiable — any clustering can be rationalized |
| **No pre-registered thresholds** | Without declaring what "good" looks like before running, you can't fail |
| **Circular ground truth** | Comparing spectral against directory structure, then celebrating when they agree, proves nothing — directory grouping is free |
| **Degenerate metrics** | NMI = 0 on flat packages. ARI is misleading at module level. No single metric works everywhere |
| **Tiny sample** | 3 codebases. Any method can look good on 3 cherry-picked examples |
| **No predictive validity** | Current tests show spectral finds *something*. They don't show that *something* matters |

---

## Experiment Design

### Experiment 1: Cross-Directory Architecture Recovery

**Question:** When the true architecture disagrees with directory structure, does spectral recover the true architecture?

This is the decisive test. If spectral only recovers directory structure, it adds no value. If it recovers architecture that *crosses* directory boundaries, it sees something directories don't.

**Method:**

1. Select codebases where architectural modules are **known to span multiple directories**.
   - Django: `django.db.models` + `django.db.backends` + `django.db.migrations` are separate directories but tightly coupled subsystems
   - CPython stdlib: `email`, `http`, `urllib` have documented architectural coupling across top-level packages
   - Projects post-major-refactoring where files were moved (git history provides ground truth of "what used to be together")

2. Construct **gold-standard labels** BEFORE running spectral:
   - For each codebase, create a mapping: `{node_id: architectural_module}` based on:
     - Official architecture documentation
     - README / ARCHITECTURE.md / developer guides
     - Maintainer blog posts describing the design
   - Labels must be created by reading documentation only, NOT by looking at spectral output
   - Label document is committed to the repo before any experiment runs

3. Run three methods on each codebase:
   - **Directory grouping** (top-level or second-level package)
   - **Louvain community detection**
   - **Spectral clustering** (topo's method)

4. Measure each method against gold-standard labels using:
   - **V-measure** (harmonic mean of homogeneity and completeness — works across different granularities, unlike ARI)
   - **Boundary F1** (precision/recall on whether pairs of nodes that should be in the same module are co-clustered)
   - **Cross-directory recovery rate** (of nodes that belong to the same architectural module but different directories, what fraction does each method correctly co-cluster?)

**The cross-directory recovery rate is the critical metric.** Directory grouping scores 0% on it by definition. Spectral must score significantly above 0%.

**Pre-registered success threshold:**
- Spectral cross-directory recovery rate > 30% on ≥ 3/5 codebases
- Spectral V-measure > directory grouping V-measure on ≥ 3/5 codebases
- Spectral V-measure > Louvain V-measure on ≥ 3/5 codebases

**Pre-registered failure threshold:**
- Spectral cross-directory recovery rate < 10% on ≥ 3/5 codebases → bet fails on marginal value
- Spectral V-measure < directory grouping on ≥ 3/5 codebases → bet fails on accuracy

---

### Experiment 2: Seeded Defect Detection

**Question:** Can spectral analysis detect architectural violations that directory grouping cannot?

Directory grouping is structurally blind — it doesn't look at edges. If spectral can detect injected violations that directory grouping misses, it proves marginal value.

**Method:**

1. Take 5 clean codebases (can overlap with Experiment 1).

2. For each, create controlled mutations with known architectural violations:
   - **Reverse dependency**: Add a call from a low-level module to a high-level module (violates layering)
   - **Dependency cycle**: Add mutual imports between two previously-independent modules
   - **God object**: Move 5+ functions from different modules into one module (creates artificial hub)
   - **Misplaced utility**: Move a utility function into an unrelated module (creates cross-cutting coupling)
   - **Boundary erosion**: Add sparse cross-module calls that shouldn't exist (weakens module boundaries)

3. For each clean/mutated pair, run:
   - **Directory grouping** + simple dependency count
   - **Louvain**
   - **Spectral clustering + anomaly detection**

4. Measure detection:
   - **True positive**: method flags the mutated region as anomalous (spectral outlier, role change, module reassignment, new cross-module coupling)
   - **False negative**: mutation is present but method doesn't flag it
   - **False positive**: method flags something in the clean version
   - Compute **precision**, **recall**, **F1** per method per mutation type

**Pre-registered success threshold:**
- Spectral recall > directory-based recall by ≥ 20 percentage points on ≥ 3/5 mutation types
- Spectral precision ≥ 0.5 (fewer than half of flagged items are false positives)

**Pre-registered failure threshold:**
- Spectral recall ≤ directory-based recall on ≥ 3/5 mutation types → spectral adds no detection value
- Spectral precision < 0.3 → too noisy to be useful

---

### Experiment 3: Predictive Validity via Git History

**Question:** Do spectral anomalies at time T predict code churn at time T+N?

This is the strongest possible evidence. If spectral anomalies are architecturally real, the code they flag should be more likely to be refactored, fixed, or changed in the future. If they're noise, they should have no predictive power.

**Method:**

1. Select 5 codebases with ≥ 3 years of git history and ≥ 500 commits.

2. For each, take a snapshot at the **midpoint** of the git history (time T).

3. Run spectral analysis at time T. Record:
   - Which nodes are flagged as anomalies (spectral outliers, cross-module violations, cycle members)
   - Which nodes are classified as bridges/hubs
   - Module assignments

4. Measure code churn from T to HEAD (time T+N):
   - For each node, count: commits touching that file, lines changed, whether the file was moved/renamed/deleted
   - Normalize by file size (churn per line)

5. Compare:
   - **Anomalous nodes at T** vs **churn from T to T+N**
   - **Bridge/hub nodes at T** vs **churn from T to T+N**
   - Control: **random nodes** vs **churn from T to T+N**

6. Statistical test: **Mann-Whitney U** (non-parametric, no normality assumption) comparing churn distributions of anomalous vs non-anomalous nodes.

7. Effect size: **Cohen's d** or **rank-biserial correlation**.

**Pre-registered success threshold:**
- Anomalous nodes have significantly higher churn (p < 0.05, Bonferroni-corrected) on ≥ 3/5 codebases
- Effect size (rank-biserial) ≥ 0.2 (small-to-medium) on ≥ 3/5 codebases

**Pre-registered failure threshold:**
- p > 0.05 on ≥ 4/5 codebases → anomalies don't predict anything
- Effect size < 0.1 on ≥ 4/5 codebases → even if significant, effect is negligible

---

### Experiment 4: Stability Under Architecture-Preserving Transformations

**Question:** Is the spectral signal robust, or is it noise?

If spectral output changes under trivial transformations (renaming, reformatting, reordering), the signal is noise. If it stays stable under trivial changes and changes only under architectural changes, it's real.

**Method:**

1. Take 5 codebases.

2. Apply architecture-preserving transformations:
   - Rename all functions to `f1`, `f2`, ... (preserves graph structure exactly)
   - Reorder function definitions within files (preserves everything)
   - Add docstrings to all functions (preserves graph structure)
   - Split a file into two files in the same directory (preserves module membership at package level)

3. Apply architecture-breaking transformations:
   - Move a function from module A to module B and update all callers (changes graph structure)
   - Merge two modules into one (changes module boundaries)
   - Add a dependency cycle between two previously independent modules

4. Run spectral analysis before and after each transformation.

5. Measure:
   - **Partition stability** (ARI between module assignments before/after) — should be HIGH for preserving, LOW for breaking
   - **Role stability** (macro F1 of role assignments before/after) — should be HIGH for preserving, LOW for breaking
   - **Anomaly overlap** (Jaccard of top-K anomalies before/after) — should be HIGH for preserving, LOW for breaking

**Pre-registered success threshold:**
- Mean partition ARI ≥ 0.8 for architecture-preserving transformations
- Mean partition ARI ≤ 0.5 for architecture-breaking transformations
- Separation: preserving ARI - breaking ARI ≥ 0.3

**Pre-registered failure threshold:**
- Mean preserving ARI < 0.6 → signal is unstable, can't trust it
- Preserving ARI ≈ breaking ARI (difference < 0.1) → spectral can't distinguish real changes from noise

---

## Codebase Selection

**Criteria for inclusion:**

1. Pure Python (parser limitation)
2. ≥ 200 nodes at module level (non-trivial)
3. Documented architecture or well-known structure
4. Different structural types:
   - Hierarchical multi-package (Django, Sphinx)
   - Flat single-package (Click, Rich)
   - Framework with sub-packages (Flask, FastAPI)
   - Library with deep nesting (SQLAlchemy)
   - Application with layered architecture (Sentry, Celery)

**Target: 8 codebases** (5 minimum for statistical validity, 8 for robustness).

| Codebase | Type | Why |
|----------|------|-----|
| Flask | Framework, sub-packages | Already tested, known baseline |
| Click | Flat single-package | Tests flat-package value proposition |
| Django | Large monolith, deep hierarchy | Tests scale, has documented architecture |
| FastAPI | Framework, sub-packages | Already tested, REST-focused |
| Rich | Flat-ish library | Tests visual/UI library structure |
| SQLAlchemy | Deep ORM hierarchy | Tests deep nesting, multiple subsystems |
| Celery | Distributed task system | Tests application architecture |
| Sphinx | Documentation tool | Tests tool/plugin architecture |

---

## Metrics Summary

| Metric | Used In | Why This Metric |
|--------|---------|-----------------|
| **V-measure** | Exp 1 | Harmonic mean of homogeneity + completeness; works across different granularities unlike ARI; doesn't degenerate on flat packages |
| **Boundary F1** | Exp 1 | Direct measure of "are the right pairs co-clustered?" |
| **Cross-directory recovery rate** | Exp 1 | The decisive metric — measures exactly the marginal value claim |
| **Precision / Recall / F1** | Exp 2 | Standard detection metrics for seeded defects |
| **Mann-Whitney U** | Exp 3 | Non-parametric comparison of churn distributions |
| **Rank-biserial correlation** | Exp 3 | Effect size for non-parametric comparisons |
| **Partition ARI** | Exp 4 | Measures agreement between two clusterings of the same nodes |

**Why not NMI?** NMI degenerates when one partition has very few groups (flat packages produce 1 group → NMI = 0 or undefined). V-measure handles this correctly because it's based on conditional entropy, not mutual information normalization. For Experiment 4 (comparing two runs on the same nodes), ARI is appropriate because both partitions have the same granularity.

---

## Execution Protocol

1. **Pre-register**: Commit this document and gold-standard labels BEFORE running any experiments.
2. **Automate**: All experiments run via scripts, not manual inspection. No post-hoc cluster evaluation.
3. **Record everything**: Raw numbers, not just summaries. Every metric for every codebase.
4. **No cherry-picking**: Report all 8 codebases. Cannot drop a codebase because "the parser didn't work well on it."
5. **Baselines run on identical data**: Same parsed graph, same projection, different clustering method.
6. **Blinding where possible**: Gold-standard labels created before seeing spectral output. Mutation effects measured automatically, not by human judgment.

---

## Decision Matrix

After all four experiments:

| Outcome | Interpretation |
|---------|----------------|
| All 4 experiments pass | **Bet validated.** Spectral analysis produces real, useful, robust structural intelligence that outperforms cheap alternatives |
| Exp 1+2+4 pass, Exp 3 fails | **Bet partially validated.** Spectral finds real structure and detects violations, but findings don't predict future problems. Tool is diagnostic, not prognostic |
| Exp 1+4 pass, Exp 2+3 fail | **Bet weakly validated.** Spectral recovers architecture but doesn't detect violations or predict churn better than baselines. Value limited to visualization/confirmation |
| Exp 1 fails | **Bet falsified on accuracy.** Spectral doesn't recover real architecture. Stop. |
| Exp 1 passes, Exp 4 fails | **Bet falsified on robustness.** Spectral output is noise. Stop. |
| Exp 1 passes but cross-directory rate < 10% | **Bet falsified on marginal value.** Spectral reproduces directory structure. No added value over `tree`. Stop. |

---

## Implementation Plan

### Phase 1: Ground Truth Construction (no code changes)
- Select 8 codebases, clone them
- For each: read architecture docs, create `{node_id: architectural_module}` mappings
- Commit label files to `benchmark/gold_labels/`

### Phase 2: Experiment Harness
- Build `experiments/` module with:
  - `run_experiment_1.py` — architecture recovery
  - `run_experiment_2.py` — seeded defect detection
  - `run_experiment_3.py` — predictive validity
  - `run_experiment_4.py` — stability
  - `metrics.py` — V-measure, boundary F1, cross-directory rate, Mann-Whitney U
  - `baselines.py` — directory grouping, Louvain (may already exist)

### Phase 3: Run & Record
- Execute all experiments
- Store raw results in `benchmark/results/`
- Compute aggregate statistics
- Apply decision matrix

### Phase 4: Write-Up
- Document results against pre-registered thresholds
- State conclusion: validated, partially validated, or falsified
- If falsified: document what was tried and why it failed (this is valuable)
- If validated: identify which claims are supported and which are not

---

## What "Proving the Bet" Actually Means

The bet is not "spectral analysis is the best possible method for code structure analysis." It's a more specific claim:

> Spectral decomposition of code dependency graphs recovers architectural structure that is (a) real (matches expert judgment), (b) non-trivial (goes beyond directory grouping), and (c) robust (stable under irrelevant changes).

If these three properties hold empirically across diverse codebases, the bet is proven. If any one consistently fails, the bet is disproven. There is no room for "conditionally validated" — the thresholds are pre-registered, the metrics are automatic, and the codebases are fixed before execution.
