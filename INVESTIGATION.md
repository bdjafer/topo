# Real-Codebase Validation Investigation

**Date:** 2026-03-13
**Updated:** 2026-03-13 (post-fix re-validation)
**Codebases tested:** Flask, Requests, Click, FastAPI

## Executive Summary

~~Running topo on four real Python codebases reveals **two critical parser bugs** that make the tool produce useless or misleading results on real code.~~

**Update (post-fix):** The parser bugs identified below have been fixed in commit `271edce`. Independent re-validation on Flask, Click, and Topo-self confirms the pipeline now works end-to-end on real codebases. See [Re-Validation Results](#re-validation-results) below for the evidence.

**Current status:** The tool produces architecturally plausible clusters on real codebases. Spectral clustering outperforms Louvain community detection on Flask. The remaining question is whether the structural insight exceeds what directory grouping provides — see [Core Bet Assessment](#core-bet-assessment).

---

## Original Investigation (Pre-Fix)

---

## Bug 1: Import edges use unresolved short names (CRITICAL)

**Impact:** 100% of import edges are silently dropped during projection.

The parser stores import edge targets as raw AST names — exactly what appears in the source code. For relative or intra-package imports, these are short names like `app`, `helpers`, `config` rather than fully-qualified names like `flask.app`, `flask.helpers`, `flask.config`.

Since no node in the graph has the short name as its ID, the projection phase drops every import edge — the edge target doesn't exist in `graph.nodes`, so the edge is filtered out as external.

**Evidence:**
```
Flask:    546 import edges, 0 with targets matching any graph node (0%)
Requests: 364 import edges, 0 with targets matching any graph node (0%)
Click:    517 import edges, 0 with targets matching any graph node (0%)
FastAPI:  973 import edges, 91 targets found (20.9% — only absolute cross-package imports work)
```

**Root cause:** `_extract_import()` in `python.py:202-218` stores `node.module` and `f"{node.module}.{alias.name}"` directly from the AST without resolving relative imports to fully-qualified names. For `from .app import Flask` in flask/__init__.py, it stores `target="app.Flask"` instead of `target="flask.app.Flask"`.

## Bug 2: Import validation kills all cross-module call edges (CRITICAL)

**Impact:** 96-100% of cross-module calls are incorrectly rejected.

The `_has_import_path()` validation in `_add_pycg_calls()` checks whether the source module has an import edge targeting the target module. But since all import edge targets are unresolved short names (Bug 1), the validation always fails for cross-module calls, silently discarding them.

**Evidence (Flask):**
```
PyCG produces:      759 raw call edges
Both endpoints resolve: 219 edges
Cross-module calls:     71 edges
Pass import validation: 0 edges (0%)
Final call edges:       153 (all same-module)
```

PyCG correctly resolves 71 cross-module calls, but every single one is rejected by the import validation because the import targets are stored as short names that don't match the fully-qualified call targets.

**The import validation was designed to prevent false positives from fuzzy suffix matching, but due to Bug 1, it prevents ALL cross-module calls.**

## Consequence: Module-level analysis is completely broken

When projected to module level:
- Call edges: 96-100% are same-module, so they become self-loops and are dropped
- Import edges: 100% targets unresolved, so all are dropped
- Inherits edges: mostly same-module (class inherits from class in same file)

Result: **0 edges** at module level for Flask, Requests, and Click.

```
Flask    MODULE level: 24 nodes, 0 edges → 24 orphans, no spectral analysis possible
Requests MODULE level: 18 nodes, 0 edges → 18 orphans, no spectral analysis possible
Click    MODULE level: 17 nodes, 0 edges → 17 orphans, no spectral analysis possible
FastAPI  MODULE level: 48 nodes, 305 edges → partial results (FastAPI has sub-packages so some absolute imports work)
```

## Bug 3: Spectral outlier detection crashes on multi-component graphs (NON-CRITICAL)

Symbol-level analysis crashes on all 4 codebases with:
```
ValueError: all the input array dimensions except for the concatenation axis must match exactly
```

The `_detect_spectral_outliers()` function in `anomalies.py` tries to `np.vstack()` fingerprint vectors from different connected components, which have different dimensionalities. This is a secondary bug — the analysis would still produce results if the anomaly detection were made robust.

---

## What the analyzer DOES produce when it gets edges (FastAPI)

FastAPI partially works because it has sub-packages (`fastapi.dependencies`, `fastapi.security`, `fastapi.openapi`) whose absolute imports do resolve. At module level with combined layers:

- **48 nodes, 305 edges** (vs 0 for the other three)
- **8 spectral clusters** with silhouette 0.596
- **NMI 0.718** against file-module baseline (reasonable)
- **Meaningful role detection:** 2 bridges, 7 entry points, 9 utilities, 17 regular

The clusters show some structural insight:
- Security modules (api_key, http, oauth2, utils) cluster together
- Routing and core app logic cluster together
- OpenAPI models cluster separately

However, with only 68.8% spectral coverage (15 isolated singleton components), the results are partial and the module assignments are incomplete.

---

## Conclusion: It's the parser, not the analyzer

| Component | Status | Evidence |
|-----------|--------|----------|
| **Spectral math** | Works | Synthetic benchmarks pass, FastAPI partial results are reasonable |
| **Module detection** | Works | When given edges, silhouette-based k-estimation produces interpretable clusters |
| **Role classification** | Works | Distribution-based roles produce sensible results on FastAPI |
| **Import resolution** | Broken | 0% of intra-package import targets resolve to graph nodes |
| **Cross-module call validation** | Broken | 0% of cross-module calls pass validation (due to broken imports) |
| **Spectral outlier detection** | Crashes | Multi-component fingerprint dimensions mismatch |

## Required Fixes

### P0: Resolve import targets to fully-qualified names
In `_extract_import()`, resolve relative and intra-package imports using the module's position in the package tree. `from .app import Flask` in `flask/__init__.py` should produce target `flask.app.Flask`, not `app.Flask`.

### P0: Fix import validation to use resolved names
Once imports are resolved, `_has_import_path()` will work correctly. Until then, cross-module calls are silently dropped.

### P1: Handle multi-component fingerprint dimensions in outlier detection
`_detect_spectral_outliers()` must handle nodes from different connected components having different fingerprint sizes.

### P2: Re-validate on real codebases after fixes
Re-run this investigation after P0 fixes to determine whether spectral analysis on real Python codebases produces architecturally meaningful results.

---

## Raw Data

### Parser output per codebase

| Metric | Flask | Requests | Click | FastAPI |
|--------|-------|----------|-------|---------|
| Nodes | 402 | 292 | 543 | 403 |
| Modules | 24 | 18 | 17 | 48 |
| Classes | 46 | 44 | 71 | 99 |
| Functions | 332 | 230 | 455 | 256 |
| Call edges | 153 | 134 | 300 | 258 |
| Import edges | 546 | 364 | 517 | 973 |
| Inherits edges | 14 | 32 | 38 | 58 |
| Calls/function | 0.46 | 0.58 | 0.66 | 1.01 |
| Cross-module calls | 3.3% | 2.2% | 0.3% | 44.6% |
| Functions with 0 call edges | 50.3% | 34.3% | 36.9% | 30.5% |

### Module-level analysis results

| Metric | Flask | Requests | Click | FastAPI |
|--------|-------|----------|-------|---------|
| Analysis nodes | 24 | 18 | 17 | 48 |
| Analysis edges | 0 | 0 | 0 | 305 |
| Spectral coverage | 0% | 0% | 0% | 68.8% |
| Modules detected | 0 | 0 | 0 | 8 |
| Silhouette | N/A | N/A | N/A | 0.596 |
| Package fallback | No | No | No | No |
| All orphans | Yes | Yes | Yes | No |

---

## Re-Validation Results

**Date:** 2026-03-13 (after fix commit `271edce`)
**Methodology:** Independent diagnostic scripts testing each pipeline stage in isolation. No trusting commit messages — raw numbers from actual runs.

### Bug Status

| Bug | Pre-Fix | Post-Fix | Evidence |
|-----|---------|----------|----------|
| **Bug 1: Import resolution** | 0% internal imports resolved | 82-94% resolved | Flask: 212/258 internal imports resolve. Remaining 46 are imports of module-level *variables* (e.g., `flask.globals.current_app`), which is expected — the parser creates nodes for functions/classes, not variables. |
| **Bug 2: Cross-module calls** | 0 cross-module calls survived | 49 (Flask), 134 (Click) | Flask: 24.3% of calls are cross-module. Click: 30.1%. Previously 0% for both. |
| **Bug 3: Spectral outlier crash** | Crashes on all codebases | No crashes observed | Multi-component fingerprint padding fix works. |

### Pipeline Stage Results

| Stage | Flask (pre) | Flask (post) | Click (post) |
|-------|-------------|--------------|--------------|
| Nodes parsed | 402 | 402 | 543 |
| Edges parsed | ~713 | 1186 | 1565 |
| Module-level edges | **0** | **263** | **430** |
| Isolated modules | 24/24 | 0/24 | 1/17 |
| Spectral coverage | 0% | **100%** | **94.1%** |
| Modules detected | 0 | **8** | **7** |
| Silhouette | N/A | **0.533** | **0.505** |

### Spectral Cluster Quality (Flask)

8 clusters detected. Qualitative assessment:

| Cluster | Members | Assessment |
|---------|---------|------------|
| 0 | `flask.app`, `flask.ctx` | Correct — app and request context are tightly coupled |
| 1 | `flask.json`, `flask.json.provider`, `flask.json.tag` | Perfect — recovers json sub-package exactly |
| 2 | `flask.sessions`, `flask.testing` | Plausible — testing exercises session machinery |
| 3 | `flask.sansio.blueprints`, `flask.sansio.scaffold` | Perfect — recovers sansio sub-package |
| 4 | `flask.__main__`, `flask.cli` | Perfect — CLI entry point |
| 5 | `flask.blueprints`, `flask.debughelpers`, `flask.globals`, `flask.helpers`, `flask.wrappers` | Reasonable — utility/glue modules |
| 6 | `flask`, `flask.config`, `flask.logging`, `flask.sansio.app`, `flask.signals`, `flask.templating` | Interesting — groups sansio.app with core init/config, crossing directory boundary |
| 7 | `flask.typing`, `flask.views` | Plausible — types and view machinery |

### Spectral vs Baselines

**NMI against sub-package directory baseline:**

| Method | Flask | Click |
|--------|-------|-------|
| Spectral | **0.351** | 0.000 (degenerate — flat package, 1 baseline group) |
| Louvain | 0.217 | 0.000 |
| Random | 0.152 | 0.000 |

Click's NMI is 0 for all methods because it's a flat package (`click.*`) — the sub-package baseline has only one group. NMI is meaningless here. For flat packages, the question is whether spectral clusters are architecturally sensible (qualitatively yes — see below).

**Spectral outperforms Louvain on Flask** (NMI 0.351 vs 0.217).

**Click spectral clusters (flat package, no directory structure to guide):**

| Cluster | Members | Assessment |
|---------|---------|------------|
| 0 | `click._compat`, `click._termui_impl`, `click._winconsole` | Platform compatibility internals |
| 1 | `click.exceptions`, `click.parser` | Parser raises exceptions — correct coupling |
| 2 | `click.formatting`, `click.testing` | Both deal with output |
| 3 | `click.core`, `click.shell_completion` | Completion extends core |
| 4 | `click`, `click.decorators` | Public API surface |
| 5 | `click._utils`, `click.globals`, `click.termui`, `click.types`, `click.utils` | Utility/infrastructure |

### Topo Self-Analysis

55 modules, 488 edges at module level. 21 spectral modules detected (silhouette 0.579). Notable clusters:

- `topo_analyzer.*` (analysis, anomalies, modules, spectral) grouped correctly as one unit
- `topo_parser.ast_resolve` + `topo_parser.python` grouped correctly
- `topo_analyzer.projection` + `topo_cli.main` grouped together — **cross-package insight**: the CLI's primary job is configuring projection
- `topo_benchmark.codegraph_io` + `topo_parser` + `topo_parser.graph` — **cross-package insight**: benchmark serialization depends on parser data structures

---

## Core Bet Assessment

**The core bet:** Spectral analysis of code graphs produces architecturally meaningful clusters that reveal structure beyond what directory grouping provides.

### Evidence FOR the bet paying off

1. **The pipeline works end-to-end on real codebases.** Flask, Click, and Topo all produce non-degenerate spectral clusters with silhouette > 0.5.

2. **Clusters are architecturally plausible.** For Flask, 6/8 clusters map cleanly to known architectural groups. No cluster contains obviously unrelated modules.

3. **Spectral outperforms Louvain** on Flask (NMI 0.351 vs 0.217). Spectral produces finer-grained, more interpretable clusters.

4. **Cross-directory groupings make sense.** Flask's cluster 6 groups `flask.sansio.app` with core init/config — architecturally correct, since `sansio.app` IS the core app implementation. Topo's grouping of `topo_analyzer.projection` with `topo_cli.main` reflects real coupling.

5. **For flat packages (Click), spectral provides structure where directories provide none.** Click has no sub-packages, so directory grouping is useless. Spectral produces 6 meaningful groups.

### Evidence AGAINST / Caveats

1. **Most clusters reproduce directory structure.** For Flask, 6/8 clusters are "PURE" (single sub-package). The spectral analysis mostly confirms what `ls` already tells you.

2. **Only 1-2 cross-directory insights per codebase.** The "hidden architecture" that spectral reveals is limited to a few connections per project.

3. **NMI is moderate, not strong.** Flask NMI of 0.35 against sub-package baseline indicates correlation but not tight alignment.

4. **Silhouette scores are moderate (0.50-0.58).** Not excellent cluster quality — there's overlap and ambiguity.

5. **Import resolution is incomplete.** ~18% of internal imports target variables/constants which have no nodes. This means the graph is missing edges, which may affect cluster quality.

### Verdict

**The bet is conditionally validated.** The spectral analysis works mechanically and produces non-random, architecturally plausible results. It outperforms Louvain. It adds genuine value for flat packages where directory structure provides no signal.

However, for hierarchical packages, the "hidden architecture" signal is thin — most of what spectral finds, you can learn from `ls`. The tool's current value proposition is:

1. **Quantifying** structural coupling that developers sense intuitively
2. **Finding structure** in flat packages where no directory hierarchy exists
3. **Identifying** the 1-2 cross-directory couplings per codebase that directory grouping misses
4. **Role classification** (hubs, bridges, orphans) which is orthogonal to clustering

The bet has NOT been falsified — the tool works and produces meaningful output. Whether this level of insight justifies the complexity is a product judgment, not a technical one.
