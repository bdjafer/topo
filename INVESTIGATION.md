# Real-Codebase Validation Investigation

**Date:** 2026-03-13
**Codebases tested:** Flask, Requests, Click, FastAPI

## Executive Summary

Running topo on four real Python codebases reveals **two critical parser bugs** that make the tool produce useless or misleading results on real code. The spectral analyzer itself is not the problem — when it receives a graph with edges, it produces reasonable clusters. The issue is that the parser delivers graphs with almost no usable edges to the analyzer.

**The tool currently does not work on real codebases at module level**, and produces fragmented, noisy results at symbol level. The root causes are entirely in the parser, not the analyzer.

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
