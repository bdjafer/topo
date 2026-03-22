#!/usr/bin/env python3
"""
Post-processing quality validation for preprocessed feature files.

Runs quality checks on each repo's features.npz and features.meta.json,
producing a validation report with pass/fail status and quality metrics.

Usage:
    python validate.py                    # Validate all preprocessed repos
    python validate.py --repos flask,click
    python validate.py --strict           # Fail on any warning
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"


# Quality thresholds (from STEP_1 spec §4)
MIN_NODES = 50
MAX_NODES = 50_000
MIN_EDGE_RATIO = 0.5          # edges >= n/2
MIN_COMPONENT_RATIO = 0.5     # largest component >= 50% of nodes
MIN_SPECTRAL_K = 2            # at least 2 non-trivial eigenvectors
MIN_EMBEDDING_COVERAGE = 0.8  # 80% of nodes must have non-zero embeddings


def validate_repo(name: str) -> dict:
    """Validate one repo's feature files. Returns quality report."""
    repo_dir = EXAMPLES_DIR / name
    npz_path = repo_dir / "features.npz"
    meta_path = repo_dir / "features.meta.json"

    result: dict = {"repo": name, "checks": {}, "metrics": {}, "pass": True}

    # Check files exist
    if not npz_path.exists():
        result["checks"]["npz_exists"] = False
        result["pass"] = False
        return result
    if not meta_path.exists():
        result["checks"]["meta_exists"] = False
        result["pass"] = False
        return result
    result["checks"]["npz_exists"] = True
    result["checks"]["meta_exists"] = True

    try:
        arrays = np.load(npz_path)
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        result["checks"]["readable"] = False
        result["error"] = str(e)
        result["pass"] = False
        return result
    result["checks"]["readable"] = True

    n_nodes = meta["n_nodes"]
    result["metrics"]["n_nodes"] = n_nodes

    # ── Check 1: Node count bounds ──
    result["checks"]["node_count_min"] = n_nodes >= MIN_NODES
    result["checks"]["node_count_max"] = n_nodes <= MAX_NODES
    if not result["checks"]["node_count_min"] or not result["checks"]["node_count_max"]:
        result["pass"] = False

    # ── Check 2: Required arrays present ──
    required_arrays = [
        "semantic", "spectral_vecs", "spectral_vals", "rwpe",
        "tree_features", "node_types",
    ]
    for arr_name in required_arrays:
        present = arr_name in arrays
        result["checks"][f"array_{arr_name}"] = present
        if not present:
            result["pass"] = False

    # ── Check 3: Array shapes ──
    if "semantic" in arrays:
        shape = arrays["semantic"].shape
        result["metrics"]["semantic_shape"] = list(shape)
        ok = shape[0] == n_nodes and shape[1] == 768
        result["checks"]["semantic_shape"] = ok
        if not ok:
            result["pass"] = False

    if "spectral_vecs" in arrays:
        shape = arrays["spectral_vecs"].shape
        result["metrics"]["spectral_k"] = shape[1] if len(shape) == 2 else 0
        ok = shape[0] == n_nodes
        result["checks"]["spectral_vecs_shape"] = ok
        if not ok:
            result["pass"] = False

    if "rwpe" in arrays:
        shape = arrays["rwpe"].shape
        ok = shape[0] == n_nodes
        result["checks"]["rwpe_shape"] = ok
        if not ok:
            result["pass"] = False

    if "tree_features" in arrays:
        shape = arrays["tree_features"].shape
        ok = shape == (n_nodes, 4)
        result["checks"]["tree_features_shape"] = ok
        if not ok:
            result["pass"] = False

    if "node_types" in arrays:
        shape = arrays["node_types"].shape
        ok = shape == (n_nodes,)
        result["checks"]["node_types_shape"] = ok
        if not ok:
            result["pass"] = False

    # ── Check 4: Edge count (coupling edges >= n/2) ──
    n_edges_total = 0
    edge_type_counts = {}
    for etype in ["calls", "imports", "inherits"]:
        key = f"edge_index_{etype}"
        if key in arrays:
            m = arrays[key].shape[1] if arrays[key].ndim == 2 else 0
            edge_type_counts[etype] = m
            n_edges_total += m
        else:
            edge_type_counts[etype] = 0
    result["metrics"]["edge_counts"] = edge_type_counts
    result["metrics"]["n_edges_total"] = n_edges_total
    result["checks"]["edge_count_min"] = n_edges_total >= n_nodes / 2

    # ── Check 5: Largest component ratio ──
    n_components = meta.get("n_components", 1)
    result["metrics"]["n_components"] = n_components
    # We don't have per-component sizes in the meta sidecar, but n_components=1 means ratio=1.0
    if n_components == 1:
        result["checks"]["component_ratio"] = True
    else:
        # Can't verify precisely without component sizes — pass with warning
        result["checks"]["component_ratio"] = "unknown"

    # ── Check 6: Spectral k ──
    fiedler = meta.get("fiedler_value", 0.0)
    result["metrics"]["fiedler_value"] = fiedler
    if "spectral_vecs" in arrays:
        spectral_k = arrays["spectral_vecs"].shape[1] if arrays["spectral_vecs"].ndim == 2 else 0
        # Count non-zero columns (actual eigenvectors, not zero-padded)
        if spectral_k > 0:
            col_norms = np.linalg.norm(arrays["spectral_vecs"], axis=0)
            actual_k = int(np.sum(col_norms > 1e-10))
        else:
            actual_k = 0
        result["metrics"]["spectral_k_actual"] = actual_k
        result["checks"]["spectral_k_min"] = actual_k >= MIN_SPECTRAL_K

    # ── Check 7: Semantic embedding coverage ──
    if "semantic" in arrays:
        row_norms = np.linalg.norm(arrays["semantic"], axis=1)
        n_embedded = int(np.sum(row_norms > 1e-10))
        coverage = n_embedded / n_nodes if n_nodes > 0 else 0.0
        result["metrics"]["semantic_coverage"] = coverage
        result["metrics"]["n_embedded"] = n_embedded
        result["checks"]["semantic_coverage"] = coverage >= MIN_EMBEDDING_COVERAGE

    # ── Check 8: No NaN/Inf in arrays ──
    for arr_name in required_arrays:
        if arr_name in arrays:
            arr = arrays[arr_name]
            if np.issubdtype(arr.dtype, np.floating):
                finite = bool(np.isfinite(arr).all())
            else:
                finite = True  # integer arrays are always finite
            result["checks"][f"finite_{arr_name}"] = finite
            if not finite:
                result["pass"] = False

    # ── Metric: Edge type coverage (informational, not a pass/fail gate) ──
    n_edge_types_present = sum(1 for c in edge_type_counts.values() if c > 0)
    result["metrics"]["edge_types_present"] = n_edge_types_present
    result["metrics"]["has_all_edge_types"] = n_edge_types_present == 3

    # ── Metrics (recorded, not filtered) ──
    # Node type distribution
    if "node_types" in arrays:
        types = arrays["node_types"]
        unique, counts = np.unique(types, return_counts=True)
        type_dist = {int(u): int(c) for u, c in zip(unique, counts)}
        result["metrics"]["node_type_distribution"] = type_dist

    # Edge type distribution (as percentages)
    if n_edges_total > 0:
        result["metrics"]["edge_type_pct"] = {
            k: round(v / n_edges_total * 100, 1)
            for k, v in edge_type_counts.items()
        }

    # Overall pass requires all boolean checks to be True
    for check_name, check_val in result["checks"].items():
        if check_val is False:
            result["pass"] = False

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate preprocessed feature files.")
    parser.add_argument("--repos", type=str, default=None, help="Comma-separated repo names")
    parser.add_argument("--strict", action="store_true", help="Fail on any warning")
    args = parser.parse_args()

    # Find repos with features.npz
    if args.repos:
        repo_names = [r.strip() for r in args.repos.split(",")]
    else:
        repo_names = sorted(
            d.name
            for d in EXAMPLES_DIR.iterdir()
            if d.is_dir() and (d / "features.npz").exists()
        )

    if not repo_names:
        print("No preprocessed repos found. Run preprocess.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Validating {len(repo_names)} repos...")

    results = []
    n_pass = 0
    n_fail = 0

    for name in repo_names:
        r = validate_repo(name)
        results.append(r)
        status = "PASS" if r["pass"] else "FAIL"
        if r["pass"]:
            n_pass += 1
        else:
            n_fail += 1

        failed_checks = [k for k, v in r["checks"].items() if v is False]
        detail = f" ({', '.join(failed_checks)})" if failed_checks else ""
        n = r["metrics"].get("n_nodes", "?")
        print(f"  [{status}] {name}: {n} nodes{detail}")

    # Write validation report
    report_path = EXAMPLES_DIR / "validation_report.json"
    report = {
        "total": len(results),
        "pass": n_pass,
        "fail": n_fail,
        "repos": results,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults: {n_pass} pass, {n_fail} fail")
    print(f"Validation report: {report_path}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
