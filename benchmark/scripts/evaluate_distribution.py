#!/usr/bin/env python3
"""
Signal 3: Cross-Codebase Distributional Sanity.

Run topo on all corpus repos and check:
1. Issue rate per diagnostic (should be 5-60%)
2. Severity distributions (should be right-skewed)
3. Diagnostic concordance (co-occurrence matrix)
4. Issue-per-node ratio (should be O(1))

Usage:
    python3 benchmark/scripts/evaluate_distribution.py
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
CORPUS_DIR = PROJECT_ROOT / "benchmark" / "corpus"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results" / "distribution"


def analyze_cached_graph(graph_path: Path) -> dict | None:
    """Run topo analysis on a cached graph."""
    try:
        result = subprocess.run(
            ["cargo", "run", "-p", "topo-cli", "--", "analyze",
             "--input", str(graph_path), "--format", "json"],
            cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return None


def main():
    if not CORPUS_DIR.exists():
        print("No corpus found. Run harvest_corpus.py first.")
        sys.exit(1)

    repos = sorted(CORPUS_DIR.glob("*/graph.json"))
    print(f"Analyzing {len(repos)} repos for distributional sanity...")

    # Collectors
    diagnostic_rates: dict[str, int] = Counter()  # diagnostic -> count of repos where it fires
    severity_values: dict[str, list[float]] = {}  # diagnostic -> list of max severities
    issue_per_node: list[float] = []
    modularity_values: list[float] = []
    total_repos = 0
    cooccurrence: dict[tuple[str, str], int] = Counter()

    for graph_path in repos:
        name = graph_path.parent.name
        meta_path = graph_path.parent / "metadata.json"

        # Check for cached analysis
        analysis_path = graph_path.parent / "analysis.json"
        if analysis_path.exists():
            with open(analysis_path) as f:
                analysis = json.load(f)
        else:
            print(f"  [{name}] analyzing...")
            analysis = analyze_cached_graph(graph_path)
            if analysis is None:
                print(f"  [{name}] FAILED")
                continue
            # Cache it
            with open(analysis_path, "w") as f:
                json.dump(analysis, f)

        total_repos += 1
        issues = analysis.get("issues", [])
        nodes_count = analysis.get("coverage", {}).get("analyzed_nodes", 1) or 1

        # Issue-per-node ratio
        issue_per_node.append(len(issues) / nodes_count)

        # Modularity
        health = analysis.get("health") or {}
        q = health.get("modularity_q")
        if q is not None:
            modularity_values.append(q)

        # Per-diagnostic stats
        kinds_in_repo: set[str] = set()
        for issue in issues:
            kind = issue.get("kind", "unknown")
            kinds_in_repo.add(kind)
            severity = issue.get("severity", 0.0)
            severity_values.setdefault(kind, []).append(severity)

        for kind in kinds_in_repo:
            diagnostic_rates[kind] += 1

        # Co-occurrence
        sorted_kinds = sorted(kinds_in_repo)
        for i, k1 in enumerate(sorted_kinds):
            for k2 in sorted_kinds[i + 1:]:
                cooccurrence[(k1, k2)] += 1

    # ── Report ──

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n── Distribution Report ({total_repos} repos) ──\n")

    # 1. Diagnostic rates
    print("Diagnostic firing rates:")
    print(f"  {'Diagnostic':<30} {'Repos':>6} {'Rate':>8}")
    print("  " + "─" * 46)
    rates_data = {}
    for kind, count in sorted(diagnostic_rates.items(), key=lambda x: -x[1]):
        rate = count / max(1, total_repos)
        flag = ""
        if rate < 0.05:
            flag = " ⚠ DEAD (<5%)"
        elif rate > 0.60:
            flag = " ⚠ NOISY (>60%)"
        print(f"  {kind:<30} {count:>6} {rate:>7.1%}{flag}")
        rates_data[kind] = {"count": count, "rate": round(rate, 4)}

    # 2. Severity distributions
    print("\nSeverity distribution (per diagnostic):")
    print(f"  {'Diagnostic':<30} {'Mean':>6} {'P50':>6} {'P90':>6} {'Max':>6}")
    print("  " + "─" * 56)
    severity_data = {}
    for kind in sorted(severity_values.keys()):
        vals = sorted(severity_values[kind])
        n = len(vals)
        mean = sum(vals) / n
        p50 = vals[n // 2]
        p90 = vals[int(n * 0.9)]
        mx = vals[-1]
        print(f"  {kind:<30} {mean:>6.2f} {p50:>6.2f} {p90:>6.2f} {mx:>6.2f}")
        severity_data[kind] = {
            "mean": round(mean, 4), "p50": round(p50, 4),
            "p90": round(p90, 4), "max": round(mx, 4), "n": n,
        }

    # 3. Issue-per-node ratio
    if issue_per_node:
        mean_ipn = sum(issue_per_node) / len(issue_per_node)
        max_ipn = max(issue_per_node)
        print(f"\nIssue-per-node ratio: mean={mean_ipn:.3f}, max={max_ipn:.3f}")
        if max_ipn > 1.0:
            print("  ⚠ Some repos have >1 issue per node (thresholds may be too loose)")

    # 4. Modularity Q
    if modularity_values:
        mean_q = sum(modularity_values) / len(modularity_values)
        min_q = min(modularity_values)
        max_q = max(modularity_values)
        print(f"\nModularity Q: mean={mean_q:.3f}, range=[{min_q:.3f}, {max_q:.3f}]")

    # Write results
    summary = {
        "total_repos": total_repos,
        "diagnostic_rates": rates_data,
        "severity_distributions": severity_data,
        "issue_per_node": {
            "mean": round(mean_ipn, 4) if issue_per_node else None,
            "max": round(max_ipn, 4) if issue_per_node else None,
        },
        "modularity_q": {
            "mean": round(mean_q, 4) if modularity_values else None,
            "min": round(min_q, 4) if modularity_values else None,
            "max": round(max_q, 4) if modularity_values else None,
        },
        "cooccurrence": {
            f"{k1}+{k2}": count
            for (k1, k2), count in sorted(cooccurrence.items(), key=lambda x: -x[1])
        },
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults written to {RESULTS_DIR}/summary.json")


if __name__ == "__main__":
    main()
