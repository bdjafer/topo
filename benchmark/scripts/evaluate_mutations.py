#!/usr/bin/env python3
"""
Signal 1: Mutation Sensitivity Evaluation.

For each repo in the corpus:
1. Load cached graph.json
2. Run clean analysis
3. Apply each mutation type × severity level
4. Run analysis on mutated graph
5. Check if expected diagnostic fires

Reports sensitivity, specificity, severity calibration, and collateral.

Usage:
    python3 benchmark/scripts/evaluate_mutations.py
    python3 benchmark/scripts/evaluate_mutations.py --repos flask,click --mutations inject_cycle
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
CORPUS_DIR = PROJECT_ROOT / "benchmark" / "corpus"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results" / "mutation_sensitivity"

# Mutation types and their expected diagnostics
MUTATIONS = {
    "inject_cycle": "circular_dependency",
    "wide_interface": "wide_interface",
    "misplaced_concern": "misplaced_concern",
    "incoherent_module": "incoherent_module",
    "shadow_dependency": "shadow_dependency",
    "layer_violation": "layer_violation",
    "near_disconnect": "near_disconnect",
    "overloaded_utility": "overloaded_utility",
    "redundant_api": "redundant_api",
}

SEVERITY_LEVELS = [1, 2, 3]


@dataclass
class MutationResult:
    repo: str
    mutation: str
    severity: int
    clean_has_diagnostic: bool
    mutated_has_diagnostic: bool
    clean_severity_max: float
    mutated_severity_max: float
    clean_issue_count: int
    mutated_issue_count: int
    collateral_new: int  # issues that appeared
    collateral_gone: int  # issues that disappeared
    attribution_hit: bool  # mutated region in top-3


@dataclass
class MutationScorecard:
    mutation_type: str
    expected_diagnostic: str
    n_repos: int = 0
    true_positives: int = 0  # diagnostic fires on mutated
    false_positives: int = 0  # diagnostic already fires on clean
    severity_pairs: list[tuple[int, float]] = field(default_factory=list)
    collateral_ratios: list[float] = field(default_factory=list)
    attribution_hits: int = 0

    @property
    def sensitivity(self) -> float:
        return self.true_positives / max(1, self.n_repos)

    @property
    def specificity(self) -> float:
        return 1.0 - (self.false_positives / max(1, self.n_repos))

    @property
    def net_sensitivity(self) -> float:
        return self.sensitivity - (1.0 - self.specificity)

    def to_dict(self) -> dict:
        return {
            "mutation_type": self.mutation_type,
            "expected_diagnostic": self.expected_diagnostic,
            "n_repos": self.n_repos,
            "sensitivity": round(self.sensitivity, 4),
            "specificity": round(self.specificity, 4),
            "net_sensitivity": round(self.net_sensitivity, 4),
            "attribution_rate": round(self.attribution_hits / max(1, self.n_repos), 4),
            "mean_collateral": round(
                sum(self.collateral_ratios) / max(1, len(self.collateral_ratios)), 4
            ),
        }


def get_issue_kinds(analysis: dict) -> set[str]:
    """Extract set of issue kinds from analysis output."""
    return {issue["kind"] for issue in analysis.get("issues", [])}


def get_max_severity_for_kind(analysis: dict, kind: str) -> float:
    """Get maximum severity for a specific issue kind."""
    severities = [
        issue.get("severity", 0.0)
        for issue in analysis.get("issues", [])
        if issue.get("kind") == kind
    ]
    return max(severities) if severities else 0.0


def analyze_graph(graph: dict) -> dict:
    """Run topo analysis on a graph dict. Returns analysis output.

    TODO: Replace with PyO3 call to topo_analyzer.analyze_full()
    when the Python bindings expose the analyze function.
    Currently uses subprocess as a bridge.
    """
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(graph, f)
        input_path = f.name

    try:
        result = subprocess.run(
            ["cargo", "run", "-p", "topo-cli", "--", "analyze",
             "--input", input_path, "--format", "json"],
            cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return {"issues": [], "error": result.stderr[:200]}
        return json.loads(result.stdout)
    except Exception as e:
        return {"issues": [], "error": str(e)}
    finally:
        Path(input_path).unlink(missing_ok=True)


def apply_mutation(graph: dict, mutation_type: str, severity: int) -> dict | None:
    """Apply a mutation to a graph.

    TODO: Replace with Rust mutation operators via PyO3.
    This is a placeholder that demonstrates the interface.
    The real operators will be in packages/topo-benchmark/src/mutations/.
    """
    # Placeholder — real implementation will use Rust operators
    # For now, return None to indicate "not yet implemented"
    return None


def evaluate_corpus(
    repos: list[str] | None = None,
    mutations: list[str] | None = None,
) -> dict[str, MutationScorecard]:
    """Run the full evaluation pipeline."""
    if mutations is None:
        mutations = list(MUTATIONS.keys())

    scorecards: dict[str, MutationScorecard] = {
        m: MutationScorecard(m, MUTATIONS[m]) for m in mutations
    }

    # Discover available repos
    available = []
    for meta_path in sorted(CORPUS_DIR.glob("*/metadata.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        name = meta["name"]
        if repos is None or name in repos:
            graph_path = meta_path.parent / "graph.json"
            if graph_path.exists():
                available.append((name, graph_path))

    print(f"Evaluating {len(available)} repos × {len(mutations)} mutations × {len(SEVERITY_LEVELS)} levels")
    print(f"Total test cases: {len(available) * len(mutations) * len(SEVERITY_LEVELS)}")
    print()

    for repo_name, graph_path in available:
        print(f"[{repo_name}]")
        with open(graph_path) as f:
            graph = json.load(f)

        # Clean analysis (once per repo)
        clean_analysis = analyze_graph(graph)
        if "error" in clean_analysis:
            print(f"  SKIP: analysis failed: {clean_analysis['error']}")
            continue

        clean_kinds = get_issue_kinds(clean_analysis)

        for mutation_type in mutations:
            expected = MUTATIONS[mutation_type]

            for severity in SEVERITY_LEVELS:
                mutated_graph = apply_mutation(graph, mutation_type, severity)
                if mutated_graph is None:
                    continue

                mutated_analysis = analyze_graph(mutated_graph)
                if "error" in mutated_analysis:
                    continue

                mutated_kinds = get_issue_kinds(mutated_analysis)

                card = scorecards[mutation_type]
                card.n_repos += 1

                if expected in mutated_kinds:
                    card.true_positives += 1
                if expected in clean_kinds:
                    card.false_positives += 1

                card.severity_pairs.append(
                    (severity, get_max_severity_for_kind(mutated_analysis, expected))
                )

                # Collateral
                new_kinds = mutated_kinds - clean_kinds
                gone_kinds = clean_kinds - mutated_kinds
                total_kinds = len(clean_kinds | mutated_kinds) or 1
                card.collateral_ratios.append(
                    (len(new_kinds) + len(gone_kinds) - (1 if expected in new_kinds else 0))
                    / total_kinds
                )

    return scorecards


def main():
    repos = None
    mutations = None

    for arg in sys.argv[1:]:
        if arg.startswith("--repos="):
            repos = arg.split("=", 1)[1].split(",")
        elif arg.startswith("--mutations="):
            mutations = arg.split("=", 1)[1].split(",")

    scorecards = evaluate_corpus(repos, mutations)

    # Write results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scorecard_data = {k: v.to_dict() for k, v in scorecards.items()}

    with open(RESULTS_DIR / "scorecard.json", "w") as f:
        json.dump(scorecard_data, f, indent=2)

    print("\n── Scorecard ──")
    print(f"{'Mutation':<24} {'Sens':>6} {'Spec':>6} {'Net':>6} {'Attr':>6} {'Coll':>6}")
    print("─" * 72)
    for card in scorecards.values():
        d = card.to_dict()
        print(
            f"{d['mutation_type']:<24} "
            f"{d['sensitivity']:>6.2f} "
            f"{d['specificity']:>6.2f} "
            f"{d['net_sensitivity']:>6.2f} "
            f"{d['attribution_rate']:>6.2f} "
            f"{d['mean_collateral']:>6.2f}"
        )


if __name__ == "__main__":
    main()
