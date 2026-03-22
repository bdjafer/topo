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
TOPO_BIN = PROJECT_ROOT / "target" / "release" / "topo"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results" / "mutation_sensitivity"

# Mutation types and their expected diagnostics.
# Only the 5 implemented ones — 4 more (misplaced_concern, incoherent_module,
# shadow_dependency, redundant_api) are designed but not yet implemented.
MUTATIONS = {
    "inject_cycle": "circular_dependency",
    "layer_violation": "layer_violation",
    "overloaded_utility": "overloaded_utility",
    "wide_interface": "wide_interface",
    "near_disconnect": "near_disconnect",
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
    """Run topo analysis on a graph dict. Returns analysis output."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(graph, f)
        input_path = f.name

    try:
        result = subprocess.run(
            [str(TOPO_BIN), "analyze-raw", "--input", input_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return {"issues": [], "error": result.stderr[:200]}
        return json.loads(result.stdout)
    except Exception as e:
        return {"issues": [], "error": str(e)}
    finally:
        Path(input_path).unlink(missing_ok=True)


def apply_mutation(
    graph: dict, mutation_type: str, severity: int, seed: int = 42,
) -> dict | None:
    """Apply a mutation to a graph via the topo CLI.

    Returns the mutated graph dict, or None if the mutation can't be applied
    (graph lacks preconditions for this mutation type).
    """
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(graph, f)
        input_path = f.name

    try:
        result = subprocess.run(
            [
                str(TOPO_BIN),
                "mutate",
                "--input", input_path,
                "--type", mutation_type,
                "--severity", str(severity),
                "--seed", str(seed),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=120,
        )

        if result.returncode == 2:
            # Exit code 2 = mutation returned None (preconditions not met).
            return None
        if result.returncode != 0:
            print(f"    WARN: mutate failed: {result.stderr[:200]}", file=sys.stderr)
            return None

        output = json.loads(result.stdout)
        return output.get("graph")
    except Exception as e:
        print(f"    WARN: mutate error: {e}", file=sys.stderr)
        return None
    finally:
        Path(input_path).unlink(missing_ok=True)


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

    # Discover available repos from examples/ (single source for all cached graphs).
    available = []
    for graph_path in sorted(EXAMPLES_DIR.glob("*/graph.json")):
        name = graph_path.parent.name
        if name == "scripts":
            continue
        if repos is None or name in repos:
            available.append((name, graph_path))

    print(f"Evaluating {len(available)} repos × {len(mutations)} mutations × {len(SEVERITY_LEVELS)} levels")
    print(f"Total test cases: {len(available) * len(mutations) * len(SEVERITY_LEVELS)}")
    print()

    for repo_name, graph_path in available:
        print(f"[{repo_name}]")
        with open(graph_path) as f:
            graph = json.load(f)

        # Clean analysis: reuse cached analysis.json if available, otherwise run fresh.
        cached_analysis = graph_path.parent / "analysis.json"
        if cached_analysis.exists():
            with open(cached_analysis) as f:
                clean_analysis = json.load(f)
        else:
            clean_analysis = analyze_graph(graph)
            if "error" in clean_analysis:
                print(f"  SKIP: analysis failed: {clean_analysis['error']}")
                continue

        clean_kinds = get_issue_kinds(clean_analysis)

        for mutation_type in mutations:
            expected = MUTATIONS[mutation_type]

            for severity in SEVERITY_LEVELS:
                seed = hash((repo_name, mutation_type, severity)) % (2**32)
                mutated_graph = apply_mutation(graph, mutation_type, severity, seed=seed)
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
