"""Compare two benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path

from topo_benchmark.bootstrap import bootstrap_delta_ci
from topo_benchmark.scorecard import Scorecard


def compare_runs(
    candidate_dir: Path,
    reference_dir: Path,
    max_regression: float = 0.02,
) -> dict:
    """Compare candidate vs reference benchmark runs.

    Returns a comparison report with deltas, CIs, and promotion decision.
    """
    candidate = Scorecard.load(candidate_dir / "scorecard.json")
    reference = Scorecard.load(reference_dir / "scorecard.json")

    # Per-dimension deltas
    dimension_deltas: dict[str, dict] = {}
    any_regression = False

    for dim in candidate.dimensions:
        c_score = candidate.dimensions.get(dim, 0.0)
        r_score = reference.dimensions.get(dim, 0.0)
        delta = c_score - r_score

        # Use per-case scores for bootstrap if available
        c_cases_path = candidate_dir / "per_case.jsonl"
        r_cases_path = reference_dir / "per_case.jsonl"

        c_scores = _load_dimension_case_scores(c_cases_path, dim)
        r_scores = _load_dimension_case_scores(r_cases_path, dim)

        if c_scores and r_scores:
            _, ci_low, ci_high = bootstrap_delta_ci(c_scores, r_scores)
        else:
            ci_low = delta
            ci_high = delta

        if delta < -max_regression:
            any_regression = True

        dimension_deltas[dim] = {
            "candidate": c_score,
            "reference": r_score,
            "delta": round(delta, 4),
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "regressed": delta < -max_regression,
        }

    overall_delta = candidate.overall_primary - reference.overall_primary

    # Promotion rules
    overall_improved = overall_delta > 0
    no_regressions = not any_regression
    guardrails_pass = all(candidate.guardrails.values())

    promotion = overall_improved and no_regressions and guardrails_pass

    return {
        "overall_delta": round(overall_delta, 4),
        "candidate_overall": candidate.overall_primary,
        "reference_overall": reference.overall_primary,
        "dimensions": dimension_deltas,
        "promotion_decision": "pass" if promotion else "fail",
        "reasons": {
            "overall_improved": overall_improved,
            "no_regressions": no_regressions,
            "guardrails_pass": guardrails_pass,
        },
    }


def _load_dimension_case_scores(path: Path, dimension: str) -> list[float]:
    """Load per-case scores for a dimension from per_case.jsonl."""
    if not path.exists():
        return []
    scores = []
    for line in path.read_text().strip().split("\n"):
        if not line:
            continue
        case = json.loads(line)
        if case.get("dimension") == dimension:
            scores.append(case.get("score", 0.0))
    return scores
