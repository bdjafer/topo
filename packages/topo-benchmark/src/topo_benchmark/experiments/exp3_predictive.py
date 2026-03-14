"""Experiment 3: Predictive Validity via Git History.

Tests whether spectral anomalies at time T predict code churn at time T+N.
This is the strongest evidence: if anomalies predict future changes, they
capture real architectural stress.

STATUS: Scaffold — git history analysis and statistics need implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodebaseChurnResult:
    codebase: str
    snapshot_commit: str = ""
    anomalous_nodes: int = 0
    normal_nodes: int = 0
    anomalous_mean_churn: float = 0.0
    normal_mean_churn: float = 0.0
    u_statistic: float = 0.0
    p_value: float = 1.0
    effect_size: float = 0.0  # rank-biserial correlation
    error: str | None = None


@dataclass
class Experiment3Result:
    per_codebase: list[CodebaseChurnResult] = field(default_factory=list)
    verdict: str = "NOT_IMPLEMENTED"
    verdict_details: list[str] = field(default_factory=list)


def checkout_at_midpoint(repo_path: Path) -> tuple[str, int]:
    """Find the midpoint commit of a repo's history.

    Returns (midpoint_commit_hash, total_commits).
    """
    raise NotImplementedError("Experiment 3 not yet implemented")


def compute_file_churn(
    repo_path: Path,
    since_commit: str,
) -> dict[str, float]:
    """Compute per-file churn (commits touching file / file size) since a commit.

    Returns {file_path: normalized_churn}.
    """
    raise NotImplementedError("Experiment 3 not yet implemented")


def mann_whitney_test(
    anomalous_churn: list[float],
    normal_churn: list[float],
) -> tuple[float, float, float]:
    """Mann-Whitney U test comparing churn distributions.

    Returns (u_statistic, p_value, rank_biserial_correlation).
    Requires scipy.
    """
    raise NotImplementedError("Experiment 3 not yet implemented")


def run_experiment_3(
    codebases_root: Path,
    labels_root: Path,
    output_dir: Path,
) -> Experiment3Result:
    """Run Experiment 3: Predictive Validity via Git History."""
    return Experiment3Result(
        verdict="NOT_IMPLEMENTED",
        verdict_details=["Experiment 3 scaffold — git history analysis needs implementation."],
    )
