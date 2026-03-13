"""Scorecard assembly and guardrail evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from topo_benchmark.scoring import geometric_mean


@dataclass
class Scorecard:
    """Benchmark scorecard matching the BENCHMARK.md spec."""

    tier: str
    split: str
    overall_primary: float
    dimensions: dict[str, float]
    guardrails: dict[str, bool]
    promotion_decision: str

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> Scorecard:
        data = json.loads(path.read_text())
        return cls(**data)


def build_scorecard(
    tier: str,
    split: str,
    dimension_results: dict[str, dict],
    baseline_results: dict[str, dict] | None = None,
) -> Scorecard:
    """Build a scorecard from dimension results."""
    dimensions = {
        name: result.get("score", 0.0)
        for name, result in dimension_results.items()
    }

    overall = geometric_mean(*dimensions.values()) if dimensions else 0.0

    # Evaluate guardrails
    arch = dimension_results.get("architecture_recovery", {})
    anomaly = dimension_results.get("anomaly_precision_calibration", {})

    coverage_ok = arch.get("guardrails", {}).get("coverage_ok", True)
    baseline_ok = arch.get("guardrails", {}).get("baseline_ok", True)
    calibration_ok = anomaly.get("calibration_score", 1.0) > 0.3
    no_regressions = True  # Set during compare, not during run

    guardrails = {
        "coverage_ok": coverage_ok,
        "calibration_ok": calibration_ok,
        "baseline_ok": baseline_ok,
        "no_material_regressions": no_regressions,
    }

    all_pass = all(guardrails.values()) and overall > 0
    promotion_decision = "pass" if all_pass else "fail"

    return Scorecard(
        tier=tier,
        split=split,
        overall_primary=round(overall, 4),
        dimensions={k: round(v, 4) for k, v in dimensions.items()},
        guardrails=guardrails,
        promotion_decision=promotion_decision,
    )
