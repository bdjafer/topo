"""Pre-registered thresholds and codebase registry.

All thresholds are from METHODOLOGY.md, committed before any experiments run.
Do not modify thresholds after experiments start.
"""

from __future__ import annotations

THRESHOLDS = {
    "exp1": {
        # Cross-directory recovery rate thresholds
        "cross_dir_recovery_pass": 0.30,  # >30% on >=N codebases → pass
        "cross_dir_recovery_fail": 0.10,  # <10% on >=N codebases → fail
        # V-measure must beat baselines on >= this many codebases
        "min_codebases_pass": 3,
        "min_codebases_total": 5,
    },
    "exp2": {
        "recall_advantage_pp": 0.20,  # Spectral recall > baseline by >=20pp
        "precision_floor": 0.50,
        "precision_fail": 0.30,
        "min_mutation_types_pass": 3,
        "mutation_types_total": 5,
    },
    "exp3": {
        "p_value": 0.05,  # Bonferroni-corrected
        "effect_size_pass": 0.20,  # rank-biserial >= 0.2
        "effect_size_fail": 0.10,
        "min_codebases_pass": 3,
        "min_codebases_total": 5,
    },
    "exp4": {
        "preserving_ari_pass": 0.80,
        "breaking_ari_ceiling": 0.50,
        "separation_pass": 0.30,  # preserving - breaking >= 0.3
        "preserving_ari_fail": 0.60,
        "separation_fail": 0.10,
    },
}

CODEBASES: dict[str, dict[str, str]] = {
    "flask": {
        "repo": "pallets/flask",
        "ref": "3.1.0",
        "src_root": "src/flask",
        "type": "framework_subpackages",
    },
    "click": {
        "repo": "pallets/click",
        "ref": "8.1.8",
        "src_root": "src/click",
        "type": "flat_single_package",
    },
}
