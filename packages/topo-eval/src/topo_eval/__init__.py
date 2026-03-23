"""Self-supervised evaluation pipeline for R-GIN."""

from topo_eval.tier1 import tier1_intrinsic_metrics
from topo_eval.tier2 import tier2_phase2_agreement
from topo_eval.tier3 import tier3_perturbation_test
from topo_eval.tier4 import tier4_structural_consistency
from topo_eval.baselines import run_baselines
from topo_eval.gate import go_no_go_gate

__all__ = [
    "tier1_intrinsic_metrics",
    "tier2_phase2_agreement",
    "tier3_perturbation_test",
    "tier4_structural_consistency",
    "run_baselines",
    "go_no_go_gate",
]
