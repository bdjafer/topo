"""topo-analyzer: Graph → structural intelligence via spectral analysis."""

from topo_analyzer.spectral import spectral_decomposition, spectral_decomposition_multilayer
from topo_analyzer.modules import detect_modules
from topo_analyzer.roles import classify_roles
from topo_analyzer.anomalies import detect_anomalies
from topo_analyzer.analysis import StructuralAnalysis, analyze

__all__ = [
    "StructuralAnalysis",
    "analyze",
    "classify_roles",
    "detect_anomalies",
    "detect_modules",
    "spectral_decomposition",
    "spectral_decomposition_multilayer",
]
