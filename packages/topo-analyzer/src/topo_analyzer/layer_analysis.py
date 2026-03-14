"""
Layer signal analysis — measuring each graph layer's contribution to clustering quality.

Answers the key empirical question from CLAUDE.md:
"Which graph layers contribute the most structural signal?"

Runs spectral decomposition per-layer and across weight combinations,
measuring NMI (against directory baseline) and silhouette score for each
configuration. This identifies which layers carry independent structural
information and what weight ratios maximize clustering quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np

from topo_parser.graph import CodeGraph, EdgeKind
from topo_analyzer.modules import ModuleDetection, detect_modules
from topo_analyzer.projection import (
    AnalysisLevel,
    AnalysisProjection,
    AnalysisProjectionConfig,
    build_projection,
)
from topo_analyzer.spectral import (
    SpectralResult,
    spectral_decomposition,
    spectral_decomposition_multilayer,
)


ALL_LAYERS = (EdgeKind.CALLS, EdgeKind.IMPORTS, EdgeKind.INHERITS, EdgeKind.CONTAINS)


@dataclass(frozen=True)
class LayerSignal:
    """Clustering quality metrics for a single layer configuration."""

    label: str
    weights: dict[EdgeKind, float]
    nmi: float | None
    silhouette: float | None
    n_modules: int
    coverage_ratio: float
    edge_count: int

    @property
    def quality_score(self) -> float:
        """Combined quality: weighted average of NMI and silhouette.

        NMI measures alignment with directory structure (external validity).
        Silhouette measures cluster separation (internal validity).
        Both matter; NMI is weighted higher because architectural alignment
        is the primary goal.
        """
        nmi = self.nmi if self.nmi is not None else 0.0
        sil = self.silhouette if self.silhouette is not None else 0.0
        return 0.6 * nmi + 0.4 * sil


@dataclass
class LayerAnalysisResult:
    """Complete layer signal analysis for a codebase."""

    signals: list[LayerSignal]
    best_single: LayerSignal | None = None
    best_combined: LayerSignal | None = None
    recommended_weights: dict[EdgeKind, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary of layer analysis."""
        lines = ["Layer Signal Analysis", "=" * 40]

        lines.append("")
        lines.append("Per-layer results:")
        lines.append(f"  {'Layer':<12} {'NMI':>6} {'Sil':>6} {'Edges':>6} {'Mods':>5} {'Cov':>6}")
        lines.append(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*6}")
        for signal in self.signals:
            if "+" not in signal.label:  # single layers only
                nmi = f"{signal.nmi:.3f}" if signal.nmi is not None else "  n/a"
                sil = f"{signal.silhouette:.3f}" if signal.silhouette is not None else "  n/a"
                lines.append(
                    f"  {signal.label:<12} {nmi:>6} {sil:>6} {signal.edge_count:>6} "
                    f"{signal.n_modules:>5} {signal.coverage_ratio:>5.1%}"
                )

        combined = [s for s in self.signals if "+" in s.label]
        if combined:
            lines.append("")
            lines.append("Combined configurations:")
            lines.append(f"  {'Config':<28} {'NMI':>6} {'Sil':>6} {'Score':>6}")
            lines.append(f"  {'-'*28} {'-'*6} {'-'*6} {'-'*6}")
            for signal in sorted(combined, key=lambda s: -s.quality_score):
                nmi = f"{signal.nmi:.3f}" if signal.nmi is not None else "  n/a"
                sil = f"{signal.silhouette:.3f}" if signal.silhouette is not None else "  n/a"
                lines.append(
                    f"  {signal.label:<28} {nmi:>6} {sil:>6} {signal.quality_score:>5.3f}"
                )

        if self.best_single:
            lines.append("")
            lines.append(f"Best single layer: {self.best_single.label} "
                         f"(score={self.best_single.quality_score:.3f})")
        if self.best_combined:
            lines.append(f"Best combined:     {self.best_combined.label} "
                         f"(score={self.best_combined.quality_score:.3f})")

        if self.recommended_weights:
            lines.append("")
            lines.append("Recommended weights:")
            for kind, weight in sorted(self.recommended_weights.items(), key=lambda x: -x[1]):
                lines.append(f"  {kind.value}: {weight:.2f}")

        return "\n".join(lines)


def _nmi_vs_directory(
    graph: CodeGraph,
    modules: list,
) -> float | None:
    """Compute NMI between module assignments and directory-based baseline."""
    from collections import defaultdict
    from math import log

    predicted: dict[str, int] = {}
    for module in modules:
        if module.unassigned:
            continue
        for node_id in module.node_ids:
            predicted[node_id] = module.id

    if len(predicted) < 3:
        return None

    baseline = {nid: nid.split(".", 1)[0] for nid in predicted}

    common = sorted(set(predicted) & set(baseline))
    if len(common) < 3:
        return None

    labels_a = [predicted[nid] for nid in common]
    labels_b = [baseline[nid] for nid in common]
    n = len(labels_a)

    joint: dict[tuple, int] = defaultdict(int)
    count_a: dict[object, int] = defaultdict(int)
    count_b: dict[object, int] = defaultdict(int)
    for la, lb in zip(labels_a, labels_b):
        joint[(la, lb)] += 1
        count_a[la] += 1
        count_b[lb] += 1

    mi = 0.0
    for (la, lb), nij in joint.items():
        if nij == 0:
            continue
        pij = nij / n
        pi = count_a[la] / n
        pj = count_b[lb] / n
        mi += pij * log(pij / (pi * pj))

    ha = -sum((c / n) * log(c / n) for c in count_a.values() if c > 0)
    hb = -sum((c / n) * log(c / n) for c in count_b.values() if c > 0)

    if ha + hb == 0:
        return 1.0
    return 2 * mi / (ha + hb)


def _evaluate_config(
    graph: CodeGraph,
    projection: AnalysisProjection,
    label: str,
    weights: dict[EdgeKind, float],
    n_modules: int | None = None,
) -> LayerSignal:
    """Run spectral decomposition + clustering for a weight configuration."""
    analysis_graph = projection.graph

    # Count edges for the active layers
    edge_count = 0
    for kind, weight in weights.items():
        if weight > 0:
            edge_count += sum(
                1 for e in analysis_graph.edges_by_kind(kind)
                if e.source in analysis_graph.nodes and e.target in analysis_graph.nodes
            )

    # Single layer vs multilayer
    active_layers = {k: w for k, w in weights.items() if w > 0}
    if len(active_layers) == 1:
        layer = next(iter(active_layers))
        spectral = spectral_decomposition(analysis_graph, edge_kind=layer)
    else:
        spectral = spectral_decomposition_multilayer(analysis_graph, layer_weights=weights)

    if spectral is None:
        return LayerSignal(
            label=label,
            weights=dict(weights),
            nmi=None,
            silhouette=None,
            n_modules=0,
            coverage_ratio=0.0,
            edge_count=edge_count,
        )

    module_detection = detect_modules(spectral, n_modules=n_modules)
    nmi = _nmi_vs_directory(analysis_graph, module_detection.modules)

    return LayerSignal(
        label=label,
        weights=dict(weights),
        nmi=nmi,
        silhouette=module_detection.silhouette,
        n_modules=len([m for m in module_detection.modules if not m.unassigned]),
        coverage_ratio=spectral.coverage_ratio,
        edge_count=edge_count,
    )


def analyze_layer_signal(
    graph: CodeGraph,
    *,
    level: AnalysisLevel = AnalysisLevel.MODULE,
    scope_roots: tuple[Path, ...] = (),
    weight_steps: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    n_modules: int | None = None,
) -> LayerAnalysisResult:
    """
    Systematically measure each layer's contribution to clustering quality.

    For each layer, runs spectral decomposition independently, then sweeps
    weight combinations to find which mix maximizes NMI and silhouette.

    Args:
        graph: Parsed code graph.
        level: Analysis level (package, module, symbol).
        scope_roots: Source roots for scope filtering (empty = no filtering).
        weight_steps: Weight values to try in the combinatorial sweep.
        n_modules: Force a specific number of modules (auto if None).

    Returns:
        LayerAnalysisResult with per-layer and combined metrics.
    """
    # Build projection with all edge kinds available
    config = AnalysisProjectionConfig(
        level=level,
        edge_kinds=ALL_LAYERS,
        scope_roots=scope_roots,
        internal_only=True,
    )
    projection = build_projection(graph, config)

    signals: list[LayerSignal] = []

    # Phase 1: Per-layer analysis
    for layer in ALL_LAYERS:
        weights = {layer: 1.0}
        signal = _evaluate_config(
            graph, projection, layer.value, weights, n_modules=n_modules,
        )
        signals.append(signal)

    # Phase 2: Pairwise and full combinations
    # Use a coarser grid for efficiency — test key weight ratios
    layers_with_edges = [s for s in signals if s.edge_count > 0]
    active_layers = [
        next(k for k in ALL_LAYERS if k.value == s.label)
        for s in layers_with_edges
    ]

    if len(active_layers) >= 2:
        # Test all pairs
        for i, layer_a in enumerate(active_layers):
            for layer_b in active_layers[i + 1:]:
                for wa, wb in [(1.0, 0.5), (1.0, 1.0), (0.5, 1.0)]:
                    weights = {layer_a: wa, layer_b: wb}
                    label = f"{layer_a.value}({wa})+{layer_b.value}({wb})"
                    signal = _evaluate_config(
                        graph, projection, label, weights, n_modules=n_modules,
                    )
                    signals.append(signal)

    # Test the full combination with a weight sweep on the secondary layers
    # (CALLS always anchored at 1.0 since it's expected to be primary)
    if len(active_layers) >= 2 and EdgeKind.CALLS in active_layers:
        other_layers = [l for l in active_layers if l != EdgeKind.CALLS]
        # Sweep secondary weights
        for step_combo in product(weight_steps, repeat=len(other_layers)):
            if all(s == 0.0 for s in step_combo):
                continue  # calls-only already tested
            weights = {EdgeKind.CALLS: 1.0}
            label_parts = ["calls(1.0)"]
            for layer, w in zip(other_layers, step_combo):
                if w > 0:
                    weights[layer] = w
                    label_parts.append(f"{layer.value}({w})")
            if len(weights) <= 1:
                continue
            label = "+".join(label_parts)
            signal = _evaluate_config(
                graph, projection, label, weights, n_modules=n_modules,
            )
            signals.append(signal)

    # Find best configurations
    single_signals = [s for s in signals if "+" not in s.label and s.nmi is not None]
    combined_signals = [s for s in signals if "+" in s.label and s.nmi is not None]

    best_single = max(single_signals, key=lambda s: s.quality_score) if single_signals else None
    best_combined = max(combined_signals, key=lambda s: s.quality_score) if combined_signals else None

    # Determine recommended weights
    best_overall = best_combined if (
        best_combined and best_single
        and best_combined.quality_score > best_single.quality_score + 0.02
    ) else best_single

    recommended = {}
    if best_overall:
        recommended = dict(best_overall.weights)
        # Zero out unused layers explicitly
        for layer in ALL_LAYERS:
            if layer not in recommended:
                recommended[layer] = 0.0

    return LayerAnalysisResult(
        signals=signals,
        best_single=best_single,
        best_combined=best_combined,
        recommended_weights=recommended,
    )
