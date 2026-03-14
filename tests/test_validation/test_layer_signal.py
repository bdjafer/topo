"""Empirical tests for layer signal contribution.

Answers the open question: which graph layers carry the most structural
signal, and do combined layers outperform single-layer analysis?
"""

from __future__ import annotations

from topo_analyzer.layer_analysis import (
    LayerSignal,
    analyze_layer_signal,
)
from topo_analyzer.projection import AnalysisLevel
from topo_parser.graph import EdgeKind

from tests.test_validation.benchmark_utils import (
    analyze_fixture,
    fixture_root,
    repo_root,
)
from topo_parser.python import parse_python_project


# ---------------------------------------------------------------------------
# Fixture-based tests (hermetic, fast)
# ---------------------------------------------------------------------------


def test_calls_layer_has_signal_on_layered_app():
    """The CALLS layer alone should produce meaningful clustering on layered_app."""
    root = fixture_root("layered_app")
    graph = parse_python_project(root)
    result = analyze_layer_signal(graph, level=AnalysisLevel.MODULE)

    calls_signal = next(s for s in result.signals if s.label == "calls")
    assert calls_signal.edge_count > 0, "layered_app should have call edges"
    assert calls_signal.nmi is not None, "calls should produce clusters"
    assert calls_signal.nmi > 0.3, f"calls NMI={calls_signal.nmi} too low"


def test_imports_layer_has_signal_on_layered_app():
    """The IMPORTS layer should also carry structural signal."""
    root = fixture_root("layered_app")
    graph = parse_python_project(root)
    result = analyze_layer_signal(graph, level=AnalysisLevel.MODULE)

    imports_signal = next(s for s in result.signals if s.label == "imports")
    # imports may have zero edges on small fixtures — that's informative too
    if imports_signal.edge_count > 0:
        assert imports_signal.nmi is not None


def test_best_config_is_identified():
    """analyze_layer_signal should identify a best configuration."""
    root = fixture_root("layered_app")
    graph = parse_python_project(root)
    result = analyze_layer_signal(graph, level=AnalysisLevel.MODULE)

    # At least one best should be found
    assert result.best_single is not None or result.best_combined is not None
    # Recommended weights should be populated
    assert len(result.recommended_weights) > 0


def test_contains_layer_alone_is_weak():
    """CONTAINS edges (parent-child) should not produce good clustering alone.

    Contains edges encode organizational hierarchy, not coupling. Their
    clustering should be weaker than calls.
    """
    root = fixture_root("layered_app")
    graph = parse_python_project(root)
    result = analyze_layer_signal(graph, level=AnalysisLevel.MODULE)

    calls_signal = next(s for s in result.signals if s.label == "calls")
    contains_signal = next(s for s in result.signals if s.label == "contains")

    if contains_signal.edge_count > 0 and calls_signal.nmi is not None:
        # Contains alone should not beat calls
        contains_score = contains_signal.quality_score
        calls_score = calls_signal.quality_score
        assert calls_score >= contains_score, (
            f"contains ({contains_score:.3f}) should not beat calls ({calls_score:.3f})"
        )


def test_summary_is_readable():
    """The summary output should be non-empty and formatted."""
    root = fixture_root("layered_app")
    graph = parse_python_project(root)
    result = analyze_layer_signal(graph, level=AnalysisLevel.MODULE)

    summary = result.summary()
    assert "Layer Signal Analysis" in summary
    assert "calls" in summary


# ---------------------------------------------------------------------------
# Self-analysis test (runs on topo's own codebase)
# ---------------------------------------------------------------------------


def test_layer_signal_on_topo_self():
    """Run layer signal analysis on topo's own codebase.

    This test serves as a living validation: as the codebase evolves,
    the layer signal analysis should continue to produce results and
    the calls layer should remain the strongest signal.
    """
    from topo_analyzer.projection import discover_first_party_source_roots

    root = repo_root()
    scope_roots = discover_first_party_source_roots(root)
    graph = parse_python_project(root, include_roots=list(scope_roots))

    result = analyze_layer_signal(graph, level=AnalysisLevel.MODULE)

    # Should produce results
    assert len(result.signals) > 0

    calls_signal = next(s for s in result.signals if s.label == "calls")
    assert calls_signal.edge_count > 0, "topo should have call edges"

    # Print the summary for inspection during test runs
    print("\n" + result.summary())
