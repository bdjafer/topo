"""Tests for the top-level analysis orchestrator."""

import textwrap
from pathlib import Path

from topo_analyzer.analysis import analyze
from topo_analyzer.anomalies import AnomalyKind
from topo_analyzer.projection import (
    AnalysisLevel,
    AnalysisProjectionConfig,
    load_analysis_policy,
)
from topo_parser.graph import EdgeKind
from topo_parser.python import parse_python_project


def _make_two_package_project(tmp_path: Path) -> Path:
    """Create a tiny two-package project with one directional dependency."""
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    (alpha / "__init__.py").write_text("")
    (beta / "__init__.py").write_text("")
    (beta / "tools.py").write_text(textwrap.dedent("""\
        def helper():
            pass
    """))
    (alpha / "core.py").write_text(textwrap.dedent("""\
        from beta.tools import helper

        def run():
            helper()
    """))
    return tmp_path


def _make_reverse_two_package_project(tmp_path: Path) -> Path:
    """Create a two-package project with a reverse dependency."""
    root = _make_two_package_project(tmp_path)
    (root / "beta" / "tools.py").write_text(textwrap.dedent("""\
        from alpha.core import run

        def helper():
            run()
    """))
    return root


def test_analyze_reports_projection_health_and_dependencies(tmp_path: Path):
    """analyze() should expose coverage, health, and package flow."""
    root = _make_two_package_project(tmp_path)
    graph = parse_python_project(root)
    config = AnalysisProjectionConfig.for_analysis(
        edge_kind=EdgeKind.CALLS,
        combined=True,
        level=AnalysisLevel.MODULE,
    )

    result = analyze(graph, combined=True, projection_config=config)

    assert result.coverage is not None
    assert result.health is not None
    assert result.graph.node_count >= 4
    assert result.health.call_count == 1
    assert any(
        dependency.source_package == "alpha" and dependency.target_package == "beta"
        for dependency in result.cross_package_dependencies
    )


def test_summary_and_json_include_confidence_and_findings(tmp_path: Path):
    """Text and JSON output should include findings and confidence metadata."""
    root = _make_two_package_project(tmp_path)
    graph = parse_python_project(root)
    config = AnalysisProjectionConfig.for_analysis(
        edge_kind=EdgeKind.CALLS,
        combined=True,
        level=AnalysisLevel.MODULE,
    )

    result = analyze(graph, combined=True, projection_config=config)
    summary = result.summary()
    data = result.to_dict()

    assert "Findings:" in summary
    assert "Coverage:" in summary
    assert "Package flow:" in summary
    assert "scope" in data
    assert "coverage" in data
    assert "findings" in data
    assert "clustering" in data


def test_reverse_dependency_is_reported_from_graph(tmp_path: Path):
    """Bidirectional package flow should produce a reverse-dependency finding."""
    root = _make_reverse_two_package_project(tmp_path)
    graph = parse_python_project(root)
    config = AnalysisProjectionConfig.for_analysis(
        edge_kind=EdgeKind.CALLS,
        combined=True,
        level=AnalysisLevel.MODULE,
    )

    result = analyze(graph, combined=True, projection_config=config)

    reverse_findings = [finding for finding in result.findings if finding.kind == "reverse_dependency"]
    assert reverse_findings
    assert "alpha and beta" in reverse_findings[0].title


def test_cycle_is_reported_from_graph(tmp_path: Path):
    """A real call cycle should remain visible to the analyzer."""
    root = _make_reverse_two_package_project(tmp_path)
    graph = parse_python_project(root)
    config = AnalysisProjectionConfig.for_analysis(
        edge_kind=EdgeKind.CALLS,
        combined=True,
        level=AnalysisLevel.MODULE,
    )

    result = analyze(graph, combined=True, projection_config=config)

    assert any(anomaly.kind == AnomalyKind.CYCLE_MEMBER for anomaly in result.anomalies)
    assert any(finding.kind == "cycle_member" for finding in result.findings)


def test_repo_policy_file_is_discovered_and_parsed(tmp_path: Path):
    """A nearby `topo.toml` file should define analysis defaults."""
    (tmp_path / "topo.toml").write_text(textwrap.dedent("""\
        [analysis]
        scope = "first-party"
        level = "module"
    """))
    project = tmp_path / "project"
    project.mkdir()

    policy = load_analysis_policy(project)

    assert policy is not None
    assert policy.scope == "first-party"
    assert policy.level == AnalysisLevel.MODULE
