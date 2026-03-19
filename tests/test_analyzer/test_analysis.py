"""Tests for the top-level analysis orchestrator."""

import textwrap
from pathlib import Path

from topo_analyzer.analysis import (
    analyze,
    _aggregate_roles_to_report_level,
)
from topo_analyzer.anomalies import AnomalyKind
from topo_analyzer.projection import (
    AnalysisLevel,
    AnalysisProjectionConfig,
    load_analysis_policy,
)
from topo_formatter.text import format_text
from topo_analyzer.roles import RoleAssignment, StructuralRole
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
    assert result.graph.node_count >= 2
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
    data = result.to_dict()
    summary = format_text(data)

    assert "Issues" in summary
    assert "Health" in summary
    assert "scope" in data
    assert "issues" in data
    assert "architecture" in data


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


def test_findings_have_stable_ids(tmp_path: Path):
    """Every finding should have a deterministic issue ID."""
    root = _make_reverse_two_package_project(tmp_path)
    graph = parse_python_project(root)
    config = AnalysisProjectionConfig.for_analysis(
        edge_kind=EdgeKind.CALLS,
        combined=True,
        level=AnalysisLevel.MODULE,
    )

    result = analyze(graph, combined=True, projection_config=config)

    for finding in result.findings:
        assert finding.id, f"Finding {finding.kind} has no id"
        assert ":" in finding.id, f"Finding id should be kind:target, got {finding.id}"

    reverse = [f for f in result.findings if f.kind == "reverse_dependency"]
    if reverse:
        assert reverse[0].id == "reverse-dependency:alpha,beta"


def test_orphans_appear_as_findings(tmp_path: Path):
    """Orphan nodes should be surfaced as actionable findings."""
    root = _make_two_package_project(tmp_path)
    # Add an isolated module with code but no connections to anything
    (root / "alpha" / "isolated.py").write_text("def lonely(): pass\n")
    graph = parse_python_project(root)
    config = AnalysisProjectionConfig.for_analysis(
        edge_kind=EdgeKind.CALLS,
        combined=True,
        level=AnalysisLevel.MODULE,
    )

    result = analyze(graph, combined=True, projection_config=config)

    orphan_findings = [f for f in result.findings if f.kind == "orphan"]
    assert orphan_findings, "Expected at least one orphan finding"
    for f in orphan_findings:
        assert f.id.startswith("orphan:")


def test_ignore_filtering_hides_findings(tmp_path: Path):
    """summary() should filter findings matching ignore keys."""
    root = _make_reverse_two_package_project(tmp_path)
    graph = parse_python_project(root)
    config = AnalysisProjectionConfig.for_analysis(
        edge_kind=EdgeKind.CALLS,
        combined=True,
        level=AnalysisLevel.MODULE,
    )

    result = analyze(graph, combined=True, projection_config=config)
    assert result.findings, "Need findings to test ignore filtering"

    # Ignore the first finding
    first_id = result.findings[0].id
    data = result.to_dict()
    summary_with = format_text(data)
    summary_without = format_text(data, ignores={first_id: "test justification"})

    assert first_id in summary_with
    assert first_id not in summary_without
    assert "acknowledged" in summary_without


def test_policy_loads_ignore_section(tmp_path: Path):
    """topo.toml [analysis.ignore] should populate policy.ignores."""
    (tmp_path / "topo.toml").write_text(textwrap.dedent("""\
        [analysis]
        scope = "first-party"

        [analysis.ignore]
        "spectral-outlier:foo.bar" = "expected by design"
    """))
    project = tmp_path / "project"
    project.mkdir()

    policy = load_analysis_policy(project)

    assert policy is not None
    assert policy.ignores == {"spectral-outlier:foo.bar": "expected by design"}


def test_policy_rejects_non_string_ignore_values(tmp_path: Path):
    """Non-string values in [analysis.ignore] should raise ValueError."""
    (tmp_path / "topo.toml").write_text(textwrap.dedent("""\
        [analysis]
        scope = "first-party"

        [analysis.ignore]
        "some-issue:foo" = 42
    """))
    project = tmp_path / "project"
    project.mkdir()

    import pytest
    with pytest.raises(ValueError, match="must be strings"):
        load_analysis_policy(project)


# --- Orphan unanimity tests ---


def test_orphan_not_propagated_from_single_child():
    """A module with one ORPHAN child and one non-ORPHAN child should NOT be ORPHAN."""
    symbol_roles = [
        RoleAssignment(node_id="mod.fn_a", role=StructuralRole.ORPHAN, degree=0, betweenness=0.0, in_degree=0, out_degree=0),
        RoleAssignment(node_id="mod.fn_b", role=StructuralRole.REGULAR, degree=3, betweenness=0.0, in_degree=1, out_degree=2),
    ]
    symbol_to_report = {"mod.fn_a": "mod", "mod.fn_b": "mod"}

    result = _aggregate_roles_to_report_level(symbol_roles, symbol_to_report)

    assert len(result) == 1
    assert result[0].node_id == "mod"
    assert result[0].role == StructuralRole.REGULAR
    assert result[0].degree == 3  # Max across children


def test_orphan_propagated_when_all_children_orphan():
    """A module whose children are ALL orphans should be ORPHAN."""
    symbol_roles = [
        RoleAssignment(node_id="dead.fn_a", role=StructuralRole.ORPHAN, degree=0, betweenness=0.0, in_degree=0, out_degree=0),
        RoleAssignment(node_id="dead.fn_b", role=StructuralRole.ORPHAN, degree=0, betweenness=0.0, in_degree=0, out_degree=0),
    ]
    symbol_to_report = {"dead.fn_a": "dead", "dead.fn_b": "dead"}

    result = _aggregate_roles_to_report_level(symbol_roles, symbol_to_report)

    assert len(result) == 1
    assert result[0].role == StructuralRole.ORPHAN


def test_orphan_hub_child_wins_over_orphan_siblings():
    """A HUB child should take priority even when other siblings are ORPHAN."""
    symbol_roles = [
        RoleAssignment(node_id="pkg.hub_fn", role=StructuralRole.HUB, degree=20, betweenness=0.1, in_degree=10, out_degree=10),
        RoleAssignment(node_id="pkg.dead_fn", role=StructuralRole.ORPHAN, degree=0, betweenness=0.0, in_degree=0, out_degree=0),
        RoleAssignment(node_id="pkg.util_fn", role=StructuralRole.UTILITY, degree=5, betweenness=0.0, in_degree=5, out_degree=0),
    ]
    symbol_to_report = {"pkg.hub_fn": "pkg", "pkg.dead_fn": "pkg", "pkg.util_fn": "pkg"}

    result = _aggregate_roles_to_report_level(symbol_roles, symbol_to_report)

    assert len(result) == 1
    assert result[0].role == StructuralRole.HUB
    assert result[0].degree == 20


# --- Spectral outlier quality gate tests ---


def test_spectral_outlier_suppressed_weak_clustering(tmp_path: Path):
    """When largest_module_ratio >= 0.8, spectral outlier findings should be suppressed."""
    # Create a tiny project where one mega-module dominates
    root = _make_two_package_project(tmp_path)
    graph = parse_python_project(root)
    config = AnalysisProjectionConfig.for_analysis(
        edge_kind=EdgeKind.CALLS,
        combined=True,
        level=AnalysisLevel.MODULE,
    )
    result = analyze(graph, combined=True, projection_config=config)

    # Small project => weak clustering => spectral outliers should be suppressed
    spectral_findings = [f for f in result.findings if f.kind == "spectral_outlier"]
    assert len(spectral_findings) == 0


def test_orphans_single_function_module_still_detected(tmp_path: Path):
    """A module with a single orphan function should still be flagged as orphan."""
    root = _make_two_package_project(tmp_path)
    (root / "alpha" / "isolated.py").write_text("def lonely(): pass\n")
    graph = parse_python_project(root)
    config = AnalysisProjectionConfig.for_analysis(
        edge_kind=EdgeKind.CALLS,
        combined=True,
        level=AnalysisLevel.MODULE,
    )
    result = analyze(graph, combined=True, projection_config=config)

    orphan_findings = [f for f in result.findings if f.kind == "orphan"]
    assert orphan_findings, "Single-function isolated module should still be orphan"
