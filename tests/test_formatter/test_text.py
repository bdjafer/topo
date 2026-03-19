"""Tests for the text formatter consuming analysis.json dicts."""

from topo_formatter.text import format_text


def _minimal_analysis() -> dict:
    """Return a minimal valid analysis.json dict."""
    return {
        "scope": {"level": "module", "edge_kinds": ["calls"]},
        "coverage": {
            "analyzed_nodes": 5,
            "analyzed_edges": 3,
            "parsed_nodes": 10,
            "parsed_edges": 8,
        },
        "spectral": None,
        "architecture": {
            "modules": [
                {"id": 0, "label": "core", "size": 3, "members": ["core.a", "core.b", "core.c"],
                 "cohesion": 0.2, "separation": 0.8, "confidence": 0.7, "unassigned": False},
                {"id": 1, "label": "util", "size": 2, "members": ["util.x", "util.y"],
                 "cohesion": 0.3, "separation": 0.6, "confidence": 0.5, "unassigned": False},
            ],
            "dependencies": [
                {"source": 0, "target": 1, "weight": 2, "edge_kinds": {"calls": 2}},
            ],
            "silhouette": 0.65,
            "package_fallback": False,
        },
        "roles": [
            {"node_id": "core.a", "role": "hub", "degree": 4, "betweenness": 0.5,
             "in_degree": 2, "out_degree": 2},
            {"node_id": "util.x", "role": "utility", "degree": 2, "betweenness": 0.0,
             "in_degree": 2, "out_degree": 0},
            {"node_id": "core.b", "role": "regular", "degree": 1, "betweenness": 0.0,
             "in_degree": 0, "out_degree": 1},
        ],
        "issues": [
            {"id": "test-issue:core", "kind": "god_module", "title": "God module",
             "description": "Module core is too large.", "severity": 0.8,
             "severity_label": "high", "confidence": 0.9, "confidence_label": "high",
             "anchors": []},
        ],
        "health": {"modularity_q": 0.42},
    }


def test_format_text_includes_all_sections():
    """All major sections should appear in the text output."""
    data = _minimal_analysis()
    output = format_text(data)

    assert "Issues" in output
    assert "Architecture" in output
    assert "Critical Nodes" in output
    assert "Health" in output
    assert "Modularity Q: 0.420" in output


def test_format_text_renders_issues():
    data = _minimal_analysis()
    output = format_text(data)

    assert "test-issue:core" in output
    assert "Module core is too large." in output
    assert "[high]" in output


def test_format_text_renders_critical_roles():
    data = _minimal_analysis()
    output = format_text(data)

    assert "core.a" in output
    assert "HUB" in output
    assert "util.x" in output
    assert "UTILITY" in output
    # Regular roles should be excluded
    assert "core.b" not in output


def test_format_text_ignores_filter():
    data = _minimal_analysis()
    output_with = format_text(data)
    output_without = format_text(data, ignores={"test-issue:core": "accepted"})

    assert "test-issue:core" in output_with
    assert "test-issue:core" not in output_without
    assert "acknowledged" in output_without


def test_format_text_null_spectral_and_health():
    """Formatter should handle null spectral and health gracefully."""
    data = _minimal_analysis()
    data["spectral"] = None
    data["health"] = None

    output = format_text(data)
    assert "Issues" in output
    assert "Health" not in output


def test_format_text_verbose_shows_module_members():
    data = _minimal_analysis()
    output = format_text(data, verbose=True)

    assert "core (3 nodes)" in output
    assert "util (2 nodes)" in output


def test_format_text_diagnostics():
    data = _minimal_analysis()
    data["spectral"] = {
        "fiedler_value": 0.5,
        "eigenvalues": [0.5, 1.2, 2.0],
        "nodes_covered": 4,
        "coverage_ratio": 0.8,
        "components": 1,
        "largest_component_ratio": 1.0,
    }

    output = format_text(data, diagnostics=True)

    assert "Diagnostics" in output
    assert "Algebraic connectivity: 0.5000" in output
    assert "Spectral dimensions: 3" in output
    assert "Silhouette: 0.650" in output
