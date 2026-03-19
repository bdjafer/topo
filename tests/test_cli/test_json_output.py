"""Tests for CLI JSON output format."""

import json
import textwrap
from pathlib import Path

from topo_cli.main import main


def test_json_output_structure(tmp_path: Path, capsys):
    """JSON output should have all expected top-level keys."""
    (tmp_path / "app.py").write_text(textwrap.dedent("""\
        def foo():
            bar()

        def bar():
            pass
    """))

    main([str(tmp_path), "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert "graph" in data
    assert "nodes" in data["graph"]
    assert "edges" in data["graph"]
    assert "modules" in data
    assert "roles" in data
    assert "anomalies" in data
    assert "cross_package_dependencies" in data
    assert "health" in data


def test_json_output_with_imports_layer(tmp_path: Path, capsys):
    """JSON output works with --edge-kind imports."""
    (tmp_path / "app.py").write_text(textwrap.dedent("""\
        import os
        import sys

        def main():
            pass
    """))

    main([str(tmp_path), "--json", "--edge-kind", "imports", "--level", "symbol"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert data["graph"]["nodes"] >= 2
    assert isinstance(data["roles"], list)


def test_json_roles_have_expected_fields(tmp_path: Path, capsys):
    """Each role entry should have node_id, role, degree, betweenness."""
    (tmp_path / "app.py").write_text("def f(): pass\n")

    main([str(tmp_path), "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    for role in data["roles"]:
        assert "node_id" in role
        assert "role" in role
        assert "degree" in role
        assert "betweenness" in role


def test_json_output_includes_cross_package_dependencies_and_health(tmp_path: Path, capsys):
    """JSON output should surface package flow and graph health metrics."""
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

    main([str(tmp_path), "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    alpha_to_beta = next(
        item
        for item in data["cross_package_dependencies"]
        if item["source_package"] == "alpha" and item["target_package"] == "beta"
    )

    assert alpha_to_beta["edge_counts"]["calls"] == 1
    assert alpha_to_beta["edge_counts"]["imports"] >= 1

    assert data["health"]["call_count"] == 1
    assert data["health"]["analyzed_node_count"] >= 2
    assert data["health"]["call_density"] > 0


def test_json_output_reports_reverse_dependency_descriptively(tmp_path: Path, capsys):
    """Bidirectional package flow should surface a structural reverse dependency."""
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    (alpha / "__init__.py").write_text("")
    (beta / "__init__.py").write_text("")
    (beta / "tools.py").write_text(textwrap.dedent("""\
        from alpha.core import run

        def helper():
            run()
    """))
    (alpha / "core.py").write_text(textwrap.dedent("""\
        from beta.tools import helper

        def run():
            helper()
    """))

    main([str(tmp_path), "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    alpha_to_beta = next(
        item
        for item in data["cross_package_dependencies"]
        if item["source_package"] == "alpha" and item["target_package"] == "beta"
    )
    beta_to_alpha = next(
        item
        for item in data["cross_package_dependencies"]
        if item["source_package"] == "beta" and item["target_package"] == "alpha"
    )

    assert alpha_to_beta["total"] > 0
    assert beta_to_alpha["total"] > 0
    assert "expected" not in alpha_to_beta
    assert "policy_status" not in beta_to_alpha
    assert any(finding["kind"] == "reverse_dependency" for finding in data["findings"])
    # All findings should have a stable issue ID
    for finding in data["findings"]:
        assert "id" in finding
        assert isinstance(finding["id"], str)


def test_json_output_loads_analysis_defaults(tmp_path: Path, capsys):
    """A repo-level `topo.toml` should still configure operational defaults."""
    (tmp_path / "topo.toml").write_text(textwrap.dedent("""\
        [analysis]
        level = "symbol"
    """))

    (tmp_path / "app.py").write_text(textwrap.dedent("""\
        def foo():
            bar()

        def bar():
            pass
    """))

    main([str(tmp_path), "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert data["scope"]["level"] == "symbol"
    assert data["graph"]["nodes"] >= 2
    assert "policy_file" not in data["scope"]
