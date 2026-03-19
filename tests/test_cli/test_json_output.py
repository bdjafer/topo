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

    assert "coverage" in data
    assert "analyzed_nodes" in data["coverage"]
    assert "analyzed_edges" in data["coverage"]
    assert "architecture" in data
    assert "roles" in data
    assert "issues" in data
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

    assert data["coverage"]["analyzed_nodes"] >= 2
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


def test_json_output_includes_health(tmp_path: Path, capsys):
    """JSON output should surface graph health metrics."""
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

    assert "modularity_q" in data["health"]
    assert isinstance(data["health"]["modularity_q"], (int, float))


def test_json_output_reports_reverse_dependency_as_issue(tmp_path: Path, capsys):
    """Bidirectional package flow should surface a reverse-dependency issue."""
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

    assert any(issue["kind"] == "reverse_dependency" for issue in data["issues"])
    # All issues should have a stable ID
    for issue in data["issues"]:
        assert "id" in issue
        assert isinstance(issue["id"], str)


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
    assert data["coverage"]["analyzed_nodes"] >= 2
    assert "policy_file" not in data["scope"]
