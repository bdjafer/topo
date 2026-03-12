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


def test_json_output_with_imports_layer(tmp_path: Path, capsys):
    """JSON output works with --edge-kind imports."""
    (tmp_path / "app.py").write_text(textwrap.dedent("""\
        import os
        import sys

        def main():
            pass
    """))

    main([str(tmp_path), "--json", "--edge-kind", "imports"])
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
