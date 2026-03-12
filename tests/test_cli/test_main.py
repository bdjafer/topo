"""Tests for the CLI entry point."""

import textwrap
from pathlib import Path

from topo_cli.main import main


def test_cli_runs_on_project(tmp_path: Path, capsys):
    """CLI produces output when pointed at a Python project."""
    (tmp_path / "app.py").write_text(textwrap.dedent("""\
        def foo():
            bar()

        def bar():
            pass
    """))

    main([str(tmp_path)])
    captured = capsys.readouterr()
    assert "CodeGraph" in captured.out
    assert "nodes" in captured.out
