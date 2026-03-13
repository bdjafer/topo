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
    assert "Analysis graph:" in captured.out
    assert "Projection:" in captured.out


def test_cli_summary_includes_health_and_cross_package_dependencies(tmp_path: Path, capsys):
    """CLI text output should surface health metrics and package flow."""
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

    main([str(tmp_path)])
    captured = capsys.readouterr()

    assert "Health:" in captured.out
    assert "Package flow:" in captured.out
    assert "alpha -> beta:" in captured.out
