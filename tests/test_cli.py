"""CLI regression tests: command functions must work when called directly
(the repro path), not only through typer's argument resolution."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from graphlime_rdf.cli import app, train

REPO = Path(__file__).resolve().parents[1]


def test_train_callable_directly_with_explicit_args(
    tmp_path: Path, monkeypatch: object
) -> None:
    """repro calls train() as a plain function — OptionInfo defaults must never leak."""
    getattr(monkeypatch, "chdir")(tmp_path)  # runs/ land in tmp  # noqa: B009
    train(REPO / "configs" / "synthetic.yaml", final=False, checkpoint_dir=tmp_path / "ckpt")
    assert (tmp_path / "ckpt" / "synthetic_best.pt").exists()
    assert list((tmp_path / "runs" / "scratch").glob("*/manifest.json"))


def test_info_command() -> None:
    result = CliRunner().invoke(app, ["info"])
    assert result.exit_code == 0
    assert "torch==" in result.output
