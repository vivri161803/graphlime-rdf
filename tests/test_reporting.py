"""M8 tests: figure generation and marker-based README/report injection."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphlime_rdf.config import AgreementRecord, FidelityRecord, StabilityRecord
from graphlime_rdf.pipeline import write_jsonl
from graphlime_rdf.reporting.figures import generate_all_figures
from graphlime_rdf.reporting.readme import (
    MARKER_BEGIN,
    MARKER_END,
    build_results_block,
    inject,
    refresh_documents,
)


def _provenance() -> dict[str, object]:
    return {"seed": 0, "config_hash": "x", "git_commit": "abc1234"}


def _fake_results(results: Path) -> None:
    for ds in ["aifb", "mutag"]:
        write_jsonl(
            [
                FidelityRecord(
                    dataset=ds, node_id=n, k=k, fidelity_plus=0.4 + 0.01 * k,
                    fidelity_minus=0.1, random_plus=0.05, random_minus=0.3,
                    **_provenance(),  # type: ignore[arg-type]
                )
                for n in [1, 2]
                for k in [1, 2, 5]
            ],
            results / f"fidelity_{ds}.jsonl",
        )
        write_jsonl(
            [
                StabilityRecord(
                    dataset=ds, node_id=1, kind="seeds", variant_a="seed=0",
                    variant_b="seed=1", k=5, jaccard=0.7, config_hash="x",
                    git_commit="abc1234",
                ),
                StabilityRecord(
                    dataset=ds, node_id=1, kind="hops", variant_a="hops=1",
                    variant_b="hops=2", k=5, jaccard=0.5, config_hash="x",
                    git_commit="abc1234",
                ),
            ],
            results / f"stability_{ds}.jsonl",
        )
        write_jsonl(
            [
                AgreementRecord(
                    dataset=ds, node_id=1, kind="feature_space", k=5, jaccard=0.6,
                    **_provenance(),  # type: ignore[arg-type]
                ),
                AgreementRecord(
                    dataset=ds, node_id=1, kind="baseline", k=5, jaccard=0.3,
                    **_provenance(),  # type: ignore[arg-type]
                ),
            ],
            results / f"agreement_{ds}.jsonl",
        )


def test_figures_generated(tmp_path: Path) -> None:
    _fake_results(tmp_path)
    written = generate_all_figures(tmp_path)
    assert [p.name for p in written] == [
        "fidelity_curve.png", "stability_heatmap.png", "agreement_comparison.png",
    ]
    for path in written:
        assert path.exists() and path.stat().st_size > 1000


def test_inject_replaces_only_marked_region(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text(f"# Title\n\nintro\n\n{MARKER_BEGIN}\nold\n{MARKER_END}\n\noutro\n")
    inject(doc, "NEW CONTENT\n")
    text = doc.read_text()
    assert "old" not in text
    assert "NEW CONTENT" in text
    assert text.startswith("# Title\n\nintro\n")
    assert text.endswith("\n\noutro\n")
    # Idempotent: injecting the same block twice changes nothing.
    inject(doc, "NEW CONTENT\n")
    assert doc.read_text() == text


def test_inject_requires_markers(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("no markers here")
    with pytest.raises(RuntimeError, match="markers"):
        inject(doc, "x")


def test_refresh_documents_end_to_end(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _fake_results(results)
    tables_dir = results / "tables"
    tables_dir.mkdir(parents=True)
    (tables_dir / "main_results.md").write_text("| dataset |\n|---|\n| AIFB |\n")
    readme = tmp_path / "README.md"
    readme.write_text("# P\nprose only, no markers\n")
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "relazione.md").write_text(f"# R\n{MARKER_BEGIN}\n{MARKER_END}\n")

    refreshed = refresh_documents(tmp_path)
    assert refreshed == [report_dir / "relazione.md"]
    report_text = (report_dir / "relazione.md").read_text()
    assert "Classification accuracy" in report_text
    assert "| AIFB |" in report_text
    # README carries no numbers and must be left untouched (ADR-010).
    assert readme.read_text() == "# P\nprose only, no markers\n"

    block = build_results_block(tables_dir, "results/figures")
    assert "### Classification accuracy" in block
