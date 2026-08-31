"""Marker-based results injection for README.md and report/relazione.md (plan §9.4).

Prose outside the ``<!-- RESULTS:BEGIN -->`` / ``<!-- RESULTS:END -->`` markers
is hand-written once; everything between them is regenerated from
``results/tables/*.md`` — numbers are never typed twice.
"""

from __future__ import annotations

from pathlib import Path

MARKER_BEGIN = "<!-- RESULTS:BEGIN -->"
MARKER_END = "<!-- RESULTS:END -->"

# (table file, section heading) in presentation order.
SECTIONS = [
    ("main_results.md", "Classification accuracy"),
    ("fidelity.md", "Explanation fidelity (column masking, random-k control)"),
    ("stability.md", "Explanation stability (Jaccard@5)"),
    ("agreement.md", "Feature-space sensitivity and baseline agreement"),
    ("refusals.md", "Refused explanations"),
]

FIGURES = [
    ("fidelity_curve.png", "Fidelity+ vs k against the random-k control"),
    ("stability_heatmap.png", "Stability heatmap across seeds and hops"),
    ("agreement_comparison.png", "Feature-space (A vs B) and baseline agreement"),
]


def build_results_block(tables_dir: Path, figures_prefix: str, heading_level: int = 3) -> str:
    """Concatenate all tables (+ figure links) into one injectable block."""
    hashes = "#" * heading_level
    parts = []
    for filename, heading in SECTIONS:
        path = tables_dir / filename
        if not path.exists():
            continue
        parts.append(f"{hashes} {heading}\n\n{path.read_text().rstrip()}\n")
    figure_lines = [
        f"![{caption}]({figures_prefix}/{name})"
        for name, caption in FIGURES
        if (tables_dir.parent / "figures" / name).exists()
    ]
    if figure_lines:
        parts.append("\n".join(figure_lines) + "\n")
    return "\n".join(parts)


def inject(document: Path, block: str) -> None:
    """Replace the marked region of ``document`` with ``block``."""
    text = document.read_text()
    if MARKER_BEGIN not in text or MARKER_END not in text:
        raise RuntimeError(f"{document} lacks the RESULTS markers")
    head, rest = text.split(MARKER_BEGIN, 1)
    _, tail = rest.split(MARKER_END, 1)
    document.write_text(f"{head}{MARKER_BEGIN}\n{block}{MARKER_END}{tail}")


def refresh_documents(repo_root: Path = Path()) -> list[Path]:
    """Inject fresh tables into README.md and report/relazione.md."""
    tables_dir = repo_root / "results" / "tables"
    documents = [
        (repo_root / "README.md", "results/figures"),
        (repo_root / "report" / "relazione.md", "../results/figures"),
    ]
    refreshed = []
    for document, figures_prefix in documents:
        if document.exists():
            inject(document, build_results_block(tables_dir, figures_prefix))
            refreshed.append(document)
    return refreshed
