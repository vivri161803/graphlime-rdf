"""Figures for the report (plan M8): generated only from ``results/*.jsonl``.

Deterministic output: fixed figure sizes, sorted iteration, PNG metadata
stripped of dates.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from graphlime_rdf.config import AgreementRecord, FidelityRecord, StabilityRecord
from graphlime_rdf.pipeline import read_jsonl

DATASETS = ["aifb", "mutag"]


def _save(fig: Figure, out_path: Path) -> None:
    """Deterministic PNG: fixed dpi, no embedded date."""
    fig.savefig(out_path, dpi=150, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def fidelity_curve(results_dir: Path, out_path: Path) -> None:
    """Mean fidelity+ (solid) vs random+ (dashed) as a function of k."""
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(9, 3.4), sharey=True)
    for ax, ds in zip(axes, DATASETS, strict=True):
        records = read_jsonl(results_dir / f"fidelity_{ds}.jsonl", FidelityRecord)
        by_k: dict[int, list[FidelityRecord]] = defaultdict(list)
        for record in records:
            by_k[record.k].append(record)
        ks = sorted(by_k)
        top = [statistics.mean(r.fidelity_plus for r in by_k[k]) for k in ks]
        rand = [statistics.mean(r.random_plus for r in by_k[k]) for k in ks]
        ax.plot(ks, top, marker="o", label="top-k (GraphLIME)")
        ax.plot(ks, rand, marker="s", linestyle="--", label="random-k")
        ax.set_title(ds.upper())
        ax.set_xlabel("k (features masked)")
        ax.set_xticks(ks)
    axes[0].set_ylabel("fidelity+ (drop in p)")
    axes[0].legend()
    fig.tight_layout()
    _save(fig, out_path)


def stability_heatmap(results_dir: Path, out_path: Path) -> None:
    """Mean Jaccard@5 per (dataset, comparison pair) as an annotated grid."""
    grouped: dict[str, dict[str, float]] = {}
    for ds in DATASETS:
        records = read_jsonl(results_dir / f"stability_{ds}.jsonl", StabilityRecord)
        cells: dict[str, list[float]] = defaultdict(list)
        for record in records:
            cells[f"{record.variant_a}\nvs {record.variant_b}"].append(record.jaccard)
        grouped[ds] = {pair: statistics.mean(vals) for pair, vals in sorted(cells.items())}

    pairs = sorted({pair for cells in grouped.values() for pair in cells})
    matrix = [[grouped[ds].get(pair, float("nan")) for pair in pairs] for ds in DATASETS]

    fig, ax = plt.subplots(figsize=(1.4 * len(pairs) + 2, 2.8))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(DATASETS)), [ds.upper() for ds in DATASETS])
    ax.set_xticks(range(len(pairs)), pairs, fontsize=8)
    for i in range(len(DATASETS)):
        for j in range(len(pairs)):
            value = matrix[i][j]
            if value == value:  # not NaN
                ax.text(
                    j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if value < 0.6 else "black", fontsize=9,
                )
    fig.colorbar(image, label="mean Jaccard@5")
    ax.set_title("Explanation stability")
    fig.tight_layout()
    _save(fig, out_path)


def agreement_comparison(results_dir: Path, out_path: Path) -> None:
    """Grouped bars: mean agreement per dataset for A-vs-B and vs-GNNExplainer."""
    kinds = ["feature_space", "baseline"]
    labels = {"feature_space": "space A vs B", "baseline": "vs GNNExplainer"}
    means: dict[str, list[float]] = {kind: [] for kind in kinds}
    for ds in DATASETS:
        records = read_jsonl(results_dir / f"agreement_{ds}.jsonl", AgreementRecord)
        by_kind: dict[str, list[float]] = defaultdict(list)
        for record in records:
            by_kind[record.kind].append(record.jaccard)
        for kind in kinds:
            means[kind].append(statistics.mean(by_kind[kind]) if by_kind.get(kind) else 0.0)

    x = range(len(DATASETS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for offset, kind in zip([-width / 2, width / 2], kinds, strict=True):
        ax.bar(
            [i + offset for i in x], means[kind], width, label=labels[kind]
        )
    ax.set_xticks(list(x), [ds.upper() for ds in DATASETS])
    ax.set_ylabel("mean Jaccard@5 (predicates)")
    ax.set_ylim(0, 1)
    ax.set_title("Explanation agreement")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)


def generate_all_figures(results_dir: Path = Path("results")) -> list[Path]:
    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, build in [
        ("fidelity_curve.png", fidelity_curve),
        ("stability_heatmap.png", stability_heatmap),
        ("agreement_comparison.png", agreement_comparison),
    ]:
        path = out_dir / name
        build(results_dir, path)
        written.append(path)
    return written
