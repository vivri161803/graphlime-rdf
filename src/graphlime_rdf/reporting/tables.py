"""JSONL → markdown result tables (plan §9.3, M8 crown deliverable).

The only inputs are ``results/*.jsonl`` and ``runs/final/*/manifest.json``;
tables are never hand-edited. Regeneration is idempotent: same inputs, same
bytes. All aggregations iterate in sorted order.
"""

from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from graphlime_rdf.config import (
    AgreementRecord,
    ExplanationRecord,
    FidelityRecord,
    RefusalRecord,
    RunManifest,
    StabilityRecord,
)
from graphlime_rdf.pipeline import read_jsonl

DATASETS = ["aifb", "mutag"]


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _mean_std(values: list[float]) -> str:
    if len(values) == 1:
        return _fmt(values[0])
    return f"{_fmt(statistics.mean(values))} ± {_fmt(statistics.stdev(values))}"


def _final_manifests(runs_dir: Path) -> dict[str, list[RunManifest]]:
    by_dataset: dict[str, list[RunManifest]] = defaultdict(list)
    for manifest_path in sorted(runs_dir.glob("*/manifest.json")):
        manifest = RunManifest.model_validate_json(manifest_path.read_text())
        by_dataset[manifest.dataset].append(manifest)
    return by_dataset


def _checkpoint_digest(path: Path) -> str:
    if not path.exists():
        return "—"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main_results_table(runs_dir: Path, checkpoints_dir: Path) -> str:
    by_dataset = _final_manifests(runs_dir)
    lines = [
        "| dataset | test accuracy (mean ± std, 5 seeds) | per-seed | best seed | best accuracy | checkpoint sha256 |",
        "|---|---|---|---|---|---|",
    ]
    for ds in DATASETS:
        manifests = sorted(by_dataset.get(ds, []), key=lambda m: m.seed)
        if not manifests:
            continue
        accs = [m.final_test_accuracy for m in manifests]
        best = max(manifests, key=lambda m: m.final_test_accuracy)
        per_seed = ", ".join(_fmt(a) for a in accs)
        digest = _checkpoint_digest(checkpoints_dir / f"{ds}_best.pt")
        lines.append(
            f"| {ds.upper()} | {_mean_std(accs)} | {per_seed} | {best.seed} "
            f"| {_fmt(best.final_test_accuracy)} | `{digest}` |"
        )
    return "\n".join(lines) + "\n"


def fidelity_table(results_dir: Path) -> str:
    lines = [
        "| dataset | k | fidelity+ | random+ | fidelity− | random− | nodes |",
        "|---|---|---|---|---|---|---|",
    ]
    for ds in DATASETS:
        path = results_dir / f"fidelity_{ds}.jsonl"
        if not path.exists():
            continue
        records = read_jsonl(path, FidelityRecord)
        by_k: dict[int, list[FidelityRecord]] = defaultdict(list)
        for record in records:
            by_k[record.k].append(record)
        for k in sorted(by_k):
            rows = by_k[k]
            lines.append(
                f"| {ds.upper()} | {k} "
                f"| {_fmt(statistics.mean(r.fidelity_plus for r in rows))} "
                f"| {_fmt(statistics.mean(r.random_plus for r in rows))} "
                f"| {_fmt(statistics.mean(r.fidelity_minus for r in rows))} "
                f"| {_fmt(statistics.mean(r.random_minus for r in rows))} "
                f"| {len(rows)} |"
            )
    return "\n".join(lines) + "\n"


def stability_table(results_dir: Path) -> str:
    lines = [
        "| dataset | comparison | pair | mean Jaccard@5 | pairs |",
        "|---|---|---|---|---|",
    ]
    for ds in DATASETS:
        path = results_dir / f"stability_{ds}.jsonl"
        if not path.exists():
            continue
        records = read_jsonl(path, StabilityRecord)
        grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for record in records:
            grouped[record.kind, record.variant_a, record.variant_b].append(record.jaccard)
        for (kind, a, b), values in sorted(grouped.items()):
            lines.append(
                f"| {ds.upper()} | {kind} | {a} vs {b} "
                f"| {_fmt(statistics.mean(values))} | {len(values)} |"
            )
    return "\n".join(lines) + "\n"


def agreement_table(results_dir: Path) -> str:
    labels = {
        "feature_space": "GraphLIME: space A vs space B",
        "baseline": "GraphLIME vs GNNExplainer",
    }
    lines = [
        "| dataset | comparison | mean Jaccard@5 (predicates) | nodes |",
        "|---|---|---|---|",
    ]
    for ds in DATASETS:
        path = results_dir / f"agreement_{ds}.jsonl"
        if not path.exists():
            continue
        records = read_jsonl(path, AgreementRecord)
        grouped: dict[str, list[float]] = defaultdict(list)
        for record in records:
            grouped[record.kind].append(record.jaccard)
        for kind in ["feature_space", "baseline"]:
            if kind in grouped:
                values = grouped[kind]
                lines.append(
                    f"| {ds.upper()} | {labels[kind]} "
                    f"| {_fmt(statistics.mean(values))} | {len(values)} |"
                )
    return "\n".join(lines) + "\n"


def refusals_table(results_dir: Path) -> str:
    lines = [
        "| dataset | explained | refused | refusal rate | reasons |",
        "|---|---|---|---|---|",
    ]
    for ds in DATASETS:
        refusal_path = results_dir / f"refusals_{ds}.jsonl"
        explained_path = results_dir / f"explanations_{ds}.jsonl"
        if not explained_path.exists():
            continue
        explained = len(read_jsonl(explained_path, ExplanationRecord))
        refusals = read_jsonl(refusal_path, RefusalRecord) if refusal_path.exists() else []
        total = explained + len(refusals)
        reasons: dict[str, int] = defaultdict(int)
        for record in refusals:
            reasons[record.reason.split(" < ")[-1]] += 1
        reason_text = (
            "; ".join(f"{count}× {reason}" for reason, count in sorted(reasons.items()))
            or "—"
        )
        rate = len(refusals) / total if total else 0.0
        lines.append(
            f"| {ds.upper()} | {explained} | {len(refusals)} | {_fmt(rate)} | {reason_text} |"
        )
    return "\n".join(lines) + "\n"


TABLES: dict[str, Callable[[Path, Path, Path], str]] = {
    "main_results.md": lambda results, runs, ckpt: main_results_table(runs, ckpt),
    "fidelity.md": lambda results, runs, ckpt: fidelity_table(results),
    "stability.md": lambda results, runs, ckpt: stability_table(results),
    "agreement.md": lambda results, runs, ckpt: agreement_table(results),
    "refusals.md": lambda results, runs, ckpt: refusals_table(results),
}


def generate_all_tables(
    results_dir: Path = Path("results"),
    runs_dir: Path = Path("runs/final"),
    checkpoints_dir: Path = Path("checkpoints"),
) -> list[Path]:
    """Write every table under ``results/tables/`` and return the paths."""
    out_dir = results_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, build in sorted(TABLES.items()):
        content = build(results_dir, runs_dir, checkpoints_dir)
        path = out_dir / name
        path.write_text(content)
        written.append(path)
    return written
