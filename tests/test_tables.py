"""M8 tests: tables generated only from JSONL/manifests, idempotent."""

from __future__ import annotations

from pathlib import Path

from graphlime_rdf.config import (
    ExperimentConfig,
    FeatureSpaceConfig,
    FidelityRecord,
    RefusalRecord,
    RunManifest,
)
from graphlime_rdf.pipeline import write_jsonl
from graphlime_rdf.reporting.tables import generate_all_tables


def _fake_inputs(base: Path) -> tuple[Path, Path, Path]:
    results = base / "results"
    runs = base / "runs" / "final"
    ckpt = base / "checkpoints"
    for seed, acc in [(0, 0.9), (1, 0.8)]:
        cfg = ExperimentConfig(
            dataset="aifb", feature_space=FeatureSpaceConfig(kind="predicate")
        )
        manifest = RunManifest(
            run_id=f"t_{seed}",
            dataset="aifb",
            seed=seed,
            git_commit="abc1234",
            resolved_config=cfg,
            library_versions={"torch": "0"},
            final_test_accuracy=acc,
            epochs=1,
        )
        run_dir = runs / manifest.run_id
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(manifest.model_dump_json())
    write_jsonl(
        [
            FidelityRecord(
                dataset="aifb", node_id=1, k=2, fidelity_plus=0.5, fidelity_minus=0.1,
                random_plus=0.05, random_minus=0.4, seed=0, config_hash="x",
                git_commit="abc1234",
            )
        ],
        results / "fidelity_aifb.jsonl",
    )
    write_jsonl(
        [
            RefusalRecord(
                dataset="aifb", node_id=2, reason="neighborhood has 3 nodes < min_neighborhood=10",
                neighborhood_size=3, seed=0, config_hash="x", git_commit="abc1234",
            )
        ],
        results / "refusals_aifb.jsonl",
    )
    write_jsonl([], results / "explanations_aifb.jsonl")
    ckpt.mkdir()
    return results, runs, ckpt


def test_tables_generated_and_idempotent(tmp_path: Path) -> None:
    results, runs, ckpt = _fake_inputs(tmp_path)
    written1 = generate_all_tables(results, runs, ckpt)
    contents1 = {p.name: p.read_text() for p in written1}
    written2 = generate_all_tables(results, runs, ckpt)
    contents2 = {p.name: p.read_text() for p in written2}
    assert contents1 == contents2  # idempotent
    assert set(contents1) == {
        "main_results.md", "fidelity.md", "stability.md", "agreement.md", "refusals.md",
    }
    main = contents1["main_results.md"]
    assert "AIFB" in main and "0.8500 ± 0.0707" in main
    fid = contents1["fidelity.md"]
    assert "| AIFB | 2 | 0.5000 | 0.0500 | 0.1000 | 0.4000 | 1 |" in fid
    refusals = contents1["refusals.md"]
    assert "min_neighborhood=10" in refusals
