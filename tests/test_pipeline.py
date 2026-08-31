"""M7 pipeline tests: records validate, carry provenance, and are deterministic."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphlime_rdf.config import (
    ExperimentConfig,
    ExplanationRecord,
    FidelityRecord,
    GraphLIMEConfig,
    RefusalRecord,
    SyntheticConfig,
)
from graphlime_rdf.data.loader import RDFGraph
from graphlime_rdf.data.synthetic import generate_ground_truth_graph
from graphlime_rdf.pipeline import (
    explanation_records,
    fidelity_records,
    read_jsonl,
    write_jsonl,
)
from graphlime_rdf.training import TrainResult, train_run

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def syn_result() -> tuple[TrainResult, RDFGraph, ExperimentConfig]:
    graph = generate_ground_truth_graph(SyntheticConfig(num_entities=80))
    cfg = ExperimentConfig.from_yaml(REPO / "configs" / "synthetic.yaml")
    cfg = cfg.model_copy(
        update={"graphlime": GraphLIMEConfig(min_neighborhood=5, max_neighborhood=60)}
    )
    return train_run(graph, cfg, seed=0), graph, cfg


def test_explanation_records_validate_and_carry_provenance(
    syn_result: tuple[TrainResult, RDFGraph, ExperimentConfig],
) -> None:
    result, graph, cfg = syn_result
    explanations, refusals = explanation_records(result, graph, cfg)
    assert explanations, "no explanations produced"
    for record in explanations:
        assert record.config_hash == cfg.config_hash()
        assert record.git_commit
        assert record.seed == 0
        assert 0 <= record.predicted_prob <= 1
        for name, beta in record.top_features:
            assert not name.isdigit(), "integer id where a name is expected"
            assert ":" in name
            assert beta > 0
    for record in refusals:
        assert "min_neighborhood" in record.reason


def test_jsonl_round_trip(
    syn_result: tuple[TrainResult, RDFGraph, ExperimentConfig], tmp_path: Path
) -> None:
    result, graph, cfg = syn_result
    explanations, refusals = explanation_records(result, graph, cfg)
    write_jsonl(explanations, tmp_path / "explanations.jsonl")
    write_jsonl(refusals, tmp_path / "refusals.jsonl")
    assert read_jsonl(tmp_path / "explanations.jsonl", ExplanationRecord) == explanations
    assert read_jsonl(tmp_path / "refusals.jsonl", RefusalRecord) == refusals


def test_stability_and_agreement_records(
    syn_result: tuple[TrainResult, RDFGraph, ExperimentConfig],
) -> None:
    from graphlime_rdf.pipeline import agreement_records, stability_records

    result, graph, cfg = syn_result
    other = train_run(graph, cfg, seed=1)
    stability = stability_records({0: result, 1: other}, graph, cfg)
    kinds = {record.kind for record in stability}
    assert kinds == {"seeds", "hops"}
    for record in stability:
        assert 0.0 <= record.jaccard <= 1.0
        assert record.config_hash == cfg.config_hash()

    agreement = agreement_records(result, graph, cfg, baseline_epochs=5)
    assert {record.kind for record in agreement} == {"feature_space", "baseline"}
    for record in agreement:
        assert 0.0 <= record.jaccard <= 1.0


def test_synthetic_dataset_loadable_by_name() -> None:
    from graphlime_rdf.data.loader import load_rdf_graph

    g1 = load_rdf_graph("synthetic")
    g2 = generate_ground_truth_graph(SyntheticConfig())
    assert g1.node_names == g2.node_names


def test_fidelity_records_deterministic(
    syn_result: tuple[TrainResult, RDFGraph, ExperimentConfig],
) -> None:
    result, graph, cfg = syn_result
    r1 = fidelity_records(result, graph, cfg)
    r2 = fidelity_records(result, graph, cfg)
    assert r1 == r2
    assert {record.k for record in r1} == {1, 2, 5, 10}
    for record in r1:
        FidelityRecord.model_validate(record.model_dump())
