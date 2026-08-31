"""Fast sanity tests for the ground-truth generator (M6 support)."""

from __future__ import annotations

import torch

from graphlime_rdf.config import FeatureSpaceConfig, SyntheticConfig
from graphlime_rdf.data.synthetic import generate_ground_truth_graph
from graphlime_rdf.features import build_features


def test_generator_deterministic() -> None:
    g1 = generate_ground_truth_graph(SyntheticConfig())
    g2 = generate_ground_truth_graph(SyntheticConfig())
    assert g1.node_names == g2.node_names
    assert torch.equal(g1.edge_index, g2.edge_index)
    assert torch.equal(g1.labels, g2.labels)


def test_label_rule_holds_at_zero_noise() -> None:
    graph = generate_ground_truth_graph(SyntheticConfig())
    x, vocab = build_features(
        graph, FeatureSpaceConfig(kind="predicate", min_support=1)
    )
    j = vocab.index("out:syn:p7")
    labeled = graph.labels >= 0
    has_target = x[:, j] > 0
    assert bool((has_target[labeled] == (graph.labels[labeled] == 1)).all())


def test_noise_breaks_rule_for_a_fraction() -> None:
    cfg = SyntheticConfig(noise_rate=0.2)
    graph = generate_ground_truth_graph(cfg)
    x, vocab = build_features(
        graph, FeatureSpaceConfig(kind="predicate", min_support=1)
    )
    j = vocab.index("out:syn:p7")
    labeled = graph.labels >= 0
    agree = (x[:, j][labeled] > 0) == (graph.labels[labeled] == 1)
    rate = float(agree.float().mean())
    assert 0.7 < rate < 0.9  # ≈ 1 − noise_rate


def test_distractors_present_and_correlated() -> None:
    cfg = SyntheticConfig(num_distractors=2, distractor_strength=0.9)
    graph = generate_ground_truth_graph(cfg)
    x, vocab = build_features(
        graph, FeatureSpaceConfig(kind="predicate", min_support=1)
    )
    j_t = vocab.index("out:syn:p7")
    j_d = vocab.index("out:syn:d0")
    labeled = graph.labels >= 0
    target = x[labeled, j_t] > 0
    distractor = x[labeled, j_d] > 0
    # Correlated: distractor much more frequent among target-carriers.
    p_with = float(distractor[target].float().mean())
    p_without = float(distractor[~target].float().mean())
    assert p_with > 0.75 and p_without < 0.25


def test_split_masks_partition_entities() -> None:
    graph = generate_ground_truth_graph(SyntheticConfig())
    labeled = graph.labels >= 0
    assert bool(((graph.train_mask | graph.test_mask) == labeled).all())
    assert not bool((graph.train_mask & graph.test_mask).any())
