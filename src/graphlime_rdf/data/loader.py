"""Deterministic loader: benchmark RDF dumps → typed :class:`RDFGraph` (plan M1).

Parses the raw ``<name>_stripped.nt.gz`` and TSV splits directly (ADR-001):
node / relation / label indices are assigned in lexicographic order of their
canonical string forms, so the mapping is identical across processes and
``PYTHONHASHSEED`` values. PyG is used only to download the standard archive.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from graphlime_rdf.data.leakage import assert_no_leakage

DOWNLOAD_URL = "https://data.dgl.ai/dataset/{}.tgz"

# TSV column headers used by the benchmark split files.
_NODE_HEADER = {"aifb": "person", "mutag": "bond"}
_LABEL_HEADER = {"aifb": "label_affiliation", "mutag": "label_mutagenic"}

# Published benchmark figures (R-GCN paper / PyG stats) asserted at load time.
EXPECTED_NUM_NODES = {"aifb": 8285, "mutag": 23644}
EXPECTED_NUM_CLASSES = {"aifb": 4, "mutag": 2}
EXPECTED_NUM_TRAIN = {"aifb": 140, "mutag": 272}
EXPECTED_NUM_TEST = {"aifb": 36, "mutag": 68}


@dataclass(frozen=True)
class RDFGraph:
    """A heterogeneous RDF graph with readable names and benchmark splits.

    ``edge_index``/``edge_type`` hold the *forward* triples only; the model
    layer adds inverse relations. ``labels`` is ``-1`` for unlabeled nodes.
    """

    dataset: str
    edge_index: Tensor  # (2, E) long
    edge_type: Tensor  # (E,) long
    relation_names: list[str]  # index → predicate URI
    node_names: list[str]  # index → canonical N3 term (readable)
    labels: Tensor  # (N,) long, -1 where unlabeled
    label_names: list[str]  # class index → label URI/string
    train_mask: Tensor  # (N,) bool
    test_mask: Tensor  # (N,) bool
    node_index: dict[str, int] = field(repr=False)  # canonical N3 → index

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def num_relations(self) -> int:
        return len(self.relation_names)

    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    def doubled_edges(self) -> tuple[Tensor, Tensor]:
        """Forward + inverse edges for the R-GCN: relation r → 2r, inverse → 2r+1."""
        src, dst = self.edge_index
        edge_index = torch.cat(
            [torch.stack([src, dst]), torch.stack([dst, src])], dim=1
        )
        edge_type = torch.cat([2 * self.edge_type, 2 * self.edge_type + 1])
        return edge_index, edge_type


def _ensure_raw_files(dataset: str, root: Path) -> Path:
    """Download and extract the benchmark archive if not present."""
    raw_dir = root / dataset / "raw"
    graph_file = raw_dir / f"{dataset}_stripped.nt.gz"
    if not graph_file.exists():
        from torch_geometric.data import download_url, extract_tar

        path = download_url(DOWNLOAD_URL.format(dataset), str(root / dataset))
        extract_tar(path, str(raw_dir))
    return raw_dir


def _canonical(term: object) -> str:
    """Canonical node identity: N3 form (distinguishes URIs from literals)."""
    import rdflib

    if isinstance(term, rdflib.term.Node):
        return term.n3()
    raise TypeError(f"not an RDF term: {term!r}")


def load_rdf_graph(dataset: str, root: str | Path = "data") -> RDFGraph:
    """Load AIFB or MUTAG into a typed, deterministic :class:`RDFGraph`."""
    import rdflib

    if dataset == "synthetic":
        # Deterministic default ground-truth graph (plan M6) — lets every
        # dataset-addressed code path (CLI, pipeline) run on synthetic data.
        from graphlime_rdf.config import SyntheticConfig
        from graphlime_rdf.data.synthetic import generate_ground_truth_graph

        return generate_ground_truth_graph(SyntheticConfig())
    if dataset not in _NODE_HEADER:
        raise ValueError(f"unknown dataset {dataset!r}; expected 'aifb' or 'mutag'")
    root = Path(root)
    raw_dir = _ensure_raw_files(dataset, root)

    g = rdflib.Graph()
    with gzip.open(raw_dir / f"{dataset}_stripped.nt.gz", "rb") as f:
        g.parse(data=f.read(), format="nt")

    # Blank-node ids are regenerated on every parse (AIFB has 152 of them),
    # which would break cross-process determinism. Canonicalise them from the
    # graph content (RGDA1) so identical files always yield identical names.
    if any(isinstance(t, rdflib.BNode) for t in set(g.subjects()) | set(g.objects())):
        from rdflib.compare import to_canonical_graph

        g = to_canonical_graph(g)

    # Deterministic indexing: lexicographic sort of canonical string forms.
    triples = [(_canonical(s), str(p), _canonical(o)) for s, p, o in g]
    relation_names = sorted({p for _, p, _ in triples})
    assert_no_leakage(dataset, relation_names)

    node_names = sorted({s for s, _, _ in triples} | {o for _, _, o in triples})
    node_index = {n: i for i, n in enumerate(node_names)}
    relation_index = {r: i for i, r in enumerate(relation_names)}

    triples.sort()
    src = torch.tensor([node_index[s] for s, _, _ in triples], dtype=torch.long)
    dst = torch.tensor([node_index[o] for _, _, o in triples], dtype=torch.long)
    edge_index = torch.stack([src, dst])
    edge_type = torch.tensor([relation_index[p] for _, p, _ in triples], dtype=torch.long)

    # Splits: benchmark TSVs; label ids assigned in sorted order.
    node_header, label_header = _NODE_HEADER[dataset], _LABEL_HEADER[dataset]
    train_df = pd.read_csv(raw_dir / "trainingSet.tsv", sep="\t")
    test_df = pd.read_csv(raw_dir / "testSet.tsv", sep="\t")

    label_names = sorted(
        {str(v) for v in train_df[label_header]} | {str(v) for v in test_df[label_header]}
    )
    label_ids = {name: i for i, name in enumerate(label_names)}

    n = len(node_names)
    labels = torch.full((n,), -1, dtype=torch.long)
    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    for df, mask in [(train_df, train_mask), (test_df, test_mask)]:
        for uri, lab in zip(df[node_header], df[label_header], strict=True):
            idx = node_index[f"<{uri}>"]
            labels[idx] = label_ids[str(lab)]
            mask[idx] = True

    graph = RDFGraph(
        dataset=dataset,
        edge_index=edge_index,
        edge_type=edge_type,
        relation_names=relation_names,
        node_names=node_names,
        labels=labels,
        label_names=label_names,
        train_mask=train_mask,
        test_mask=test_mask,
        node_index=node_index,
    )
    _validate_against_benchmark(graph)
    return graph


def _validate_against_benchmark(graph: RDFGraph) -> None:
    """Assert the loaded graph matches the published benchmark figures."""
    ds = graph.dataset
    checks = [
        ("num_nodes", graph.num_nodes, EXPECTED_NUM_NODES[ds]),
        ("num_classes", graph.num_classes, EXPECTED_NUM_CLASSES[ds]),
        ("num_train", int(graph.train_mask.sum()), EXPECTED_NUM_TRAIN[ds]),
        ("num_test", int(graph.test_mask.sum()), EXPECTED_NUM_TEST[ds]),
    ]
    for name, actual, expected in checks:
        if actual != expected:
            raise RuntimeError(f"{ds}: {name}={actual}, benchmark expects {expected}")
