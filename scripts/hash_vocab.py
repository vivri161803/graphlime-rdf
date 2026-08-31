"""Print vocabulary hashes for AIFB feature spaces — used to verify
determinism across processes and PYTHONHASHSEED values (M2 DoD)."""

from graphlime_rdf.config import FeatureSpaceConfig
from graphlime_rdf.data.loader import load_rdf_graph
from graphlime_rdf.features import build_features

graph = load_rdf_graph("aifb")
for kind in ("predicate", "predicate_object"):
    _, vocab = build_features(graph, FeatureSpaceConfig(kind=kind))  # type: ignore[arg-type]
    print(kind, len(vocab), vocab.hash)
