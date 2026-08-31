"""Debug the synthetic fidelity gap: per-node fid+ vs random+, by class."""

from pathlib import Path

import numpy as np

from graphlime_rdf.config import ExperimentConfig, SyntheticConfig
from graphlime_rdf.data.synthetic import generate_ground_truth_graph
from graphlime_rdf.evaluate.fidelity import fidelity_at_k
from graphlime_rdf.explain.graphlime import Explanation, explain_node
from graphlime_rdf.training import train_run

repo = Path(__file__).resolve().parents[1]
graph = generate_ground_truth_graph(SyntheticConfig())
cfg = ExperimentConfig.from_yaml(repo / "configs" / "synthetic.yaml")
result = train_run(graph, cfg, seed=0)

test_nodes = graph.test_mask.nonzero().flatten().tolist()[:30]
rows = []
for node in test_nodes:
    out = explain_node(
        result.model, int(node), result.x, result.edge_index, result.edge_type,
        result.vocabulary, cfg.graphlime,
    )
    if not isinstance(out, Explanation):
        continue
    ranked = [int(j) for j in np.argsort(-out.beta, kind="stable")]
    top_names = [result.vocabulary.names[j] for j in ranked[:2]]
    fp, fm, rp, rm = fidelity_at_k(
        result.model, result.x, result.edge_index, result.edge_type, int(node), ranked, k=2
    )
    cls = int(graph.labels[node])
    rows.append((cls, fp, rp, fp - rp, top_names))

for cls in [0, 1]:
    sub = [r for r in rows if r[0] == cls]
    print(
        f"class {cls}: n={len(sub)} mean_fid+={np.mean([r[1] for r in sub]):.4f} "
        f"mean_rand+={np.mean([r[2] for r in sub]):.4f} mean_gap={np.mean([r[3] for r in sub]):.4f}"
    )
print("overall gap:", np.mean([r[3] for r in rows]))
print("sample explanations class0:", [r[4] for r in rows if r[0] == 0][:3])
print("sample explanations class1:", [r[4] for r in rows if r[0] == 1][:3])
