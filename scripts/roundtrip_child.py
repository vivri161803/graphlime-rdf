"""Child process of the checkpoint round-trip test (M3 DoD).

Loads a checkpoint bundle in a *fresh* process, recomputes logits on the tiny
synthetic graph, and prints a digest — compared with the parent's digest.
"""

import hashlib
import sys

from graphlime_rdf.data.synthetic import tiny_overfit_graph
from graphlime_rdf.features import build_features
from graphlime_rdf.training import load_checkpoint

ckpt = load_checkpoint(sys.argv[1])
graph = tiny_overfit_graph()
x, vocab = build_features(graph, ckpt.manifest.resolved_config.feature_space)
assert vocab.hash == ckpt.manifest.vocabulary_hash
edge_index, edge_type = graph.doubled_edges()
logits = ckpt.model(x, edge_index, edge_type)
print(hashlib.sha256(logits.detach().numpy().tobytes()).hexdigest())
