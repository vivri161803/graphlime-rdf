"""Print a digest of the loaded AIFB graph — used to verify determinism
across processes and PYTHONHASHSEED values."""

import hashlib

from graphlime_rdf.data.loader import load_rdf_graph

g = load_rdf_graph("aifb")
h = hashlib.sha256()
h.update("".join(g.node_names).encode())
h.update("".join(g.relation_names).encode())
h.update(g.edge_index.numpy().tobytes())
h.update(g.edge_type.numpy().tobytes())
h.update(g.labels.numpy().tobytes())
print(h.hexdigest())
