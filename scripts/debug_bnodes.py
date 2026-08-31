"""Debug: count blank nodes in the benchmark dumps."""

import gzip

import rdflib

for name in ["aifb", "mutag"]:
    g = rdflib.Graph()
    with gzip.open(f"data/{name}/raw/{name}_stripped.nt.gz", "rb") as f:
        g.parse(data=f.read(), format="nt")
    terms = set(g.subjects()) | set(g.objects())
    bnodes = [t for t in terms if isinstance(t, rdflib.BNode)]
    literals = [t for t in terms if isinstance(t, rdflib.Literal)]
    print(name, "terms:", len(terms), "bnodes:", len(bnodes), "literals:", len(literals))
    if bnodes:
        print("  sample bnode n3:", bnodes[0].n3())
