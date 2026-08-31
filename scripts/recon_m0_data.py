"""M0 recon part 2: download raw AIFB/MUTAG files and inspect predicates.

Verifies empirically whether the known leakage predicates are present in the
`*_stripped.nt.gz` dumps, and counts nodes/triples/predicates.
"""

from __future__ import annotations

import gzip
import os.path as osp
from collections import Counter

import rdflib
from torch_geometric.data import download_url, extract_tar

URL = "https://data.dgl.ai/dataset/{}.tgz"
ROOT = "data"

LEAKAGE_CANDIDATES = {
    "aifb": ["employs", "affiliation"],
    "mutag": ["isMutagenic"],
}


def inspect(name: str) -> None:
    raw_dir = osp.join(ROOT, name, "raw")
    graph_file = osp.join(raw_dir, f"{name}_stripped.nt.gz")
    if not osp.exists(graph_file):
        path = download_url(URL.format(name), osp.join(ROOT, name))
        extract_tar(path, raw_dir)

    g = rdflib.Graph()
    with gzip.open(graph_file, "rb") as f:
        g.parse(file=f, format="nt")

    freq = Counter(g.predicates())
    print(f"\n=== {name.upper()} ===")
    print("triples:", len(g))
    subjects = set(g.subjects())
    objects = set(g.objects())
    print("nodes (subj ∪ obj):", len(subjects | objects))
    print("distinct predicates:", len(freq))
    for cand in LEAKAGE_CANDIDATES[name]:
        hits = [str(p) for p in freq if cand.lower() in str(p).lower()]
        print(f"leakage candidate '{cand}': {'PRESENT ' + str(hits) if hits else 'absent'}")
    print("top predicates by frequency:")
    for p, c in freq.most_common(12):
        print(f"  {c:6d}  {p}")


if __name__ == "__main__":
    for name in ["aifb", "mutag"]:
        inspect(name)
