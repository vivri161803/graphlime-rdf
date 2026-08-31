"""M3 fixed-grid exploration (scratch runs only, results logged to stdout).

Grid: hidden_dim x directions x epochs — small and fixed per plan §0.
Chosen configs are then written to configs/{aifb,mutag}.yaml (ADR-006).
"""

from __future__ import annotations

import statistics
import sys
import time

from graphlime_rdf.config import (
    ExperimentConfig,
    FeatureSpaceConfig,
    RGCNModelConfig,
    TrainingConfig,
)
from graphlime_rdf.data.loader import load_rdf_graph
from graphlime_rdf.training import train_run

DATASET = sys.argv[1] if len(sys.argv) > 1 else "aifb"
SEEDS = [0, 1, 2, 3, 4]

graph = load_rdf_graph(DATASET)
for directions in [["out"], ["out", "in"]]:
    for hidden in [16, 32]:
        for epochs in [50]:
            cfg = ExperimentConfig(
                dataset=DATASET,  # type: ignore[arg-type]
                feature_space=FeatureSpaceConfig(kind="predicate", directions=directions),  # type: ignore[arg-type]
                model=RGCNModelConfig(hidden_dim=hidden),
                training=TrainingConfig(epochs=epochs, seeds=SEEDS),
            )
            accs = []
            t0 = time.time()
            for seed in SEEDS:
                res = train_run(graph, cfg, seed)
                accs.append(res.manifest.final_test_accuracy)
            print(
                f"{DATASET} dirs={directions} hidden={hidden} epochs={epochs} "
                f"mean={statistics.mean(accs):.4f} std={statistics.stdev(accs):.4f} "
                f"accs={[round(a, 3) for a in accs]} ({time.time() - t0:.0f}s)"
            )
