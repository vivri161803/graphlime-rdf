"""M3 focused probe (scratch): convergence-oriented settings, 2 seeds each."""

from __future__ import annotations

import statistics
import sys

from graphlime_rdf.config import (
    ExperimentConfig,
    FeatureSpaceConfig,
    RGCNModelConfig,
    TrainingConfig,
)
from graphlime_rdf.data.loader import load_rdf_graph
from graphlime_rdf.training import train_run

DATASET = sys.argv[1] if len(sys.argv) > 1 else "aifb"
SEEDS = [0, 1]

graph = load_rdf_graph(DATASET)
for binary in [True, False]:
    for wd in [0.0005, 0.0]:
        for epochs in [200]:
            cfg = ExperimentConfig(
                dataset=DATASET,  # type: ignore[arg-type]
                feature_space=FeatureSpaceConfig(
                    kind="predicate", directions=["out", "in"], binary=binary
                ),
                model=RGCNModelConfig(hidden_dim=32),
                training=TrainingConfig(epochs=epochs, weight_decay=wd, seeds=SEEDS),
            )
            accs = []
            curves = []
            for seed in SEEDS:
                res = train_run(graph, cfg, seed)
                accs.append(res.manifest.final_test_accuracy)
                curves.append([round(h[3], 3) for h in res.history[::50]])
            print(
                f"{DATASET} binary={binary} wd={wd} epochs={epochs} "
                f"mean={statistics.mean(accs):.4f} accs={[round(a, 3) for a in accs]} "
                f"curve50={curves[0]}"
            )
