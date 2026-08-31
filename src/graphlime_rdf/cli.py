"""Typer CLI: train | explain | evaluate | tables | readme | load-and-explain | repro.

Every command is driven by a resolved :class:`ExperimentConfig`; no hidden
defaults. Commands are added milestone by milestone.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import typer

from graphlime_rdf.config import ExperimentConfig

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def train(
    config: Path = typer.Argument(..., help="Experiment YAML, e.g. configs/aifb.yaml"),
    final: bool = typer.Option(False, help="Write to runs/final/ (audit trail) or runs/scratch/"),
    checkpoint_dir: Path = typer.Option(Path("checkpoints"), help="Where the best bundle goes"),
) -> None:
    """Train all configured seeds; save the best seed as a checkpoint bundle."""
    from graphlime_rdf.data.loader import load_rdf_graph
    from graphlime_rdf.training import save_checkpoint, train_run, write_run_dir

    cfg = ExperimentConfig.from_yaml(config)
    graph = load_rdf_graph(cfg.dataset)
    base = Path("runs") / ("final" if final else "scratch")

    results = []
    for seed in cfg.training.seeds:
        result = train_run(graph, cfg, seed)
        run_dir = write_run_dir(result, base)
        results.append(result)
        typer.echo(
            f"seed {seed}: test_acc={result.manifest.final_test_accuracy:.4f} → {run_dir}"
        )

    accs = [r.manifest.final_test_accuracy for r in results]
    best = max(results, key=lambda r: r.manifest.final_test_accuracy)
    bundle_path = checkpoint_dir / f"{cfg.dataset}_best.pt"
    save_checkpoint(best, graph, bundle_path)
    typer.echo(
        f"{cfg.dataset}: mean={statistics.mean(accs):.4f} std={statistics.stdev(accs):.4f} "
        f"best_seed={best.manifest.seed} → {bundle_path}"
    )


if __name__ == "__main__":
    app()
