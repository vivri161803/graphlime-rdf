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
def info() -> None:
    """Print the pinned library versions that determine numerical results."""
    from graphlime_rdf.config import library_versions

    for pkg, version in library_versions().items():
        typer.echo(f"{pkg}=={version}")


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

    if final and base.exists():
        # Idempotent repro: one set of final runs per dataset — a rerun
        # replaces the previous audit trail instead of double-counting it
        # in the tables (run ids are timestamped).
        import shutil

        from graphlime_rdf.config import RunManifest

        for manifest_path in base.glob("*/manifest.json"):
            old = RunManifest.model_validate_json(manifest_path.read_text())
            if old.dataset == cfg.dataset:
                shutil.rmtree(manifest_path.parent)

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


@app.command()
def evaluate(
    config: Path = typer.Argument(..., help="Experiment YAML, e.g. configs/aifb.yaml"),
    results_dir: Path = typer.Option(Path("results"), help="Where JSONL records go"),
) -> None:
    """Train all seeds and write every record family (explanations, refusals,
    fidelity, stability, agreement) as validated JSONL."""
    from graphlime_rdf.pipeline import run_full_evaluation

    cfg = ExperimentConfig.from_yaml(config)
    run_full_evaluation(cfg, results_dir)
    typer.echo(f"evaluation records written to {results_dir}/*_{cfg.dataset}.jsonl")


@app.command()
def repro(
    configs_dir: Path = typer.Option(Path("configs"), help="Directory with dataset YAMLs"),
) -> None:
    """Deterministic end-to-end reproduction: train + evaluate AIFB and MUTAG,
    regenerate results/ and tables."""
    for dataset in ["aifb", "mutag"]:
        typer.echo(f"=== {dataset}: train (final) ===")
        train(configs_dir / f"{dataset}.yaml", final=True)
        typer.echo(f"=== {dataset}: evaluate ===")
        evaluate(configs_dir / f"{dataset}.yaml")
    _generate_tables()


def _generate_tables() -> None:
    from graphlime_rdf.reporting.figures import generate_all_figures
    from graphlime_rdf.reporting.tables import generate_all_tables

    for path in [*generate_all_tables(), *generate_all_figures()]:
        typer.echo(f"wrote {path}")


@app.command()
def tables() -> None:
    """Regenerate results/tables/*.md and results/figures/*.png from the JSONL
    records (single source of truth)."""
    _generate_tables()


@app.command()
def readme() -> None:
    """Inject the current tables between the RESULTS markers of README.md and
    report/relazione.md."""
    from graphlime_rdf.reporting.readme import refresh_documents

    for path in refresh_documents():
        typer.echo(f"refreshed {path}")


@app.command("load-and-explain")
def load_and_explain(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Path to a *_best.pt bundle"),
    node: int = typer.Option(..., "--node", help="Node id to explain"),
    rho: float | None = typer.Option(None, help="Override the HSIC Lasso ρ"),
    hops: int | None = typer.Option(None, help="Override the neighbourhood radius"),
    top: int = typer.Option(10, help="How many features to print"),
) -> None:
    """Load a self-contained checkpoint and print a readable explanation —
    no retraining, no external config (plan §9.2)."""
    from graphlime_rdf.data.loader import load_rdf_graph
    from graphlime_rdf.explain.graphlime import Refusal, explain_node
    from graphlime_rdf.features import build_features
    from graphlime_rdf.training import load_checkpoint

    ckpt = load_checkpoint(checkpoint)
    cfg = ckpt.manifest.resolved_config
    graph = load_rdf_graph(ckpt.manifest.dataset)
    x, vocabulary = build_features(graph, cfg.feature_space)
    if vocabulary.hash != ckpt.manifest.vocabulary_hash:
        raise typer.Exit(code=1)
    edge_index, edge_type = graph.doubled_edges()

    updates: dict[str, float | int] = {}
    if rho is not None:
        updates["rho"] = rho
    if hops is not None:
        updates["hops"] = hops
    graphlime_cfg = cfg.graphlime.model_copy(update=updates) if updates else cfg.graphlime

    probs = ckpt.model.predict_proba(x, edge_index, edge_type)
    predicted = int(probs[node].argmax())
    label_names = {i: name for name, i in ckpt.label_map.items()}
    true = int(graph.labels[node])
    typer.echo(f"node {node}: {graph.node_names[node]}")
    typer.echo(
        f"predicted: {label_names[predicted]} (p={float(probs[node, predicted]):.3f}); "
        f"true: {label_names.get(true, 'unlabeled')}"
    )

    out = explain_node(
        ckpt.model, node, x, edge_index, edge_type, vocabulary, graphlime_cfg
    )
    if isinstance(out, Refusal):
        typer.echo(f"explanation refused: {out.reason}")
        raise typer.Exit(code=0)
    typer.echo(
        f"GraphLIME explanation (hops={graphlime_cfg.hops}, rho={graphlime_cfg.rho}, "
        f"neighborhood={out.neighborhood_size}):"
    )
    for name, beta in out.top_features(top):
        typer.echo(f"  β={beta:.4f}  {name}")


if __name__ == "__main__":
    app()
