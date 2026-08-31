# Architecture

Module map and data flow for `graphlime-rdf`. The directory contract
(plan §5): `results/` holds validated records, tables and figures; `runs/`
holds the training audit trail; `checkpoints/` holds one best bundle per
dataset.

## Module map

```mermaid
flowchart TB
    subgraph data["graphlime_rdf.data"]
        loader["loader.py<br/>RDF dumps → RDFGraph<br/>(deterministic indexing)"]
        leakage["leakage.py<br/>blocklist + assertion"]
        synthetic["synthetic.py<br/>ground-truth generator"]
    end
    subgraph features["graphlime_rdf.features"]
        vocab["vocabulary.py<br/>sorted, hash-stable"]
        spaceA["predicate.py<br/>space A: indicators/counts"]
        spaceB["predicate_object.py<br/>space B: (p,o) pairs"]
    end
    subgraph models["graphlime_rdf.models"]
        rgcn["rgcn.py<br/>2-layer RGCNConv"]
    end
    subgraph explain["graphlime_rdf.explain"]
        hsic["hsic.py<br/>RBF, centering, (n)HSIC"]
        lasso["hsic_lasso.py<br/>non-negative Lasso"]
        gl["graphlime.py<br/>k-hop sampling + orchestration"]
        base["baseline.py<br/>GNNExplainer wrapper"]
    end
    subgraph evaluate["graphlime_rdf.evaluate"]
        fid["fidelity.py"]
        stab["stability.py"]
        agree["agreement.py"]
    end
    subgraph reporting["graphlime_rdf.reporting"]
        tables["tables.py<br/>JSONL → markdown"]
        readme["readme.py<br/>marker injection"]
        figures["figures.py"]
    end
    config["config.py — pydantic contracts (single source of truth)"]
    training["training.py — seeded runs, manifests, checkpoint bundles"]
    pipeline["pipeline.py — records → results/*.jsonl"]
    cli["cli.py — typer entry points"]

    loader --> leakage
    loader --> features
    features --> training
    rgcn --> training
    training --> pipeline
    hsic --> lasso --> gl
    gl --> pipeline
    base --> pipeline
    evaluate --> pipeline
    pipeline --> reporting
    config -.-> training
    config -.-> pipeline
    cli --> training
    cli --> pipeline
    cli --> reporting
```

## Data flow

```mermaid
flowchart LR
    dumps["benchmark dumps<br/>*_stripped.nt.gz + TSV"] --> loader2["loader<br/>+ leakage assert"]
    loader2 --> graph["RDFGraph"]
    graph --> fb["FeatureBuilder<br/>(space A / B)"]
    fb --> X["X: interpretable features"]
    X --> model2["R-GCN (input = X)"]
    model2 --> ckpt["checkpoints/*_best.pt<br/>(self-contained bundle)"]
    model2 --> Y["predicted probabilities"]
    X --> glime["GraphLIME<br/>HSIC Lasso on k-hop nodes"]
    Y --> glime
    glime --> recs["ExplanationRecords<br/>results/*.jsonl"]
    recs --> tab["tables.py → results/tables/*.md"]
    tab --> docs2["report/relazione.md<br/>(marker injection)"]
```

Key invariant: the model input and the explanation space share **one
vocabulary** (plan §6) — fidelity is column masking on `X`, and every feature
index maps back to a readable predicate name.
