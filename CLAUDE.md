# CLAUDE.md — project rules for graphlime-rdf

GraphLIME applied to node classification on RDF knowledge graphs (R-GCN on
AIFB and MUTAG). XAI laboratory course project. All artifacts are in English.

## Non-negotiable rules

"Explanations are expressed in the vocabulary of the knowledge graph. Any
output containing raw integer feature indices where a predicate name is
expected is a bug."

"Tests assert correctness and invariants, never experimental outcomes. No
threshold is ever lowered to make progress. Nothing on AIFB/MUTAG is reported
before the synthetic gate passes."

## Working rules

- `just check` (lint + types + test) must be green before every commit.
- Conventional commits (`feat:`, `test:`, `fix:`, `docs:`, `chore:`).
- Milestones close with a tagged commit (`m0`…`m9`) whose body lists the
  verified DoD items.
- Every non-trivial decision gets a dated ADR-style entry in
  `docs/decisions.md`.
- No magic numbers outside `src/graphlime_rdf/config.py` / `configs/*.yaml`.
- Directory contract: `results/` = validated ExplanationRecords + tables +
  figures; `runs/` = training audit trail; `checkpoints/` = best weights only.
- All numbers in the report are generated from `results/*.jsonl` via
  `cli.py tables` and injected between `<!-- RESULTS:BEGIN/END -->` markers.
  The README carries no numbers at all (ADR-010).
