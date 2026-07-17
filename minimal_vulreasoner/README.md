# Minimal PyReason example (VulReasoner core)

A single, self-contained script that exercises **only** the PyReason features
VulReasoner actually depends on — for profiling / optimization work without the
sampling, multi-KG, trace-filtering, and result-schema machinery in
`src/reasoning.py` getting in the way.

## Contents

| File | Purpose |
|------|---------|
| `run_minimal_reasoner.py` | The example. Loads the KG, one ruleset, the annotation fn, facts, and runs the fixpoint. |
| `run_srdatalog_reasoner.py` | Runs those same inputs through SRDatalog and prints every temporal `AnalystAt` result. |
| `validate_minimal_parity.py` | Runs both engines and compares their complete temporal result maps. |
| `graphml_ingest.py` | Streaming, PyReason-compatible GraphML-to-relation adapter. |
| `analyst_rule_loader.py` | Parser for the constrained six-rule analyst CSV shape. |
| `srdatalog_query.py` | The same six delayed analyst rules encoded in the SRDatalog DSL. |
| `stress_workload.py` | Deterministic layered workloads shared by both engines. |
| `benchmark_pyreason.py` | PyReason oracle and timing harness. |
| `benchmark_srdatalog.py` | SRDatalog build, GPU timing, and result-digest harness. |
| `graphml/CWE_121_MVP2.graphml` | Knowledge graph — label→label ontology edges (`can_cause`, `contributes_to`, `manifestation_of`, …). Copied from `graphml/`. |
| `rules/analyst_rules.csv` | The one ruleset. Analyst rules whose heads call the annotation function. Copied from `inputs/rules/`. |
| `annotation_fn.py` | `paired_minimum_bounds_ann_fn`, copied verbatim from `src/annotation_fn.py`. |

`output/` is created at runtime and holds the rule-trace CSVs (`pr.save_rule_trace`).

## PyReason surface used

`pr.reset` → `pr.load_graphml` → `pr.settings.{atom_trace, allow_ground_rules,
save_graph_attributes_to_trace}` → `pr.add_annotation_function` →
`pr.add_rule_from_csv` → `pr.add_closed_world_predicate` → `pr.add_fact` /
`pr.Fact` (hasLabel, analystAt, stepFrom) → `pr.reason` → `pr.save_rule_trace`.

That is the complete set VulReasoner's reasoning loop touches.

## PyReason terms in database and provenance language

PyReason uses `annotation` for several ideas that database systems normally
name separately.  The mapping used by the SRDatalog port is:

| PyReason term | Database/provenance term |
|---|---|
| satisfied or grounded rule body | join result; one derivation witness for the head |
| `qualified_nodes` / `qualified_edges` | the supporting input tuples belonging to each witness |
| body `annotations` | interval-valued payloads carried by those input tuples |
| head annotation function | user-defined grouped aggregate over the witnesses for one rule and head key |
| `paired_minimum_bounds_ann_fn` | rule-local `ARG MAX` by lower bound, carrying the winning witness's interval |
| rule trace | explanation graph / why-provenance trace |
| world | the current keyed relation state at one logical time, not a probabilistic-database possible world |
| world update | duplicate-key interval-lattice merge into that state |
| `<-1` | logical-time shift, represented in SRDatalog by a successor join |

The callback is **provenance-aware** because it compares alternative
derivation witnesses and returns the interval belonging to the selected
witness.  It is not full semiring provenance: classical why-provenance retains
all alternative supporting derivations, whereas the callback deliberately
keeps one maximum-scoring witness.  The parity encoding retains its stable rank
and interval, but does not currently materialize a complete provenance
polynomial.

The compiler-level denotation and the reason for the separate candidate
relation are specified in
[`docs/ir_lowering_semantics.md`, Section 6.2](../docs/ir_lowering_semantics.md#62-provenance-aware-witness-selection).

## The example workflow

Four code blocks chained so the analyst's `analystAt` "control" atom propagates
one hop per timestep, from the first block to the `CWE_121` vulnerability class:

```
b1 (computed_write_length)
   --contributes_to-->   b2 (incorrect_length_calculation)   [analyst-rule-2]
   --can_cause-->        b3 (return_address_overwrite)        [analyst-rule-1]
   --manifestation_of--> b4 (CWE_121)                         [analyst-rule-5]
```

## Run

```bash
pyenv activate vulreasoner-venv   # Python 3.10.19, pyreason==3.6.0
cd scripts/minimal_pyreason
PYTHONHASHSEED=0 python run_minimal_reasoner.py
```

Expected tail:

```
Converged at time: 4
analystAt(b4) reached: True
```

`PYTHONHASHSEED=0` isn't required for this script, but VulReasoner pins it so its
on-disk traces are byte-reproducible (ADR 0005).

## Run the existing inputs on SRDatalog

This command consumes the checked-in `CWE_121_MVP2.graphml`,
`analyst_rules.csv`, and the workflow facts declared in `example_config.py`:

```bash
PYTHONPATH=src:. python3 minimal_vulreasoner/run_srdatalog_reasoner.py \
  --jobs "$(nproc)"
```

The runner streams GraphML edge attributes into ordinary relation columns,
parses the six PyReason rules into the supported analyst-rule contract, compiles
the DSL query, loads the relations, runs the GPU fixpoint, and prints one
`RESULT_JSON=...` line. After the first build, reuse the cached shared library:

```bash
PYTHONPATH=src:. python3 minimal_vulreasoner/run_srdatalog_reasoner.py \
  --no-compile
```

The exact result is:

```json
{
  "b1@0": [1.0, 1.0],
  "b1@1": [1.0, 1.0],
  "b2@2": [1.0, 1.0],
  "b3@3": [1.0, 1.0],
  "b4@4": [1.0, 1.0]
}
```

For a fresh differential comparison against the sibling PyReason checkout:

```bash
PYTHONPATH=src:. python3 minimal_vulreasoner/validate_minimal_parity.py \
  --no-compile
```

It compares every `(node,time) -> [lower,upper]` entry and exits nonzero on any
difference. The validated input contains 188 graph nodes, 403 graph edges, and
138 edge-attribute facts selected by the six analyst rules. On an RTX 6000 Ada,
streaming ingestion took about 4 ms, cached GPU execution about 17 ms, and the
one-time CUDA build about 39 seconds.

## SRDatalog semantics

The PyReason delay `<-1` is source-level data, not a special GPU scheduling
primitive.  `Successor(t,t1)` connects an `AnalystAt` state at `t` to the state
produced at `t1`, and ordinary semi-naive evaluation carries the change.

`AnalystAt(node,time,lower,upper)` is a functional lattice relation keyed by
`(node,time)`.  Duplicate knowledge is joined by interval intersection:

```
[l1,u1] join [l2,u2] = [max(l1,l2), min(u1,u2)]
```

Each connector join first emits the raw derivation witnesses for one PyReason
rule.  A functional candidate relation groups them by `(node,time)` and
materializes the rule-local `ARG MAX`: maximum lower bound, with stable
connector rank as the tie-break key.  This is required to reproduce
`paired_minimum_bounds_ann_fn`; intersecting all raw witnesses is not
equivalent.

Candidate insertion performs the selection during keyed `NEW -> FULL`
maintenance.  The following candidate-to-`AnalystAt` rule is only a projection
of the already-selected witness.  Insertion into `AnalystAt` then applies the
different, cross-rule interval-intersection operation.  A future head aggregate
could fuse these stages, but it could not remove their semantic grouping
boundary.

Bounds are stored in two columns as bit-cast IEEE-754 binary32 words.  All
generated values are in `[0,1]`, where unsigned word order preserves float
order.  MIR sees only a generic functional-lattice merge/delta operation;
sorted-array GPU maintenance is one physical realization of it.

## Stress parity and benchmark

Run the PyReason oracle in one warm process:

```bash
python minimal_vulreasoner/benchmark_pyreason.py \
  --warmup \
  --case 4,4,2 \
  --case 8,32,4 \
  --case 12,128,8
```

Build once and run the same shapes on SRDatalog:

```bash
PYTHONPATH=src:. python minimal_vulreasoner/benchmark_srdatalog.py \
  --case 4,4,2 \
  --case 8,32,4 \
  --case 12,128,8 \
  --repeat 2
```

Both harnesses print `target_bounds_sha256`, a SHA-256 digest over every final
target and interval.  Equal digests establish exact target-set parity, while
`target_bounds_sample` leaves a few decoded bounds visible for diagnosis.

On an NVIDIA RTX 6000 Ada, the exact implementation produced these warm
measurements:

| Depth × width × fanout | Transitions | PyReason reason | SRDatalog run | Digest parity |
|---|---:|---:|---:|---|
| 4 × 4 × 2 | 32 | 8.1 ms | 8.2 ms | exact |
| 8 × 32 × 4 | 1,024 | 351.8 ms | 32.5 ms | exact |
| 12 × 128 × 8 | 12,288 | 29.39 s | 32.7 ms | exact |
| 16 × 256 × 8 | 32,768 | — | 44.8 ms | GPU stress only |

Compilation is intentionally reported separately (about 49 seconds on this
machine) because generated CUDA is cached and reused.  The GPU harness also
uses an explicit process exit after database shutdown to avoid a known
ctypes/CUDA shared-library static-teardown crash; this does not affect query
results or measured execution time.
