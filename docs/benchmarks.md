# Benchmarks

Every benchmark from the upstream Nim reference under
`integration_tests/examples/` is auto-translated to Python DSL via
`tools/nim_to_dsl.py` and lives in `examples/`. All compile through
HIR/MIR cleanly; the smoke-tested ones also compile + run end-to-end
on this box.

## Canonical benchmarks

| Benchmark | Relations | Rules | Strata | MIR steps | Notes |
|---|---|---|---|---|---|
| `triangle` | 4 | 1 | 1 | 2 | Classic 3-way self-join |
| `tc` | 3 | 3 | 3 | 6 | Transitive closure |
| `sg` | 2 | 2 | 2 | 4 | Same-generation |
| `andersen` | 5 | 4 | 2 | 4 | Pointer analysis |
| `cspa` | 7 | 12 | 3 | 10 | Context-sensitive points-to |
| `galen` | 8 | 8 | 2 | 6 | Ontology closure |
| `crdt` | 22 | 24 | 17 | 39 | CRDT replication semantics |
| `polonius_test` | 38 | 38 | 32 | 68 | Rust borrow-checker |
| `ddisasm` | 39 | 23 | 11 | 30 | Binary disassembly (needs `--meta`) |
| `reg_scc` | 16 | 10 | 2 | 9 | Register-SCC subquery of ddisasm |
| `doop` | 76 | 84 | 16 | 60 | Java points-to (needs `--meta`) |

## LSQB triangle variants

| Benchmark | Notes |
|---|---|
| `lsqb_q3_triangle` | Basic 3-way triangle count |
| `lsqb_q6_2hop` | 2-hop path |
| `lsqb_q6_count` | Count variant of q6 |
| `lsqb_q7_optional` | Left-join / optional pattern |
| `lsqb_q9_neg2hop` | 2-hop with negation |
| `lsqb_triangle_count` | Count-only triangle |

## Running one

```bash
# Synthetic, no input data:
python examples/run_benchmark.py triangle

# Needs a CSV dir:
python examples/run_benchmark.py tc --data /path/to/edges

# Needs a CSV dir + meta JSON:
python examples/run_benchmark.py doop \
    --data /path/to/doop_input_dir \
    --meta /path/to/batik_meta.json

# Compile only (no run):
python examples/run_benchmark.py galen --no-run

# Cap fixpoint for a sanity-check run:
python examples/run_benchmark.py polonius_test --data /path/to/data --max-iter 3
```

Each invocation prints one line per phase (DSL build → emit →
compile → load → run) with wall-clock timings — useful when
diagnosing where time is going on your box.

## Regenerating from Nim

When upstream Nim sources change, regenerate every benchmark with:

```bash
for nim in integration_tests/examples/*/*.nim; do
  python tools/nim_to_dsl.py "$nim" --out examples/$(basename "$nim" .nim).py
done
```

The translator is deliberately noisy — it raises on any syntax it
hasn't been taught, so you'll see exactly which benchmarks regressed.
See `tools/nim_to_dsl.py`'s header for the full supported subset.
