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

## Twelve-dataset DOOP corpus

`examples/doop_benchmark.py` prepares twelve real DaCapo
23.11-MR2-chopin applications from the
[published FlowLog facts](https://huggingface.co/datasets/NemoYuu/flowlog_benchmark/tree/main/dataset/csv).
`examples/doop_suite/datasets.json` pins the corpus revision, archive SHA-256,
archive size, and the source of the upstream reference cardinalities.
These are fresh Chopin datasets, not aliases for the older five local datasets.

The following **local scheduling tiers** use upstream reference `VarPointsTo`
cardinality, not input size or measured SRDatalog results. They are not official
DOOP dataset editions. H2O, for example, has substantially more raw input than
Jython but a much smaller upstream points-to result.

| Tier | Reference VPT rows | Applications |
|---|---:|---|
| small | < 15 million | xalan, zxing, biojava, pmd, sunflow |
| medium | 15–<30 million | h2o, spring |
| large | 30–<100 million | batik, eclipse, fop, h2 |
| xlarge | >= 100 million | jython |

```bash
python examples/doop_benchmark.py list
python examples/doop_benchmark.py fetch --all --root /path/to/doop-data
python examples/doop_benchmark.py prepare --all --root /path/to/doop-data

# Select named applications or a workload tier instead:
python examples/doop_benchmark.py prepare --dataset xalan jython \
    --root /path/to/doop-data
python examples/doop_benchmark.py prepare --tier medium \
    --root /path/to/doop-data
```

Python 3.10+ and the external `sort` command are required for preparation.
Downloading all archives requires approximately 1.74 GB; the complete raw facts
require approximately 28.3 GB before normalized inputs, dictionaries, build
caches or result exports. Keep all data outside the source checkout.
`--archive-cache DIR` optionally reuses a read-only archive cache after checksum
verification. `DOOP_SORT_TMPDIR` can select an existing scratch directory.

Preparation preserves the complete `MainClass` set and uses one shared symbol
dictionary per dataset. It derives all 39 declared integer TSV input files,
including descriptors and heap types, then materializes each projected relation
as a set with `sort -u`; it never samples rows or adds synthetic roots.
Required files, arities, signed-int32 numeric domains, and functional attributes
needed by the normalization are checked explicitly.
The instantiated program currently uses 37 of those inputs and has 37 derived
relations; `Var_DeclaringMethod` and `isVirtualMethodInvocation_Insn` are declared
but unused. Preparation reports all declared files; execution reports the
actually consumed input rows and bytes separately.

Each `prepared/APP/` contains the input CSV files (tab-delimited despite their
extension), `meta.json`, `str2num.json`, and `manifest.json`. The manifest records
raw/prepared hashes, prepared rows and bytes, entrypoints, and provenance.
The status `prepared_not_engine_validated` deliberately does not claim query
correctness. An incomplete preparation is not published; existing prepared
datasets are verified rather than overwritten. Do not reuse another dataset's
interned constants or compare old-generation facts under the same application
name without recording that change.

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
