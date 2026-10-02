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
| `doop` | 75 | — | — | — | Corrected Java points-to query: 37 consumed inputs + 38 IDBs; other compile metrics not remeasured (needs `--meta`) |

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

## Real-application DOOP corpus

> **DOOP evidence correction:** the historical v2 exact comparisons validate
> GPU execution against the CPU translation of the same canonical Python query,
> which contained an **unguarded `Object[]` store bug**. They do **not** establish
> equivalence to the upstream DOOP query. The historical computation-only and
> Nsight Systems (`nsys`) results likewise measure that old query, not a
> performance baseline for the corrected query. Existing release assets remain
> immutable historical evidence; no corrected-query timings are claimed here.

The corrected query factors the `Object[]` store eligibility check into a unary
`EligibleObjectArrayHeap(heap)` relation. It requires the stored heap's
`HeapAllocation_Type(heap, heaptype)`, the `Object[]` component type from
`ComponentType(ObjectArray, comptype)`, and
`SupertypeOf(comptype, heaptype)`. `AIPT_Store_ObjectArray` joins this unary
eligibility filter, preserving the stored-heap-type, component-type, and subtype
guards without materializing an `Object[]` array-heap/stored-heap pair Cartesian
product. The current instantiated contract has **75 relations and 38 IDBs**,
using the same **39 prepared input files**, of which 37 are consumed.

The corrected Eclipse run uses unchanged prepared tuples and produces
`VarPointsTo=39,139,806`, `InstanceFieldPointsTo=9,676,775`, and
`ArrayIndexPointsTo=321,887`, matching the independently executed upstream
raw-fact oracle's cardinalities. Corrected CPU/GPU exports match all **38 IDB
tuple sets** exactly. This is not a claim of full upstream tuple equivalence
or corrected validation for the other 20 applications.
`EligibleObjectArrayHeap` has 52,237 rows; the non-Object-array compatibility
helper remains 16,979,586 rows rather than adding 76,004,835 Object-array pairs.

An earlier shared-workstation session recorded 41.78/40.95/41.47 s on Soufflé
(12 threads) and 117.10/119.48/121.36 s on the GPU bitmap plan. Loading,
H2D and exports are outside these timers; the final repetition exports
separately for correctness. The unchanged historical GPU binary also took
117.29 s in the same session, versus its historical 2.83 s median.
Therefore these timings do **not** establish a guard-induced regression or
speedup; the large environment/timing shift remains unexplained.
Speculative outer-variable-order overrides produced 117.33/122.53/117.44 s
and were not retained. A smaller eligibility set alone does not establish a
faster physical plan. The [guard correction evidence](https://github.com/harp-lab/srdatalog-python/releases/tag/doop-object-array-guard-v1)
preserves both attempts and the control. Existing manifests pin the old query:
prepare into a **fresh root**, without rewriting historical identities.

Subsequent local runs with no competing GPU compute process before each run
used the **same corrected logical query** to compare compiler generations.
The semijoin-aware compiler recognizes eligible static unary sets as bound-key
filters, preserves filtered DELTA-key traversal, and emits one-sided membership
probes instead of generic intersections. It creates no extra recursive relation
and introduces no DOOP-specific planner branch.

| Eclipse GPU compiler | Median fixedpoint seconds |
|---|---:|
| Previous guarded compiler | 3.228 |
| Semijoin-aware compiler | 2.928 |

One warmup and five measured fresh processes per compiler, alternating order,
on RTX 6000 Ada with device-only `cuda_async`: **9.3% less fixedpoint time**.
Loading, H2D, counts and tuple export are outside the timer. All 75 counts
matched; a separate final export matched all 38 CPU IDB tuple sets exactly.
This is an Eclipse result, not a corpus-wide performance claim or a globally
optimal cost model. [Samples, build identities and correctness evidence](https://github.com/harp-lab/srdatalog-python/releases/tag/doop-semijoin-planner-v1)
also retain the unsuccessful intermediate implementation's measurements.

`examples/doop_benchmark.py` combines eleven retained DaCapo
23.11-MR2-chopin applications from the
[published FlowLog facts](https://huggingface.co/datasets/NemoYuu/flowlog_benchmark/tree/main/dataset/csv)
with new DaCapo 2006 and independent JVM applications.
`examples/doop_suite/datasets.json` pins each archive's SHA-256, size, source
revision, and per-application provenance. The published
[initial expansion](https://github.com/harp-lab/srdatalog-python/releases/tag/doop-corpus-expanded-v1)
is immutable. The catalog's
[native-admission revision](https://github.com/harp-lab/srdatalog-python/releases/tag/doop-corpus-expanded-v2)
replaces H2O with Groovy and Kotlin 1.5.31 with 1.4.32. It contains the
replacement raw archives, extraction provenance, historical CPU measurements,
native CPU/GPU agreement evidence for the old query, selection audit and memory diagnoses.
No binaries or facts are stored in Git.

The **local scheduling tiers** mostly use historical canonical SRDatalog CPU
`VarPointsTo` cardinalities measured with the unguarded query, not archive size
or mixed upstream analyses. Eclipse alone has a corrected measurement and
entry-level query hash; the others retain the catalog's historical reference
query hash. The thresholds are unchanged and are not official DOOP editions.
The retained Chopin archives are unchanged, and their upstream cardinalities
remain separately recorded for traceability. In the historical v2 validation,
every selected application completed the old canonical CPU query with 74 relation
counts and all 37 IDB exports. Zero warmups and one repetition establish historical
cardinality evidence, not comparative performance or upstream equivalence.

The catalog has **21 distinct applications: 6 small, 5 medium, 6 large, 4 xlarge**,
compared with the original 5/2/4/1 distribution.

| Tier | Historical unguarded-query VPT rows | Applications |
|---|---:|---|
| small | < 15 million | xalan, zxing, biojava, pmd, bloat, sunflow |
| medium | 15–<30 million | clojure, chart, groovy, javac, spring |
| large | 30–<100 million | batik, eclipse, h2, fop, jruby, pdfbox |
| xlarge | >= 100 million | soot, jython, scala, kotlin |

| New application | Version / source | Historical unguarded-query VPT rows |
|---|---|---:|
| bloat | DaCapo 2006 | 11,227,250 |
| chart | DaCapo 2006 | 17,221,612 |
| clojure | 1.8.0 | 16,591,860 |
| groovy | 2.4.21 | 19,565,192 |
| javac | OpenJDK 8u312 | 24,163,325 |
| jruby | 1.7.27 | 60,512,570 |
| pdfbox | 2.0.20 | 69,315,812 |
| soot | 4.3.0 | 412,802,921 |
| scala | 2.11.12 | 612,741,889 |
| kotlin | 1.4.32 | 719,597,196 |

New inputs use DOOP 4.24.9 and `java_8`, with the container image, platform,
application and dependency hashes recorded in each provenance asset.
They preserve all extracted facts and genuine application main methods:
no multiplied facts, synthetic roots, sampled inputs, or duplicate versions
counted as new applications. Raw archive preparation was checked for byte-identical
normalized relations and metadata against the immutable CPU-validated inputs.
Recorded absolute paths in the evidence identify the original runs; portable
use goes through the catalog commands below.

The memory diagnoses and VPT counts below also describe the historical
unguarded-query runs, not new measurements of the corrected query.

VPT tiers do **not** predict GPU memory requirements. H2O was removed from the
local 48 GB catalog after tracing its nonrecursive `CastTo` precomputation:
`Precompute0_String` produces 1,456,822,017 tuples, including
8,559 string casts × 153,657 string heaps. Together with `Precompute0`,
`CastTo` contains 1,631,555,384 four-column tuples. Its two required full-column
index orders occupy **48.624 GiB alone**, above the card's 47.363 GiB
CUDA-addressable capacity; `HeapAllocSuperType` adds another 8.989 GiB.
The initial failure also exposed unnecessary ownership copies, which are engine
issues rather than reasons to discard otherwise runnable workloads. Groovy
replaces H2O in the medium tier: 19,565,192 VPT rows and 1.935 GiB of single-copy
tuple payload. No rows were sampled or relabeled to make either dataset fit.

Kotlin 1.5.31 (1,076,040,872 canonical VPT rows) was also retired from this
48 GB catalog after device-only memory exhaustion during recursive
`VarPointsTo` compaction. Its early pool failure involved fragmentation and
unreleased allocations, not proof that the workload's live data exceeded VRAM;
the later device-only async attempt still failed. Kotlin 1.4.32 is a smaller
genuine compiler release with its original `org.jetbrains.kotlin.cli.jvm.K2JVMCompiler`
entrypoint and complete extracted facts, not a sample of the old input.
It yields 719,597,196 canonical VPT rows and remains the single Kotlin
application in the xlarge tier.

In the **historical v2 unguarded-query validation**, all **21 selected applications**
completed native SRDatalog's **bitmap** plan with
`SRDATALOG_RMM_RESOURCE=cuda_async` on an RTX 6000 Ada (48 GB).
Each exited normally, matched all 74 relation counts, and matched all 37 CPU
IDB tuple sets exactly: **777 complete relation-set comparisons**, without
sampling, CPU fallback, managed memory, or host spill.
The [native validation evidence](https://github.com/harp-lab/srdatalog-python/releases/download/doop-corpus-expanded-v2/native-validation.json)
records per-application builds, input identities, counts and exact comparisons.
These zero-warmup, single-repeat runs establish CPU/GPU agreement for the old
query, not upstream equivalence, corrected-query validation, or speedups.

The [baseline audit](https://github.com/harp-lab/srdatalog-python/releases/download/doop-corpus-expanded-v2/baseline-validation.json)
contains 20 exact native passes (19 earlier ownership-fixed `pool` runs, plus
the historical Kotlin `cuda_async` run), not a same-build performance comparison.
Jython's historical baseline run failed: at recursive step 26, zero-based iteration 113,
its combined `VarPointsTo` NEW output has 3,026,761,641 rows before deduplication.
Index sorting requests 22.562 GiB of scratch while 32.045 GiB is live:
54.607 GiB exceeds the GPU's 47.363 GiB addressable memory.
This is baseline intermediate/sort overhead, not grounds to discard Jython:
the bitmap plan passed all 37 exact comparisons for the historical unguarded
query on the same complete input.
The [allocation trace](https://github.com/harp-lab/srdatalog-python/releases/download/doop-corpus-expanded-v2/jython-baseline-memory.json)
records the failing operation and live/reserved storage separately.

These are static extracted-fact workloads, not claims of complete Java or
reflection coverage. Phantom diagnostics are retained, not suppressed:
some name generated outer prefixes whose dollar-suffixed classes exist, while
optional missing types remain in some applications. Scala retains phantom-based
methods. Kotlin's official embeddable artifact omits shaded IntelliJ UI classes;
after supplying its dependencies, JNA and Java 8 tools, 23 phantom methods and
3 phantom-based methods remain. Groovy retains two optional phantom methods.
Consult the full provenance rather than assuming a phantom-free corpus.
Failed Soot extraction attempts were not promoted merely
because the process exited zero. Upstream software retains its own licenses.

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
The catalog archives total **2,369,170,952 bytes (2.37 GB)**; their complete
raw `.facts` files total **43,585,790,298 bytes (43.59 GB / 40.59 GiB)**,
before normalized inputs, dictionaries, build caches or result exports.
These totals include the replacement Groovy and Kotlin 1.4.32 inputs, not the
retired H2O or Kotlin 1.5.31 generations. Keep all data outside the source checkout.
`--archive-cache DIR` optionally reuses a read-only archive cache after checksum
verification. `DOOP_SORT_TMPDIR` can select an existing scratch directory.

Preparation preserves the complete `MainClass` set and uses one shared symbol
dictionary per dataset. It derives all 39 declared integer TSV input files,
including descriptors and heap types, then materializes each projected relation
as a set with `sort -u`; it never samples rows or adds synthetic roots.
Required files, arities, signed-int32 numeric domains, and functional attributes
needed by the normalization are checked explicitly.
The instantiated program currently uses 37 of those inputs and has 38 derived
relations, for 75 relations in total; `Var_DeclaringMethod` and
`isVirtualMethodInvocation_Insn` are declared but unused. Preparation reports all
declared files; execution reports the actually consumed input rows and bytes
separately.

Each `prepared/APP/` contains the input CSV files (tab-delimited despite their
extension), `meta.json`, `str2num.json`, and `manifest.json`. The manifest records
raw/prepared hashes, prepared rows and bytes, entrypoints, and provenance.
The status `prepared_not_engine_validated` deliberately does not claim query
correctness. An incomplete preparation is not published; existing prepared
datasets are verified rather than overwritten. Do not reuse another dataset's
interned constants or compare old-generation facts under the same application
name without recording that change.
Prepared directories are immutable snapshots: their manifests retain the exact
adapter hash used to create them. Reuse verifies those stored tuples, not whether
the adapter source is unchanged; use a new data root when applying normalization
changes. Execution conservatively requires the recorded canonical query source
hash to match the current query, even for source-only edits.

## Running the DOOP correctness and timing suite

`examples/run_doop_suite.py` runs the canonical Python query, with each dataset's
own metadata. The CPU backend translates that instantiated logical program to
Soufflé; it does not substitute a separately maintained query. It requires
Soufflé and its development headers, a C++17/OpenMP compiler, and zlib/SQLite
development libraries. `SOUFFLE`, `SOUFFLE_INCLUDE_DIR`, `CXX`, `CPPFLAGS`,
`CXXFLAGS`, and `LDFLAGS` support nonstandard installations.
The GPU backend requires the normal [CUDA build setup](getting_started).

The default RMM resource is `pool`. On CUDA, `SRDATALOG_RMM_RESOURCE=cuda_async`
selects CUDA's stream-ordered **device-memory** pool to reduce fragmentation;
it does not enable managed memory, host spill, or reduced input data.
`SRDATALOG_RMM_POOL_INITIAL_SIZE` controls the initial reservation for either
resource. A nonzero `SRDATALOG_RMM_POOL_MAX_SIZE` is supported only by `pool`;
`cuda_async` rejects that combination rather than silently ignoring the cap.
The selected resource is recorded in GPU reports. Use the same resource when
comparing execution times across plans.

```bash
# Prepare first; each run needs a new external output directory.
CXX=g++ python examples/run_doop_suite.py --all \
    --root /path/to/doop-data --output /path/to/results/cpu \
    --backend cpu --threads 12

python examples/run_doop_suite.py --all \
    --root /path/to/doop-data --output /path/to/results/gpu-baseline \
    --backend gpu --plan baseline --jobs 2 \
    --reference /path/to/results/cpu/suite.json

SRDATALOG_RMM_RESOURCE=cuda_async \
python examples/run_doop_suite.py --all \
    --root /path/to/doop-data --output /path/to/results/gpu-bitmap \
    --backend gpu --plan bitmap --jobs 2 \
    --reference /path/to/results/cpu/suite.json

# The historical v2 full-catalog validation used device-only cuda_async:
SRDATALOG_RMM_RESOURCE=cuda_async \
python examples/run_doop_suite.py --dataset kotlin \
    --root /path/to/doop-data --output /path/to/results/kotlin-bitmap-async \
    --backend gpu --plan bitmap --warmups 0 --repeats 1 \
    --reference /path/to/results/cpu/suite.json
```

Use `--dataset NAME ...` or `--tier TIER` instead of `--all` for smaller runs.
The bitmap variant applies the opt-in plan only to `VPT_Assign`'s two recursive
variants; the baseline and logical query remain unchanged.
Defaults are one warmup and three measured repetitions, each in a fresh process.
`--timeout` bounds each build/execution process; `--warmups 0 --repeats 1` is
useful for validation but is not a stable performance measurement.

The suite records build, load, GPU preparation, execution, and export separately.
GPU preparation constructs a fresh device database, transfers the inputs and
synchronizes **before** the fixedpoint timer starts. GPU execution ends with
checked device synchronization; CPU execution times Soufflé after loading.
Both fixedpoint timers exclude input loading, tuple serialization/export and
cardinality checks. Internal RAM/VRAM accesses and rule/index work remain part
of execution; this is complete fixedpoint timing, not kernel-only timing.
Historical v2 GPU reports include H2D setup and must not be mixed with this
new timing boundary. They also executed the old unguarded query, so changing
the timer boundary alone does not make them a corrected-query baseline.
Every run reaches an unlimited fixedpoint and must exit normally.
Failures and timeouts retain logs and appear explicitly in `suite.json`;
the command exits unsuccessfully if any selected dataset fails.

By default, the final measured run records all 75 relation cardinalities and
exports all 38 derived relation sets. With `--reference`, comparison requires
matching query/input identities, complete relation coverage, equal cardinalities,
and exact integer tuple sets after external sorting—not just matching VPT counts.
Without a reference, correctness is `not_compared`, even when execution passes.
Reports distinguish `input_rows`/`input_bytes` for the 37 consumed inputs from
`prepared_input_rows`/`prepared_input_bytes` for all 39 prepared files.
The runner rejects a prepared benchmark with no selected main method rather than
accepting a vacuous empty-analysis match. The per-process timeout also covers
loading and final tuple export; increase it for large exports.

For computation-only performance runs, add `--no-export` on **both** backends.
Soufflé uses `.printsize` instead of `.output` to keep every IDB observable;
the embedded driver disables generated I/O and reads cardinalities after the
timed fixedpoint. The GPU skips tuple export and reads sizes after its timer.
No tuple files are created, including on the final repetition. With
`--reference`, this mode reports `cardinality_match`, **not** `exact_match`;
retain a separate exact-export validation run for tuple-level correctness.

```bash
python examples/run_doop_suite.py --tier xlarge \
    --root /path/to/doop-data --output /path/to/results/cpu-compute \
    --backend cpu --threads 12 --warmups 1 --repeats 3 --no-export
SRDATALOG_RMM_RESOURCE=cuda_async \
python examples/run_doop_suite.py --tier xlarge \
    --root /path/to/doop-data --output /path/to/results/gpu-compute \
    --backend gpu --plan bitmap --warmups 1 --repeats 3 --no-export \
    --reference /path/to/results/cpu-compute/suite.json
```

Reserve substantial disk space for derived results and sort scratch, especially
Jython; small compressed inputs do not imply small fixedpoints. Run performance
measurements without competing CPU/GPU workloads. Compilation and a small GPU
smoke test do not establish that a complete dataset fits in available VRAM.

### Historical computation-only large-workload comparison (unguarded query)

These measurements and associated `nsys` profiles predate the `Object[]`
eligibility correction. They are retained for traceability only: neither their
timings nor their CPU/GPU ratios describe the corrected query. A corrected-query
performance baseline requires fresh runs and matching query/input identities.

[Recorded measurements](https://github.com/harp-lab/srdatalog-python/releases/download/doop-compute-only-v1/measurements.json)
use Soufflé 2.4 compiled with `-O3` and 12 threads on an AMD Threadripper PRO
5945WX, versus SRDatalog's bitmap plan on an RTX 6000 Ada with `cuda_async`.
Each engine runs in a fresh process: one warmup and three measured repetitions,
alternating CPU/GPU per workload after **all compilation finishes**.
The table reports median complete-fixedpoint seconds, excluding input loading,
H2D database preparation, counts and tuple export. No tuple files are written.
Internal rule/index RAM/VRAM accesses remain included.

| Dataset | Soufflé CPU seconds | SRDatalog GPU seconds | CPU/GPU median ratio |
|---|---:|---:|---:|
| batik | 31.41 | 2.96 | 10.6× |
| eclipse | 15.34 | 2.83 | 5.4× |
| h2 | 25.52 | 2.82 | 9.1× |
| fop | 36.48 | 2.58 | 14.1× |
| jruby | 56.81 | 5.62 | 10.1× |
| pdfbox | 48.34 | 3.82 | 12.6× |
| soot | 164.98 | 21.10 | 7.8× |
| jython | 323.90 | 14.31 | 22.6× |
| scala | 235.21 | 19.05 | 12.3× |
| kotlin 1.4.32 | 306.03 | 18.43 | 16.6× |

All 80 historical processes (including warmups) exited normally and matched all
74 old-query cardinalities against the immutable old-query CPU oracles. These
count-only runs do **not** establish fresh tuple equality or upstream equivalence;
the separate v2 exact-export proofs also concern only the old unguarded query.
Raw samples, preparation timings, build/input fingerprints and hardware
configuration are in the evidence release.
Three samples are not a comprehensive scalability study: CPU times varied
notably for PDFBox (46.83–59.72 s) and Soot (164.41–190.31 s). Filesystem input
cache and CPU/GPU clock state are uncontrolled. Only Soufflé is compared here,
not other CPU engines. GPU progress diagnostics remain enabled; their small
console-output overhead is included, rather than subtracted or hidden.

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

For DOOP, regeneration must preserve the corrected unary `Object[]` eligibility
guard described above; a successful translation or CPU/GPU match alone does not
prove equivalence to the upstream query.
