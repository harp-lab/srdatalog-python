# Architecture

SRDatalog's Python frontend compiles a Datalog program through five
phases. Each phase is a Python subpackage; transitions between them
are plain function calls.

| Phase | Package | Primary types |
|---|---|---|
| DSL | {py:mod}`srdatalog.dsl` | {py:class}`Program`, {py:class}`Relation`, {py:class}`Rule`, {py:class}`Var`, {py:class}`Filter` |
| HIR | {py:mod}`srdatalog.hir` | {py:class}`HirProgram`, {py:class}`HirStratum`, {py:class}`HirRuleVariant` |
| MIR | {py:mod}`srdatalog.mir` | {py:class}`~srdatalog.mir.types.MirNode`, {py:class}`~srdatalog.mir.types.ExecutePipeline`, {py:class}`~srdatalog.mir.types.FixpointPlan` |
| Emit | {py:mod}`srdatalog.codegen` | schema/runner/main-file/JIT-batch generators |
| Compile | {py:mod}`srdatalog.codegen.jit.compiler_ninja` | emits `build.ninja`, runs ninja + clang++ |

## DSL → HIR

{py:func}`srdatalog.hir.compile_to_hir` runs a fixed pass pipeline:

1. **Constant rewrite** — `R(1, x)` becomes `R(_c0, x) & Filter((_c0,), "return _c0 == 1;")`.
2. **Head-constant rewrite** — same, but for constants in rule heads.
3. **Semi-join optimization** — opt-in via `rule.with_semi_join()`; rewrites 3+ body-atom rules into a semi-join form when profitable.
4. **Stratification** — partitions rules into strata, handles negation / aggregation dependencies.
5. **Semi-naive variant generation** — one variant per delta position.
6. **Join planning** — builds a var-order / clause-order / access-pattern per variant.
7. **Temp-rel synthesis** (pass 4.5) — splits rules with `SPLIT` markers.
8. **Index selection** — picks the minimal set of indexes to build per relation.
9. **Temp-rel index registration** (pass 5.5) — merges temp-rel indexes back into the global index map.

## HIR → MIR

{py:func}`srdatalog.hir.compile_to_mir` calls
{py:func}`~srdatalog.hir.lower.lower_hir_to_mir_steps` to flatten every
variant into a sequence of steps. Each step is either:

- `ExecutePipeline` — a single rule variant's join pipeline (scan →
  joins → filter → materialize).
- `FixpointPlan` — a recursive stratum with delta-merge bookkeeping.
- `ParallelGroup` — a set of pipelines safe to run concurrently.

MIR passes then run:

- `pre_reconstruct_rebuilds` — inserts index rebuilds before
  non-incremental reads.
- `clause_order_reorder` — applies user-specified clause orderings.
- `prefix_source_reorder` — hoists prefix-sharing sources.
- `apply_balanced_scan_pass` — experimental skew-split support.

## MIR → C++ tree

{py:func}`srdatalog.build.build_project` stitches the codegen layer
together:

- {py:func}`srdatalog.codegen.jit.complete_runner.gen_complete_runner`
  emits a `JitRunner_<rule>` struct per rule — fully concrete, no C++
  templates for the kernels themselves.
- {py:func}`srdatalog.codegen.jit.orchestrator_jit.gen_step_body` emits
  each `step_N` as a template member of the host-side `_Runner` struct.
- {py:func}`srdatalog.codegen.jit.main_file.gen_main_file_content`
  composes the main.cpp (schemas → DB alias → GPU includes → runner
  fwd decls → `_Runner` struct).
- {py:func}`srdatalog.codegen.jit.main_file.gen_extern_c_shim` appends
  the ten `extern "C"` entries used by native callers:
  `srdatalog_init`, `srdatalog_load_all`, `srdatalog_load_csv`,
  `srdatalog_prepare`, `srdatalog_run`, `srdatalog_size`, `srdatalog_get_size`,
  `srdatalog_export_tsv`, `srdatalog_synchronize`, `srdatalog_shutdown`.
- {py:func}`srdatalog.codegen.jit.cache.write_jit_project` writes the
  `.cpp` tree to `<cache_base>/jit/<Project>_<hash>/`.

**Behavioral validation** checks MIR index availability and DELTA lifetimes,
including secondary-index reads before ownership transfer and retention across
recursive iterations. Native CUDA regressions exercise result tuples, consuming
index ownership, provenance, wide column addressing, and process teardown;
enable them with `SRDATALOG_JIT_RUN_COMPILE_TESTS=1`. Historical Nim source
formatting is not a compatibility contract.

`build_project(index_only_outputs=True)` enables proven-dead raw FULL
reconstruction elimination for closed, index-backed C-ABI consumers such as the
DOOP suite. Generic and custom C++ builds keep raw output storage by default.
The opt-in analysis preserves reconstruction for raw readers, opaque hooks,
unsupported representations, and nonterminal recursive consumers. Terminal
recursive results may omit raw exit copies only when no later MIR consumer needs
them; live recursive DELTA remains intact during iteration, and dead scratch is
released only after synchronized convergence. Counts and TSV exports observe the
complete logical relations through their canonical indexes. Built-in
NoProvenance two-level indexes export both base and HEAD in bounded host chunks,
without allocating a compacted device-sized output table.

## Compile → load

{py:func}`srdatalog.compile_jit_project` emits a `build.ninja` in the
cache dir and shells out to the ninja binary from the `ninja` PyPI
wheel. The rule structure:

- `pch_host` / `pch_device` — PCH emit rules (currently behind
  `use_pch=True`; see [CUDA PCH blocker](cuda_pch_blocker) for why
  they're off by default).
- `cxx_host_only` — `-x cuda --cuda-host-only` compile for TUs that
  don't define `__global__` kernels (main.cpp, step-body shards).
  Halves per-TU compile time by skipping the redundant device pass.
- `cxx` — full two-pass CUDA compile for `jit_batch_*.cpp`.
- `link` — clang++ `-shared` of every `.o` + `-lcudart -lcuda
  -lboost_container`.

If `ccache` is on `$PATH`, it's automatically prepended to the `cxx`
variable — so warm rebuilds after `rm -rf build/jit/` drop from ~100s
to ~3s on doop.

The resulting `.so` is loadable via `ctypes.CDLL(path,
mode=RTLD_GLOBAL)`. Symbols:

| `extern "C"` symbol | Signature | Notes |
|---|---|---|
| `srdatalog_init()` | `int()` | `SRDatalog::GPU::init_cuda()` |
| `srdatalog_load_csv(rel, path)` | `int(const char*, const char*)` | Per-relation CSV load; only dispatches to relations declared with `input_file`. |
| `srdatalog_load_all(dir)` | `int(const char*)` | Convenience — iterates every `input_file` relation. |
| `srdatalog_prepare()` | `int()` | Construct a fresh device DB and finish H2D transfer before timing. |
| `srdatalog_run(max_iters)` | `int(uint64_t)` | Consume a prepared DB once, or copy fresh host→device; run complete fixedpoint, `0` means unlimited. |
| `srdatalog_size(rel)` | `uint64_t(const char*)` | Canonical-index size on the device DB after execution; legacy zero-on-missing probe. |
| `srdatalog_get_size(rel, out)` | `int(const char*, uint64_t*)` | Checked size read; reports missing indexes rather than masking errors. |
| `srdatalog_export_tsv(rel, path)` | `int(const char*, const char*)` | Export logical tuples after execution. |
| `srdatalog_synchronize()` | `int()` | Checked completion boundary for GPU work. |
| `srdatalog_shutdown()` | `int()` | Synchronize and free host and device DBs. |

Preparation is a one-shot snapshot, not a result cache. Loading more host inputs
invalidates it; repeated runs without preparation still reconstruct empty IDBs.
The DOOP benchmark calls preparation outside its fixedpoint timer and collects
counts/optional exports afterward.

## Nim ↔ Python parity

`tools/nim_to_dsl.py` auto-translates upstream Nim programs to Python
DSL. Every benchmark under `integration_tests/examples/` in the
upstream tree is already translated; regenerate with:

```bash
for nim in integration_tests/examples/*/*.nim; do
  python tools/nim_to_dsl.py "$nim" --out examples/$(basename "$nim" .nim).py
done
```

The translator is conservative — it fails loudly on any syntax it
hasn't been taught. See the header comment of `tools/nim_to_dsl.py`
for the supported subset.
