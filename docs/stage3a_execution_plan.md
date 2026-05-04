---
orphan: true
---

# Stage 3A — execution plan

Operational companion to [`milestones.md` § Stage 3](./milestones.md). The
milestones doc indexes the work; this doc is the binding design + per-task
plan.

Binding specs (read first):
- [`design_principles.md`](./design_principles.md) — D-rules and A-rules
- [`ir_lowering_semantics.md`](./ir_lowering_semantics.md) Part I §3–§4 — dialect ABI + P1/P2/P3
- `src/srdatalog/ir/core/CLAUDE.md` — `ir/core/` invariants (in-tree, not built into sphinx)

## 1. Vocabulary (binding)

The conversation that produced this plan kept tripping on three operations
that the current code conflates. They are different and stay different:

| Operation | Type | Target-aware? | Dispatched by | Test surface |
|---|---|---|---|---|
| **Lowering** | `D1.Op → D2.Op*` (data → data) | No | `PassDriver` (typed) | byte-equal s-expr of D2 |
| **Render** | `D.Op → str` (data → target text) | Yes | Codegen's render registry | byte-equal target text |
| **Print** | `D.Op → str` (data → canonical s-expr) | No | Dialect-owned printer | byte-equal s-expr |

**Dialect ≠ Codegen.** A dialect owns IR data: $\langle T, O, L^{\text{out}}, R, V \rangle$
plus Print. A codegen consumes IR data and produces target-specific text:
a Render registry plus target configuration (block size, stream count, etc.).
Codegens have no ops, no lowerings, no rewrites. They are pure renderers
plus parameter providers — "dumb printers" by design.

## 2. IR topology

```
                                                  ┌─→ s-expr   (Print: canonical, debug/test)
                                                  │
HIR ── lower ──→ MIR ── lower ──→ IIR ────────────┤
                                                  │
                                                  ├─→ C++      (Codegen.cuda Render)
                                                  ├─→ C++/TBB  (Codegen.cpp_tbb Render — future)
                                                  └─→ ...      (other codegens)
```

Each IR layer carries the same five-tuple plus Print. Only IIR has Renders
(one per codegen that consumes it). HIR and MIR don't go to target text;
they only have Print.

```
HIR  = ⟨T_h, O_h, L_h^out, R_h, V_h⟩ + Print_h
MIR  = ⟨T_m, O_m, L_m^out, R_m, V_m⟩ + Print_m
IIR  = ⟨T_i, O_i, L_i^out, R_i, V_i⟩ + Print_i + Render_cuda + Render_tbb + …
```

`L_h^out` = HIR → MIR. `L_m^out` = MIR → IIR (per-source dialect, P2 dispatch).
`L_i^out` is empty today — IIR is the lowest IR; below it is target text via Render.

## 3. Target file layout (post-S3A.0)

```
src/srdatalog/ir/
├── core/                                  # framework: Compiler, PassDriver, Lowering, Rewrite, Verifier
├── hir/                                   # dialect: HIR
│   ├── types.py                           # T_h, O_h
│   ├── lowerings.py                       # L_h^out — Lowering instances
│   ├── rewrites.py                        # R_h
│   ├── verifier.py                        # V_h
│   └── print.py                           # Print_h
├── mir/
│   ├── types.py                           # T_m, O_m  (already exists)
│   ├── lowerings.py                       # L_m^out — per-source dispatch
│   ├── passes.py                          # R_m  (exists, ad-hoc — formalize)
│   ├── verifier.py                        # V_m
│   └── print.py                           # Print_m  (exists today as mir/emit.py — RENAME)
├── dialects/                              # IIR-contributing dialects (own T, O, L^out, R, V)
│   ├── iir/cf/
│   ├── relation/sorted_array/
│   ├── relation/d2l/
│   └── parallel/data/
└── codegen/                               # NOT dialects — renderers
    └── cuda/
        ├── render/                        # per-IIR-op renderer functions, registry-dispatched
        │   ├── iir_cf.py
        │   ├── sorted_array.py
        │   ├── d2l.py
        │   └── parallel_data.py
        ├── strategy.py                    # block size, stream count, scheduling policy
        ├── orchestrator.py                # MIR → step-body C++
        ├── complete_runner.py             # MIR + IIR → kernel struct C++
        ├── main_file.py                   # main.cpp shape
        ├── plugin.py                      # index plugin registry (per-Codegen instance)
        └── api.py                         # public entry point
```

## 4. The renderer-completeness safety contract

The framework enforces that **adding a new op is safe to register** —
forgetting a renderer fails loudly at `Compiler()` construction, not at
first emit.

```python
# src/srdatalog/ir/codegen/cuda/__init__.py

class CudaCodegen:
    supported_dialects = (
        iir_cf.DIALECT,
        sorted_array.DIALECT,
        d2l.DIALECT,
        parallel_data.DIALECT,
    )
    render_handlers: dict[type[Op], RenderFn] = {}

    @classmethod
    def verify_completeness(cls):
        for dialect in cls.supported_dialects:
            for op_cls in dialect.ops:
                if op_cls not in cls.render_handlers:
                    raise CodegenIncompleteError(
                        f"{cls.__name__} supports {dialect.name} but "
                        f"has no renderer for {op_cls.__name__}"
                    )
```

`Compiler.register_codegen(CudaCodegen)` calls `verify_completeness()`. Mismatch
= every test that constructs a `Compiler` fails immediately.

Workflow when adding a new op (e.g. `SaSkipList` to `sorted_array`):
1. Add the dataclass to `dialects/relation/sorted_array/ops.py`.
2. Register a renderer in `codegen/cuda/render/sorted_array.py`:
   ```python
   @register_render(CudaCodegen, SaSkipList)
   def _render(op: SaSkipList, ctx: RenderCtx) -> str: ...
   ```
3. Add a `Lowering` if any upper dialect's L^out should produce it.

Skip step 2 → completeness check raises at Compiler construction → CI red.

## 5. Test surfaces

Each layer has its own correctness gate. Lower layers don't depend on text
emit going through:

| Layer | Test | Granularity |
|---|---|---|
| HIR | byte-equal Print_h s-expr (vs frozen golden) | per pipeline pass |
| MIR | byte-equal Print_m s-expr (vs frozen golden) | per MIR pass / lowering |
| IIR | byte-equal Print_i s-expr (vs frozen golden) | per MIR→IIR lowering / IIR rewrite |
| C++ | byte-equal Codegen.cuda output (vs Nim reference) | per kernel / per runner — *existing* gate |
| Renderer completeness | `tests/test_codegen_completeness.py` | per (codegen, dialect) pair, in CI |

The Print_i test gate is what makes IIR an inspectable layer rather than
the fleeting intermediate it is today.

## 6. Pre-flight: pin violations as failing tests

Land these *before* any S3A task. They turn cleanup into red→green so
"done" has an unambiguous meaning.

### 6.1 `tests/test_ir_no_import_side_effects.py` (new)

Importing a module under `ir/dialects/` or `ir/codegen/` must not mutate
state in any *other* module. Currently fails: importing `relation.d2l`
mutates `target.cuda.plugin._PLUGIN_REGISTRY` via the side-effect import
at [d2l/__init__.py:56](../src/srdatalog/ir/dialects/relation/d2l/__init__.py#L56).
**S3A.8 turns this green.**

### 6.2 Extension to `tests/test_ir_core_discipline.py`

No module under `ir/dialects/` or `ir/codegen/` defines a top-level
mutable dict or list. Currently fails on
[plugin.py:147](../src/srdatalog/ir/dialects/target/cuda/plugin.py#L147).
**S3A.8 turns this green.**

### 6.3 `tests/test_codegen_completeness.py` (new)

For each registered Codegen, every op in every supported dialect has a
renderer. Pre-S3A this test cannot exist (codegen registry doesn't exist).
**S3A.0 + S3A.3 enable it; pinned thereafter.**

## 7. Task plans

### S3A.0 — rename `ir/dialects/target/cuda/` → `ir/codegen/cuda/`

**Goal.** Resolve the category error: codegen is not a dialect. Establish
the right vocabulary in the codebase before any other Stage 3A work.

**Approach.** `git mv` the directory. Update every import. Optionally keep
a thin re-export shim at the old path for one release.

**Risk.** Very low. Pure namespace move, no semantics change.

**Test gate.** Full suite green; no behavior change.

**Commit shape.** 1 commit; mostly mechanical.

---

### S3A.1 — add Print_i (IIR s-expression)

**Goal.** Give IIR a canonical text form so MIR→IIR lowerings can be
tested without involving C++ codegen.

**Files.** New: `dialects/iir/cf/print.py`,
`dialects/relation/sorted_array/print.py`,
`dialects/relation/d2l/print.py`,
`dialects/parallel/data/print.py`.

**Approach.** Mirror [`mir/emit.py`](../src/srdatalog/ir/mir/emit.py)
(the MIR s-expr printer) — one dispatch function per dialect, called from
a top-level `print_iir(op)` that dispatches by op's owning dialect.

**Risk.** Low. Pure addition; no existing code changes.

**Test gate.** New `tests/test_iir_print_roundtrip.py` — for each op
class, construct an instance and assert the s-expr is non-empty + parses
back. Frozen goldens added as IIR shapes stabilize.

**Commit shape.** 1 commit.

---

### S3A.2 — sorted_array/lowerings returns IIR data, not C++ text

**Goal.** Make IIR exist as inspectable data between lowering and render.
Today [sorted_array/lowerings.py:353](../src/srdatalog/ir/dialects/relation/sorted_array/lowerings.py#L353)
calls `target.cuda.emit.emit(op, ctx)` directly — IIR exists for ~one
stack frame.

**Approach.** Split the lowering into two phases:
1. Build IIR tree (returns `IirOp` data).
2. Caller invokes Render on the IIR tree separately.

The caller (currently `complete_runner.py`) becomes:
```python
iir_body = sorted_array.lower_pipeline_body(ep)
cuda_text = codegen.cuda.render(iir_body, ctx)
```

**Risk.** Medium. Requires identifying every site that consumes "MIR
pipeline body C++" today and routing through the new two-phase path.
Mitigation: Print_i (S3A.1) provides an inspection point for the
intermediate IIR tree.

**Test gate.** Existing C++ byte-equivalence suite still green (no
semantic change). New `tests/test_mir_to_iir_lowering.py` asserts IIR
shape via Print_i for representative pipelines.

**Commit shape.** 2 commits: 1 (introduce two-phase API alongside
existing path) + 1 (migrate callers, delete old path).

---

### S3A.3 — split `codegen/cuda/emit.py` into `codegen/cuda/render/` package

**Goal.** Eliminate the 41-case match
([emit.py:95](../src/srdatalog/ir/dialects/target/cuda/emit.py#L95)).
Adding a new dialect should not require editing codegen core code (P1 fix).

**Approach.**

1. Create `codegen/cuda/render/` with one file per IIR-contributing
   dialect: `iir_cf.py`, `sorted_array.py`, `d2l.py`, `parallel_data.py`.
2. Move each match arm from `emit.py` into the corresponding file as a
   handler function.
3. Each handler self-registers via `@register_render(CudaCodegen, OpClass)`.
4. The dispatcher in `codegen/cuda/render/__init__.py` becomes
   `CudaCodegen.render_handlers[type(op)](op, ctx)`.
5. Delete `emit.py` (or convert to a re-export of the new dispatcher).

**Risk.** Medium-high. Touches the byte-equivalence hot path. Mitigation:
move handlers in groups (one dialect at a time, one commit per dialect),
running the byte-equivalence suite between each.

**Test gate.** All 1009 tests still green. New
`tests/test_codegen_completeness.py` (gate 6.3) green. Smoke test:
register a dummy `relation.lsm` dialect and verify that adding a renderer
for its ops in `codegen/cuda/render/` is the only change needed.

**Commit shape.** ~6 commits: 1 (registry infra + dispatcher) + 4 (one
per dialect's handlers moved) + 1 (delete old emit.py).

---

### S3A.4 — formalize MIR→IIR lowerings as `Lowering` instances

**Goal.** Realize P3 — typed pass dispatch via the framework.

**Files.**
- [sorted_array/lowerings.py](../src/srdatalog/ir/dialects/relation/sorted_array/lowerings.py) — main MIR→IIR lowerings
- [d2l/__init__.py](../src/srdatalog/ir/dialects/relation/d2l/__init__.py) — D2L-specific lowerings
- Per-dialect `L^out` registration on the dialect's `DIALECT` record

**Approach.** For each existing ad-hoc Python lowering function:
1. Wrap it in a `Lowering` instance with `matches`, `apply`, `name` fields.
2. Add to the source dialect's `DIALECT.lowerings` list.
3. Update the `Lowering` dataclass with `consumes: tuple[str, ...]` and
   `produces: tuple[str, ...]` (the dialect names of input and output ops).

**Risk.** Low-medium. Lowerings exist; this is wrapping + registering.

**Test gate.** Unchanged C++ byte-equivalence. New `tests/test_lowering_registry.py`
asserts: every documented MIR op has a registered lowering on every relation
dialect that supports it.

**Commit shape.** 2-3 commits, one per dialect.

---

### S3A.5 — formalize R1–R5 as `Rewrite` instances on `sorted_array.R`

**Goal.** Realize P3 for rewrites. R1–R5 from
[`ir_lowering_semantics.md`](./ir_lowering_semantics.md) §11 exist in
code (count-as-product, hint introduction, negation pre-narrow, etc.) but
not as `Rewrite` instances.

**Approach.** For each R1–R5:
1. Identify the existing implementation.
2. Wrap as a `Rewrite` instance with `matches`, `apply`, `name`.
3. Register on `sorted_array.DIALECT.rewrites`.

**Risk.** Low. Rewrites already work; this is wrapping + registering.

**Test gate.** Unchanged byte-equivalence. New unit tests apply each
rewrite in isolation and assert the output IIR via Print_i.

**Commit shape.** 1-2 commits.

---

### S3A.6 — `PassDriver.run` dispatches by type

**Goal.** Make the framework actually run, instead of being a no-op stub.

**Files.** [core/passes.py:73-87](../src/srdatalog/ir/core/passes.py#L73-L87).

**Approach.**

1. Replace the no-op loop with: walk the program; for each op, look up
   matching `Lowering` / `Rewrite` instances in the registered dialects;
   apply them (rewrites to fixpoint, then lowerings).
2. Add `consumes` / `produces` topo-ordering for lowerings.
3. On unmet dependency, raise `PassDependencyError` with the missing
   dialect name.

**Open design decision** — see §9 below.

**Risk.** Medium. New behavior, but starts from a no-op so nothing regresses.
Production code keeps using direct calls until S3A.4 / S3A.5 land lowerings
into the registry.

**Test gate.** New `tests/test_pass_driver_dispatch.py` — register a few
toy lowerings on a toy dialect; assert the driver dispatches them in
topo order and raises on unmet deps.

**Commit shape.** 1-2 commits.

---

### S3A.7 — wire `Dialect.verifier` into PassDriver

**Goal.** Enforce D10 — verification at every level.

**Approach.**

1. After each pass run, `PassDriver` invokes `d.verifier(prog)` for every
   dialect. On `VerificationError`, abort with the pass name attached.
2. Add a no-op verifier to each existing dialect as scaffolding (real
   per-dialect verification grows as a follow-up).

**Risk.** Low.

**Test gate.** New `tests/test_pass_driver_verify.py` — register a dialect
whose verifier always fails; assert `PassDriver.run` aborts cleanly.

**Commit shape.** 1 commit.

---

### S3A.8 — per-`Codegen` plugin registry; explicit `register_*` calls

**Goal.** Kill A6 (module-global mutable state) and A7 (import-time
side-effect registration).

**Files.**
- [target/cuda/plugin.py:147-148](../src/srdatalog/ir/dialects/target/cuda/plugin.py#L147-L148) — `_PLUGIN_REGISTRY`, `_DEFAULT_PLUGIN`
- [d2l/__init__.py:56](../src/srdatalog/ir/dialects/relation/d2l/__init__.py#L56) — side-effect import

**Approach.**

1. Move plugin registry off the module onto a `CudaCodegen` instance
   (post-S3A.0 location).
2. Replace `register_index_plugin(plugin)` →
   `register_index_plugin(codegen, plugin)`.
3. In `d2l/cuda.py`, replace module-level registration with an explicit
   `register_d2l_cuda_plugin(codegen)` function.
4. Delete the side-effect import at `d2l/__init__.py:56`.
5. Find every code path that needs D2L registered; add the explicit call
   at the right level (likely `codegen/cuda/api.py`).

**Risk.** Medium. Threading the codegen instance is mechanical but touches
many call sites. Risk of missing a register call → silent fallback to
default plugin → wrong output. Mitigation: temporarily log the
"fell back to default" branch; run full suite; ensure no test trips it.

**Test gate.** Pre-flight tests 6.1 + 6.2 turn green. Existing
byte-equivalence still passes.

**Commit shape.** 2 commits: 1 (introduce per-codegen registry, keep
module-level as deprecated shim) + 1 (delete shim, fail discipline tests
on remaining usage).

---

### S3A.9 — independent cleanups

Two small items unrelated to the framework realization but worth bundling:

**S3A.9a — stop double-running HIR.**
[pipeline.py:87-88](../src/srdatalog/ir/pipeline.py#L87-L88) calls
`compile_to_hir` then `compile_to_mir`, which re-runs `compile_to_hir`.
Change `compile_to_mir(program, *, hir=None, ...)` to accept an existing
HIR. **Commit shape:** 1 commit.

**S3A.9b — relocate `block_group.py` emit functions.** The dialect file
contains target-specific emit functions
([block_group.py:107+](../src/srdatalog/ir/dialects/parallel/data/block_group.py#L107))
that pull in cuda context helpers. Move emit functions out into
`codegen/cuda/render/parallel_data.py` (post-S3A.0). The dialect file
keeps only its op definitions. **Commit shape:** 1 commit.

## 8. Suggested execution order

Smallest mechanical change first; risky / high-blast-radius last.

| # | Task | Why this slot |
|---|---|---|
| 0 | Pre-flight tests (6.1, 6.2) | Define "done" for S3A.8 |
| 1 | **S3A.0** | Foundational rename; resolves vocabulary; everything below assumes it |
| 2 | **S3A.9a** | Tiny mechanical warm-up |
| 3 | **S3A.1** | Print_i unblocks testing all later tasks |
| 4 | **S3A.6** | PassDriver becomes real (no-op → real) — needed for later tasks to register against |
| 5 | **S3A.4** | Lowerings as data; uses S3A.6 |
| 6 | **S3A.5** | Rewrites as data; uses S3A.6 |
| 7 | **S3A.2** | IIR-as-data refactor; uses S3A.1 + S3A.4 |
| 8 | **S3A.3** | Renderer registry — biggest blast radius; do after lowering is data |
| 9 | **S3A.9b** | block_group cleanup; piggybacks on S3A.3's renderer file layout |
| 10 | **S3A.7** | Verifier wiring — last because verifiers grow over time |
| 11 | **S3A.8** | Per-Codegen registry + explicit register; piggybacks on S3A.0 layout |

Each task ships as its own PR.

## 9. Open design decision (resolve before S3A.6)

**Question.** When a pass `P` declares `consumes=("dialect_X",)` but
`dialect_X` is not registered with the `Compiler`, what should
`PassDriver` do?

| Behavior | Pros | Cons |
|---|---|---|
| **Abort with `PassDependencyError`** | Loud failure; explicit | Pipelines must spell out every dialect they use |
| **Auto-skip the pass with a warning** | Convenient; pipelines work with subsets | Silent under-application of optimization; bug-bait |
| **Auto-skip silently** | Matches LLVM PassManager defaults | Worst of both — neither loud nor explicit |

**Recommendation: Abort.** Matches the framework's existing posture (loud
failure over silent fallback — see [`design_principles.md`](./design_principles.md)
A4 / A8). A pipeline opting out of a dialect's passes does so by not
registering those passes, not by not registering the dialect.

**Decide explicitly before writing S3A.6 code.**

## 10. What this plan deliberately does NOT include

- **A new IR layer (e.g. LIR for explicit scheduling).** Considered and
  rejected: would not help P1/P3 realization. Defer until a concrete
  forcing function appears (e.g. multi-target, scheduling rewrites).
- **Target dialect ops** (Model A from earlier discussion). GPU codegen
  stays a "dumb printer" — no ops, no lowerings. If CUDA-specific
  transformations as data become valuable later, introduce
  `dialects/target/cuda/` then; it's a distinct concern from `codegen/cuda/`.
- **`iir.cf.GridStrideLoop` / `iir.cf.ParallelFor` cleanup.** These are
  CUDA-flavored ops in the cross-target IIR layer — a known smell, but
  fixing them doesn't help P1/P3. Defer (likely until a second codegen
  exists, then choose: move to target dialect ops, or parameterize with
  a strategy field).
- **Stage 3B** — HIR/MIR types onto `Op`/`Type` subclasses with
  `@dataclass(frozen=True, slots=True)` per D1–D4. Planning only until
  Stage 3A lands; revisited after.
- **CPU/WASM target.** Strong forcing function for the design but out
  of scope for the cleanup work.
- **`complete_runner.py` templating.** Orthogonal — separate radar item.
