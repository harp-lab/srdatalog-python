---
orphan: true
---

# Phase Decomposition Redesign — Unified Spec

This is the architectural document that supersedes the piecemeal A3 +
Layer 3 cleanup work. It frames every remaining migration PR against
ONE load-bearing invariant — the ACID test in section 1 — and shows
where today's compiler fails that invariant and what shape the work
takes to satisfy it.

Companion docs (plain refs; sphinx-myst breaks on local markdown
links in some build modes — never use `[name](path.md)`):

- Spec: docs/compiler_redesign.md — the spine; defines the three Pass
  kinds, the LowerCtx contract, the pragma-as-partial-evaluation
  model. This redesign is the next architectural layer above it.
- Spec: docs/code_discipline.md — D-rules + R-rules + CI enforcement.
  This redesign proposes D20 (additive-contract rule) in section 7.
- Spec: docs/concept_glossary.md — IR / Dialect / Pass / Plugin
  definitions; section 13 introduces the four topology checks; this
  spec extends them to the per-target axis.
- Spec: docs/phase_a_mir_onto_op.md — MIR-onto-Op (A1 + A2). Stays
  merged unchanged.
- Spec: docs/phase_b_lowering_dispatcher.md — per-MIR-op @lowering
  (the kernel-body slice). Stays merged unchanged. Phase R below
  extends the same pattern to the runner-shell slice.
- Spec: docs/phase_c_pragma_materialization.md — typed pragmas. Stays
  merged unchanged.
- Spec: docs/phase_d_hir_passes.md — HIR ProgramPass. Stays merged
  unchanged.
- Spec: docs/phase_e_plugin_extensibility.md — entry-point discovery.
  Extended in Phase T (see section 3.2): each dialect plugin must
  split into a data contribution + per-target render contribution.
- Spec: docs/phase_a3_remove_deprecated_bool_fields.md — A3 PRs.
  **Superseded by this spec.** See section 8 for the disposition
  rationale.
- Spec: docs/phase_f5_declarative_pipeline.md — declarative pipeline
  shims. The current `DEFAULT_KERNEL_PIPELINE` is the surface this
  spec generalizes from "1 target hard-wired" to "N targets selected
  by the Compiler at run time".

## 1. The ACID test (load-bearing invariant)

> **For ANY new addition — new algorithm, new pragma, new index type,
> new aggregation, or new backend target (CPU/TBB, OpenMP, SYCL,
> ROCm, MLIR, plain interpreter, WASM) — the addition must be purely
> additive: zero edits to existing source files. The contribution is
> a plugin: new typed ops + `@lowering` rules + `@register_render`
> registrations + an entry-point declaration.**

This is the single criterion the redesign must satisfy. Every other
constraint (the three Pass kinds, the small LowerCtx, the four
topology checks, the typed-pragma model) is in service of this one
invariant. The compiler is correct iff every reasonable extension
the user might want is a purely additive plugin.

> **Sub-clause (Amendment 1).** Every operational semantic —
> termination conditions, cross-stratum dependencies, delta-variant
> ordering, multi-head emission, kernel fusion, pragma compatibility,
> view-slot layout — must be expressed as **typed IR ops or typed
> metadata**, not as string-tagged dispatch inside renderers.
> Renderers may only dispatch on op TYPE, never on a string field of
> an op. If a backend needs to know "is this a count phase or a
> materialize phase?", that question is answered by which IR-op type
> (e.g. `CountKernelDef` vs `MaterializeKernelDef`) is being
> rendered — never by reading `op.phase == "count"`. See § 10 for
> the full set of typed concerns this sub-clause covers and the
> renderer-side discipline test D21 in § 7.

> **Sub-clause (Amendment 2 — no-context principle).** Information
> flows between passes ONLY through the IR. No pass-to-pass
> through-state contexts (`LoweringCtx`, `KernelCtx`, `InitialProg`,
> `CudaRenderCtx`, or any successor with named-field schema). Three
> kinds of data, three disjoint mechanisms:
>
> 1. **Cross-pass data** (produced by pass A, consumed by pass B):
>    must be a typed IR op. If pass B reads it from a `ctx.field`,
>    that field is a missed IR op — the IR pass A produced is
>    incomplete. Adding new cross-pass data = new IR op, never new
>    ctx field.
> 2. **Intra-pass scratch** (fresh-name counters, lexically-scoped
>    variables threaded into one render's nested emit): function-
>    local state on the stack frame. Never a `ctx` parameter.
> 3. **Services** (`compiler`, `name_gen`, `plugin_registry`):
>    immutable, injected at the start of a pass. May travel in a
>    `Services` handle whose schema is service registrations, NOT
>    feature-by-feature accumulated fields. Adding a service = new
>    typed registration, not a new attribute.
>
> Codified as discipline rule D22 (see § 7). The shape-fix for
> Phase T is § 3.2.1 (rewritten by Amendment 2).

### 1.1 What the ACID test forbids

After this redesign lands, any PR that needs to EDIT an existing
source file to add a feature is a discipline violation. The PR must
instead extend by REGISTRATION:

- New algorithm → new typed MIR op + `@lowering(target=IIR_*, source=NewOp)`.
- New pragma → new `Pragma` subclass + `@pragma_handler(NewPragma, on=mir.X)`.
- New index type → new `Op` vocabulary in a new `relation.<name>` dialect +
  per-target `@register_render(Op, target=T)` for each target the user wants.
- New aggregation → same as new algorithm.
- New backend target → new plugin package with `@register_render(Op, target='cpu_tbb')`
  for every existing IIR op + a new scheduler op family for the RIR
  (see section 3.1).

CI mechanism: discipline rule D20 (proposed in section 7) blocks PRs
that edit existing `src/srdatalog/ir/dialects/**` or
`src/srdatalog/ir/codegen/**` files in service of a feature add.

### 1.2 What the ACID test does NOT forbid

The redesign-internal cleanup PRs DO edit existing files — that is
the work of getting to the additive end state. D20 is enforced only
on PRs landing AFTER this spec is approved AND the migration plan
in section 6 is complete. Migration PRs themselves are exempt.

It also does not forbid:

- Bug fixes that touch one existing branch (e.g., a typo, an
  off-by-one).
- Performance optimizations that don't change the feature surface
  (e.g., caching a registry lookup).
- Discipline-rule amendments (per code_discipline.md §9).

The criterion is "adding feature X edits no existing file". Not "no
PR ever touches anything".

## 2. Diagnosis — concrete violations (line-numbered)

Today the redesign satisfies the ACID test for ONE slice of ONE
target: per-MIR-op `@lowering` migrations for kernel-body lowering
on CUDA (Phase B, Wave 2A). Outside that slice the compiler is
imperative monoliths threading pragma state, typed ops with CUDA
C++ shape baked into their fields, and pipeline scaffolding hard-
wired to one target. Three sub-sections, each with file:line cites.

### 2.1 Runner monolith (Phase R targets this)

Three modules under `src/srdatalog/ir/codegen/cuda/` form an
imperative monolith of branches keyed on pragma / phase / scheduler
concerns. Total ~198 if-statements across the three files; the
table below lists the ones that gate emission shape (i.e., the ones
that would each need a parallel `if target == 'cpu_tbb'` branch
under the current shape).

`orchestrator.py` (895 LOC) — drives `step_N()` method bodies, fixpoint
loops, parallel-group dispatch:

| File:line | Branch key | Emits when true |
|---|---|---|
| orchestrator.py:79 | `isinstance(src_spec, (m.ColumnSource, m.Scan, m.Negation, m.Aggregate))` | source-info extraction |
| orchestrator.py:108-114 | `isinstance(instr, m.ExecutePipeline)` + `dest.rel_name in count_only_rels` | count-only pipeline classification |
| orchestrator.py:132/135/138 | `isinstance(instr, m.{ComputeDelta,ComputeDeltaIndex,MergeIndex})` | canonical-spec collection |
| orchestrator.py:202 | `isinstance(op, m.ExecutePipeline)` | ParallelGroup dispatch |
| orchestrator.py:288/389/428/439/448 | `isinstance(dest, m.InsertInto)` | dest-stream-map collection (5 sites in `_gen_execute_pipeline`) |
| orchestrator.py:494-588 | 14-way `isinstance(instr, m.X)` chain in `gen_instruction_code` (one branch per maintenance op type: ExecutePipeline, ParallelGroup, ComputeDelta, ComputeDeltaIndex, MergeIndex, CheckSize, RebuildIndex, RebuildIndexFromIndex, CreateFlatView, ClearRelation, MergeRelation, InjectCppHook, PostStratumReconstructInternCols) | per-MIR-op CUDA C++ |
| orchestrator.py:633/635/637 | `isinstance(instr, m.ExecutePipeline / m.ParallelGroup / m.ExecutePipeline)` | fixpoint-body dispatch |
| orchestrator.py:705/812 | `isinstance(instr, m.ParallelGroup)` (twice) | sync-after-parallel decisions |
| orchestrator.py:726 | `isinstance(instr, maint_kinds)` | post-maintenance sync |
| orchestrator.py:768/781/792/847 | `isinstance(plan, m.{Block, ExecutePipeline, FixpointPlan, PostStratumReconstructInternCols})` | non-recursive block dispatch |

`complete_runner.py` (1123 LOC) — emits the per-rule
`JitRunner_<rule>` C++ struct + kernel definitions + phase methods +
execute(). Side-effect import of `srdatalog.ir.dialects.relation.d2l`
at line 37 (cross-tree coupling — the runner imports the data
dialect just to trigger CUDA-plugin registration). Branches keyed
on pragma/phase/scheduler concerns:

| File:line | Branch key | Emits when true |
|---|---|---|
| complete_runner.py:99-126 | source-type isinstance fan-out (5 sites: `_src_schema`, `_src_version_cpp`, `_src_mir_version`, `_src_index`, `_root_is_scan`) | per-source-type C++ |
| complete_runner.py:141 | `_root_is_scan` (Scan dispatch) | scan-rooted runner shape |
| complete_runner.py:219 | `if node.dedup_hash:` (count-kernel) | dedup args + per-thread fast path |
| complete_runner.py:286/312/335 | `if node.dedup_hash:` (materialize-kernel, 3 sites) | dedup-table struct + atomicAdd write path |
| complete_runner.py:322/426 | `if tiled_cartesian_eligible:` (2 sites) | tiled-Cartesian materialize variant |
| complete_runner.py:691 | `isinstance(src, m.Negation)` | `force_rebuild=true` view setup |
| complete_runner.py:795 | `if node.block_group:` (setup phase) | BG histogram setup |
| complete_runner.py:861/876/879/882 | `if node.dedup_hash:` (setup phase, 4 sites) | dedup table allocation + memset |
| complete_runner.py:925-926 | `if node.work_stealing:` | `NotImplementedError("WS not yet ported")` |
| complete_runner.py:937 | `is_fused_eligible = not is_count and not node.dedup_hash` | fused-kernel eligibility |
| complete_runner.py:955-963 | 4-way conjunction (`not work_stealing and not block_group and not dedup_hash`) | tiled-Cartesian eligibility |
| complete_runner.py:987-1017 | `if node.dedup_hash` / `not node.block_group and not node.dedup_hash` (fused dispatch gating) | dispatch into BG/dedup/baseline launch sequences |
| complete_runner.py:1029-1108 | BG vs baseline dispatch via `is_block_group`, `is_dedup_hash` kwargs threaded into 6 phase-method emitters (`emit_launch_count`, `emit_launch_fused`, `emit_launch_materialize`, `emit_execute`, …) | per-pragma phase methods |

`runner.py` (802 LOC) — phase-method templates + execute()
dispatcher. Pure-template emitters parameterized by bools:

| File:line | Branch key | Emits when true |
|---|---|---|
| runner.py:47-57 | `is_dedup_hash` (kwarg → branch) | dedup arg threading + fast-path shape |
| runner.py:70 | `if is_block_group:` | BG launch variant |
| runner.py:180 | `if ep.dedup_hash:` (launch_materialize) | dedup kernel args |
| runner.py:204/240 | `if ep.block_group:` (launch_materialize + launch_fused) | BG histogram launch |
| runner.py:373-374 | `is_block_group=False, is_dedup_hash=False` kwargs (emit_launch_params_struct) | per-pragma struct field generation |
| runner.py:395-417 | `if is_block_group` / `if is_dedup_hash` (2 sites in struct emit) | per-pragma launch-param fields |
| runner.py:494-617 | `is_count`, `is_block_group`, `is_dedup_hash` kwargs in `emit_execute` (4 branches) | per-pragma execute() dispatcher |
| runner.py:654 | `if ep.block_group:` (emit_execute_fused) | BG fused dispatch |

The `RunnerGenState` dataclass (`context.py:179-195`) carries five
pragma/phase bools (`is_balanced`, `is_work_stealing`, `is_block_group`,
`is_dedup_hash`, `is_count`) directly into every runner-emit hook —
exactly the imperative-monolith pattern `compiler_redesign.md` §1
diagnoses for the kernel-body lowering, applied at the runner-shell
layer. Five bools × ~6 phase-method emitters = ~30 conditional sites.

**Total imperative branches in the runner shell that key on
pragma / phase / scheduler concerns: ~57** (the count cited in the
brief — confirmed by the tables above; some isinstance sites
collapse to one row, some bool sites span multiple emitters).

#### 2.1.1 The `ep_has_X(...)` helper anti-pattern (parked A3 work)

The parked A3-1/A3-2/A3-3 branches (`feat/a3-1-remove-dedup-hash-bool`,
`feat/a3-2-remove-work-stealing-bool`, `feat/a3-3-remove-block-group-bool`)
proposed introducing `ep_has_pragma(ep, PragmaCls) -> bool` helpers
and routing each runner branch through them (per
`phase_a3_remove_deprecated_bool_fields.md` §3.1, step 2). Concretely:

```python
# A3-1 proposed:
if ep_has_pragma(ep, DedupHash):  # was: if ep.dedup_hash:
  emit_dedup_table_struct(...)
```

This is the SAME imperative anti-pattern in different clothes. The
branch key changed from `ep.dedup_hash` to `ep_has_pragma(ep, DedupHash)`,
but every emit site still keys on a pragma to choose a code shape.
The code shape (BG launch vs baseline launch, dedup arg threading
vs none) is still hard-wired into the runner monolith. The pragma
flag is still a branch trigger, not a partial-evaluation trigger.

Phase R below dissolves both the bool reads AND the `ep_has_X(...)`
reads by lowering the runner shell through a new RIR (Runner IR)
with one typed `KernelLaunch(scheduler=BGScheduler(...))` op per
pragma-induced shape. Each scheduler variant gets its own
`@register_render`. The branch becomes an op-type dispatch — exactly
the shape Phase B established for kernel bodies.

### 2.2 Target leakage into IR (Phase T targets this)

The IIR op vocabulary, the LowerCtx, and the index-plugin registry
all leak target-specific shape into ostensibly target-agnostic IR
layers. Three categories:

#### 2.2.1 IIR ops with CUDA-specific names and field shapes

`relation/sorted_array/ops.py:193-258` defines `SaTiledCartesian2D`
with twenty fields encoding the exact CUDA C++ template (smem
tile, ballot writes, lane var, warp_in_block index, kCartTileSize
constant). The docstring at lines 196-232 IS the C++ text. The op
is named for what it renders to, not for what it semantically does
(a tiled Cartesian dispatch over two sources, which on CPU/TBB
would render as nested for-loops with manual SIMD, on SYCL as
nd_range parallel_for, etc.).

`parallel/data/block_group.py:66-107` defines `BgRootCjMulti`
("BG dispatch shape for root multi-source ColumnJoin"). The shape
is the CUDA warp-cumulative-work-balance dispatch, encoded as
field types: `BgSourceSpec` per-source descriptor (line 46) carries
`view_var`, `handle_var`, `degree_var` — all C++ identifiers, all
chosen by the CUDA renderer.

Other IIR ops with CUDA shape baked in:

| Op | File | What's CUDA-specific |
|---|---|---|
| `SaTiledCartesian2D` | sorted_array/ops.py:193 | smem tile layout, kCartTileSize, lane var, warp-in-block index |
| `BgRootCjMulti` | parallel/data/block_group.py:66 | warp-cumulative dispatch, BgSourceSpec C++ identifier fields |
| `BgSourceSpec` | parallel/data/block_group.py:46 | C++ identifier fields (view_var, handle_var) |
| `SaPrefCoop` | sorted_array/ops.py:107 | cooperative warp-tile execution model |
| `SaPrefSeq` | sorted_array/ops.py:124 | per-thread sequential execution model |
| `SaChildRange` | sorted_array/ops.py:156 | warp-tile parameter `tile` in signature |
| `GridStrideLoop` | iir/cf/ops.py:237 | CUDA grid-stride convention |
| `LaneZeroGuard` | iir/cf/ops.py:282 | CUDA lane-zero predicate |
| `TiledBallotBlock` | iir/cf/ops.py:341 | CUDA ballot intrinsic surface |

Phase T renames each to a target-agnostic semantic name; the CUDA
shape moves into the CUDA renderer's per-op handler. Section 3.2
gives the rename table.

#### 2.2.2 LowerCtx CUDA-render state

The current `LoweringCtx` in
`src/srdatalog/ir/dialects/relation/sorted_array/lowerings/__init__.py:92-155`
has ~15 fields. The compiler_redesign.md §5 target is 5 fields,
target-agnostic. Today's fields:

| Field | Line | Target-agnostic or CUDA-render? |
|---|---|---|
| `view_var_names: dict[str, str]` | 102 | CUDA — string names of C++ view variables |
| `is_counting: bool` | 103 | (pragma scratch; Phase A3-5 keeps as scratch) |
| `output_var: str` | 105 | CUDA — name of C++ OutputContext variable |
| `output_var_overrides: dict[str, str]` | 108 | CUDA — per-relation OutputContext name map |
| `rel_index_types: dict[str, str]` | 112 | CUDA — per-relation C++ index template name |
| `view_slot_bases: dict[str, int]` | 116 | CUDA — per-source positional slot offset |
| `dedup_hash: bool` | 122 | (pragma scratch; pragma's own lowering should flip) |
| `tiled_cartesian: bool` | 129 | (pragma scratch) |
| `tiled_cartesian_valid_var: str` | 134 | CUDA — name of C++ ballot validity var |
| `ws_enabled: bool` | 141 | (pragma scratch) |
| `bg_enabled: bool` | 155 | (pragma scratch) |
| `ws_cartesian_valid_var: str` | (~155+) | CUDA — name of C++ WS ballot validity var |
| `ws_cartesian_bound_vars: list[str]` | (~155+) | CUDA — names of C++ WS-loop-bound vars |
| `neg_pre_narrow: dict[int, NegPreNarrowInfo]` | (~155+) | CUDA — per-handle pre-narrow info threaded into C++ |
| `dedup_hash_vars: list[str]` | (~155+) | CUDA — names of C++ vars feeding dedup hash key |

The `view_var_names`, `output_var`, `output_var_overrides`,
`view_slot_bases`, and `dedup_hash_vars` fields all carry CUDA
identifier strings; they are render-state, not lowering-state. They
belong in a target-private `CudaRenderCtx`, NOT in the generic
`LowerCtx`. Phase T splits them out.

Note: the legacy `CodeGenContext` in
`src/srdatalog/ir/codegen/cuda/context.py:255-338` is the CUDA-emit-
side equivalent of the same disease — 35+ fields, half of them
pragma scratch (`bg_enabled`, `ws_enabled`, `dedup_hash_enabled`,
`tiled_cartesian_enabled`, …), half of them CUDA identifier scratch.
Phase T collapses both into `LowerCtx (5 fields, target-agnostic) +
CudaRenderCtx (~15 fields, target-private)` — preserving the
information but separating concerns.

#### 2.2.3 CUDA-private helpers masquerading as IR-side scaffolding

Two large files under `codegen/cuda/` are CUDA render-side helpers
that the IIR ops call into:

- `view_slots.py` (324 LOC) — deduplicates pipeline source views,
  computes view-slot offsets, emits `auto view_X = views[i];`
  blocks. Every line is CUDA C++ string composition.
- `materialized.py` (531 LOC) — materialized binary-join emission
  using Thrust primitives + three `__global__` CUDA kernels. Every
  line is CUDA C++ / Thrust string composition.

These belong under `codegen/cuda/` (where they already live) — but
the IIR-layer lowerings call them as if they were target-agnostic
helpers (e.g., `view_count` lookups from `relation.d2l.view_counts_for_specs`
are consumed by `EmitViewDeclsShim` in `default_pipelines.py:241`).
The right shape is: `view_counts` is a CUDA-render concern, not a
lowering-time concern — only the renderer knows how many CUDA view
slots a FULL D2L source occupies. Phase T moves the view-count
inquiry into the renderer's setup pass.

#### 2.2.4 Index plugins reaching into the CUDA-target's private registry

`src/srdatalog/ir/dialects/relation/d2l/cuda.py:65` calls
`register_index_plugin(two_level_plugin)`, where
`register_index_plugin` and `_PLUGIN_REGISTRY` are defined in
`src/srdatalog/ir/codegen/cuda/plugin.py:147-155`. The data dialect
(`relation/d2l/`) reaches across the tree into the CUDA-target's
private module-global registry.

Two coupling failures:

1. Adding a new render target (CPU/TBB) would require the D2L
   dialect to ALSO register a new CPU plugin in a new
   `codegen/cpu_tbb/plugin.py` registry — meaning the data dialect
   knows about every target.
2. The `register_index_plugin` call at module import time creates
   a cross-tree, side-effecting dependency that
   `complete_runner.py:37` makes explicit (a direct
   `import srdatalog.ir.dialects.relation.d2l` for its side
   effects).

Phase T splits index-plugin registration: the data dialect ships
the data semantics (the `D2lSegmentLoop` IIR op + its print form)
target-agnostic; the per-target render template is shipped as a
target-side render contribution: `@register_render(D2lSegmentLoop,
target='cuda')` in `srdatalog/codegen/cuda/render/relation_d2l.py`,
parallel `@register_render(...)` in `srdatalog_cpu_tbb/codegen/...`.

### 2.3 Target monopolization (Phase B2 targets this)

The pipeline is hard-wired to CUDA. The `target` parameter exists
on one entry point (`compile_pipeline`), is validated against the
literal `'cuda'`, and is dropped on the floor; the rest of the
pipeline calls CUDA-only helpers directly.

| File:line | Target-monopolization site | What it does |
|---|---|---|
| `src/srdatalog/ir/codegen/cuda/api.py:38` | `Target = Literal['cuda']` | One-target type alias |
| `src/srdatalog/ir/codegen/cuda/api.py:41` | `compile_pipeline(ep, *, target: Target = 'cuda')` | Accepts target; rejects anything else |
| `src/srdatalog/ir/codegen/cuda/api.py:48-49` | `if target != 'cuda': raise ValueError(...)` | The target check IS the entire target dispatch |
| `src/srdatalog/ir/codegen/cuda/api.py:68` | `compile_runner(ep, ...)` | No `target` parameter at all |
| `src/srdatalog/ir/codegen/cuda/api.py:98` | `compile_kernel_body(ep, *, is_counting, ...)` | No `target` parameter |
| `src/srdatalog/ir/codegen/cuda/api.py:149-150` | `from srdatalog.ir.core.dialect import Compiler` + `from srdatalog.ir.default_pipelines import DEFAULT_KERNEL_PIPELINE` | Pipeline is a global singleton, not per-target |
| `src/srdatalog/ir/default_pipelines.py:343-364` | `class CudaRenderShim(ProgramPass):` — name has `Cuda` baked in | Shim name encodes target |
| `src/srdatalog/ir/default_pipelines.py:264-298` | `LowerScanPipelineShim` constructs `LoweringCtx(...)` directly | No target dispatch — single LoweringCtx assumed |
| `src/srdatalog/ir/default_pipelines.py:301-340` | `VerifyRenderabilityShim` hardcodes `target='cuda'` (line 339) | Verifier is per-target but the target is a string literal |
| `src/srdatalog/ir/default_pipelines.py:380-387` | `DEFAULT_KERNEL_PIPELINE = [AssignHandlesShim(), CollectViewSpecsShim(), EmitViewDeclsShim(), LowerScanPipelineShim(), VerifyRenderabilityShim(), CudaRenderShim()]` | Five of six shims are CUDA-specific |
| `src/srdatalog/ir/pipeline.py:17-23` | `from srdatalog.ir.codegen.cuda.{batchfile,complete_runner,main_file,orchestrator} import …` | The whole-program pipeline imports CUDA modules directly |
| `src/srdatalog/ir/pipeline.py:80` | `compile_program(program, project_name)` | No target parameter; CUDA hardcoded |
| `src/srdatalog/ir/codegen/cuda/complete_runner.py:37` | `import srdatalog.ir.dialects.relation.d2l  # noqa: F401` | Side-effect import couples data dialect to CUDA target |
| `src/srdatalog/ir/codegen/cuda/plugin.py:147` | `_PLUGIN_REGISTRY: dict[str, IndexPlugin] = {}` | Module-global, target-private — not per-Compiler |
| `pyproject.toml:64-70` | `[project.entry-points."srdatalog.plugins"]` | One plugin group — no separation between data dialects and render targets |

Concretely: even though `compile_kernel_body(ep, target='cuda')`
has been refactored to call `Compiler().run(..., pipeline=DEFAULT_KERNEL_PIPELINE)`
(api.py:152-164), the pipeline list itself contains
`LowerScanPipelineShim` and `CudaRenderShim` as named subclasses;
swapping a CUDA target for CPU/TBB requires constructing a different
pipeline list — and the user has no API to do so. The Compiler
doesn't choose the pipeline; the entry-point hardcodes it.

`DEFAULT_KERNEL_PIPELINE` shim class names that encode CUDA:

| Shim | Suggested target-parametric name |
|---|---|
| `CollectViewSpecsShim` | `CollectViewSpecsShim` (already neutral; behavior is CUDA-shaped though — view_specs are CUDA C++ slots) |
| `EmitViewDeclsShim` | `EmitViewDeclsShim` (CUDA C++ string emission) |
| `LowerScanPipelineShim` | `LowerKernelBodyShim` (drop CUDA-specific `LoweringCtx` construction; pass `target` through to construct a per-target LowerCtx) |
| `CudaRenderShim` | `RenderShim` (target-parametric; resolves the per-op `@register_render(target=T)` at render time) |
| `VerifyRenderabilityShim` | `VerifyRenderabilityShim` (already parametric in shape; just stop hardcoding `target='cuda'` at line 339) |

Phase B2 makes the Compiler the target authority: `Compiler.run(prog,
pipeline=DEFAULT, target='cuda')` chooses the target at run time;
the render-stage shim consults the registered `@register_render(Op,
target=T)` for the chosen T. Multi-target compilation
(`targets=['cuda', 'cpu_tbb']`) emits one runner per target from the
same IR.

## 3. The three-axis fix

The three violations in section 2 are independent on the surface but
deeply interleaved in execution order. The fix is three named phases
running in a specific sequence:

- **Phase R** (Runner IR) — replaces the runner monolith
  (section 2.1) with a typed RIR + per-RIR-op `@lowering` and
  `@register_render`. Mirrors Phase B's shape, one IR layer up.
- **Phase T** (Target abstraction) — splits LowerCtx (section 2.2.2),
  renames CUDA-shaped IIR ops to target-agnostic semantic ops
  (section 2.2.1), moves CUDA-private helpers under `codegen/cuda/`
  (section 2.2.3), splits index plugins (section 2.2.4).
- **Phase B2** (Target parametricity) — makes `Compiler.run(..., target=T)`
  the authoritative target dispatch (section 2.3); the pipeline
  becomes target-parametric; multi-target compilation enabled.

Sequencing rationale: Phase T must precede Phase B2 (the per-target
selection mechanism in Phase B2 needs the target-agnostic op
vocabulary that Phase T produces). Phase R is independent of T and
B2 at the IR-vocabulary level (RIR is a new layer; its ops are
new), but Phase R's renders are per-target via the same mechanism
B2 introduces, so the LAST PRs of Phase R land after Phase B2's
target-parametricity is in place. The migration plan in section 6
reconciles this.

### 3.1 Phase R — Runner IR (RIR)

A new IR layer between MIR and rendered text, dedicated to runner-
shell concerns (the per-rule C++ struct + kernel launch sequencing
+ fixpoint loop scaffolding + step dispatch). RIR is target-agnostic;
its renders are per-target.

#### 3.1.1 RIR op vocabulary (initial proposal — refine in implementation)

```python
# srdatalog/ir/dialects/rir/ops.py

@dataclass(frozen=True, slots=True)
class RunnerStruct(Op):
  '''Per-rule runner struct shape. Renders as `struct JitRunner_<rule>`
  on CUDA; as `class TbbRunner_<rule>` on CPU/TBB; as `function <rule>_run`
  on a WASM target.'''
  rule_name: str
  members: tuple[RunnerMember, ...]
  ctor: RunnerCtor

@dataclass(frozen=True, slots=True)
class StepDispatch(Op):
  '''Top-level step dispatch (one switch arm per step_N method).'''
  steps: tuple[StepRef, ...]
  switch_arms: tuple[SwitchArm, ...]

@dataclass(frozen=True, slots=True)
class FixpointLoop(Op):
  '''Recursive-step fixpoint loop. Body is RIR; termination is RIR.'''
  setup: tuple[Op, ...]
  body: Block
  termination: Op  # e.g. CheckEmptyDelta

@dataclass(frozen=True, slots=True)
class KernelDef(Op):
  '''Definition of one kernel by name and phase (count / materialize /
  fused). Body is IIR (the existing per-MIR-op lowering output).'''
  name: str
  phase: KernelPhase  # COUNT | MATERIALIZE | FUSED
  body: Op  # IIR

@dataclass(frozen=True, slots=True)
class KernelLaunch(Op):
  '''Launch a previously-defined kernel with a per-launch layout +
  scheduling strategy. `scheduler` is a typed op (see below).'''
  kernel: KernelRef
  layout: LaunchLayout  # grid/block dims (CUDA) | thread count (TBB)
  scheduler: Op  # one of: PlainScheduler, WSScheduler, FanOutScheduler, BGScheduler

@dataclass(frozen=True, slots=True)
class PlainScheduler(Op):
  '''Baseline launch — one kernel call per phase per stream.'''
  grid_dim: str
  block_dim: str

@dataclass(frozen=True, slots=True)
class WSScheduler(Op):
  '''Work-stealing launch — kernel-side task queue + steal loop.'''
  task_queue: TaskQueueRef
  steal_loop: SchedulerLoopShape

@dataclass(frozen=True, slots=True)
class FanOutScheduler(Op):
  '''Fan-out launch — explore-then-emit kernel split (jit_fanout_executor).'''
  queue: TaskQueueRef
  dispatcher: SchedulerDispatcherShape

@dataclass(frozen=True, slots=True)
class BGScheduler(Op):
  '''Block-group launch — per-warp work-balance histogram + dispatch.'''
  per_warp_blocks: PerWarpDispatchShape
  balance_setup: BalanceSetupShape

@dataclass(frozen=True, slots=True)
class DedupTable(Op):
  '''Per-runner dedup-table allocation. Capacity is a known-at-compile-time
  power of two; renders as a CUDA `DeviceArray<uint64_t>`.'''
  struct_name: str
  key_type: TypeRef
  capacity: int
```

The shape is one op per pragma-induced runner shape. Today
`complete_runner.py` keys 14+ branches off
`ep.dedup_hash / ep.block_group / ep.work_stealing / is_counting`;
after Phase R the same information is one op type per branch (a
`KernelLaunch(scheduler=BGScheduler(...))` for block-group, a
`KernelLaunch(scheduler=WSScheduler(...))` for work-stealing, …).
The render dispatches per op type.

> **Amendment 1.** The vocabulary listed above is the structural
> skeleton; § 3.1.1.1 expands it into the full ~30-op typed
> vocabulary required by the § 1 sub-clause. In particular,
> `MaintenanceCall` (one op with 11 string-tagged variants),
> `FixpointLoop.termination: Op` (a black-box untyped slot),
> `KernelDef.phase: KernelPhase` (a string-ish enum), and the
> implicit-list-order semantics for steps / variants / multi-head
> emission are replaced with one typed op per concern.

#### 3.1.1.1 Amendment 1 — typed-vocabulary expansion

Eight semantic gaps in the § 3.1.1 vocabulary above leave
operational meaning either string-tagged (`MaintenanceCall.kind`,
`KernelDef.phase`) or implicit (list ordering for steps, delta
variants, multi-head emission; opaque `termination: Op` slot; render-
time view-slot allocation). Amendment 1 fills them. The total RIR
op count grows from ~10 to ~30; per-op work is mechanical.

##### Gap 1 — `MaintenanceCall` decomposes into 11 typed ops

The single `MaintenanceCall(kind: str, args)` op is replaced with
one typed op per MIR maintenance op. Source-of-truth count: the 11
`isinstance(instr, m.X)` branches in
`src/srdatalog/ir/codegen/cuda/orchestrator.py` lines 506-588 (i.e.,
the 14-branch chain cited in § 2.1 minus `ExecutePipeline` and
`ParallelGroup`, which are kernel/group ops, not maintenance).

| RIR op | Signature | MIR source op |
|---|---|---|
| `ComputeDeltaCall` | `(rel_name: str, index: IndexSpec)` | `mir.ComputeDelta` |
| `ComputeDeltaIndexCall` | `(rel_name: str, canonical_index: IndexSpec)` | `mir.ComputeDeltaIndex` |
| `MergeIndexCall` | `(rel_name: str, index: IndexSpec)` | `mir.MergeIndex` |
| `MergeRelationCall` | `(rel_name: str)` | `mir.MergeRelation` |
| `CheckSizeCall` | `(rel_name: str, version: VersionTag)` | `mir.CheckSize` |
| `RebuildIndexCall` | `(rel_name: str, version: VersionTag, index: IndexSpec)` | `mir.RebuildIndex` |
| `RebuildIndexFromIndexCall` | `(rel_name: str, source_index: IndexSpec, target_index: IndexSpec, version: VersionTag)` | `mir.RebuildIndexFromIndex` |
| `CreateFlatViewCall` | `(rel_name: str, version: VersionTag, index: IndexSpec)` | `mir.CreateFlatView` |
| `ClearRelationCall` | `(rel_name: str, version: VersionTag)` | `mir.ClearRelation` |
| `InjectCppHookCall` | `(code: str, rule_name: str)` | `mir.InjectCppHook` |
| `PostStratumReconstructInternColsCall` | `(rel_name: str, canonical_index: IndexSpec)` | `mir.PostStratumReconstructInternCols` |

Each gets its own `@register_render(<Op>, target='cuda')`. Renders
dispatch on op TYPE, never on a `kind: str` field. (No
`MaintenanceCall` op survives the lowering.)

Maintenance-op inventory finding: the spec's § 3.1.2 lowering table
cites "~14 maintenance ops total per orchestrator.py:494-588
dispatch chain". Direct grep of the chain yields **11** maintenance
ops (the 14 figure includes `ExecutePipeline`, `ParallelGroup`, and
one duplicate branch). The 11 typed ops above are the complete set.
§ 3.1.2 is updated below to cite the corrected count.

##### Gap 2 — Fixpoint termination is a typed union

The current `FixpointLoop.termination: Op` slot is implicitly a
CUDA-text-emitting op. Amendment 1 replaces it with a typed
`TerminationCheck` union:

```python
# srdatalog/ir/dialects/rir/ops.py

class TerminationCheck(Op):
  '''Abstract base; concrete subclasses below.'''

@dataclass(frozen=True, slots=True)
class ConvergenceCheck(TerminationCheck):
  '''Halt when all named deltas are empty.'''
  deltas: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class MaxIterationFuel(TerminationCheck):
  '''Halt after N iterations regardless of fixpoint state.'''
  limit: int

@dataclass(frozen=True, slots=True)
class EitherFirst(TerminationCheck):
  '''Halt as soon as ANY contained check fires.'''
  checks: tuple[TerminationCheck, ...]

@dataclass(frozen=True, slots=True)
class AndAll(TerminationCheck):
  '''Halt only when ALL contained checks fire simultaneously.'''
  checks: tuple[TerminationCheck, ...]
```

`FixpointLoop.termination` is retyped from `Op` to `TerminationCheck`.
Each leaf check has its own `@register_render`; composers
(`EitherFirst`, `AndAll`) recurse. Backends render the preferred
shape (CUDA: a `do { ... } while (!converged);` with conjunction
inlined; CPU/TBB: same scaffold, different sync primitive). The
semantic is in the IR.

##### Gap 3 — Cross-stratum dependencies become typed DAG edges

The current `Program.steps: list[Step]` ordering is the dependency
relation, implicitly. Amendment 1 promotes the relationship to
typed metadata:

```python
@dataclass(frozen=True, slots=True)
class Step(Op):
  '''One stratum's step. id is scoped to the Program; deps name
  prior steps whose results this step reads.'''
  id: StepId  # int | str — scoped to the Program
  body: Op    # FixpointLoop | KernelLaunch | Block
  deps: tuple[StepId, ...] = ()

StepId = int  # alias; str also acceptable for human-readable ids
```

`Program.steps` remains a `tuple[Step, ...]`; the list ordering is a
DEFAULT topological order, but backends MAY honor `Step.deps` for
speculative execution / cross-stratum pipelining. (CUDA today
executes sequentially; CPU/TBB may parallelize disjoint subtrees.)

##### Gap 4 — Delta-variant ordering is typed

Semi-naive delta variants today emit in `FixpointLoop.body` list
order. The ordering RELATIONSHIP isn't typed — a backend can't know
"these variants may fuse" vs "these must run sequentially".
Amendment 1:

```python
class VariantOrdering(Enum):
  SEQUENTIAL  = 'sequential'   # order matters; emit in declaration order
  INDEPENDENT = 'independent'  # any order; backend may permute
  FUSABLE     = 'fusable'      # backend MAY fuse into one kernel

@dataclass(frozen=True, slots=True)
class DeltaVariantSet(Op):
  '''A set of semi-naive delta-variant kernel definitions schedules
  together. The unit a backend dispatches.'''
  variants: tuple[KernelDef, ...]   # KernelDef = Count/Materialize/Fused (Gap 6)
  ordering: VariantOrdering
```

`FixpointLoop.body` becomes a tuple of `DeltaVariantSet` (and other
RIR ops). Each set carries its own ordering tag; the render
contract is: SEQUENTIAL renders one launch per variant; INDEPENDENT
permits stream-parallel launches; FUSABLE permits the backend's
fusion pass.

##### Gap 5 — Multi-head insertion is a typed op

A multi-head rule today is a `KernelDef.body` containing N
`InsertInto` ops; the emission ordering between heads is invisible.
Amendment 1:

```python
class HeadOrdering(Enum):
  SEQUENTIAL = 'sequential'   # emit heads in declaration order
  CONCURRENT = 'concurrent'   # disjoint heads; backend may permute

@dataclass(frozen=True, slots=True)
class HeadSpec:
  '''Per-head emission descriptor; not an Op (no render — it's a
  data carrier referenced from MultiHeadInsert).'''
  rel_name: str
  vars: tuple[VarRef, ...]
  dedup_table: DedupTableRef | None

@dataclass(frozen=True, slots=True)
class MultiHeadInsert(Op):
  '''One IR op for the entire multi-head emission of a rule body.'''
  heads: tuple[HeadSpec, ...]
  ordering: HeadOrdering
```

CUDA render: emits one `OutputContext::emit(...)` per head in
declared order for SEQUENTIAL, or in `__syncthreads()`-free
interleaved form for CONCURRENT. CPU/TBB render: similar shape with
TBB primitives. The choice is typed.

##### Gap 6 — Kernel fusion is typed, not string-tagged

The current `KernelDef(name, phase: str, body)` keys fusion off a
`phase` enum (`"count"` / `"materialize"` / `"fused"`). Amendment 1
splits into three distinct ops:

```python
@dataclass(frozen=True, slots=True)
class CountKernelDef(Op):
  '''Count-only phase kernel.'''
  name: str
  body: Op  # IIR

@dataclass(frozen=True, slots=True)
class MaterializeKernelDef(Op):
  '''Materialize-only phase kernel.'''
  name: str
  body: Op  # IIR

@dataclass(frozen=True, slots=True)
class FusedKernelDef(Op):
  '''Two-body fused kernel: count + materialize in one launch.'''
  name: str
  count_body: Op         # IIR
  materialize_body: Op   # IIR
```

The legacy `KernelDef` and `KernelPhase` enum from § 3.1.1 are
removed. Each new op has its own `@register_render`; renders
dispatch on type, never on `op.phase == "count"`.

##### Gap 7 — Pragma composability is typed metadata

Pragma handlers today are unaware of each other; conflicting
pragmas on the same EP (e.g., a work-stealing schedule plus a
block-group schedule) surface as cryptic codegen errors. Amendment 1
adds typed composability metadata to the `Pragma` class itself —
this lives in `core/pragma.py`, NOT in RIR, but is referenced from
RIR for explanation:

```python
# srdatalog/ir/core/pragma.py — extends the existing typed Pragma
# (cf. Spec: docs/pragma_as_typed_object.md § 2 for the base class)

@dataclass(frozen=True, slots=True)
class ComposabilityMeta:
  '''Declares this pragma's compatibility with other pragmas on the
  same EP. Optional; default = composes with everything.'''
  composes_with:     frozenset[type[Pragma]] = frozenset()
  requires_before:   frozenset[type[Pragma]] = frozenset()
  incompatible_with: frozenset[type[Pragma]] = frozenset()

class Pragma:
  # ... existing fields ...
  composability: ComposabilityMeta = ComposabilityMeta()
```

Framework check at `MirPragmaPass.apply` time: for every pair of
pragmas attached to the same EP, if either declares the other in
`incompatible_with`, raise `IncompatiblePragmasError(p1, p2)` with
the (rule, EP, pragma classes) named. `requires_before` enforces
declared application order. This extends the typed-pragma model in
Spec: docs/pragma_as_typed_object.md § 8 (discipline implications)
with a typed-composability rule.

##### Gap 8 — View-slot allocation is typed IR metadata

`view_slots.py` (324 LOC, `src/srdatalog/ir/codegen/cuda/`) computes
view-slot bindings at render time, CUDA-specifically. Amendment 1
promotes the binding to RIR:

```python
@dataclass(frozen=True, slots=True)
class ViewBinding:
  '''Explicit IR-level binding from a source handle to a render-
  target view slot. Targets that don't use slot indexing (CPU/TBB)
  ignore view_slot and base_offset; CUDA renders consume both.
  Carrier struct, not an Op (no render — it's referenced from kernel-
  def metadata).'''
  handle_idx: int
  view_slot: int
  base_offset: int
```

Each `CountKernelDef` / `MaterializeKernelDef` / `FusedKernelDef`
gains a `view_bindings: tuple[ViewBinding, ...]` metadata field.
The MIR→RIR lowering for `mir.ExecutePipeline` populates it (i.e.,
the slot-allocation algorithm moves from CUDA-render-time to
MIR→RIR-lowering-time, and the result is a typed IR field, not a
mutable side-channel). CUDA renders read `view_bindings` directly;
non-slot-using targets ignore it. The Phase T move of
`view_slots.py` (per § 3.2.3) becomes simpler: only the slot-
allocation algorithm moves to MIR→RIR; the per-binding render code
moves to CUDA-render.

##### Final RIR op count

After Amendment 1, the typed RIR vocabulary grouped by concern:

| Concern | Ops | Count |
|---|---|---|
| Structure | `RunnerStruct`, `StepDispatch`, `FixpointLoop`, `Step` | 4 |
| Termination | `ConvergenceCheck`, `MaxIterationFuel`, `EitherFirst`, `AndAll` | 4 |
| Scheduling | `KernelLaunch`, `PlainScheduler`, `WSScheduler`, `FanOutScheduler`, `BGScheduler` | 5 |
| Kernel bodies | `CountKernelDef`, `MaterializeKernelDef`, `FusedKernelDef` | 3 |
| Variant ordering | `DeltaVariantSet` (+ `VariantOrdering` enum) | 1 |
| Multi-head | `MultiHeadInsert` (+ `HeadOrdering` enum, `HeadSpec` carrier) | 1 |
| Dedup | `DedupTable` | 1 |
| Maintenance | 11 typed calls (per Gap 1) | 11 |
| **Total typed Ops** | | **30** |

Carrier dataclasses (not Ops; no `@register_render`):
`HeadSpec`, `ViewBinding`, `ComposabilityMeta`, `StepId` alias,
`VariantOrdering` enum, `HeadOrdering` enum. These are referenced
from Ops above but never rendered directly — they are read by the
render that consumes the parent Op.

#### 3.1.2 MIR → RIR lowerings

One `@lowering(target=RIR, source=mir.X)` per MIR maintenance op,
mirroring Phase B's shape:

| MIR op | Phase R lowering | Produces (RIR) |
|---|---|---|
| `mir.FixpointPlan` | `lower_fixpoint_plan` | `FixpointLoop(...)` |
| `mir.ParallelGroup` | `lower_parallel_group` | `Block(...)` of `KernelLaunch` |
| `mir.ExecutePipeline` (no pragmas) | `lower_execute_pipeline_plain` | `KernelLaunch(scheduler=PlainScheduler(...))` |
| `mir.WSScope` (post-pragma materialization) | `lower_ws_scope` | `KernelLaunch(scheduler=WSScheduler(...))` |
| `mir.FanOut` | `lower_fan_out` | `KernelLaunch(scheduler=FanOutScheduler(...))` |
| `mir.BlockGroupRoot` | `lower_block_group_root` | `KernelLaunch(scheduler=BGScheduler(...))` + `BgWorkBalanceSetup` |
| `mir.DedupGate` | `lower_dedup_gate` (runner-side) | `DedupTable(...)` |
| `mir.ComputeDelta` | `lower_compute_delta` | `ComputeDeltaCall(...)` (per Amendment 1 Gap 1) |
| `mir.RebuildIndex` | `lower_rebuild_index` | `RebuildIndexCall(...)` (per Amendment 1 Gap 1) |
| (11 maintenance ops total per orchestrator.py:506-588 dispatch chain — see § 3.1.1.1 Gap 1 for the corrected count; Amendment 1 emits one typed RIR op per MIR op) | one lowering each | one typed RIR op each |

The kernel body — already produced by Phase B's `MirToIirLowering`
— is wrapped inside a `KernelDef(body=<IIR>)` RIR op by the same
pragma's RIR-lowering. Phase R does NOT touch kernel body lowering;
that work is Phase B's, complete.

#### 3.1.3 RIR → text renders

One `@register_render(RIR-op, target='cuda')` per RIR op. Mirrors
what Phase B did for IIR: every op type has a per-target render
function; the render is the textual output.

| RIR op | CUDA renderer | Output |
|---|---|---|
| `RunnerStruct` | `render_runner_struct_cuda` | `struct JitRunner_<rule> { ... };` |
| `StepDispatch` | `render_step_dispatch_cuda` | `void step_N(...) { switch(...) { ... } }` |
| `FixpointLoop` | `render_fixpoint_loop_cuda` | `for (uint32_t it = 0; ; ++it) { ... if (empty) break; }` |
| `CountKernelDef` (Amendment 1 Gap 6) | `render_count_kernel_def_cuda` | `__global__ void kernel_count(...) { <body-text> }` |
| `MaterializeKernelDef` (Amendment 1 Gap 6) | `render_materialize_kernel_def_cuda` | `__global__ void kernel_materialize(...) { <body-text> }` |
| `FusedKernelDef` (Amendment 1 Gap 6) | `render_fused_kernel_def_cuda` | `__global__ void kernel_fused(...) { <count_body>; <materialize_body>; }` |
| `KernelLaunch + PlainScheduler` | `render_kernel_launch_plain_cuda` | `kernel_count<<<g, b, 0, stream>>>(...)` |
| `KernelLaunch + WSScheduler` | `render_kernel_launch_ws_cuda` | WS task-queue setup + launch |
| `KernelLaunch + FanOutScheduler` | `render_kernel_launch_fanout_cuda` | FanOut explore-then-emit dispatch |
| `KernelLaunch + BGScheduler` | `render_kernel_launch_bg_cuda` | BG histogram + per-warp dispatch |
| `DedupTable` | `render_dedup_table_cuda` | `DeviceArray<uint64_t> dedup_hash_arr; ...` |
| `ConvergenceCheck` / `MaxIterationFuel` / `EitherFirst` / `AndAll` (Amendment 1 Gap 2) | one render each | termination predicate inlined into `FixpointLoop` body |
| `DeltaVariantSet` (Amendment 1 Gap 4) | `render_delta_variant_set_cuda` | sequential / parallel-stream / fused launch sequence per `ordering` |
| `MultiHeadInsert` (Amendment 1 Gap 5) | `render_multi_head_insert_cuda` | sequential or interleaved per-head emit |
| 11 typed maintenance ops (Amendment 1 Gap 1) | one `@register_render` each | `db.<rel>.merge_delta(...)` etc. — no string dispatch |
| `Step` (Amendment 1 Gap 3) | `render_step_cuda` | one `step_N(...)` method; `deps` consulted for any cross-stratum sync |

#### 3.1.4 What Phase R deletes

After all Phase R lowerings + renders land:

- `src/srdatalog/ir/codegen/cuda/orchestrator.py` (895 LOC) — deleted.
  Every site is replaced by a registered renderer.
- `src/srdatalog/ir/codegen/cuda/complete_runner.py` (1123 LOC) —
  deleted. The 14-branch pragma dispatcher is replaced by op-type
  dispatch in the RIR renderer.
- `src/srdatalog/ir/codegen/cuda/runner.py` (802 LOC) — deleted. The
  template emitters become per-RIR-op render functions, refactored
  to take an RIR op instead of pragma kwargs.

Total LOC removed: ~2820. New code: ~2400 (estimate) split across
~20 single-concern files under `src/srdatalog/ir/dialects/rir/`
and `src/srdatalog/ir/codegen/cuda/render/rir/`.

### 3.2 Phase T — Target abstraction

Three sub-categories, each its own wave of PRs.

#### 3.2.1 Dissolve LoweringCtx (no-context principle)

> **Rewritten by Amendment 2.** The original § 3.2.1 proposed
> splitting `LoweringCtx` into `LoweringCtx` (planning) + a new
> `CudaRenderCtx` (CUDA-private). PR-1d (#81) shipped that split
> and was reverted (#82) because the new `CudaRenderCtx` was the
> same anti-pattern reskinned — a fixed-schema attribute carrier
> where adding a new CUDA render feature still requires editing
> `lower_ctx.py` to add a field. Per § 1 sub-clause (Amendment 2),
> through-state contexts dissolve. This section now describes the
> dissolution shape, not a split shape.

Today `LoweringCtx` (`src/srdatalog/ir/dialects/relation/sorted_array/
lowerings/__init__.py:93`) has ~15 fields. The dissolution sorts every
field into one of three buckets, per the § 1 sub-clause.

##### Bucket 1: cross-pass data → IR op

Fields here are produced by an earlier pass and consumed by a later
one. They become typed IR ops (or typed metadata on existing ops).

| Field | Producer (today) | Consumer (today) | Replacement |
|---|---|---|---|
| `view_var_names` | `EmitViewDeclsShim` (kernel-pipeline pass) | every IIR render | `ViewBinding(handle_idx, var_name)` metadata on `KernelDef` (already typed per Amendment 1 Gap 7); render reads `op.view_bindings[handle_idx].var_name` |
| `view_slot_bases` | `EmitViewDeclsShim` | D2L segment-loop render | same as above — `ViewBinding.slot_base` |
| `output_var` + `output_var_overrides` | kernel-pipeline driver | `InsertInto` render | `KernelDef.output_binding: OutputBinding` (NEW carrier) |
| `rel_index_types` | `compile_kernel_body` caller | D2L / sorted-array renders | `KernelDef.rel_index_bindings: dict[str, IndexBinding]` (NEW carrier per Phase T-3 / index-plugin split, § 3.2.4) |

No `ctx.<field>` reads anywhere. Each render takes the op as its only
data input; the op carries the typed binding metadata it needs.

##### Bucket 2: intra-pass scratch → function-local

Fields here exist for the duration of ONE render call (often one nested
emit). They become local variables in the render function, not ctx
fields.

| Field | Why it's local-only | Replacement |
|---|---|---|
| `tiled_cartesian_valid_var` | Set by `_lower_nested_cart` immediately before emitting the body; read only by the body's `InsertInto`. Never crosses a render-call boundary. | Pass as explicit kwarg from `render_tiled_cartesian_dispatch` to the inner `InsertInto` emitter — a named parameter, not a ctx mutation. |
| `ws_cartesian_valid_var` | Same shape — set right before a WS Cartesian body emits. | Same — explicit kwarg. |
| `cartesian_bound_vars` | Stack of vars bound by the current Cartesian; popped on exit. Pure lexical scope. | A local list passed down the render call chain, or returned as part of a `RenderResult` if needed. |
| `neg_pre_narrow` | Populated by `_lower_nested_cart` right before its body so the Negation handler can read the pre-allocated handle. Lifetime = one nested-Cart body. | Explicit kwarg from `render_nested_cart` to `render_negation`. |
| `bound_vars`, `handle_vars` | Same shape — render-call-local lookup tables. | Local dicts on the render call chain. |
| `name_counter` (+ `fresh(prefix)` method) | Fresh-name source. | Lives on the `NameGen` service (already in `LowerCtx`); render takes `NameGen` as a service injection. |
| `debug`, `tile_var` | Per-render configuration. | Constructor / kwarg to the render function. |

No ctx parameter required. The renderer's signature is `(op, services,
**kwargs) → emitted_text` where `kwargs` carries the lexical-scope
bindings the caller is threading.

##### Bucket 3: pragma scratch flags → DELETE outright

`is_counting`, `dedup_hash`, `tiled_cartesian`, `ws_enabled`,
`bg_enabled` were threaded from `ExecutePipeline` bool fields (the A3
series removed the EP fields; the helpers `ep_has_X(...)` and these
ctx flags are vestigial reskins per the spec § 2.1 anti-pattern note).
Each lives ONLY inside its own pragma's lowering call. After
dissolution they become explicit kwargs from the pragma-lowering to
its inner emitter — but they also disappear from any place that ISN'T
the pragma's own lowering chain. Other lowerings stop reading them.

##### Bucket 4: services → `Services` handle (immutable)

| Field | Today | Replacement |
|---|---|---|
| `compiler` (currently on `LowerCtx`) | Field on `LowerCtx` | Stays a service; passed via `Services` |
| `name_gen` (currently on `LowerCtx`) | Field on `LowerCtx` | Same |
| `plugin_registry` (currently on `LowerCtx`) | Field on `LowerCtx` | Same |
| `view_layout` (currently on `LowerCtx`) | Field on `LowerCtx` | NOT a service — it is data. Becomes IR op output (see Bucket 1; `ViewBinding` carrier on `KernelDef`). |
| `target` (currently on `LowerCtx`) | String field | Stays a per-`Compiler.run` constant; threaded as kwarg, NOT mutated through-state. |

The `Services` handle has only registered services; adding a new
service = new typed `@register_service(MyService)`, never a new field
on a context class.

##### Final shape

After Amendment 2 lands its PR series, the codebase has:

- **Zero through-state contexts.** `LoweringCtx`, `KernelCtx`,
  `InitialProg`, `CudaRenderCtx` (if reintroduced — it won't be) are
  all gone.
- **Renderer signatures** look like `def render_<op>(op, services,
  **scope) → emitted_text`. `op` carries all cross-pass data via
  typed bindings; `services` is the injected immutable handle;
  `**scope` is whatever lexical state the parent render is threading
  into this child render.
- **Pipeline shims** (Compiler.run's pass list) communicate solely
  by transforming IR. `EmitViewDeclsShim` produces a `KernelDef`
  with populated `view_bindings`; the next pass reads the bindings
  off the op. There is no intermediate dataclass carrying the bindings
  as a side channel.

##### Migration shape (Phase T-1 redesigned)

The Phase T-1 work now spans roughly three to four PRs (final count
set by the audit + re-plan; see § 6 amendment):

- **PR T1-α** Introduce typed binding carriers (`ViewBinding`,
  `OutputBinding`, `IndexBinding`) on `KernelDef`. Populate them in
  the producing pass; provide back-compat readers on the legacy ctx
  fields so call sites can migrate one at a time.
- **PR T1-β** Migrate all renderer call sites that read `ctx.view_var_names`,
  `ctx.view_slot_bases`, `ctx.output_var*`, `ctx.rel_index_types`
  to read from the op's typed bindings. Once a bucket-1 field has zero
  callers, delete it from `LoweringCtx`.
- **PR T1-γ** Convert bucket-2 fields to explicit kwargs / function-
  local state. One field per PR or a small batch; delete each field
  from `LoweringCtx` as call sites migrate.
- **PR T1-δ** Delete `LoweringCtx` entirely. Delete `KernelCtx` and
  `InitialProg` (their fields are either now on the IR or moved to
  the `Services` handle). The pipeline shims communicate solely via
  the IR.

Discipline ratchet **D22** (introduced this amendment): the count of
named fields across all `*Ctx` dataclasses is strictly monotonically
decreasing PR-over-PR. New PRs that add a ctx field fail CI. See § 7.

#### 3.2.1.1 Amendment 3 — full carrier dissolution scope

Amendment 2 named the principle. The post-revert audit revealed
the dissolution scope is significantly broader than `LoweringCtx`
alone. **The entire `src/srdatalog/ir/codegen/cuda/context.py` file
(543 LOC) dissolves under the no-context principle.** This
sub-section enumerates the scope so the dissolution PR series has
a complete target.

##### Carriers that dissolve entirely

| Carrier | Location | Fields | Disposition |
|---|---|---|---|
| `LoweringCtx` | `dialects/relation/sorted_array/lowerings/__init__.py:93` | ~17 fields | Dissolves — § 3.2.1 four-bucket plan |
| `CodeGenContext` | `codegen/cuda/context.py:254` | ~38 fields | Dissolves — see field-by-field bucket assignment below |
| `RunnerGenState` | `codegen/cuda/context.py:179` | 14 fields incl. 4 typed bools (`is_balanced`, `is_block_group`, `is_dedup_hash`, `is_count`) | Dissolves — bools become op-type dispatch (`BalancedScanKernelDef` vs `BlockGroupKernelDef` etc., per Amendment 1 Gap 6) |
| `CodeGenHooks` | `codegen/cuda/context.py:203` | 10 named `Callable \| None` slots | Dissolves — the "hooks" abstraction IS the anti-pattern. Replaced by `@register_render(NewOp, target=T)` — the new op type IS the extension point |
| `KernelCtx` | `default_pipelines.py:66` | 12 fields incl. 2 bools (`tiled_cartesian`, `bg_enabled`) | Dissolves — strip flags, let pragma materialization put a typed wrap op in `ep.pipeline` |
| `InitialProg` | `default_pipelines.py:48` | 5 fields | Dissolves — each pass's output IS the IR; no carrier needed |

##### `CodeGenContext` field-by-field bucket assignment

- **Bucket 1 (cross-pass data → typed attribute on IR op):**
  `bound_vars`, `handle_vars`, `view_vars`, `output_vars`,
  `rel_index_types`, `view_slot_offsets`. All become typed attributes
  attached to `KernelDef` via the typed-attribute-dict (§ 3.2.1.2 Risk 2).
- **Bucket 2 (intra-pass scratch → function-local / explicit kwarg):**
  `balanced_idx1/2`, `tiled_cartesian_valid_var`,
  `tiled_cartesian_ballot_done`, `bg_warp_begin_var`, `bg_warp_end_var`,
  `bg_cumulative_var`, `bg_done_var`, `ws_queue_var`,
  `ws_range_board_var`, `ws_live_handles`, `ws_cartesian_valid_var`,
  `ws_cartesian_bound_vars`, `neg_pre_narrow`, `cartesian_bound_vars`.
  Lifetime = ONE render call (§ 3.2.1.2 Risk 3 boundary).
- **Bucket 3 (pragma scratch flags → DELETE via Phase P):**
  `is_counting`, `dedup_hash_enabled`, `dedup_hash_vars`, `bg_enabled`,
  `ws_enabled`, `ws_has_cartesian`, `ws_level`, `tiled_cartesian_enabled`,
  `bg_histogram_mode`, `is_fan_out_explore`, `cartesian_as_product`,
  `scalar_mode`, `is_leaf_level`, `inside_cartesian`. All disappear
  when the owning pragma migrates to the plugin contract (§ 3.2.1.3).
- **Bucket 4 (services → typed-key Services dict):** `name_counter` (→
  `services.get(NameGen).fresh()`), `indent` (→ render-local
  `StringBuilder.indent_context()`), `debug` / `is_jit_mode` (→
  per-`Compiler` config), `output_var_name` (→ Bucket 1 attribute),
  `tile_var` / `parent_tile_var` (→ Bucket 2), `group_size` (→
  Bucket 1).

##### `CodeGenHooks` dissolution rationale

The 10 hook slots map 1:1 to existing pragma/feature dialects. Each
is an extension point baked into the framework. The ACID test
requires that adding a new pragma/feature does NOT edit this
framework file — so each hook slot is itself a § 1 violation.

After Phase P (§ 3.2.1.3) migrates each pragma to the plugin
contract, each `ctx.hooks.<hook_name>(...)` callsite becomes a
`@register_render(MatchingOpType, target=T)` dispatch on a typed op
the pragma plugin contributes. `default_hooks()` + `CodeGenHooks`
deleted when the last hook slot's owner has migrated.

##### Helper functions that take `ctx` (~30)

Located in `context.py:350-543`. Each mechanically replaced:
`ind` / `inc_indent` / `dec_indent` → render-local `StringBuilder`;
`gen_unique_name` → `services.get(NameGen).fresh()`;
`with_bound_var` / `is_var_bound` → local scope passed explicitly;
`get_rel_index_type` / `get_view_slot_base` → read from `op.attributes`;
`gen_handle_var_name` → `services.get(NameGen).fresh(...)`.
All pure-string template wrappers (`gen_view_access`, `gen_index_spec_key`,
`gen_handle_state_key`, `gen_root_handle*`, `gen_degree`, `gen_valid`,
`gen_get_value*`, `gen_child*`, `gen_iterators`, `gen_chained_prefix_calls*`)
take primitives and move to `codegen/cuda/templates.py` with no `ctx`
parameter.

##### What survives from `context.py`

`NegPreNarrowInfo` (typed scratch dataclass; passed as kwarg) and the
pure-string template wrappers (the `plugin_gen_*` thunks). Both move
to `codegen/cuda/templates.py`. `context.py` itself is deleted in
the final dissolution PR.

##### Other anti-pattern sites folded in

Seven additional sites identified by the audit:

| Site | Anti-pattern | Disposition |
|---|---|---|
| Surviving `mir.ExecutePipeline` bools (`count`, `concurrent_write`, `use_fan_out`, `dedup_hash`, `block_group`, `tiled_cartesian`) | Bool-field gate on typed op | Disappears when each owning pragma migrates to Phase P |
| `ep_has_work_stealing(...)` helper (survives A3-2) | Helper anti-pattern (§ 2.1.1) | Delete with WorkStealing migration to Phase P |
| `print_iir.py:78-94` — 4-way `isinstance(op, X_OPS)` ladder | Hardcoded dialect enumeration in framework code | Each dialect self-registers via `@register_print(dialect_id)`; framework reads registry. Phase P prerequisite. |
| `render/__init__.py:120-138` — `_eager_register_all()` hardcodes dialect imports | Hardcoded dialect enumeration in framework code | Plugin entry-point discovery. Phase P prerequisite. |
| `pipeline.py:35-38` — `_INDEX_HEADER` hardcodes index plugin names | Hardcoded enumeration | Index plugin contributes `cpp_headers`; look up dynamically. § 3.2.4 work. |
| `_TERMINAL_WRAP_OPS` in `sorted_array/__init__.py` | Hardcoded enumeration of 3 specific wrap-op types | Disappears when those pragmas migrate to Phase P |
| `core/verifier.py:121` — `if target != 'cuda': return set()` in framework `core/` | String-tagged dispatch + D6 violation | Per-`Compiler` codegen registry. Phase B2-2 work. |

#### 3.2.1.2 Self-audit — does the proposed solution actually solve the issue?

Before writing the dissolution PRs, the proposed solution was
self-audited for the same anti-pattern at the next architectural
floor up. Five risks identified + their fixes. **These fixes are
prerequisites for the dissolution work — without them, the
dissolution rebuilds the same anti-pattern one layer up.**

| Risk | Threat | Fix |
|---|---|---|
| 1. `Services` becomes a named-field carrier | `services.name_gen`, `services.compiler`, ... — each new service edits the `Services` class. Same fixed-schema anti-pattern. | `Services` is a typed-key dict: `services.get[T](T) → T`. Adding a service = new type, no edit. |
| 2. IR op classes grow per-feature with named binding fields | `KernelDef.view_bindings`, `KernelDef.output_binding`, `KernelDef.rel_index_bindings`, ... — each feature edits `KernelDef`. Bool-field anti-pattern at the IR layer. | MLIR-style typed attribute dict on every Op: `op.attributes[ViewBinding]`. Pragma plugins register their own attribute types; Op classes don't grow. |
| 3. Bucket-2 `**scope` kwargs become stack-allocated context | If three nested renders all forward the same kwarg, you've reinvented context on the call stack. | Sharpen the bucket boundary: single-render scope only = Bucket 2. Multi-render scope IS cross-pass data → Bucket 1. If you find yourself threading the same kwarg through 3 frames, you've discovered a missed IR attribute. |
| 4. `@register_X` decorators are module-global mutable state | `_PRAGMA_REGISTRY`, `_PLUGIN_REGISTRY`, render registry, lowering registry — all module-global. Decorator wraps mutable state. | Per-`Compiler` registries. Decorators are sugar that stages into a thread-local "current compiler" set during `compiler.register_pragma_plugin(...)`. Multiple Compiler instances never share registries. |
| 5. Op classes themselves carry fixed schemas | `@dataclass class KernelDef` with named fields IS a fixed-schema attribute carrier at the IR layer. | Ops are typed shells declaring only name + region structure + verification. Per-feature data lives in `op.attributes`. (Risk 2 fix generalized.) |

All five fixes are themselves the MLIR/LLVM/Cascades shape — see § 11.

#### 3.2.1.3 The pragma plugin contract — Phase P (load-bearing)

> **The deepest finding of this amendment.** Today's pragma surface
> (`@pragma_handler(MyPragma, on=mir.X)`) is a constrained
> *materialization-handler* primitive, not a plugin. A pragma cannot
> contribute its own IR ops, cannot run as a general program pass,
> cannot bundle its own lowerings + renders, and cannot declare
> dependencies. The four built-in pragmas (BlockGroup, WorkStealing,
> DedupHash, FanOut) work today only by leaking knowledge into the
> framework. **Without a real pragma plugin contract, every other
> phase of this redesign is shuffling code without changing the
> extensibility surface.**
>
> Phase P (pragma plugin contract) is therefore the highest-priority
> work in the redesign, ahead of Phase R / T / B2. The dissolution
> work in § 3.2.1 + § 3.2.1.1 becomes a CONSEQUENCE of pragmas
> migrating to the plugin contract — each pragma migration deletes
> its corresponding framework leakage in the same commit.

##### What today's `@pragma_handler` CANNOT do

1. **Contribute its own IR ops** — the handler can only return an
   existing op type; new IR ops require `Dialect(name=..., ops=[...])`
   at dialect-construction time, NOT by a pragma.
2. **Run as a general program pass** — `MirPragmaPass` is a SPECIFIC
   framework-owned pass that only iterates `op.pragmas` and only
   for ops where `on=ExecutePipeline` (per `pragma_pass.py:167`).
3. **Bring its own lowerings + renders** — `@lowering` and
   `@register_render` are decoupled from `@pragma_handler`. Three
   separate module-global registries; no pragma association.
4. **Express dependencies** — `PragmaCtx` carries only `compiler`.
   No way to declare "I need NameGen + ViewLayout"; no way to
   declare "I produce `DedupGate` ops; downstream passes that
   consume them should run after me."
5. **Bundle as one plugin** — a pragma is split across N files
   registered against N module-globals (`_PRAGMA_REGISTRY`,
   `_PLUGIN_REGISTRY`, render registry, lowering registry,
   `Dialect.ops=[...]`). No `PragmaPlugin` object that names "this
   pragma owns these ops + passes + lowerings + renders."

##### Target shape — `PragmaPlugin`

ONE atomic registration that bundles ALL of:

```python
@final
@dataclass(frozen=True, slots=True)
class BlockGroupPlugin(PragmaPlugin):
  pragma_cls:        type[Pragma]              = BlockGroupPragma
  new_ops:           tuple[type[Op], ...]      = (BgRoot, BgScope, BgScheduler)
  new_attributes:    tuple[type[Attribute], ...] = (BgPlacementAttr,)
  passes:            tuple[Pass, ...]          = (MaterializeBlockGroup(),)
  lowerings:         tuple[Lowering, ...]      = (LowerBgRoot(), LowerBgScope(), LowerBgScheduler())
  renders:           dict[str, tuple[Render, ...]] = field(default_factory=lambda: {
    'cuda': (RenderBgRootCuda(), RenderBgScopeCuda(), RenderBgSchedulerCuda()),
  })
  requires_services: tuple[type, ...]          = (NameGen, ViewLayout)
  produces_ops:      tuple[type[Op], ...]      = (BgRoot, BgScope, BgScheduler)
  consumes_ops:      tuple[type[Op], ...]      = (mir.ExecutePipeline,)
  preserves:         tuple[type, ...]          = (semi_naive_safety, pragma_compat)

def register(compiler: Compiler) -> None:
  compiler.register_pragma_plugin(BlockGroupPlugin())
```

`compiler.register_pragma_plugin(...)` unpacks the plugin's
declarations into the per-Compiler registries (ops, attributes,
passes, lowerings, renders, services) in one atomic transaction. If
any registration conflicts (op-name collision, pass dependency
cycle, render double-registration), the whole call fails atomically.

##### What the framework knows about pragmas after Phase P

**Nothing pragma-specific.** The framework owns:

1. The `Pragma` base class + `PragmaPlugin` schema (typed shells).
2. `Compiler.register_pragma_plugin(plugin)` (one-line API).
3. Per-Compiler registries (typed dicts per § 3.2.1.2 Risk 4 fix).
4. The pass scheduler (LLVM-style topo-sort on
   produces/consumes/preserves; no hardcoded order).
5. The verifier hooks.

Every line of framework code that mentions a specific pragma name
(`bg_enabled`, `ws_enabled`, `dedup_hash`, `is_fan_out_explore`,
`_BUILTIN_BOOL_SHADOW_PRAGMAS`, `_TERMINAL_WRAP_OPS`, the 10
`CodeGenHooks` slots, the surviving `ep.dedup_hash` / `ep.block_group`
/ `ep.count` bool fields) is a § 1 violation and is deleted by the
corresponding pragma's migration to the plugin contract.

##### Per-pragma migration PR shape (parallel)

One PR per pragma, run in parallel:

1. Write the `PragmaPlugin` subclass with the pragma's new ops,
   passes, lowerings, renders.
2. Switch registration from legacy `@pragma_handler` to
   `compiler.register_pragma_plugin(MyPlugin)`.
3. Delete every framework leakage site for this pragma in the same
   commit (bool field on EP, `CodeGenHooks` slot, `CodeGenContext`
   flags, `LoweringCtx` fields, `ep_has_X` helper).
4. Byte-equivalence gate: 535+ goldens green.

`LoweringCtx`, `CodeGenContext`, `CodeGenHooks`, `RunnerGenState`
shrink ATOMICALLY with each migration. After the fourth pragma
migrates, `CodeGenHooks` has zero slots (delete the class),
`CodeGenContext` has only target-agnostic scratch (Phase T-1 buckets
dispose), `_TERMINAL_WRAP_OPS` is empty (delete the set).

##### Why Phase P is load-bearing

Until Phase P lands:

- **Phase R** (RIR work) would design new typed runner ops without a
  plugin contract for them. The new RIR ops would re-leak into the
  framework the same way today's pragma ops do.
- **Phase T** (LoweringCtx dissolution) can delete fields, but the
  underlying need for those fields (pragma-specific scratch) doesn't
  go away until pragmas own their own scratch via the plugin contract.
- **Phase B2** (target parametricity) adds multi-target dispatch,
  but a non-CUDA target still cannot ship as a plugin if pragmas
  themselves cannot ship as plugins.

After Phase P lands, Phases R / T / B2 all become *consequences* of
the plugin contract being correct, rather than independent
restructuring efforts.

##### Re-prioritization

The PR plan in § 6 is amended (see § 6.0) to add a **PR-P series**
ahead of PR-2. PR-1a/b (already merged) are retained — they ship
plumbing the plugin contract needs (per-Compiler dialect/target
entry-point discovery). PR-1c (target threading) is retained but
the `target` kwarg moves to the typed `Services` dict in Phase P
(it's per-`Compiler.run` metadata, not per-feature data). PR-1d
remains reverted. The Phase T-1 dissolution PRs from Amendment 2
(T1-α/β/γ/δ) are RECLASSIFIED: each is now authored as part of the
corresponding pragma's migration PR, not as standalone work.

#### 3.2.2 Rename CUDA-shaped IIR ops to target-agnostic semantic names

| Today (CUDA-shape name) | Target-agnostic name | Semantic meaning |
|---|---|---|
| `SaTiledCartesian2D` | `TiledCartesianDispatch` (in `relation.sorted_array`) | 2-source nested Cartesian with per-source-pair tile reuse |
| `BgRootCjMulti` | `BlockGroupRootDispatch` (in `parallel.block_group`) | Root multi-source ColumnJoin under block-group scheduling |
| `BgSourceSpec` | `BlockGroupSourceSpec` (rename only — already semantic) | Per-source descriptor for block-group dispatch |
| `SaPrefCoop` | `SaPrefixCooperative` (in `relation.sorted_array`) | Cooperative warp-tile prefix lookup |
| `SaPrefSeq` | `SaPrefixSequential` (in `relation.sorted_array`) | Per-thread sequential prefix lookup |
| `SaChildRange` | `SaChildRange` (rename only — but drop `tile` from signature; pass via scope) | Range lookup on a sorted-array node |
| `GridStrideLoop` | `ParallelStridedLoop` (in `iir.cf`) | Parallel loop with strided indexing |
| `LaneZeroGuard` | `LaneFirstGuard` (in `iir.cf`) | Predicate guard for one lane per warp/wave/group |
| `TiledBallotBlock` | `TiledValidBlock` (in `iir.cf`) | Tiled block emitting valid-mask writes |

Per-rename PR shape:

1. New op class (target-agnostic name) added alongside the old.
2. Lowerings emit the new op; old op deprecated.
3. Per-target render registers against the new op name.
4. Old op class deleted in same PR (assuming no external plugin
   depends on it).

CPU/TBB render will produce nested for-loops with manual SIMD
intrinsics. SYCL render will produce `nd_range` parallel_for with
`sub_group` ballots. Each render owns its target's C++ template;
the IIR op carries only the semantic shape.

#### 3.2.3 Move CUDA-private helpers under `codegen/cuda/`

`view_slots.py` (324 LOC) and `materialized.py` (531 LOC) already
live under `src/srdatalog/ir/codegen/cuda/`. The fix is removing
their cross-tree reach into ostensibly target-agnostic call sites:

- `EmitViewDeclsShim` (default_pipelines.py:224-261) calls
  `view_counts_for_specs` from `relation.d2l` — the count is a CUDA-
  render concern (number of CUDA `views[]` slots a FULL D2L source
  consumes). Move the call into a CUDA-render-side pass; the
  `KernelCtx` carries only the dialect-level source spec.
- `materialized.py` is already CUDA-only (uses Thrust). Move it
  into `codegen/cuda/render/materialized.py` as a per-RIR-op
  renderer (after Phase R lands `MaterializedRunnerShape` as an RIR
  op).

#### 3.2.4 Split index-plugin registration: data vs render

Today (`d2l/cuda.py:65`):

```python
# d2l/cuda.py  -- data dialect reaches into target's private registry
from srdatalog.ir.codegen.cuda.plugin import register_index_plugin
register_index_plugin(two_level_plugin)
```

Target shape:

```python
# src/srdatalog/ir/dialects/relation/d2l/__init__.py  -- data dialect only
DIALECT = Dialect(name='relation.d2l', ops=[D2lSegmentLoop, ...])

# srdatalog/codegen/cuda/render/relation_d2l.py  -- per-target render
@register_render(D2lSegmentLoop, target='cuda')
def render_d2l_segment_loop_cuda(op, ctx): ...

@register_render(D2lSegmentLoop, target='cuda')
def render_d2l_view_setup_cuda(op, ctx):
  '''Emits the FULL=2-slot view setup that the legacy plugin's
  gen_host_view_setup did. CUDA-render-side; CPU/TBB will register
  its own.'''
  ...
```

External index plugins ship a data dialect package + a per-target
render package; the user installs both. The CUDA target's
`_PLUGIN_REGISTRY` becomes a `Compiler`-instance registry of
`@register_render` entries; the side-effect import at
`complete_runner.py:37` disappears.

### 3.3 Phase B2 — Target parametricity

#### 3.3.1 Target chosen at run time

```python
# src/srdatalog/__init__.py
compiler = Compiler.with_default_plugins()

# Existing surface (post-redesign):
result = compiler.run(prog, pipeline=DEFAULT, target='cuda')

# New surface — second-class target dispatch is gone:
result = compiler.run(prog, pipeline=DEFAULT, target='cpu_tbb')  # plugin loaded; works
result = compiler.run(prog, pipeline=DEFAULT, target='wasm')     # plugin not loaded; loud error
```

The Compiler resolves `target=T` against its registered
`@register_render(Op, target=T)` table; if no renderer is registered
for any op type that appears in the IIR/RIR tree post-fixpoint,
`verify_renderability(target=T)` raises `UnrenderableOpError(op, T)`
before any render runs. Per the four topology checks in
`concept_glossary.md` §13.3, this is check #4 (renderability closure),
parameterized over T.

#### 3.3.2 Multi-target compilation

```python
result = compiler.run(prog, pipeline=DEFAULT, targets=['cuda', 'cpu_tbb'])
# result.per_target = {'cuda': '<cuda C++ text>', 'cpu_tbb': '<cpu C++ text>'}
```

The Compiler runs the rendering pass once per target; the IR
preceding the render pass is shared. Use cases:

- A user wants both a GPU binary and a CPU fallback shipped in one
  build.
- A test fixture wants byte-equiv goldens for both targets from the
  same source.
- A benchmark wants the cross-target perf comparison from identical
  IR.

#### 3.3.3 Pipeline shims become target-parametric

Per section 2.3, the shim class names encode CUDA. Phase B2 renames:

| Old (CUDA-baked) | New (target-parametric) | Behavior change |
|---|---|---|
| `LowerScanPipelineShim` | `LowerKernelBodyShim` | Constructs a per-target `LowerCtx`; resolves per-target `@lowering` |
| `CudaRenderShim` | `RenderShim` | Dispatches `@register_render(op, target=ctx.target)` per op |
| `EmitViewDeclsShim` | (moved into CUDA-render-side pass per §3.2.3) | — |
| `CollectViewSpecsShim` | (moved into CUDA-render-side pass per §3.2.3) | — |
| `AssignHandlesShim` | `AssignHandlesShim` (target-agnostic — handle assignment is a property of the MIR pipeline) | unchanged |
| `VerifyRenderabilityShim` | `VerifyRenderabilityShim` (already parametric; just stop hardcoding the literal) | reads target from KernelCtx |

#### 3.3.4 Plugin discovery split: dialects vs targets

Today the entry-point group is one group:
`[project.entry-points."srdatalog.plugins"]`. Phase B2 splits:

```toml
# pyproject.toml — post-redesign
[project.entry-points."srdatalog.dialects"]
sorted_array   = "srdatalog.ir.dialects.relation.sorted_array:register"
d2l            = "srdatalog.ir.dialects.relation.d2l:register"
iir_cf         = "srdatalog.ir.dialects.iir.cf:register"
parallel_data  = "srdatalog.ir.dialects.parallel.data:register"
parallel_atomic_ws    = "srdatalog.ir.dialects.parallel.atomic_ws:register"
parallel_block_group  = "srdatalog.ir.dialects.parallel.block_group:register"
rir            = "srdatalog.ir.dialects.rir:register"

[project.entry-points."srdatalog.targets"]
cuda           = "srdatalog.codegen.cuda:register"
# external packages add:
# cpu_tbb     = "srdatalog_cpu_tbb:register"
# sycl        = "srdatalog_sycl:register"
```

`Compiler.with_default_plugins()` walks both groups; `target=T`
selects which targets' render contributions are used at render time.
External packages declare one or both groups depending on whether
they ship a data dialect, a render target, or both. The jaccard
example (PR #68) would ship under `srdatalog.dialects` only (it's
a data dialect; relies on existing CUDA target for render).

#### 3.3.5 verify_renderability per-target

R3 (`code_discipline.md`) requires every concrete IIR/RIR op has
either `@register_render(target=cuda)` or a `@rewrite`. Phase B2
makes this per-target: for each registered target T in the running
Compiler, the closure check (`verify_renderability(target=T)`) runs
once. A target that doesn't yet have renders for every op type
fails at compiler-bootstrap time, NOT at the user's first
`compiler.run(...)` call. (Bootstrap-time check, not run-time.)

External plugins that add new ops without per-target renders are
caught at install + first-import; the error names exactly which
(op, target) pair is missing a render.

## 4. Worked example: ship a CPU/TBB backend

End-to-end: an external party wants the compiler to additionally
emit CPU/TBB code for the same source rules. ZERO edits to
`srdatalog/`.

### 4.1 Before this redesign

The user would need to edit ~7 files:

1. `src/srdatalog/ir/codegen/cuda/api.py` — change the `Target`
   alias from `Literal['cuda']` to `Literal['cuda', 'cpu_tbb']`.
2. `src/srdatalog/ir/default_pipelines.py` — introduce a parallel
   `CpuTbbRenderShim` next to `CudaRenderShim`.
3. `src/srdatalog/ir/codegen/cuda/orchestrator.py` — add a parallel
   `gen_step_body_tbb` (~900 LOC of duplicated branches).
4. `src/srdatalog/ir/codegen/cuda/complete_runner.py` — add a
   parallel `gen_complete_runner_tbb` (~1100 LOC of duplicated
   branches).
5. `src/srdatalog/ir/codegen/cuda/runner.py` — add a parallel set
   of `emit_*_tbb` template emitters (~800 LOC).
6. Per IIR op: a parallel `_render_tbb_*` function in a new
   `codegen/cpu_tbb/` module — but with the current IIR ops carrying
   CUDA-specific shape (e.g., `SaTiledCartesian2D` with smem fields),
   the TBB render has to reverse-engineer the semantic intent from
   the C++ template fields.
7. `src/srdatalog/ir/codegen/cuda/plugin.py` — the index plugin
   registry needs a per-target variant, AND every data dialect's
   `cuda.py` registration module needs a parallel `cpu_tbb.py`.

Plus the inherent risk: the per-target divergence is 7+ files, each
with internal branching; the byte-equivalence harness covers only
CUDA, so a TBB break is silent until a user runs the new target.

### 4.2 After this redesign

The user creates ONE plugin package, ships it to PyPI:

```
srdatalog_cpu_tbb/
  pyproject.toml                            # entry-point declaration
  src/srdatalog_cpu_tbb/
    __init__.py                             # register(compiler)
    scheduler_ops.py                        # CpuTbb{Plain,WS,FanOut,BG}Scheduler RIR ops
    codegen/
      render/
        iir_cf.py                           # @register_render(target='cpu_tbb') per iir.cf op
        iir_expr.py                         # @register_render(target='cpu_tbb') per iir.expr op
        sorted_array.py                     # @register_render(target='cpu_tbb') per sorted_array op
        d2l.py                              # @register_render(target='cpu_tbb') per d2l op
        parallel_data.py                    # @register_render(target='cpu_tbb') per parallel.data op
        parallel_block_group.py             # @register_render(target='cpu_tbb') per parallel.block_group op
        parallel_atomic_ws.py               # @register_render(target='cpu_tbb') per parallel.atomic_ws op
        rir.py                              # @register_render(target='cpu_tbb') per RIR op
```

`pyproject.toml`:

```toml
[project]
name = "srdatalog-cpu-tbb"
dependencies = ["srdatalog>=2.0"]

[project.entry-points."srdatalog.targets"]
cpu_tbb = "srdatalog_cpu_tbb:register"
```

`src/srdatalog_cpu_tbb/__init__.py`:

```python
def register(compiler):
  from . import scheduler_ops  # noqa  -- @op registrations fire
  from .codegen.render import (  # noqa  -- @register_render registrations fire
    iir_cf, iir_expr, sorted_array, d2l,
    parallel_data, parallel_block_group, parallel_atomic_ws, rir,
  )
```

End-user code:

```python
from srdatalog import compile_runner

result = compile_runner(prog, target='cpu_tbb')
# Renders to TBB C++; the registered @register_render(...,
# target='cpu_tbb') for every op type was contributed by the
# srdatalog_cpu_tbb plugin. Zero edits to srdatalog/.
```

The Compiler bootstrap verifies: for every op type reachable in the
post-fixpoint IIR + RIR tree, `@register_render(op, target='cpu_tbb')`
is registered. If not, `UnrenderableOpError` names the missing
(op_type, 'cpu_tbb') pair — the user knows exactly which render
function to write.

## 5. Worked example: ship a new algorithm (semi-ring aggregation)

A new algorithm is a different shape from a new target. The user
adds a new MIR op type, a new IIR op type, and renders for each
registered target.

```
srdatalog_semiring/
  pyproject.toml
  src/srdatalog_semiring/
    __init__.py                             # register(compiler)
    mir_ops.py                              # SemiringAggregate (MIR op)
    iir_ops.py                              # SrAggBody (IIR op)
    lowerings.py                            # @lowering(target=IIR, source=SemiringAggregate)
    codegen/
      render/
        cuda.py                             # @register_render(SrAggBody, target='cuda')
        # cpu_tbb.py, sycl.py, ... if other targets are installed in the same env
```

`__init__.py`:

```python
def register(compiler):
  from .mir_ops import SemiringAggregate  # noqa
  from .iir_ops import SrAggBody          # noqa
  from . import lowerings                  # noqa  -- @lowering fires
  from .codegen.render import cuda         # noqa  -- @register_render(target='cuda') fires
  # cpu_tbb / sycl renders register IFF the user installed those
  # target plugins; the plugin gracefully degrades — UnrenderableOpError
  # at run time names the missing pair.
```

End-user code:

```python
from srdatalog import Rule, Var, compile_runner

x, y = Var('x'), Var('y')
sum_rule = Rule(...).aggregate(SemiringAggregate(semiring='tropical'))
result = compile_runner(Program(rules=[sum_rule]), target='cuda')
```

Zero edits to `srdatalog/`. The new algorithm flows through:

1. DSL `aggregate(...)` attaches the typed op.
2. HIR → MIR lowering emits the new `SemiringAggregate` MIR op
   (the plugin registers it as the MIR op for `aggregate(...)`).
3. `MirToIirLowering` finds `@lowering(target=IIR, source=SemiringAggregate)`
   and lowers to `SrAggBody`.
4. `verify_renderability(target='cuda')` finds the registered
   `@register_render(SrAggBody, target='cuda')` and confirms the
   tree closes.
5. `CudaRenderPass` walks the tree; the new op type gets its
   plugin-supplied renderer.

## 6. Per-PR migration plan

**~9 PRs across the three phases.** Each PR is large (1500-3500 LOC of code +
tests) but bundled around one architectural concern. Byte-equivalence on CUDA
stays green throughout (zero target migration in scope yet — CUDA is the
regression anchor).

> **Earlier draft had 35 PRs (one per MIR-op + one per RIR-op render). Compressed
> here per `git log` review: the per-op granularity duplicated framework
> infrastructure across PRs and made rebases noisy. Bundling by architectural
> concern keeps the PR count manageable while the byte-equivalence harness
> remains the per-PR safety net.**

### 6.0 PR-P series — Pragma plugin contract (Phase P, Amendment 3)

> **Highest-priority phase, ahead of everything else in § 6.** Per
> § 3.2.1.3, without this series every other phase shuffles code
> without changing the extensibility surface. The series consists of
> five PRs: one foundation + four per-pragma migrations that can run
> in parallel.

#### 6.0.0 PR-P0 — Framework primitives (serial, blocks all P1-P4)

| Change | Where |
|---|---|
| `PragmaPlugin` schema (typed dataclass: `pragma_cls`, `new_ops`, `new_attributes`, `passes`, `lowerings`, `renders`, `requires_services`, `produces_ops`, `consumes_ops`, `preserves`) | NEW `src/srdatalog/ir/core/pragma_plugin.py` |
| `Compiler.register_pragma_plugin(plugin)` — atomic unpack into per-Compiler registries; conflict detection | `src/srdatalog/ir/core/dialect.py` (extends `Compiler`) |
| Per-Compiler registries: `compiler.ops` / `attributes` / `passes` / `lowerings` / `renders` / `services` — typed dicts (§ 3.2.1.2 Risk 4 fix) | `src/srdatalog/ir/core/dialect.py` |
| `Services` typed-key dict: `services.get[T](T) → T` (§ 3.2.1.2 Risk 1 fix) | NEW `src/srdatalog/ir/core/services.py` |
| `Op.attributes: AttributeDict` typed attribute container (§ 3.2.1.2 Risks 2 + 5 fix) | `src/srdatalog/ir/core/op.py` (extends `Op` base) |
| `Pass` base + `produces`/`consumes`/`preserves` declaration; LLVM-style topo-sort scheduler | `src/srdatalog/ir/core/passes.py` |
| Decorator sugar — `@register_render`, `@lowering`, `@pragma_handler` stage into thread-local "current compiler" set during `register_pragma_plugin(...)` (Risk 4 fix); module-globals deprecated with WARN | `src/srdatalog/ir/core/decorators.py` |
| Legacy `@pragma_handler` continues to work as an explicit single-handler PragmaPlugin sugar form during the per-pragma migrations | back-compat shim in `pragma.py` |

**Acceptance**: byte-equiv 535+ goldens green. Existing pragmas
continue to work via the back-compat shim. New `PragmaPlugin` API
documented with one demo plugin (jaccard? — pending #68 cleanup per
§ 10.3 audit note).

**Why bundle (not split)**: every primitive listed depends on the
others structurally. Splitting would require introducing temporary
back-compat scaffolding that gets immediately deleted. Estimate
~2500 LOC + tests.

#### 6.0.1 PR-P1 / P2 / P3 / P4 — Per-pragma migrations (parallel after P0)

Four PRs, one per built-in pragma, all parallel (file-disjoint):

| PR | Pragma | Touches |
|---|---|---|
| PR-P1 | BlockGroup | `dialects/parallel/block_group/` + delete `ep.block_group` bool + delete `bg_*` fields in `CodeGenContext`/`LoweringCtx` + delete BG `CodeGenHooks` slots + delete `ep_has_block_group_root` helper |
| PR-P2 | WorkStealing | `dialects/parallel/atomic_ws/` + delete `ep.work_stealing` (already done? confirm) + `ep_has_work_stealing(...)` helper (survives A3-2) + `ws_*` fields in contexts + WS `CodeGenHooks` slots |
| PR-P3 | DedupHash | dialect TBD + `ep.dedup_hash` bool + `dedup_hash_*` fields + dedup `CodeGenHooks` slots + `ep_has_dedup_gate` helper |
| PR-P4 | FanOut | dialect TBD + `ep.use_fan_out` bool + `is_fan_out_explore` field + FanOut `CodeGenHooks` slots |

Per-PR shape:

1. Write `<Pragma>Plugin(PragmaPlugin)` with the pragma's new ops,
   passes, lowerings, renders.
2. Switch registration: legacy `@pragma_handler` → `compiler.register_pragma_plugin(MyPlugin)`.
3. **Same commit**: delete every framework leakage site for this
   pragma. The framework grep for the pragma's name returns ZERO
   matches after this PR lands.
4. Byte-equiv 535+ goldens green.

**Why parallel**: each pragma's migration is file-disjoint from the
others. Each PR touches its own dialect package + the leakage sites
unique to its pragma. Merge conflicts limited to shared dissolution
targets (`CodeGenContext`, `LoweringCtx`) which use atomic-decrement-
field semantics (each PR deletes its own fields, the PR list shrinks
both contexts in parallel without conflict).

**Acceptance**: after all 4 land, `CodeGenHooks` has zero slots
(delete the class), `_TERMINAL_WRAP_OPS` is empty (delete the set),
`mir.ExecutePipeline` has zero pragma-named bool fields. Byte-equiv
535+ green.

#### 6.0.2 PR-P5 — Context cleanup (consequence)

After PR-P0 + PR-P1/2/3/4 land:

| Change | Where |
|---|---|
| Delete `CodeGenHooks` class | `codegen/cuda/context.py` |
| Delete `RunnerGenState` class (replaced by typed IR metadata) | same |
| Reduce `CodeGenContext` to bucket-2 + bucket-4 surface (target-agnostic intra-pass scratch + service handles) | same |
| Reduce `LoweringCtx` similarly | `dialects/relation/sorted_array/lowerings/__init__.py` |
| Move `NegPreNarrowInfo` + pure-string template wrappers (the `plugin_gen_*` thunks) to `codegen/cuda/templates.py` | NEW file |
| Delete `_TERMINAL_WRAP_OPS` set | `dialects/relation/sorted_array/__init__.py` |
| Delete `_PRAGMA_REGISTRY` / `_PLUGIN_REGISTRY` module-globals (registries are now per-Compiler) | `core/pragma.py` + `codegen/cuda/plugin.py` |
| Delete the back-compat shim for legacy `@pragma_handler` (every pragma is now a PragmaPlugin) | `core/pragma.py` |

**Acceptance**: byte-equiv 535+ green. `context.py` (the remnant)
is renamed to reflect its shrunk role, OR deleted entirely with the
remaining helpers absorbed into `templates.py`. `LoweringCtx` is
reduced to its target-agnostic core (or also deleted; depends on
how many call sites remain).

#### 6.0.3 PR-P series total

| PR | Concern | Est. size | Sequence |
|---|---|---|---|
| PR-P0 | Framework primitives (PragmaPlugin schema, per-Compiler registries, typed Services, Op.attributes, Pass.produces/consumes, decorator sugar) | ~2500 LOC | first, serial |
| PR-P1 | BlockGroup migration | ~1500 LOC | parallel with P2/P3/P4 |
| PR-P2 | WorkStealing migration | ~1500 LOC | parallel |
| PR-P3 | DedupHash migration | ~1000 LOC | parallel |
| PR-P4 | FanOut migration | ~1000 LOC | parallel |
| PR-P5 | Context cleanup (consequence) | ~800 LOC (mostly deletion) | last, serial |

**Total: 6 PRs, ~8300 LOC**, with PR-P1/P2/P3/P4 in parallel.

After PR-P series lands, the existing PR-2/3/4/5/6 plan from § 6.1
onwards re-applies with reduced scope — most of the dissolution
work the original plan called for is already done as Phase P
consequences. PR-2/3 (RIR) now adds runner ops as additional
PragmaPlugins (the runner is essentially a meta-pragma); PR-4
(IIR rename) is unchanged; PR-5 (index plugin split) is reduced
to the `pipeline.py:_INDEX_HEADER` dynamic lookup; PR-6 (multi-
target + D20) is unchanged.

### 6.1 PR-1 — Foundation: LowerCtx split + target parametricity (Wave T1 + B2-1)

| Change | Where |
|---|---|
| Add `CudaRenderCtx` (9 fields, per § 3.2.1) | `src/srdatalog/ir/codegen/cuda/lower_ctx.py` |
| Migrate all 9 CUDA-render fields off `LoweringCtx` | `lowerings/__init__.py` + every pragma lowering site |
| Pin `LoweringCtx` at 5 fields strictly; D10 starts asserting | `core/lower_ctx.py` + `tests/test_discipline_lower_ctx_pinned.py` |
| Pragma-scratch flags become lowering-local kwargs | `pragmas/*.py` |
| `Compiler.run(prog, pipeline=..., target='cuda')` accepts `target` kwarg | `core/dialect.py` + `default_pipelines.py` |
| `KernelCtx.target` field threaded through; `RenderShim` (renamed from `CudaRenderShim`) dispatches `@register_render(op, target=ctx.target)` | `default_pipelines.py` + `codegen/cuda/render/*.py` |
| `LowerScanPipelineShim` → `LowerKernelBodyShim` | `default_pipelines.py` |
| Plugin entry-point group split: `srdatalog.plugins` → `srdatalog.dialects` + `srdatalog.targets`. Back-compat shim reads legacy group for one release. | `pyproject.toml` + `core/plugin.py` |

**Acceptance**: 535 byte-equivalence goldens green; D10 + R3 (per-target `verify_renderability`) extended with target param; existing `register()` plugin shape continues to work via back-compat.

**Why bundle**: T1 (LowerCtx split) and B2-1 (target param) interact — the target kwarg has to thread through the same ctx surface that T1 just cleaned up. Shipping them together avoids a double-touch of every pragma lowering.

### 6.2 PR-2 — Phase R framework + MIR→RIR lowerings (Wave R1 + R2)

> **Amendment 1 split.** The original PR-2 was scoped at "10 op
> families" (~2500 LOC). Amendment 1's typed-vocabulary expansion
> (§ 3.1.1.1) grows the lowering surface to ~30 typed Ops + 5 typed
> carrier dataclasses, with proportional growth in `@lowering`
> rules. Estimated combined size ~4500 LOC, which exceeds the
> ~3500-LOC per-PR ceiling honored by Phase B's largest shipped PRs.
> PR-2 is split into PR-2a (framework + structural / scheduling /
> kernel-body lowerings) and PR-2b (the remaining lowerings:
> maintenance × 11, termination union, variant ordering, multi-head,
> view bindings). Both PRs land before PR-3 (RIR renders + monolith
> deletion), strictly serial: PR-2a → PR-2b → PR-3. The "one
> architectural concern per PR" principle is preserved — both halves
> are MIR→RIR lowering, partitioned by op family for review-load
> reasons only.

#### 6.2a PR-2a — RIR framework + structural / scheduling / kernel-body lowerings

| Change | Where |
|---|---|
| RIR framework: dialect skeleton, `Op` base, `print()` infrastructure, `MirToRirLowering(LoweringPass)` scaffold, `USE_DECLARATIVE_RUNNER` ratchet, `LowerRunnerShim`, `verify_runner_completeness(rir_program, target=T)` | NEW `src/srdatalog/ir/dialects/rir/{ops,types,print,__init__}.py` + `default_pipelines.py` + `core/verifier.py` |
| Dialect registration via entry point (in the new `srdatalog.dialects` group from PR-1) | NEW `pyproject.toml` entry + `src/srdatalog/ir/dialects/rir/__init__.py` `register(compiler)` |
| Structure ops + lowerings: `RunnerStruct`, `StepDispatch`, `FixpointLoop`, `Step` (Amendment 1 Gap 3). `mir.Program` / `mir.FixpointPlan` → these. | NEW `src/srdatalog/ir/dialects/rir/ops_structure.py` + `lowerings/structure.py` |
| Scheduling ops + lowerings: `KernelLaunch`, `PlainScheduler`, `WSScheduler`, `FanOutScheduler`, `BGScheduler`. `mir.ExecutePipeline-plain` / `mir.WSScope` / `mir.FanOut` / `mir.BlockGroupRoot` → these. | NEW `src/srdatalog/ir/dialects/rir/ops_scheduling.py` + `lowerings/scheduling.py` |
| Kernel-body ops + lowerings: `CountKernelDef`, `MaterializeKernelDef`, `FusedKernelDef` (Amendment 1 Gap 6). Replaces the legacy `KernelDef(phase: KernelPhase)` proposal in § 3.1.1. | NEW `src/srdatalog/ir/dialects/rir/ops_kernel_def.py` + `lowerings/kernel_def.py` |
| Dedup: `DedupTable`. `mir.DedupGate` → `DedupTable(...)`. | `ops_scheduling.py` + `lowerings/dedup.py` |
| Legacy runner pass remains untouched in this PR — RIR layer flows through but partial-coverage (maintenance / termination / variant-set / multi-head ops still missing); `verify_runner_completeness` runs in WARN mode pending PR-2b. | unchanged |

**Acceptance**: structural / scheduling / kernel-body MIR→RIR
lowerings pass at 100% coverage for the ops in scope; PR-2b
completes the remaining lowerings; 535 byte-equiv goldens green
(legacy runner still emits C++).

#### 6.2b PR-2b — Maintenance + termination + variant ordering + multi-head + view bindings

| Change | Where |
|---|---|
| 11 typed maintenance ops + lowerings (Amendment 1 Gap 1): `ComputeDeltaCall`, `ComputeDeltaIndexCall`, `MergeIndexCall`, `MergeRelationCall`, `CheckSizeCall`, `RebuildIndexCall`, `RebuildIndexFromIndexCall`, `CreateFlatViewCall`, `ClearRelationCall`, `InjectCppHookCall`, `PostStratumReconstructInternColsCall`. One `@lowering` per MIR op. | NEW `src/srdatalog/ir/dialects/rir/ops_maintenance.py` + `lowerings/maintenance_*.py` (11 modules — one per op for D-discipline) |
| Termination typed union (Amendment 1 Gap 2): `TerminationCheck` base + `ConvergenceCheck`, `MaxIterationFuel`, `EitherFirst`, `AndAll`. `FixpointLoop.termination` retyped from `Op` to `TerminationCheck`. | NEW `src/srdatalog/ir/dialects/rir/ops_termination.py` + `lowerings/termination.py` |
| Variant ordering (Amendment 1 Gap 4): `DeltaVariantSet` + `VariantOrdering` enum. Lowering from MIR semi-naive variant lists. | NEW `src/srdatalog/ir/dialects/rir/ops_variants.py` + `lowerings/variants.py` |
| Multi-head (Amendment 1 Gap 5): `MultiHeadInsert` + `HeadOrdering` enum + `HeadSpec` carrier. Lowering replaces N `InsertInto` ops with one `MultiHeadInsert(heads=(...))`. | NEW `src/srdatalog/ir/dialects/rir/ops_multi_head.py` + `lowerings/multi_head.py` |
| View bindings (Amendment 1 Gap 8): `ViewBinding` carrier added to `Count/Materialize/FusedKernelDef.view_bindings` field. The MIR→RIR lowering for `mir.ExecutePipeline` populates it via the slot-allocation algorithm hoisted out of CUDA-render-time `view_slots.py`. | `ops_kernel_def.py` (field add) + `lowerings/kernel_def.py` (slot-alloc call) + `core/view_layout.py` (the hoisted algorithm) |
| `USE_DECLARATIVE_RUNNER` reaches 100% MIR-op coverage in this PR. `verify_runner_completeness` flips from WARN to HARD. | `src/srdatalog/ir/dialects/rir/__init__.py` |

**Acceptance**: every MIR program lowers to a complete RIR program
with no `MaintenanceCall`-style string-tagged ops, no opaque
`termination: Op` slots, no implicit variant / head ordering, no
render-time view-slot allocation; `verify_runner_completeness`
passes at HARD level; 535 byte-equiv goldens green (legacy runner
still emits the C++).

**Why bundle (2a + 2b)**: each half stands on its own architectural
boundary (2a = structural skeleton; 2b = the typed-semantics filling
that Amendment 1 added). Splitting further would fragment the
lowering surface across too many PRs and re-introduce the per-op
parallelism that the original 35-PR plan suffered from. Both halves
are MIR→RIR work, gated by the same `verify_runner_completeness`;
2b's HARD flip is contingent on 2a's framework being in place.

### 6.3 PR-3 — RIR renders + delete legacy runner (Wave R3 + Cleanup C-1)

| Change | Where |
|---|---|
| `@register_render(RIR-op, target='cuda')` for every RIR op: structural (RunnerStruct, StepDispatch, FixpointLoop, KernelDef) + scheduler (PlainScheduler, WSScheduler, FanOutScheduler, BGScheduler, DedupTable) + 14 MaintenanceCall variants | NEW `src/srdatalog/ir/codegen/cuda/render/rir_*.py` (5-6 modules) |
| `RenderShim` (renamed in PR-1) now dispatches RIR ops alongside IIR ops | already wired in PR-1 |
| **DELETE** `src/srdatalog/ir/codegen/cuda/orchestrator.py` (895 LOC) | git rm |
| **DELETE** `src/srdatalog/ir/codegen/cuda/complete_runner.py` (1123 LOC) | git rm |
| **DELETE** `src/srdatalog/ir/codegen/cuda/runner.py` (802 LOC) | git rm |
| Update `compile_runner(prog, target='cuda')` to invoke the new RIR-render path | `codegen/cuda/api.py` |
| Update `compile_pipeline(ep, target='cuda')` similarly | same |
| Drop the cross-tree side-effect import at the old `complete_runner.py:37` (file is gone) | n/a |
| D19 v2 helper-count ratchet: legacy `_lower_*` function count drops to 0 in the same commit | `tests/test_discipline_obsolete_code_ratchet.py` |

**Acceptance**: 535 byte-equivalence goldens green; the 2820 LOC of imperative monolith is gone; emission is entirely via `@register_render`.

**Why bundle**: R3 renders are atomic — partial coverage means the legacy runner has to coexist with the new renders, doubling code paths. Shipping all renders + the monolith deletion in one PR is the only way to maintain "renders are the sole source of truth" as a strict invariant from this PR forward.

**This is the biggest PR in the plan** (~3500 LOC delta — most of it deletion). Reviewer focus: byte-equiv goldens + the `verify_runner_completeness` per-target gate.

### 6.4 PR-4 — IIR rename (Wave T2)

| Change | Where |
|---|---|
| Rename CUDA-shaped IIR ops to semantic names, atomically: | |
| `SaTiledCartesian2D` → `TiledCartesianDispatch` | `ops.py` + every lowering / render site |
| `BgRootCjMulti` → `BlockGroupRootDispatch`, `BgSourceSpec` → `BlockGroupSourceSpec` | same |
| `SaPrefCoop` → `SaPrefixCooperative`, `SaPrefSeq` → `SaPrefixSequential` | same |
| `GridStrideLoop` → `ParallelStridedLoop`, `LaneZeroGuard` → `LaneFirstGuard`, `TiledBallotBlock` → `TiledValidBlock` | same |
| Drop CUDA-tile params from `SaChildRange` signature (CUDA render encodes the tile shape via the new `TiledCartesianDispatch` op carrying it) | same |
| Op docstrings rewritten: remove literal C++ template text (move that to the renderer's docstring) | `ops.py` |
| Per-rename: new op next to old; lowerings emit new; renders register against new; OLD op deleted in same PR (no deprecation window — internal-only change) | per-op |

**Acceptance**: 535 byte-equivalence goldens green (rename is a no-op for emitted text); the IIR vocabulary is target-agnostic; ready for a non-CUDA renderer to register against the same op set.

**Why bundle**: ~5-8 op renames, each a mechanical search-and-replace. Per-rename PRs would each touch ~30 files; bundling means one ~100-file touch, but the rename is internal-only with byte-equiv as the regression check. Trivial to review (search-and-replace diff).

### 6.5 PR-5 — Index-plugin data/render split (Wave T3)

| Change | Where |
|---|---|
| Split `relation/sorted_array/` such that data dialect (ops, types, print, lowerings) is target-agnostic; CUDA renders move to `codegen/cuda/render/relation_sorted_array.py` | data dialect stays at `dialects/relation/sorted_array/`; renders move |
| Split `relation/d2l/` similarly | renders move to `codegen/cuda/render/relation_d2l.py` |
| The `_PLUGIN_REGISTRY` global in `codegen/cuda/plugin.py` is **deleted** (replaced by per-Compiler render registries keyed on `(op_type, target)`, which already exist after PR-1) | `codegen/cuda/plugin.py` shrinks to deletion |
| External plugins (jaccard demo #68): single-`register()` form continues to work via shim; new multi-target plugins split into data + per-target render contributions. **Phase E retroactively versioned E1 (single-register form) + E2 (data + render split form)** per § 8.1. | `phase_e_plugin_extensibility.md` updated |

**Acceptance**: 535 byte-equivalence goldens green; no `_PLUGIN_REGISTRY` global anywhere; external plugins (jaccard #68) verified to still load via E1 shim.

### 6.6 PR-6 — Multi-target + D20 discipline (Wave B2-2 + Cleanup C-2)

| Change | Where |
|---|---|
| `Compiler.run(prog, pipeline=..., targets=['cuda', 'cpu_tbb'])` accepts a target list; render pass runs once per target; result is `MultiTargetResult(per_target={'cuda': ..., 'cpu_tbb': ...})`. | `core/dialect.py` + `default_pipelines.py` |
| `verify_renderability` runs once per registered target at compiler-bootstrap time (not run time). Failure naming: `(op_type, target)`. | `core/verifier.py` |
| Discipline rule D20 (additive-contract; § 7): PRs labelled `feature:` must not edit existing files under `src/srdatalog/ir/dialects/` or `src/srdatalog/ir/codegen/`. PRs labelled `migration:` / `refactor:` / `bugfix:` are exempt. | `docs/code_discipline.md` + GitHub PR labels |
| `tests/test_discipline_d20_additive_contract.py` — git-history check (`git diff --name-only` between PR head and base): if PR is `feature:` labelled AND modifies existing files in the locked-down trees, fail. | new test |
| Initial state: ratchet-only (warning) for one release cycle, then hard-block. | configurable in test |
| Delete parked A3 branches (do NOT merge their PRs); their work is subsumed by PR-1's LowerCtx split + Phase R's lowering wiring. | git push --delete origin |

**Acceptance**: A second target plugin can be installed via entry point and the same MIR program emits two C++ trees. D20 ratchet active. Parked A3 branches deleted from origin.

### 6.7 Per-PR byte-equivalence anchor

Every PR through PR-3 (RIR renders + monolith deletion) is byte-equivalence-gated on the CUDA target: the 535+ goldens (`tests/test_runner_byte_equivalence.py` 272 + `tests/test_byte_equivalence_jit.py` 253 + lowering goldens) must remain green. PR-3 is the only PR allowed to delete golden divergence-tolerance handlers — the new RIR renders are specified to match the existing CUDA goldens byte-for-byte. PR-2a and PR-2b (post-Amendment 1) are pure-additive at the runtime emission layer (legacy runner still emits the C++; the new RIR layer flows through but is not yet rendered), so byte-equivalence is trivially preserved across both halves.

PR-4 / PR-5 / PR-6 are similarly byte-gated; the rename + split is a no-op for the rendered CUDA C++ text. PR-6 introduces the cpu_tbb mock target (synthetic; minimal renderers; not byte-checked against any reference).

### 6.8 Total ledger

| PR | Concern | Est. size |
|---|---|---|
| PR-1 | Foundation: LowerCtx split + target parametricity | ~1500 LOC |
| PR-2a | RIR framework + structural / scheduling / kernel-body lowerings (Amendment 1) | ~2400 LOC |
| PR-2b | Maintenance + termination + variant ordering + multi-head + view bindings lowerings (Amendment 1) | ~2100 LOC |
| PR-3 | RIR renders + delete legacy runner (3 files, 2820 LOC) | ~3800 LOC (mostly deletion; render-side ops grew with Amendment 1) |
| PR-4 | IIR rename (5-8 op renames) | ~500 LOC (mechanical) |
| PR-5 | Index-plugin data/render split + Phase E re-versioning | ~800 LOC |
| PR-6 | Multi-target + D20 + D21 discipline | ~700 LOC (D21 ratchet added per Amendment 1) |

**Total: 7 PRs.** (Was 6 before Amendment 1; PR-2 split to 2a + 2b.
Previous draft pre-spec: 35. The compression still holds — each PR
is one architectural change, not one mechanical op. Per-PR work
envelope is ~1500-3800 LOC of code + tests, comparable to the
largest Phase B PRs that shipped successfully.)

**Sequencing**: PR-1 → PR-2a → PR-2b → PR-3 → PR-4 → PR-5 → PR-6, strictly serial. PR-2a depends on PR-1's target param + CudaRenderCtx; PR-2b depends on PR-2a's RIR framework being live; PR-3 depends on PR-2b's full RIR coverage; PR-4 depends on PR-3's renderer set being stable (no rename mid-monolith-deletion); PR-5 depends on PR-3's `_PLUGIN_REGISTRY` deletion path being live; PR-6 depends on PR-5's plugin split.

**Parallelism is intentionally NOT used in this plan.** Phase B's per-op parallelism worked because each B-PR was file-disjoint. This redesign's PRs each touch broad swaths of the codebase (PR-1: every pragma lowering site; PR-3: every codegen module). Serializing them keeps merge conflicts to zero.

## 7. Acceptance gates

The redesign is complete iff:

- [ ] **CUDA byte-equivalence preserved end-to-end.** All 535+
  goldens green at every PR. No divergence-tolerance was added.
- [ ] **Synthetic minimal CPU/TBB backend demonstrates the additive
  contract.** Lands as a separate package
  `examples/srdatalog_cpu_tbb_demo/` (mirroring the jaccard demo in
  PR #68). Renders to mock sequential C++; no real perf. Pure
  validation of the extension mechanism.
- [ ] **`verify_renderability` runs per target.** At
  `Compiler.with_default_plugins()` bootstrap, the check runs once
  per registered target; any (op, target) gap is named loudly.
- [ ] **D20 discipline rule blocks feature-edit PRs.** PRs that edit
  existing files in `src/srdatalog/ir/dialects/` or
  `src/srdatalog/ir/codegen/` to add a feature are blocked at CI.
  Adding a feature requires a new plugin module.

D20 proposed text (style-matched to `code_discipline.md` §2):

| ID | Forbidden pattern | Why | Discipline test |
|---|---|---|---|
| **D20** | A PR adding a new feature (algorithm, pragma, index type, aggregation, target) edits existing source files under `src/srdatalog/ir/dialects/**.py` or `src/srdatalog/ir/codegen/**.py`. Edits to designated registry sets (`DEFAULT_PIPELINE`, plugin entry points, `__all__` re-exports) are allowed; new functions, new branches in existing functions, new fields on existing types are NOT. | The ACID test from `phase_decomposition_redesign.md` §1: every new feature is a purely additive plugin. Editing an existing file is the imperative-monolith failure mode the redesign reverses. | `test_discipline_d20_additive_contract.py` — for PRs labeled `feature:` (new functionality), parse the diff; assert that no `.py` file under `src/srdatalog/ir/dialects/` or `src/srdatalog/ir/codegen/` has lines added outside the designated registry-set markers. |

D20 mechanics: enforced only on PRs labeled `feature:` (a separate
GitHub label). Bug-fix PRs, refactor PRs, and discipline-amendment
PRs are exempt. The label is set by the author; reviewers verify
the label-vs-content match. (This is the same procedural-vs-
mechanical split as the `code_discipline.md` §5 enforcement
architecture.)

### 7.1 D21 — no string-tagged dispatch in renderers (Amendment 1)

The § 1 sub-clause introduced by Amendment 1 requires that
renderers dispatch on op TYPE only, never on a string field of an
op. D21 enforces this mechanically.

| ID | Forbidden pattern | Why | Discipline test |
|---|---|---|---|
| **D21** | A `@register_render` function body matches on a STRING field of its op (e.g. `if op.kind == 'count':`, `match op.phase: case 'fused': ...`, `if op.scheduler_kind in {'ws', 'bg'}:`). String fields naming render targets, struct names, or relation names are NOT discipline-violating reads (they're identifier-shaped); branch-on-string is. | The § 1 Amendment 1 sub-clause: every operational semantic is typed. A string-tag dispatch in a renderer is the same imperative-monolith anti-pattern this redesign reverses, regressed inside the new typed shell. | `tests/test_discipline_d21_no_string_dispatch_in_renders.py` — AST scan of every module that imports `register_render`; for each decorated function body, walk for `if attr ==` / `match attr:` patterns where the matched expression resolves to a field whose declared type is `str` (or `KernelPhase` / similar enum-of-strings). Whitelist: ID-shaped string comparisons (target name, rule name) — reviewed exceptions noted in the test fixture. |

D21 lands in PR-6 alongside D20 (same ratchet-then-block cadence:
WARN for one release, then HARD). Initial discovery: scan the
codebase as of PR-3 completion; the legacy renderer is gone, so the
allowlist should be empty. Any new violation introduced by PR-4 /
PR-5 / external plugins fails at CI.

## 8. Relationship to existing phases / parked work

### 8.1 Stays merged unchanged

- **Phase F1-F5** (foundation). The framework infrastructure
  (Pass kinds, declarative pipeline, plugin discovery, small
  LowerCtx) is correct as designed; only the dialect + codegen
  layers need decomposition. Phase B2 extends F4/F5's pipeline-as-
  data to per-target; the framework itself is unchanged.
- **Phase A1-A2** (MIR onto Op). The frozen-Op work is target-
  agnostic and correct; Phase T renames some IIR ops the MIR ops
  feed into, but doesn't touch MIR.
- **Phase B** (per-MIR-op `@lowering` for kernel-body). Stays
  merged. Phase R extends the exact same pattern to the runner
  layer; the migration is the same shape (per-MIR-op PR, with a
  `USE_DECLARATIVE_RUNNER` ratchet).
- **Phase C** (typed pragmas). Stays merged. The Pragma classes are
  target-agnostic; Phase T may rename some IIR ops the pragmas
  materialize to, but the pragma-handler topology is unchanged.
- **Phase D** (HIR `ProgramPass`). Stays merged. The HIR planning
  passes are pre-MIR and target-agnostic; no impact from Phase R/T/B2.

**Amendment 1 note (Phase E re-versioning).** The new
`ComposabilityMeta` field on the `Pragma` class introduced by
Amendment 1 Gap 7 is target-AGNOSTIC — it lives on the framework's
`Pragma` base, NOT on any per-target render contribution. It does
NOT introduce a new dimension that would require a third Phase E
sub-version (E3); E1 + E2 (data + per-target render split) remains
the right partition. The composability check fires inside
`MirPragmaPass.apply` (pre-target-dispatch), so target-side plugins
inherit it for free.

### 8.2 Stays merged but extended

- **Phase E** (built-ins as plugins). Stays merged. Phase T's
  per-target render split (§3.2.4) means each dialect plugin now
  splits into a data contribution + per-target render contribution.
  Existing built-in dialects need a one-time refactor (Wave T3)
  to match. **External plugins (PR #68 jaccard demo) are NOT
  retroactively broken** — the existing single-`register(compiler)`
  entry point continues to work; the split affects only how
  built-ins are reorganized. External plugins that ALSO want to
  ship to multiple targets opt into the per-target split.
  (Open question: should Phase E be retroactively versioned to
  e.g. "Phase E1" + "Phase E2" — the original Phase E + the
  per-target split? Recommendation: yes, when the spec PR for
  Wave T3 lands. See section 9.)

### 8.3 Superseded

- **Phase A3** (parked branches A3-1/A3-2/A3-3 + the open spec at
  `phase_a3_remove_deprecated_bool_fields.md`). **SUPERSEDED.** The
  bool-field removals + the `ep_has_X(...)` helpers they introduced
  both disappear under Phase R. Don't merge the parked A3 branches;
  they're net-negative work post-redesign (every line they add is
  deleted by R3-8).
  - A3-1/A3-2/A3-3 sitting on local branches (per `git branch -a`):
    `feat/a3-1-remove-dedup-hash-bool`, `feat/a3-2-remove-work-stealing-bool`,
    `feat/a3-3-remove-block-group-bool`. **Mark as
    Closed-Won't-Merge on the PR tracker** once this spec is
    approved.
  - A3-4/A3-5/A3-6 (the harder ones — `use_fan_out`, `count`,
    `tiled_cartesian`) NOT YET implemented. Their goals are
    satisfied by Phase R: the bool reads disappear into typed RIR
    op dispatch.
  - The `MirTiledEligibilityPass` proposal in A3-6 §3.6 stays
    valuable independently: it's a real MIR pre-pass that attaches
    a `TiledCartesian` pragma to eligible EPs (target-agnostic).
    Salvage as a separate PR under Wave T1 (rename to
    `MirEligibilityPass` if eligibility for additional pragmas
    accretes).

### 8.4 Stays open + extended

- **PR #68 jaccard demo** (`feat/e-d2l-as-plugin` ancestor —
  external plugin proof of concept). Stays as the customizability
  proof. Gets extended once Wave T3 lands: an updated jaccard
  package ships the data dialect + a CUDA render module (the same
  shape Phase T applies to built-in dialects). No urgency; the
  current single-`register(compiler)` form continues to work.

- **PR #72 dispatcher gap fix.** Stays merged. It's framework
  infrastructure (closing the per-MIR-op dispatch gap for
  `USE_DECLARATIVE`); correct independent of Phase R/T/B2. No
  impact.

### 8.5 Disposition summary table

| Phase / PR | Status | Rationale |
|---|---|---|
| Phase F1-F5 | Stays merged | Foundation is correct |
| Phase A1-A2 | Stays merged | MIR-onto-Op is target-agnostic |
| Phase B (Wave 2A) | Stays merged | Per-MIR-op @lowering pattern extends to Phase R |
| Phase C | Stays merged | Typed pragmas are target-agnostic |
| Phase D | Stays merged | HIR ProgramPass is target-agnostic |
| Phase E | Stays merged + extended | Per-target render split in Wave T3 |
| Phase A3 (parked branches) | **DO NOT MERGE** | Superseded by Phase R |
| Phase A3 (spec doc) | Mark superseded; do not delete | Historical record of the diagnosis |
| PR #68 jaccard demo | Stays open + extended | Customizability proof; extends per Wave T3 |
| PR #72 dispatcher gap | Stays merged | Framework infra; orthogonal |
| Phase R / T / B2 (this spec) | Replaces parked A3 + Layer 3 | The unified plan |

## 9. Sign-off

This document represents an architectural commitment. Sign-off
includes:

- [ ] §1 ACID test is the load-bearing invariant for every future
  feature PR.
- [ ] §2 diagnosis (file:line tables) is accurate; reviewers spot-
  check a sample.
- [ ] §3 three-axis fix (R + T + B2) is the right decomposition;
  no fourth axis missing.
- [ ] §4 worked example (CPU/TBB backend) is achievable with the
  proposed shape; one reviewer prototypes the package skeleton.
- [ ] §5 worked example (semiring) is similarly achievable.
- [ ] §6 per-PR partition: 35 PRs is the right granularity;
  per-PR byte-equivalence gate is the right discipline.
- [ ] §7 acceptance gates are sufficient (especially D20).
- [ ] §8 phase disposition is correct; the parked A3 branches are
  closed (not merged).
- [ ] Open question from §8.2: Should Phase E be retroactively
  versioned to E1+E2 to mark the per-target-render-split
  amendment? Reviewer answers yes/no with one-paragraph rationale.
- [ ] **Amendment 1 (this revision) acknowledgment.** Reviewer
  acknowledges that this is the FIRST of likely SEVERAL spec
  refinements as PR-1's implementation surfaces additional
  semantic gaps. The spec is a LIVING DOCUMENT during the redesign;
  subsequent amendments are PR-shaped (one Amendment per amendment
  PR, numbered sequentially, each adding a § 3.1.1.N expansion for
  the new typed concerns it introduces) and reviewed under the
  same gates as this one. Amendment 1 covers: 11 typed maintenance
  ops; typed `TerminationCheck` union; typed `Step.deps` DAG
  edges; typed `DeltaVariantSet` + `VariantOrdering`; typed
  `MultiHeadInsert` + `HeadOrdering`; three typed kernel-def ops
  replacing `KernelDef.phase: str`; typed `ComposabilityMeta` on
  `Pragma`; typed `ViewBinding` IR carrier. Renderer-side
  discipline D21 ratchets the sub-clause.

Reviewer initials and date below.

## 10. Amendment log

Amendments are appended below as they are merged. Each amendment
identifies the spec sections it touches and the typed concerns it
introduces. The spec is consumed top-to-bottom; amendments may
modify earlier sections in place, but the log here is the canonical
record of WHEN and WHY a change landed.

### Amendment 1 — typed RIR semantics

| Field | Value |
|---|---|
| Date | 2026-05-15 |
| Trigger | User review of merged PR #74 identified 8 semantic gaps where the proposed RIR vocabulary still expressed operational meaning as string-tagged dispatch or implicit list ordering, in violation of the § 1 ACID test. |
| Sections touched | § 1 (sub-clause added); § 3.1.1 (new § 3.1.1.1 sub-section); § 3.1.2 (lowering table updated); § 3.1.3 (render table updated); § 6.2 (split to 6.2a + 6.2b); § 6.7 (byte-equiv note); § 6.8 (total grew 6 → 7 PRs); § 7 (D21 added as § 7.1); § 8.1 (re-versioning note); § 9 (living-document acknowledgment). |
| Concerns typed | Maintenance ops (11 typed ops replacing one string-tagged `MaintenanceCall`); fixpoint termination (typed `TerminationCheck` union: `ConvergenceCheck`, `MaxIterationFuel`, `EitherFirst`, `AndAll`); cross-stratum deps (typed `Step.deps`); delta-variant ordering (typed `DeltaVariantSet` + `VariantOrdering`); multi-head emission (typed `MultiHeadInsert` + `HeadOrdering` + `HeadSpec`); kernel fusion (three typed kernel-def ops: `CountKernelDef`, `MaterializeKernelDef`, `FusedKernelDef`); pragma composability (typed `ComposabilityMeta` on `Pragma`); view-slot binding (typed `ViewBinding` carrier on kernel-def metadata). |
| RIR op count | grew from ~10 to **30** typed Ops (+ 5 carrier dataclasses / enums). See § 3.1.1.1 final table. |
| PR-2 split | YES — PR-2 split to PR-2a (framework + structural / scheduling / kernel-body) + PR-2b (maintenance + termination + variant ordering + multi-head + view bindings). Total ledger: 6 PRs → 7 PRs. |
| Discipline added | D21 (no string-tagged dispatch in renderer bodies). Ratchets per the standard WARN-then-HARD cadence. |
| Forward-compat | Each spec section retains its original heading; new sub-sections (§ 3.1.1.1, § 7.1, § 6.2a, § 6.2b, § 10) are additive. PR-1 scope (§ 6.1 + § 3.2 + § 3.3) is UNCHANGED — PR-1's foundation work is orthogonal to the RIR vocabulary expansion. |
| Scope EXEMPT | PR-1 (in flight). Amendment 1 deliberately does not touch any text PR-1 depends on. |

### Amendment 2 — no-context principle

| Field | Value |
|---|---|
| Date | 2026-05-19 |
| Trigger | User review of merged PR-1d (#81) identified that the proposed `CudaRenderCtx` (10 named fields) was itself the same fixed-schema-attribute-carrier anti-pattern that the redesign is supposed to eliminate. Adding any new CUDA render feature (a new tile mode, gate, or ballot variant) would still require editing `lower_ctx.py` to add a field — a clear § 1 ACID-test violation. PR-1d reverted via #82. |
| Sections touched | § 1 (sub-clause added — three-bucket discipline); § 3.2.1 (REWRITTEN — was "Split LoweringCtx from CudaRenderCtx", now "Dissolve LoweringCtx (no-context principle)"); § 6 PR plan (PR-1d / PR-1e replanned as PR T1-α / β / γ / δ — final count TBD post-audit); § 7 (D22 added — strict monotonic decrease of ctx-field count); § 10 (this entry). |
| Principle | Information flows between passes ONLY through the IR. Three disjoint mechanisms: (1) cross-pass data → typed IR op (never a ctx field); (2) intra-pass scratch → function-local / explicit kwarg (never a ctx mutation); (3) services → immutable injected `Services` handle (never a feature-by-feature accumulated attribute set). |
| Carriers dissolved | `LoweringCtx`, `KernelCtx`, `InitialProg`, `CudaRenderCtx` (if reintroduced — it won't be). All four disappear by end of Phase T-1. |
| Discipline added | D22 (no through-state contexts; ctx-field count strictly monotonically decreasing). Ratchets per the standard WARN-then-HARD cadence. |
| Forward-compat | PR-1a, PR-1b already merged and unaffected. PR-1c (target threading via carrier dataclass fields) is now technically a discipline violation under D22 — the `target` field on `InitialProg` / `KernelCtx` is a small bucket-1 leak that the Phase T-1 dissolution work will sweep up. Not reverted; the dissolution PRs will move `target` to the `Services` handle (it's per-`Compiler.run` metadata, not per-feature data). |
| Scope NOT exempt | Unlike Amendment 1, Amendment 2 directly invalidates a merged PR's design. The audit (in flight) is sweeping for any other fixed-schema carriers that should be added to the dissolution list. Per-PR replan of the Phase T-1 work happens after the audit lands. |

### Amendment 3 — full carrier dissolution + pragma plugin contract + architectural precedent

| Field | Value |
|---|---|
| Date | 2026-05-19 |
| Trigger (1) | User asked "see if entire `context.py` is unnecessary under correct design?" — analysis confirmed the entire `src/srdatalog/ir/codegen/cuda/context.py` (543 LOC) dissolves. |
| Trigger (2) | User noted "this declarative + context data managing sounds like some familiar things in database or some other dev" — the pattern is exactly relational query compilation (Volcano/Cascades/Calcite/DuckDB), MLIR dialects, LLVM analyses, FRP UI. 40 years of validation. |
| Trigger (3) | User asked "also please review if your planned solution actually solve the issue, not introduce just new tangled non-declarative and global state" — self-audit identified 5 risks where the proposed solution would have rebuilt the anti-pattern one layer up. |
| Trigger (4) | **The load-bearing finding.** User asked "we have pragma which is actually meta programming will effect compiler, they should be no hardcode, but some pass facility allow the pragma with these package like compiler plugin actually works" + "is current pragma contain a written pass, an customized IR, and a rewrite all doable as a pragma?" — answer: NO. Today's `@pragma_handler` is a constrained materialization-handler, NOT a plugin. A pragma cannot contribute IR ops, run as a general pass, bundle lowerings + renders, or declare dependencies. Without a real pragma plugin contract, every other phase shuffles code without changing the extensibility surface. |
| Trigger (5) | User: "what is next stage and the work can be parallel working on?... without it none of work is correct, just moving code make increasing code loc?... where we are just increasing useless entropy?" — this re-prioritizes the entire PR plan. |
| Sections touched | § 3.2.1.1 (NEW — full carrier dissolution scope); § 3.2.1.2 (NEW — self-audit: 5 risks + their fixes — typed-key Services dict, MLIR-style attribute dict on ops, sharpened bucket-2 boundary, per-Compiler registries, ops as typed shells); § 3.2.1.3 (NEW — the pragma plugin contract: `PragmaPlugin` schema, `register_pragma_plugin()` API, what the framework knows about pragmas after Phase P = nothing pragma-specific); § 6.0 (NEW — PR-P series: PR-P0 framework primitives serial; PR-P1/2/3/4 per-pragma migrations parallel; PR-P5 context cleanup as consequence; 6 PRs total ~8300 LOC); § 11 (NEW — architectural precedent across 4 domains); § 10.3 (this entry). |
| Entropy retrospective | PR-1a/b (entry-point split + shim renames) ship plumbing the plugin contract needs; legitimately load-bearing. PR-1c (target threading via carrier dataclass fields) is small-scope plumbing but creates a bucket-1 leak that Phase P sweeps up. PR-1d (CudaRenderCtx) was pure entropy — moving fields between two equally-anti-pattern carriers; reverted in #82. The original PR-2/3/4 plan (RIR work) is NOT entropy but is BLOCKED by Phase P — running it first would design RIR ops around the wrong plugin contract and require a second rewrite. **Net assessment: ~10% of the PR work to date was entropy (PR-1d), ~30% was useful-but-out-of-order (B-series, A3-series — useful as inputs to Phase P), ~60% was load-bearing (PR-1a/b foundation, C-pragma typed-class series, the kernel-body declarative lowerings).** |
| Highest-priority work | Phase P (§ 3.2.1.3 + § 6.0). All other phases become consequences. |
| Carriers added to dissolution scope | `CodeGenContext` (38 fields), `CodeGenHooks` (10 callable slots), `RunnerGenState` (14 fields), plus 30+ ctx-taking helper functions in `context.py`. All of `context.py` (543 LOC) deleted by PR-P5. |
| Other anti-pattern sites folded in | 7 additional sites identified by audit: surviving `mir.ExecutePipeline` bool fields, `ep_has_work_stealing` survivor, `print_iir.py` isinstance ladder, `render/__init__.py` hardcoded dialect imports, `pipeline.py` `_INDEX_HEADER` map, `_TERMINAL_WRAP_OPS` enumeration, `core/verifier.py` string-tagged dispatch + D6 violation. Most dissolve when their owning pragma migrates to Phase P. |
| Precedent recognized | Relational query compilation (Volcano 1993 → Cascades → Calcite → DuckDB 2024); MLIR (dialects + typed Op attributes + conversion patterns, no `ConversionContext`); LLVM (analyses via `getAnalysis<T>()`, not ctx fields); FRP UI (React/Solid — components are data, hooks are lexical, "Context Providers" carry services not feature accumulators). § 11 spells out the mapping per domain. |
| PR #68 jaccard demo audit note | Concurrent audit of PR #68 found PASS-WITH-CAVEATS: zero edits to existing files (ACID-test letter satisfied), but the demo internally uses `ctx.dedup_hash` bool-field flip and imports private `_lower_insert_into` — the same bool-on-typed-op anti-pattern the redesign is retiring. Demo cannot yet be cited as proof of the additive contract; four follow-ups: (1) public re-export of the lowering helper, (2) drop the `dedup_hash` flip, (3) close the `_lower_inner_chain` registry gap, (4) replace `compiler._plugins_loaded` private access. Phase P cleans up the framework side; the demo updates separately. |
| Discipline added | Amendment 3 reaffirms Amendment 2's D22 (ctx-field count monotonic decrease) and adds, implicitly, D23 (no pragma name in framework code — every framework reference to a specific pragma class or pragma-named field is a § 1 violation, blocked once Phase P lands). |
| Forward-compat | All amendments compose. Amendment 1 = typed semantics. Amendment 2 = no through-state contexts. Amendment 3 = full dissolution scope + Phase P pragma plugin contract + architectural precedent. PR-1a/b retained; PR-1c retained but `target` moves to `Services` in Phase P; PR-1d reverted. |

## 11. Architectural precedent (Amendment 3)

The "everything is IR data; passes communicate only via IR; services
are injected, not accumulated" shape we are converging on is not a
new invention. It has 40 years of validation in four adjacent fields.
This section names the precedent so future contributors and reviewers
can pattern-match against existing literature rather than re-deriving.

### 11.1 Relational query compilation

The closest analogue. A SQL query is data (an AST). A logical plan is
IR. A physical plan is IR. The emitted operator tree (or compiled
machine code, for compiled query engines) is the final text. Every
stage transforms IR; no mutable "context" carries information across
stages. Precedents:

- **Volcano** (Graefe, 1993) introduced the iterator-based execution
  model and the optimizer-generator framework. Optimizer rules are
  declarative pattern→pattern transformations. The "context" of an
  expression is its pattern + its cost annotation, attached to the
  expression itself — not an external state object.
- **Cascades** (Graefe, 1995) refined Volcano with the "Memo"
  structure: every logical expression and its equivalent physical
  alternatives live in groups indexed by logical equivalence. Cost
  estimates, properties, and bound plans are attached to the
  group/expression — IR-as-data carrying its own metadata. The
  framework is the optimizer; the rules are the plugins.
- **Apache Calcite** (Hyde et al., 2010s) made Cascades-style
  optimization pluggable. New SQL dialects, new optimization rules,
  new physical conventions, new cost models all register via
  `@RelOpt*` annotations. The framework knows nothing about specific
  dialects; the registries are the source of truth.
- **DuckDB** (Raasveldt & Mühleisen, 2020s) takes the same shape to
  a vectorized engine: expressions are typed IR nodes, optimization
  passes consume + produce IR, the executor walks the final IR. No
  global context carries dialect-specific state.

**Mapping to srdatalog:** MIR is our logical plan; IIR + RIR are our
physical plans; the rendered C++ is our compiled output. `@lowering`
rules are Cascades-style transformations. `@register_render` is our
emission-side plugin point. The Memo group has no analogue here
because we don't do cost-based optimization — but the IR-attached
metadata shape is the same.

### 11.2 MLIR

The most direct architectural sibling. MLIR (Lattner et al., 2019,
LLVM project) is the modern reference for IR-centric compilation:

- **Dialects** are plugins. A dialect ships its `Op`s, their
  attributes, type system, and verification rules. The framework
  knows nothing about specific dialects.
- **Op attributes** carry all per-op metadata. No "context" carries
  per-op state across passes. If pass B needs information that pass A
  computed, that information is an attribute on the op.
- **Conversion patterns** are typed: `@register_conversion_pattern(
  SourceOp, TargetOp)`. The framework's `DialectConversion` driver
  consults the registry. There is no `ConversionContext` carrying
  feature flags.
- **Pass managers** are services. They schedule passes; they do not
  accumulate state on behalf of features.
- Adding a new dialect or a new conversion pattern is purely
  additive — the canonical example is the affine + linalg + vector +
  llvm dialect stack, every dialect added without editing the
  framework.

**Mapping to srdatalog:** MIR/IIR/RIR are our dialects. `Pragma` ops
are our op attributes (typed, post Amendment 1). `@lowering` and
`@register_render` are our conversion patterns. The `Compiler` is
our pass manager. The Phase R/T/B2 work brings srdatalog into
structural alignment with MLIR's plugin contract.

### 11.3 LLVM (legacy pass manager + analyses)

The classical compiler architecture for IR + pull-based analyses:

- **Passes** declare their dependencies on **Analyses**. The
  `PassManager` resolves and caches analysis results.
- An Analysis is pulled, never pushed: a pass that needs the
  dominator tree calls `getAnalysis<DominatorTreeAnalysis>()`. The
  result is typed, cached, and invalidated explicitly when its
  inputs change. There is no `ctx.dominator_tree` field.
- New analyses are additive: a plugin declares a new Analysis class
  with a public ID; passes opt in by name.

**Mapping to srdatalog:** the four `topology_*` checks
(`semi_naive_safety`, `pragma_compat`, etc.) are analyses today, but
they execute imperatively at pipeline construction time. Phase R
makes them pull-based via the verifier infrastructure
(`verify_renderability`, `verify_runner_completeness`). Future work
generalizes to a typed Analysis registry, mirroring LLVM's shape.

### 11.4 Functional reactive UI (React, Solid, Elm)

The shape is not compiler-specific. The same pattern in UI:

- **Components are data** (JSX trees, virtual DOM nodes).
- **State is lexical** — `useState`, `useReducer`, signals — never
  a global context attribute.
- **"Context Providers"** carry services (theme, router, auth,
  i18n) — not feature accumulators. A new feature does NOT add a
  field to the theme context; it adds its own provider.

**Mapping to srdatalog:** the no-context principle here is the
direct analogue of React's "props down, events up" + Context-for-
services discipline. Rendering a `KernelDef` op is rendering a
component; the op's typed bindings are its props; the `Services`
handle is the Context Provider for `name_gen` / `compiler` /
`plugin_registry`.

### 11.5 Pattern summary

| Domain | Data | Pipeline | Per-op metadata | Services |
|---|---|---|---|---|
| Relational compilation | SQL → logical → physical plan | Optimizer rules | Memo group annotations | Catalog |
| MLIR | dialect ops | Conversion patterns | Op attributes | PassManager |
| LLVM | IR | Passes | (none — IR + Analyses) | PassManager + AnalysisManager |
| FRP UI | JSX | Render tree | Component props | Context Providers |
| **srdatalog (target)** | MIR/IIR/RIR ops | `@lowering` / `@register_render` | Typed `Pragma` + `ViewBinding` etc. | `Services` handle |

The shape is consistent across all five rows: **data carries its own
typed metadata; transformations are declarative pattern→pattern;
services are injected, not accumulated.** When this redesign feels
like reinventing something, that something is one of these four
established models — and the design choices have known names. Reach
for the literature when in doubt.

### 11.6 What this does NOT mean

This precedent does NOT claim srdatalog should adopt MLIR's exact
API surface or recompile to MLIR IR (it could, in a future phase —
spec § 8.6 already sketches that as out-of-scope-but-aligned). It
claims only that the SHAPE we are converging on is well-trodden,
and naming the precedent helps reviewers + future contributors
avoid re-deriving design choices from first principles. The four
references above are the recommended background reading for anyone
shipping a non-trivial Phase R / T / B2 PR.
