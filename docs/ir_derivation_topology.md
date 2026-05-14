---
orphan: true
---

# IR Derivation Topology

The graph every contributor needs to share. Defines:

- The IR layers and how they relate.
- The dialects within each layer.
- The Pass kinds that move ops between dialects (`LoweringPass`) or
  rewrite within a dialect (`RewritePass` / `ProgramPass`).
- Where each extension point lives (sub-dialect creation, pragma
  registration, target plugin).

Companion to [`compiler_redesign.md`](compiler_redesign.md). Read that
first for the architectural spine; this doc fills in the per-dialect
detail and the dataflow graph.

## 1. The IR derivation graph

```
                      ┌──────────────────────────┐
                      │  FRONTEND (plugin)       │
                      │  Default: Python DSL     │
                      │  srdatalog.dsl.Program   │
                      └────────────┬─────────────┘
                                   │ no IR pass — direct construction
                                   ▼
                      ┌──────────────────────────┐
                      │  HIR (planning records)  │
                      │  srdatalog.ir.hir.types  │
                      │                          │
                      │  Mutable. Not Op-subclassed. │
                      │  Contains:               │
                      │   HirProgram             │
                      │   HirStratum             │
                      │   HirRuleVariant         │
                      │   AccessPattern          │
                      │   RelationDecl           │
                      └────────────┬─────────────┘
                                   │
                ┌──────────────────┴────────────────┐
                ▼                                   ▼
   ┌──────────────────────────┐         ┌──────────────────────────┐
   │ ProgramPass chain (HIR)  │  loop   │ Each pass re-emits a     │
   │  StratifyPass            │  ────→  │ HirProgram (mutated      │
   │  SplitPass               │         │ in-place per pass).      │
   │  SemiNaivePass           │         │                          │
   │  PlanPass                │         │ Each pass is a single    │
   │  IndexSelectionPass      │         │ file under hir/passes/.  │
   │  TempRelSynthesisPass    │         │                          │
   │  TempIndexRegistration   │         │                          │
   └──────────────────────────┘         └──────────────────────────┘
                │
                │ LoweringPass: HirToMirLowering
                │   (maps HirRuleVariant → mir.ExecutePipeline,
                │    carrying pragmas as a generic dict)
                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  MIR (frozen Op tree)                                        │
   │  srdatalog.ir.dialects.mir                                   │
   │                                                              │
   │  All Op-subclassed (Phase A). Two op groups:                 │
   │                                                              │
   │  mir.types  ─ generic ops:                                   │
   │     Program, ExecutePipeline, Scan, ColumnJoin,              │
   │     CartesianJoin, Filter, ConstantBind, Aggregate,          │
   │     Negation, InsertInto, InjectCppHook, ColumnSource, ...   │
   │                                                              │
   │  mir.pragma_ops  ─ wrap ops inserted by @pragma rewrites:    │
   │     DedupGate, BlockGroupRoot, WSScope, ...                  │
   │     (+ ops shipped by external @pragma plugins)              │
   └────────────┬─────────────────────────────────────────────────┘
                │
                ├─ RewritePass: MirPragmaPass (one-shot)
                │   walks @pragma registry, each registration fires
                │   when its trigger condition matches; inserts wrap ops
                │   from mir.pragma_ops; clears the pragma key.
                │
                ├─ RewritePass: MirOptPass (fixpoint)
                │   R1–R5 from ir_lowering_semantics.md §11
                │   (count-as-product, hint introduction, etc.)
                │
                │ LoweringPass: MirToIirLowering
                │   per-MIR-op @lowering registrations live in
                │   dialects/relation/<ds>/lowerings/lower_mir_*.py
                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  IIR (multi-dialect frozen Op tree)                          │
   │                                                              │
   │  STRUCTURAL dialects (LEAF, all-renderable):                 │
   │    iir.cf       Block, BracedBlock, Bind, If, IfReturn,      │
   │                 IfContinue, ParallelFor, GridStrideLoop,     │
   │                 IntersectIter, LaneZeroGuard, Phase,         │
   │                 Assign, IndexedAssign, StmtExpr, Comment,    │
   │                 BlankLine, IndentBlock, OuterAnchor,         │
   │                 TiledBallotBlock, WriteOutput, AddCount,     │
   │                 RawString (transition-only), UserCode,       │
   │                 VarRef                                       │
   │    iir.expr     BinOp, UnaryOp, Parens, Ternary, IntLit,     │
   │                 MemberAccess, MemberCall, FuncCall,          │
   │                 IndexExpr, CCast, StaticCast,                │
   │                 PostfixIncrement                             │
   │                                                              │
   │  DATA-STRUCTURE dialects (mostly LEAF, some COMPOUND):       │
   │    relation.sorted_array  SaRoot, SaPref{Coop,Seq}, SaHint,  │
   │                           SaDegree, SaValid, SaGetVal*,      │
   │                           SaIterators, SaChildRange,         │
   │                           SaTiledCartesian2D,                │
   │                           DedupTryInsert (COMPOUND)          │
   │    relation.d2l           D2lSegmentLoop                     │
   │    relation.lsm⟨K⟩  (future, plugin-shipped)                 │
   │    relation.uf      (future, plugin-shipped)                 │
   │                                                              │
   │  PARALLELISM dialects (LEAF, codegen-aware):                 │
   │    parallel.data           ParallelFor, GridStrideLoop       │
   │    parallel.block_group    BgRootCJ, BgWorkBalance           │
   │    parallel.atomic_ws      WSCount, WSEmit, WSScope          │
   │                                                              │
   │  Each plugin can ship any number of dialects.                │
   └────────────┬─────────────────────────────────────────────────┘
                │
                ├─ RewritePass: IirCanonicalizePass (fixpoint)
                │   COMPOUND ops decomposed to LEAF
                │   (e.g. DedupTryInsert → BracedBlock + Bind + If +
                │    MemberCall, per the rewrite registered on
                │    sorted_array)
                │
                │ verify_renderability(target='cuda') runs after
                │ fixpoint; loud failure on any unhandled op.
                ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  TARGET RENDER (per plugin-shipped target)                   │
   │                                                              │
   │   target.cuda      @register_render(Op) per dialect          │
   │                    in codegen/cuda/render/*.py               │
   │   target.cpp_tbb   (future, plugin-shipped)                  │
   │   target.metal     (future, plugin-shipped)                  │
   │                                                              │
   │  Each target ships its own render module; the @register_render │
   │  registry is per-target. Adding a target = ship a plugin.    │
   └──────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
                       Source code text
```

## 2. Edge legend

- **Vertical arrows between layer boxes** = `LoweringPass` (cross-IR
  transformation; source dialect → target dialect).
- **Self-loops at each layer** = `RewritePass` (within-dialect; same
  source and target dialect).
- **HIR has only `ProgramPass`es.** HIR types are not `Op`-subclassed
  (planning records); per-op dispatch isn't applicable.
- **MIR & IIR have full `LoweringPass + RewritePass` capability.** Both
  are frozen Op trees walkable by the framework.
- **Render is the terminal stage.** It's a target-plugin-defined
  per-op dispatch (the existing `@register_render` registry). It
  produces text, not IR.

## 3. Per-dialect tables

### 3.1 HIR (planning, not Op-subclassed)

| Type | Role |
|---|---|
| `HirProgram` | Top-level container: strata + relation_decls + global indices. |
| `HirStratum` | A single SCC + its rule variants. |
| `HirRuleVariant` | One delta variant of a rule (semi-naive). Carries `pragmas` dict. |
| `AccessPattern` | Per-clause access pattern (after planning). |
| `RelationDecl` | Relation declaration (name, types, semiring, index_type, ...). |

| Pass | Kind | Notes |
|---|---|---|
| `StratifyPass` | ProgramPass | SCC analysis on the rule dependency graph. |
| `SplitPass` | ProgramPass | Splits multi-head rules into single-head variants. |
| `SemiNaivePass` | ProgramPass | Generates delta variants for recursive rules. |
| `PlanPass` | ProgramPass | Chooses var_order / clause_order per variant (heuristic + user overrides). |
| `IndexSelectionPass` | ProgramPass | Computes required indices per relation. |
| `TempRelSynthesisPass` | ProgramPass | Synthesizes temp relations (e.g. semi-join helpers). |
| `TempIndexRegistrationPass` | ProgramPass | Registers indices for synthesized temps. |

Plugin extension: a third party can ship a new HIR pass by registering
a new `ProgramPass` instance and inserting it into the pipeline at a
specified position (per [`phase_d_hir_passes.md`](phase_d_hir_passes.md)).

### 3.2 MIR (Phase A: all Op-subclassed)

#### 3.2.1 Generic ops (`mir.types`)

| Op | Role | Has pragmas dict? |
|---|---|---|
| `Program` | Top-level container; sequence of (`ExecutePipeline | InjectCppHook`, is_recursive) tuples. | No |
| `ExecutePipeline` | A single rule variant lowered to a pipeline. | **Yes** |
| `Scan` | Iterate a relation. | No |
| `ColumnJoin` | Multi-source column-join (CJ_multi). | No |
| `CartesianJoin` | Cartesian join. | No |
| `Filter` | User-supplied predicate. | No |
| `ConstantBind` | User-supplied expression bound to a var. | No |
| `Aggregate` | Aggregation clause. | No |
| `Negation` | Negation clause. | No |
| `InsertInto` | Materialize / count into output relation. | No |
| `InjectCppHook` | Inline raw C++ from pragma. | No |
| `ColumnSource` | Source descriptor (rel_name, version, index, prefix_vars). Field, not stand-alone op. | No |

#### 3.2.2 Pragma wrap ops (`mir.pragma_ops`)

These are inserted by `@pragma` rewrites during `MirPragmaPass`. They
live in `dialects/mir/pragma_ops/<pragma>.py`.

| Op | Inserted by pragma | Wraps |
|---|---|---|
| `DedupGate` | `dedup_hash` | `InsertInto` |
| `BlockGroupRoot` | `block_group` | Root op of pipeline (Scan / ColumnJoin) |
| `WSScope` | `work_stealing` | Body of ExecutePipeline |
| (future) | (any external pragma) | (anything that pragma's rewrite touches) |

| Pass | Kind | Fixpoint? | Notes |
|---|---|---|---|
| `MirPragmaPass` | RewritePass | one-shot | Walks `@pragma` registry; each registration fires per its trigger; pragma key cleared after firing. |
| `MirOptPass` | RewritePass | fixpoint | R1–R5: count-as-product, unused-var elision, hint introduction, negation pre-narrow, phase specialization. |
| `HirToMirLowering` | LoweringPass | source=HIR, target=MIR | Per-`HirRuleVariant` `@lowering` registration. |

### 3.3 IIR (multi-dialect)

#### 3.3.1 Structural dialects (LEAF, all-renderable)

##### `iir.cf` (control flow + scaffolding)

| Op | Renderer | Notes |
|---|---|---|
| `Block`, `BracedBlock`, `IndentBlock`, `BlankLine` | LEAF | Sequencing / scoping. |
| `Bind`, `Assign`, `IndexedAssign`, `StmtExpr` | LEAF | Statement-form bindings. |
| `If`, `IfReturn`, `IfReturnIfNot`, `IfContinue`, `IfContinueIfNot` | LEAF | Branches. |
| `ParallelFor`, `GridStrideLoop`, `IntersectIter` | LEAF | Loops. |
| `CartesianFlatLoop`, `Cartesian2DDecompose`, `CartesianNDecompose` | LEAF | Cartesian ops. |
| `LaneZeroGuard`, `OuterAnchor`, `Phase` | LEAF | Indent / phase. |
| `TiledBallotBlock` | LEAF | Tiled ballot write. |
| `WriteOutput`, `AddCount` | LEAF | Output emission. |
| `Comment` | LEAF | Comment line. |
| `RawString` | LEAF (transition-only) | See `compiler_redesign.md` §11. |
| `UserCode` | LEAF (Category J) | User-supplied expression text. |
| `VarRef` | LEAF (both stmt + expr modes) | Identifier reference. |

##### `iir.expr` (expression vocabulary)

| Op | Renderer | Notes |
|---|---|---|
| `BinOp` | LEAF | `<lhs> <op> <rhs>` |
| `UnaryOp` | LEAF | `<op><expr>` |
| `Parens` | LEAF | `(<expr>)` |
| `Ternary` | LEAF | `<c> ? <t> : <e>` |
| `IntLit` | LEAF | `<value><suffix>` (suffix for `1u`, `0ULL`, ...) |
| `MemberAccess`, `MemberCall` | LEAF | `<obj>.<member>`, `<obj>.<method>(...)` |
| `FuncCall` | LEAF | `<name>(...)` |
| `IndexExpr` | LEAF | `<arr>[<idx>]` |
| `CCast`, `StaticCast` | LEAF | `(T)x`, `static_cast<T>(x)` |
| `PostfixIncrement` | LEAF | `<expr>++` |

#### 3.3.2 Data-structure dialects

##### `relation.sorted_array`

| Op | Renderer | Notes |
|---|---|---|
| `SaRoot`, `SaPrefCoop`, `SaPrefSeq`, `SaHint` | LEAF | Handle construction / narrowing. |
| `SaDegree`, `SaValid`, `SaGetVal`, `SaGetValAt`, `SaGetValAtPos` | LEAF | Handle queries. |
| `SaIterators`, `SaChildRange` | LEAF | Iteration support. |
| `SaTiledCartesian2D` | LEAF | Tiled-Cartesian emission scaffold. |
| `DedupTryInsert` | **COMPOUND** | Decomposes to `BracedBlock + Bind + If + MemberCall` via `@rewrite`. |

##### `relation.d2l`

| Op | Renderer | Notes |
|---|---|---|
| `D2lSegmentLoop` | LEAF | HEAD/FULL segment-loop wrapping. |

#### 3.3.3 Parallelism dialects

##### `parallel.data` (existing, LEAF)

`ParallelFor`, `GridStrideLoop` (re-exports from iir.cf historically;
to be moved here in Phase B).

##### `parallel.block_group` (Phase 2C; new)

| Op | Renderer | Notes |
|---|---|---|
| `BgRootCJ` | LEAF | Block-group root iteration (per-warp work-balanced). |
| `BgWorkBalance` | LEAF | Per-warp work-balance setup. |

##### `parallel.atomic_ws` (Phase 2C; new)

| Op | Renderer | Notes |
|---|---|---|
| `WSCount` | LEAF | Per-thread `local_count++`. |
| `WSEmit` | LEAF | Atomic-WS emission. |
| `WSScope` | LEAF | Outer scope for WS body. |

| Pass | Kind | Fixpoint? | Notes |
|---|---|---|---|
| `MirToIirLowering` | LoweringPass | source=MIR, target=IIR | Per-MIR-op `@lowering` registrations. |
| `IirCanonicalizePass` | RewritePass | fixpoint | Decomposes COMPOUND ops to LEAF; verifies renderability after. |
| `CudaRenderPass` | RenderPass (terminal) | n/a | Walks the IIR tree; emits text via `@register_render`. |

## 4. Sub-dialect criteria (re-stated for reference)

A new sub-dialect is justified when **at least two** of the following
hold:

| Criterion | Example |
|---|---|
| New op vocabulary other dialects might consume | `parallel.atomic_ws.WSScope` referenced by IIR rendering AND potential future runners |
| Materially different codegen scaffolding | `parallel.block_group` requires per-warp work-balance setup outside the kernel body |
| Owns a registered codegen-target plugin / index plugin | `relation.d2l` ships its own `gen_root_handle` / `view_count` |
| Independent rewrite vocabulary | `parallel.tiled_cartesian` has tiled-emission rewrites touching multiple ops |
| Could plausibly be authored by an external party | future `relation.lsm⟨K⟩` |

A new sub-dialect is NOT justified for:

- A flag that just toggles two render branches (one rewrite is enough).
- A small per-rule helper op that only one lowering site uses (just an
  op in an existing dialect).
- A pure renderer variation (additional `@register_render` for the
  same op).

## 5. Extension points (where plugins hook in)

| Want to add | Where | How |
|---|---|---|
| **A new pragma** (e.g. `my_dedup_v2`) | `pragmas/<pragma>.py` (built-in) or external package | `@pragma(name=..., on=mir.X, value_type=T)` decorator. May ship a wrap op in a new `mir/pragma_ops/<pragma>.py`. May ship a sub-dialect if criteria met. |
| **A new index type** (e.g. LSM⟨K⟩) | `dialects/relation/<index>/` (built-in) or external package | New dialect package with ops, types, lowerings, render module. Register via `Compiler.register_dialect(...)` or entry point. |
| **A new parallelism strategy** | `dialects/parallel/<strategy>/` (built-in) or external package | New sub-dialect with ops + lowering. |
| **A new render target** (e.g. CPU/WASM) | `codegen/<target>/` (built-in) or external package as `srdatalog-target-<name>` | New render module; `@register_render(Op, target='<target>')` per IIR op. Plus a `<Target>RenderPass` instance and any target-specific PluginRegistry. |
| **A new HIR pass** | `dialects/hir/passes/<pass>.py` or external package | New `ProgramPass` instance; insert into pipeline at specified position. |
| **A new DSL frontend** | New entry point `srdatalog.frontend` | Module that produces `HirProgram` from some source format (text, JSON, ...). |

See [`phase_e_plugin_extensibility.md`](phase_e_plugin_extensibility.md)
for the worked example.

## 6. Discipline boundaries this topology enforces

The topology graph defines what's allowed where:

- `core/` knows only about `Op`, `Type`, `Dialect`, `Compiler`, the
  three Pass kinds, and the dispatchers. It does NOT import from any
  `dialects/`. Discipline test: `test_core_has_no_dialect_imports`.
- `dialects/<X>/` knows about `core/` and (for cross-dialect lowerings)
  about the dialects it produces ops for. It does NOT know about
  `codegen/`. Discipline test:
  `test_dialects_have_no_codegen_imports`.
- `codegen/<target>/` knows about `core/` and ALL of `dialects/`
  (because it renders IR ops). It does NOT mutate dialect state.
- `pragmas/<pragma>.py` knows about `core/` and ONLY the dialects
  whose ops it wraps / inserts. Discipline test:
  `test_pragma_module_imports_are_minimal`.

## 7. Versioning + back-compat

- The dialect interface (`Dialect` dataclass shape, registration
  protocol) is part of `core/`'s public API. Breaking changes to it
  require a major version bump.
- Specific dialect ops are part of the dialect's API. Breaking changes
  to op fields require a minor version bump on the owning plugin.
- Pragma names registered by built-in plugins are part of the language
  surface. Adding new built-in pragmas is additive; removing them is
  breaking.

## 8. Open extensions noted but out of scope

- **Cross-target rewrites** (rewrites parameterized by render target)
  — possible future addition; not in scope of this redesign.
- **Multi-target single-compile** (compile once, render to N targets)
  — possible future addition; the IIR' subset can be rendered by any
  target plugin in principle, but no production use case yet.
- **Incremental compilation** (re-lower only changed rules) — possible
  future addition; would extend `Compiler.run` with a memoization
  layer.

These are noted to clarify they are not load-bearing for the redesign
and not in any current phase.
