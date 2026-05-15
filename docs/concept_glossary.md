---
orphan: true
---

# Concept Glossary

Precise definitions of the four load-bearing concepts in the
redesigned compiler: **IR**, **Dialect**, **Pass**, **Plugin**.
Plus how **Pragma** and **Op** fit alongside.

This doc exists because conflating these concepts is the original
sin of the pre-redesign codebase: the imperative monolith
(`lower_scan_pipeline`) smushed *data shape*, *execution*, and
*registration* together into one 2500-LOC function. The clean
separation here is what every Phase A/B/C/D/E PR enforces.

Subagent prompts and PR reviewers reference this doc.

## 1. One-sentence definitions

| Concept | What it is | Lifetime |
|---|---|---|
| **IR** (Intermediate Representation) | A program's representation at a specific compilation stage. HIR / MIR / IIR are the three; CUDA C++ text is the terminal "stage". | Per-stage during one compile. |
| **Dialect** | A named vocabulary of `Op` types + `Type` types + their per-dialect lowerings + rewrites + verifier, **scoped to one IR layer**. | Compiler-lifetime (registered at startup; the `Dialect` instance is frozen). |
| **Pass** | An executable transformation step in the compile pipeline. Three concrete kinds: `LoweringPass`, `RewritePass`, `ProgramPass`. | Pipeline-execution time (`Compiler.run` calls `pass.apply()`). |
| **Plugin** | A package (in-tree or external) that registers dialects, pragmas, render handlers, and pass instances with a `Compiler` at bootstrap. | Compiler-bootstrap time (entry-point auto-load OR explicit `Compiler.register_plugin`). |

## 2. Containment + dependency graph

```
PLUGIN  (e.g. srdatalog/plugins/default_lang or external my-srdatalog-jaccard)
  │
  │ at bootstrap, register(compiler) calls:
  │   compiler.register_dialect(SomeDialect)         ← contributes a DIALECT
  │   @pragma_handler(...)                             ← pragma materialization
  │   @register_render(Op, target='cuda')              ← target rendering
  │   (and provides Pass instances for DEFAULT_PIPELINE)
  ▼
COMPILER  (holds the registries)
  │
  ├── dialects: dict[str, Dialect]
  │       │
  │       └── Dialect = ⟨name, types, ops, lowerings, rewrites, verifier⟩
  │           │
  │           └── ops/types appear inside an IR. A dialect is a
  │               SUBSET of an IR's vocabulary.
  │
  ├── pragma registrations  (typed Pragma subclasses + handlers)
  ├── render registries     (per target — @register_render)
  └── pipeline: list[Pass]
        │
        └── Compiler.run(prog, pipeline=[Pass1, Pass2, ...])
             calls pass.apply() in order.

IR  (HIR / MIR / IIR — the data flowing between Pass instances)
  │
  └── Composed of Op instances from one or more DIALECTS.
      e.g. IIR contains:
        iir.cf.Block        + sorted_array.SaRoot
        + parallel.data.GridStrideLoop
      = three dialects, all coexisting in the IIR layer.
```

## 3. Concrete examples (from this codebase)

| Concept | Examples |
|---|---|
| **IR** | HIR (planning records: `HirProgram`, `HirStratum`, ...); MIR (frozen Op tree: `Program`, `ExecutePipeline`, `Scan`, `ColumnJoin`, ...); IIR (multi-dialect frozen Op tree). CUDA C++ text is the terminal "stage" produced by `CudaRenderPass`. |
| **Dialect** | `iir.cf` (control flow within IIR); `iir.expr` (expressions within IIR); `relation.sorted_array` (data structure within IIR); `relation.d2l` (alternate data structure within IIR); `mir` (the only dialect in MIR); `parallel.data`, `parallel.block_group`, `parallel.atomic_ws` (parallelism dialects within IIR). |
| **Pass** | `StratifyPass()`, `SemiNaivePass()` (HIR planning — `ProgramPass`); `HirToMirLowering()` (`LoweringPass`); `MirPragmaPass()` (`RewritePass`, one-shot); `MirOptPass()` (`RewritePass`, fixpoint); `MirToIirLowering()` (`LoweringPass`); `IirCanonicalizePass()` (`RewritePass`, fixpoint); `CudaRenderPass()` (terminal). Each is an instance in `DEFAULT_PIPELINE`. |
| **Plugin** | `srdatalog/plugins/default_lang/` (registers HIR/MIR/IIR dialects + built-in pragmas); `srdatalog/plugins/cuda_target/` (registers CUDA renderers + index plugins); hypothetical `my-srdatalog-jaccard` (PyPI package — registers a `Jaccard` `Pragma` + its dialect + its CUDA renderer). |

## 4. The four orthogonal axes meeting

| Cardinality | Example |
|---|---|
| One **plugin** can contribute to MULTIPLE **dialects** | `default_lang` plugin registers `iir.cf`, `iir.expr`, `relation.sorted_array`, ... |
| One **dialect** lives in EXACTLY ONE **IR** | `iir.cf` lives in IIR only; `mir` lives in MIR only |
| One **IR** contains MULTIPLE **dialects** | IIR contains `iir.cf` + `iir.expr` + `relation.sorted_array` + `parallel.*` (and more) |
| One **pass** consumes / produces ops from MULTIPLE **dialects** | `MirToIirLowering` consumes `mir` ops, produces ops in `iir.cf` + `iir.expr` + `sorted_array` + `parallel.*` |
| One **pass** belongs to NO dialect | `MirPragmaPass`, `MirToIirLowering` are top-level pipeline entries; they're NOT *owned* by any dialect even though they consume/produce dialect ops |
| Per-dialect **transformations** (`@lowering`, `@rewrite`) ARE owned by a dialect | `@lowering(target=IIR_CF, source=mir.Scan)` registers ON the `sorted_array` dialect (per `phase_b_lowering_dispatcher.md` §4) — the dialect is the registration site, even though the lowering CONSUMES `mir` and PRODUCES `iir.cf` ops |

## 5. Lifetime and mutability

| Concept | Lifetime | Mutability |
|---|---|---|
| **IR data** | One compile pass | Immutable (frozen Op tree); transformations create new instances via `dataclasses.replace` |
| **Dialect instance** | Compiler-lifetime | Frozen at construction; ops list is fixed; per-dialect registrations (`@lowering`/`@rewrite`/`@verifier`) are added by decorators that target the dialect, but the `Dialect(...)` instance itself doesn't mutate after registration |
| **Pass instance** | Pipeline-instance lifetime | Frozen dataclass — describes WHAT to do; `apply()` is a pure-ish method (transforms `prog` into new `prog`) |
| **Plugin** | Bootstrap (one call) | Pure side-effect: `register(compiler)` — no runtime state on the plugin itself |

## 6. The clean version

- **An IR is a *stage*.** It has a data shape (op tree, planning record). It contains nothing executable.
- **A dialect is a *vocabulary slice within ONE IR stage*.** It declares which ops/types belong to it. It is the registration site for per-dialect transformations.
- **A pass is a *step in the pipeline*.** It takes one IR shape and produces another (or transforms within the same shape). It's the executable side. It dispatches via the registries that dialects contributed.
- **A plugin is a *packaging unit*.** It contributes content (dialects, pragmas, render handlers, pass instances) to the compiler at bootstrap. It carries no runtime state itself.

## 7. Where Pragma fits

Pragma is a NEW concept introduced by `pragma_as_typed_object.md`. It's adjacent to, not the same as, the four above:

- **Pragma is to MIR ops as Op is to IR.** Pure typed data, attached to ops.
- A `Pragma` instance lives in `mir.ExecutePipeline.pragmas: tuple[Pragma, ...]`.
- Each `Pragma` subclass has a `@pragma_handler(PragmaCls, on=MirOpCls)` registration (provided by a plugin).
- `MirPragmaPass` (a `RewritePass`) consumes pragmas: walks the registry, applies each pragma's handler in topo-sorted order, removes the pragma instance from `op.pragmas`.
- Pragmas don't survive past `MirPragmaPass`. Their effect is a typed-op insertion into MIR.

So: **`Pragma` = data; `@pragma_handler` = registration (per-dialect convention); `MirPragmaPass` = the pass that drives them; plugin = ships the pragma class + handler.**

## 8. Where Op fits

`Op` (and `Type`) are the framework BASES that dialects subclass. They live in `core/op.py`.

- An `Op` is a `@dataclass(frozen=True, slots=True)` marker — pure data.
- A `Dialect` declares which `Op` subclasses belong to its vocabulary.
- An `IR` is composed of `Op` instances (and other plain Python data — strings, ints, tuples).
- The framework's tree-walker (`_walk(prog)` in `core/passes.py`) recurses through any field that's an `Op` or list/tuple of `Op`.

So **`Op` is the data substrate; `Dialect` is the vocabulary that groups Op subclasses; `IR` is the composition.**

## 9. The discipline rules each concept enforces

Each concept has clear ownership rules (pinned by discipline tests in `code_discipline.md`):

| Concept | Discipline |
|---|---|
| **IR data** | D2: immutable (`frozen=True` on every Op subclass); mutation requires `dataclasses.replace` (or transition shim per D18). |
| **Dialect** | D6: `core/` has no imports from `dialects/`; D8: `core/` has no imports of concrete `Pragma` subclasses. Dialects register themselves; core doesn't know specific names. |
| **Pass** | D17: `LoweringPass.apply` uses table-build dispatch (no isinstance bypass); R8: every Pass in `DEFAULT_PIPELINE` has `consumes` satisfied at its position. |
| **Plugin** | D6 + D11: `core/` doesn't import from `plugins/`; the file set under `core/` is pinned. |

## 10. Practical implications for adding new things

| Want to add | Conceptual move | Concrete steps |
|---|---|---|
| New index type (e.g., LSM⟨K⟩) | New **dialect** | `dialects/relation/lsm/{ops,types,print,__init__}.py` + plugin registration. No edits to existing dialects. |
| New parallelism strategy (e.g., SIMD) | New **dialect** + new **pragma** | `dialects/parallel/simd/` + `pragmas/simd.py` (typed `SIMD(width=...)` pragma) + `@pragma_handler` for materialization into MIR wrap op. |
| New compile target (e.g., WASM) | New **plugin** + per-target render module | New PyPI package `srdatalog-target-wasm`. Registers `target.wasm` codegen module + `@register_render(Op, target='wasm')` per IIR op. No edits to IR / dialects / passes. |
| New compiler optimization (e.g., CSE for IIR) | New **pass** + per-op `@rewrite` registrations | Define `IirCseRewritePass()` (`RewritePass`); insert into `DEFAULT_PIPELINE`; add `@rewrite` registrations on relevant dialects. Doesn't touch existing passes. |
| New source-language frontend (e.g., SQL) | New **plugin** | Plugin ships a SQL parser that produces `HirProgram`. No edits to anything past HIR. |

## 11. The conflation failure mode this redesign reverses

Pre-redesign, the distinction was lost:

- `lower_scan_pipeline` (the imperative monolith) was framed as "the MIR→IIR lowering", but was actually:
  - Per-MIR-op DISPATCH (which is pass-level) →
  - Mixed with IIR Op CONSTRUCTION (which is data-shape level) →
  - Mixed with PRAGMA flag READS (`if ctx.dedup_hash:`, ...) (which should be MIR-level rewrite consumption) →
  - Mixed with PARALLELISM STRATEGY decisions (`if ctx.bg_enabled:`, ...) (which should be sub-dialect selection at MIR rewrite time).

Four concepts in one function. The redesign separates them:

- DISPATCH → `LoweringPass.apply` (a pass)
- DATA-SHAPE CONSTRUCTION → `@lowering` registrations on dialects (per-dialect transformation)
- PRAGMA CONSUMPTION → `@pragma_handler` registrations + `MirPragmaPass` (pragma materialization at MIR time, before lowering)
- PARALLELISM STRATEGY → MIR wrap ops (`BlockGroupRoot`, `WSScope`) + their per-sub-dialect lowerings (`parallel.block_group`, `parallel.atomic_ws`)

When in doubt, ask: **am I describing data shape (IR/dialect), execution (pass), or packaging (plugin)?** That's the test for whether a piece of code lives in the right place.

## 13. Boundary semantics — soft vs hard, multi-dialect-in-one-stage, topology checks

### 13.1 The two kinds of boundary

| Boundary | Within an IR stage | Between IR stages |
|---|---|---|
| **Nature** | **SOFT** — multiple dialects coexist freely in one op tree | **HARD** — explicit `LoweringPass` mediates the transition |
| **Example** | An IIR tree contains `iir.cf.Block` (parent) → `parallel.block_group.BgRootCJ` (child) → `sorted_array.SaRoot` (grandchild) — three dialects, one tree | MIR ops never appear inside an IIR tree; `MirToIirLowering` walks MIR and produces IIR |
| **What enforces it** | `verify_renderability` (every op in the tree has a renderer) + `@lowering` consumes/produces declarations | `Compiler.run` pre-flight: each Pass's `consumes` satisfied by earlier `produces` |

**The mental model:** dialects within an IR are like neighboring rooms in a house — same building, walls are decorative grouping, ops freely cross between rooms. **IR stages are like floors** — you can only move between floors via the staircase (a Pass).

### 13.2 Why multi-dialect-in-one-stage is the COMMON case

The pragma materialization pattern produces multi-dialect IIR by design. A rule with three pragmas (`DedupHash` + `BlockGroup` + `WorkStealing`) generates IIR that mixes ops from `sorted_array` + `parallel.block_group` + `parallel.atomic_ws` + `iir.cf` + `iir.expr` — five dialects, one tree, one IR stage.

End-to-end walkthrough:

```
Stage 1 — HIR (planning records)
   HirRuleVariant.pragmas = (DedupHash(), BlockGroup(...), WorkStealing())

Stage 2 — HIR → MIR (LoweringPass)
   ExecutePipeline.pragmas = (DedupHash(), BlockGroup(...), WorkStealing())
   ExecutePipeline.pipeline = [Scan, ColumnJoin, ..., InsertInto]

Stage 3 — MirPragmaPass (RewritePass within MIR)
   Topo-sorted handlers fire:
     dedup_hash:    each InsertInto → DedupGate(InsertInto)
     block_group:   root ColumnJoin → BlockGroupRoot(ColumnJoin)
     work_stealing: each (Dedup-wrapped) InsertInto → WSScope(...)
   Result: pragmas=() ; pipeline contains typed wrap ops (all in mir
   dialect)

Stage 4 — MIR → IIR (LoweringPass; per-op @lowering dispatch)
   Each MIR op lowers to ops that may belong to multiple IIR dialects:
     mir.Scan            → iir.cf + sorted_array ops
     mir.BlockGroupRoot  → parallel.block_group + iir.cf ops
     mir.WSScope         → parallel.atomic_ws + iir.cf ops
     mir.DedupGate       → sorted_array.DedupTryInsert (COMPOUND)
     mir.InsertInto      → iir.cf + sorted_array ops
   Result: IIR tree with FIVE dialects coexisting

Stage 5 — IirCanonicalizePass (RewritePass within IIR, fixpoint)
   COMPOUND ops decompose to LEAFs. Tree still mixes dialects.

Stage 6 — verify_renderability(target='cuda')   ← THE CLOSURE CHECK
   For every op type reachable in tree, assert a registered renderer.
   Multi-dialect tree, multi-plugin contributions — must all close.

Stage 7 — CudaRenderPass (terminal)
   Walks tree; @register_render dispatches per (op, target).
```

**The IIR tree mixes 5 dialects throughout stages 4-7. That's normal.**

### 13.3 The four topology checks

Each operates at a different scope. Together they enforce both soft (within-stage) and hard (between-stage) boundaries.

```
┌──────────────────────────────────────────────────────────────┐
│ #1 — Plugin loading (compiler bootstrap)                     │
│   Kahn sort plugins by `provides` / `requires` attrs         │
│   Cycle → PluginCycleError                                   │
│   Conflict → PluginConflictError (unless `replaces=()`)      │
│   When: Compiler.with_default_plugins()                      │
│   Scope: BOOTSTRAP-LEVEL — registration order                │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ #2 — Pipeline ordering (pre-flight, per Compiler.run)        │
│   For each Pass, set(consumes) ⊆ available_dialects          │
│   available = registered ∪ produces of earlier passes        │
│   Mismatch → PassOrderingError (NEVER reaches apply())       │
│   When: Compiler.run(prog, pipeline=[...])                   │
│   Scope: PIPELINE-LEVEL — ordered list of Passes             │
│   Polices the HARD boundary between IR stages                │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ #3 — Pragma materialization (within MirPragmaPass)           │
│   Kahn sort @pragma_handler regs by before / after           │
│   Cycle → PragmaOrderingError                                │
│   Pragma key survives pass → UnconsumedPragmaError           │
│   When: MirPragmaPass.apply()                                │
│   Scope: WITHIN ONE PASS — handler invocation order          │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ #4 — Renderability closure (after IIR fixpoint)              │
│   For each op type in IIR tree post-rewrite:                 │
│     ∃ @register_render(op_type, target=...)                  │
│   No → UnrenderableOpError                                   │
│   When: end of IirCanonicalizePass / start of CudaRenderPass │
│   Scope: TREE-LEVEL — multi-dialect closure                  │
│   Polices the SOFT boundary within an IR stage               │
└──────────────────────────────────────────────────────────────┘
```

### 13.4 Where each check fires for the multi-pragma case

| Stage | Topology check | Catches |
|---|---|---|
| Compiler bootstrap | #1 — Plugin loading | `parallel.block_group` plugin requires `iir.cf` plugin and the latter isn't loaded → bootstrap fails. |
| `Compiler.run` pre-flight | #2 — Pipeline ordering | User puts `MirToIirLowering` BEFORE `MirPragmaPass` in custom pipeline → fail before any work. |
| `MirPragmaPass.apply` | #3 — Pragma topo-sort | `dedup_hash` declares `before=(WorkStealing,)` and `WorkStealing` declares `before=(DedupHash,)` → fail at MirPragmaPass start. |
| `MirPragmaPass.apply` end | #3 — Unconsumed pragma | DSL produced a pragma name no `@pragma_handler` claims → loud failure with "did you mean" hint. |
| `MirToIirLowering.apply` | Per-op `@lowering` lookup | `mir.BlockGroupRoot` was inserted but no plugin registered `@lowering(target=IIR, source=BlockGroupRoot)` → `LoweringMissingError`. |
| `IirCanonicalizePass.apply` | Per-op `@rewrite` lookup | A COMPOUND op (`sorted_array.DedupTryInsert`) without a registered decomposing `@rewrite` survives the fixpoint pass — caught by check #4. |
| Post-fixpoint | #4 — Renderability closure | After fixpoint, any op type without `@register_render(target='cuda')` → `UnrenderableOpError`. **This is where "the dialect's render contribution is missing" gets caught.** |
| `CudaRenderPass` | Per-op render dispatch | Should never fail at this point (#4 is the gate); `KeyError` is the loud-fallback. |

### 13.5 Where dialect boundaries actually matter

Within an IR stage, dialect boundaries are visible only in:

1. **`@lowering` / `@rewrite` registration sites.** A `@lowering(target=IIR_CF, source=mir.X)` registers ON the dialect that owns it (typically `sorted_array`, where most MIR-to-IIR lowerings live). The dialect identity matters for *who is responsible*, not for *what the lowering can produce*. A lowering on `sorted_array` can produce ops in `iir.cf` + `parallel.block_group` + `sorted_array` — it's not constrained to its own dialect.

2. **Plugin packaging.** `pip install my-srdatalog-jaccard` gets you one plugin that registers ONE dialect (`relation.jaccard`). The package's bounding box is the dialect. Once loaded, the dialect's ops mix freely with everyone else's in the IIR tree.

3. **Discipline rules.** D6 ("core has no imports from dialects") enforces that the framework infra doesn't know about specific dialects. But within an IR tree, there's no such restriction.

### 13.6 The closure invariant (D-rule R3 + the renderability check)

The load-bearing invariant for multi-dialect IIR:

> **For every op type reachable in the post-fixpoint IIR tree, EITHER a `@register_render(op_type, target=T)` is registered OR a `@rewrite(dialect, op_type)` is registered (and the rewrite's output is closed under the same condition).**

`verify_renderability` enforces it; `UnrenderableOpError` is the loud failure. This is what makes multi-dialect-in-one-stage SAFE — the closure check guarantees that whatever combination of dialects ends up in the tree, the codegen has a path for every op.

### 13.7 TL;DR

Multi-pragma → multi-dialect in one IR stage is the design's COMMON case, not an edge case. The dialect boundary within an IR is intentionally soft — it's a NAMESPACE distinction (who owns each op type), not a containment distinction (what tree shapes are valid).

The four topology checks operate at four different scopes:
- **#1 Plugin loading** — bootstrap, dialect-name level.
- **#2 Pipeline ordering** — per-run, dialect-name level. Polices HARD boundary (IR stages).
- **#3 Pragma materialization** — within `MirPragmaPass`, pragma-class level.
- **#4 Renderability closure** — within `verify_renderability`, op-type level. Polices SOFT boundary (multi-dialect tree).

The IR-stage boundary (HARD) is policed by check #2; the within-stage multi-dialect boundary (SOFT) is policed by check #4. Together they cover both axes.

## 14. References

- `docs/compiler_redesign.md` — architectural spine; defines the three Pass kinds.
- `docs/ir_derivation_topology.md` — IR layer + dialect graph; per-dialect tables.
- `docs/phase_e_plugin_extensibility.md` — plugin discovery + packaging mechanics.
- `docs/pragma_as_typed_object.md` — typed Pragma model.
- `docs/phase_zero_prerequisites.md` — locked design contracts (Layer 1+ baseline).
- `docs/code_discipline.md` — D-rules + R-rules with discipline tests.
