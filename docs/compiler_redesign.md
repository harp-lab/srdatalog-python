---
orphan: true
---

# Compiler Redesign — Spec

This is the architectural spine for the redesign committed to in May 2026.
It supersedes the implicit "imperative monolith with framework decoration"
shape of the current compiler. The companion docs are:

- [`ir_derivation_topology.md`](ir_derivation_topology.md) — the IR layer
  graph + per-dialect tables + sub-dialect criteria.
- [`code_discipline.md`](code_discipline.md) — code structure, forbidden
  and required patterns, per-PR Definition of Done, CI enforcement.
- [`phase_a_mir_onto_op.md`](phase_a_mir_onto_op.md) — MIR onto `Op`
  migration plan.
- [`phase_a3_remove_deprecated_bool_fields.md`](phase_a3_remove_deprecated_bool_fields.md) —
  remove deprecated bool fields on ExecutePipeline (Layer 3 prep).
- [`phase_b_lowering_dispatcher.md`](phase_b_lowering_dispatcher.md) —
  per-MIR-op `@lowering` migration plan.
- [`phase_c_pragma_materialization.md`](phase_c_pragma_materialization.md)
  — `@pragma` decorator + sub-dialect topology for built-in pragmas.
- [`phase_d_hir_passes.md`](phase_d_hir_passes.md) — per-HIR-pass migration
  to `ProgramPass`.
- [`phase_e_plugin_extensibility.md`](phase_e_plugin_extensibility.md) —
  entry-point discovery, `register_plugin`, what's "core" vs "default
  plugin set", how to ship an external pragma / dialect.
- [`phase_f5_declarative_pipeline.md`](phase_f5_declarative_pipeline.md) —
  declarative pipeline shim (last Foundation piece): reduces
  `compile_to_mir` + `compile_kernel_body` to `Compiler.run(prog,
  pipeline=DEFAULT_PIPELINE)`.
- [`phase_decomposition_redesign.md`](phase_decomposition_redesign.md) —
  unified Phase R/T/B2 spec: supersedes the piecemeal Phase A3 +
  Layer 3 cleanup work; frames every remaining migration PR against
  ONE load-bearing invariant (the ACID test — zero edits to existing
  source files for any new algorithm / pragma / index type /
  aggregation / backend target). The next layer above this doc.

## 1. The diagnosis

The current compiler has framework infrastructure on the renderer and
plugin axes, and an imperative monolith on the lowering axis:

- `_STMT_HANDLERS[type(op)]` dispatch — real, P1-shaped, additive.
- `PluginRegistry.resolve(index_type)` — real, P2-shaped, additive.
- `lower_scan_pipeline(...)` — 2500-LOC monolith with `if isinstance(...)`
  chains and `if ctx.<pragma>:` branches threaded through every code
  path.

The `@lowering(SA_DIALECT, mir.ExecutePipeline)` decoration is metadata
only — the registered Lowering instance is never queried by production;
production calls `lower_scan_pipeline(...)` directly. Adding a new
pragma today still means: edit `LoweringCtx`, edit `compile_kernel_body`
to thread it, edit `lower_scan_pipeline` branches.

This is the "framework on top of imperative monolith" anti-pattern.
The framework adds cost (extra abstraction surface, dual dispatch
mental model) without delivering its benefit (additive extensibility).

## 2. The target

**100 % dialect-pass-driven compiler.** Every transformation is a
registered `Pass`. `Compiler.run` drives them. Source code does
nothing imperative — it composes passes and supplies pragma-driven
rewrite rules.

```
Compiler.run(prog, pipeline=[Pass1, Pass2, ...])
   each Pass operates on registered (Lowering | Rewrite | ProgramPass) instances
   no imperative if/isinstance dispatch in user code
   pragma → typed-op-insertion → typed-op-lowering → typed-op-rendering
```

Pragma flow becomes:

```
Rule(pragmas={"dedup_hash": True})
  → HirRuleVariant(pragmas={"dedup_hash": True})
  → HirToMirLowering: produces ExecutePipeline(pragmas={"dedup_hash": True})
  → MirPragmaPass: walks @pragma registrations
       @pragma(name="dedup_hash") fires, wraps InsertInto in DedupGate,
       removes "dedup_hash" from op.pragmas
  → MirOptPass: R1–R5 fixpoint
  → MirToIirLowering: per-MIR-op @lowering, including DedupGate →
       sorted_array.DedupTryInsert
  → IirCanonicalizePass: COMPOUND ops decomposed to LEAF
  → CudaRenderPass: target-plugin-defined render
  → "C++ source text"
```

No `if ctx.dedup_hash` anywhere. The flag is materialized as a typed
op insertion at MIR time and consumed by the lowering for that op type.

## 3. The language is a configuration of the compiler

This is the load-bearing extensibility constraint. The core ships only
the framework. Every language feature — every pragma, every relation
type, every aggregation, every parallelism strategy, every DSL frontend
— is a plugin registration against the core, including the things we
currently consider "built-in".

```
srdatalog-core (this repo, ~1000 LOC):
  Op, Type, Dialect, Compiler
  Pass kinds: LoweringPass, RewritePass, ProgramPass
  PassDriver, dispatchers, registries
  Plugin discovery (entry points + explicit register_plugin)

srdatalog-lang (this repo, default plugin set):
  HIR / MIR / IIR dialects + their default ops
  Default rewrites for built-in pragmas (in pragmas/*.py)
  Default DSL frontend (Python Program(rules=[...]))

srdatalog-target-cuda (this repo, default target plugin):
  target.cuda dialect + render registry
  Per-IIR-op renderers
  PluginRegistry of index types

External package (some-third-party on PyPI):
  pip install some-third-party
  → entry-point auto-loads
  → registers a new @pragma + sub-dialect + lowering
  → end-users write Rule(pragmas={"that_pragma": True}) and it works
  → core never knew about it
```

This forces:

- **No hardcoded pragma names anywhere in core.** `Rule.pragmas` is an
  open dict (`tuple[tuple[str, Any], ...]` after Phase A freezing).
  Each pragma's `@pragma` registration declares which key it reads.
- **No central pragma list.** `MirPragmaPass` walks the registry of
  registered pragmas; their order is data (declared `before:`/`after:`
  topology).
- **No central dialect list in core.** Plugins register dialects with
  the running `Compiler`. Dependency declarations on Lowering/Rewrite
  (`consumes`/`produces`) let the driver validate that required dialects
  are registered.
- **Plugin discovery via entry points.** `pyproject.toml`:
  ```toml
  [project.entry-points."srdatalog.plugins"]
  cuda_target = "srdatalog_target_cuda:register"
  ```
  `Compiler.with_default_plugins()` walks entry points to load. For
  tests / custom configs, plugins register manually.

See [`phase_e_plugin_extensibility.md`](phase_e_plugin_extensibility.md)
for the worked-example.

## 4. The three Pass kinds

The framework recognizes exactly three kinds of `Pass`. Every step in
every `Compiler.run` pipeline is one of these; nothing else.

```python
class Pass(ABC):
    """Abstract base for every compile-pipeline step."""
    name: str
    consumes: tuple[str, ...]   # dialect names (for dependency validation)
    produces: tuple[str, ...]
    def apply(self, prog, compiler) -> NewProg: ...


class LoweringPass(Pass):
    """Cross-dialect, source-driven, per-op dispatch.

    Walks the input tree. For each op, looks up the registered
    @lowering(target=<this pass's target>, source=type(op)) and applies
    it. Result is in the target dialect (or a recursive call back into
    the dispatcher for child ops).

    Used for HIR → MIR, MIR → IIR.
    """
    target_dialect: Dialect


class RewritePass(Pass):
    """Intra-dialect, op-driven, per-op dispatch.

    Walks the input tree. For each op, looks up registered
    @rewrite(<this pass's dialect>, type(op)) and applies it. The
    result must be in the same dialect (or a known set of declared
    consumes/produces dialects).

    Optionally fixpoint: applies until a full pass produces no change.

    Used for MIR pragma materialization (one-shot), MIR opts (R1–R5,
    fixpoint), IIR canonicalization (fixpoint).
    """
    dialect: Dialect
    until_fixpoint: bool


class ProgramPass(Pass):
    """Whole-program. Used for HIR planning passes that operate on
    HirProgram as a unit (stratify, semi-naive variant generation,
    plan, index selection).

    HIR types are mutable planning records (not Op-subclassed), so
    they can't be walked by LoweringPass / RewritePass per-op
    dispatch. ProgramPass is the explicit escape hatch for
    whole-program transformations.

    Used ONLY for HIR planning. MIR / IIR transformations must be
    LoweringPass or RewritePass.
    """
    fn: Callable[[Any, Compiler], Any]
```

Pipelines are composed declaratively as data:

```python
DEFAULT_PIPELINE = [
    # HIR planning (ProgramPass)
    StratifyPass(), SplitPass(), SemiNaivePass(),
    PlanPass(), IndexSelectionPass(),
    TempRelSynthesisPass(), TempIndexRegistrationPass(),

    # HIR → MIR (LoweringPass)
    HirToMirLowering(),

    # MIR — pragma materialization (one-shot RewritePass)
    MirPragmaPass(),

    # MIR — opts (fixpoint RewritePass)
    MirOptPass(),

    # MIR → IIR (LoweringPass)
    MirToIirLowering(),

    # IIR canonicalization (fixpoint RewritePass)
    IirCanonicalizePass(),

    # Render (terminal target-plugin pass)
    CudaRenderPass(),
]

result = Compiler.with_default_plugins().run(prog, pipeline=DEFAULT_PIPELINE)
```

Users (and tests) can swap, reorder, insert custom passes. **The
pipeline is data**, not buried inside `compile_kernel_body` control flow.

## 5. `LowerCtx` — small (5 fields)

`LowerCtx` is the per-pass state carried into every `@lowering`
function. It is intentionally small. Anything not on this list is
either:

- Passed explicitly via lowering function arguments (lexical scope).
- Materialized as a typed op insertion (pragma flags).
- Looked up on the `Compiler` (registries).

```python
@dataclass(frozen=True)
class LowerCtx:
    """Per-pass state. 5 fields. Frozen — replace via dataclasses.replace.

    See docs/phase_b_lowering_dispatcher.md §3 for field-by-field design.
    """
    compiler: Compiler         # for cross-dialect dispatch + registries
    name_gen: NameGen          # name_gen.fresh(prefix) → str
    view_layout: ViewLayout    # per-relation view_var_names + slot bases
    plugin_registry: PluginRegistry  # per-Compiler plugin registry
    target: str                # render target (e.g. 'cuda') for cross-target hooks

    # The dispatch method (NOT a field — a method using `compiler`):
    def lower(self, op) -> Op:
        """Look up @lowering(target=this_pass, source=type(op)) and apply.
        Recursive entry; lowerings call this for child ops."""
        ...
```

What is NOT on `LowerCtx`:

- ❌ `dedup_hash`, `tiled_cartesian`, `bg_enabled`, `ws_enabled`, ...
  (pragma flags) — these are op insertions; once materialized, they
  vanish.
- ❌ `is_counting`, `inside_cartesian` — phase / scope flags, become
  op-level (`Phase(C, body)` vs `Phase(M, body)` per the spec
  §6, lexical via explicit `Scope` parameter to lowerings that need it).
- ❌ `output_var`, `output_var_overrides` — output naming becomes a
  property of the `WriteOutput` op or its containing scope.
- ❌ `bound_vars`, `cartesian_bound_vars`, `handle_vars`,
  `pre_narrow_infos` — lexical scope, passed explicitly via
  `Scope` parameter.

This decomposition is the point of the redesign. The current
`LoweringCtx` (~25 fields) IS the imperative monolith — every flag is
a branch trigger. The new `LowerCtx` removes the branch triggers; the
removed information is either materialized as ops or passed explicitly.

## 6. Pragmas are partial evaluation

> **NOTE — supersedes the string-keyed sketch below.**
> The canonical spec is now [`pragma_as_typed_object.md`](pragma_as_typed_object.md).
> Pragmas are first-class **typed compile-time objects** (Python class
> instances), not string keys. The `@pragma(name=..., value_type=...)`
> decorator below has been replaced by `@pragma_handler(PragmaCls, on=MirOpCls)`.
> Read `pragma_as_typed_object.md` first; the example below is kept here
> for the conceptual framing only.

A pragma is a known-at-compile-time fact about a rule (e.g. an
instance of `DedupHash(hash_size=2**20)`). The compiler
partially-evaluates the rule's lowering for that fact, producing
specialized code. The "specialization" is the substitution of one
MIR op (generic) with a wrap op + lowering chain (specialized). The
fact itself is a typed `Pragma` subclass instance — see
[`pragma_as_typed_object.md`](pragma_as_typed_object.md) for the
full contract.

```python
# The conceptual contract (typed-object form — see pragma_as_typed_object.md):

@pragma_handler(DedupHash, on=mir.ExecutePipeline,
                before=(CountAsProduct,))
def materialize_dedup_hash(op, pragma: DedupHash, ctx) -> Op | None:
    """Per docs/compiler_redesign.md §6 (partial evaluation): when
    op.pragmas contains a DedupHash instance, specialize the IR by
    inserting DedupGate wrap ops, then remove the DedupHash from
    op.pragmas. Downstream passes never see DedupHash — they only see
    the typed wrap ops.
    """
    new_pipeline = tuple(
        mir.DedupGate(inner=child, hash_size=pragma.hash_size)
        if isinstance(child, mir.InsertInto)
        else child
        for child in op.pipeline
    )
    new_pragmas = tuple(p for p in op.pragmas if not isinstance(p, DedupHash))
    return op.replace(pipeline=new_pipeline, pragmas=new_pragmas)
```

`MirPragmaPass` walks the registry of `@pragma` registrations,
topologically sorts them by `before:`/`after:`, and applies each in
order. After all pragmas have run, `op.pragmas` is empty for every
op in the tree. Any remaining pragma key is a registration error
(a pragma name appeared in DSL but no `@pragma` claims it) — caught
by a discipline test (see [`code_discipline.md`](code_discipline.md)).

External plugins use the same `@pragma` decorator. The core knows
about no specific pragmas — even built-in ones (dedup_hash,
block_group, work_stealing, tiled_cartesian) live in `pragmas/*.py`
as plugin modules.

See [`phase_c_pragma_materialization.md`](phase_c_pragma_materialization.md)
for the per-pragma design.

## 7. Sub-dialect criteria

A new sub-dialect is justified when **at least two** of the following
hold:

| Criterion | Example |
|---|---|
| New op vocabulary that other dialects might consume | `parallel.atomic_ws.WSScope` referenced by IIR rendering AND by potential future runners |
| Materially different codegen scaffolding | `parallel.block_group` requires per-warp work-balance setup outside the kernel body |
| Owns a registered codegen-target plugin / index plugin | `relation.d2l` ships `gen_root_handle` / `view_count` |
| Independent rewrite vocabulary | `parallel.tiled_cartesian` has tiled-emission rewrites touching multiple ops |
| Could plausibly be authored by an external party | future `relation.lsm⟨K⟩` is per-spec a separate dialect |

A new sub-dialect is **not** justified for:

- A flag that just toggles two render branches (one rewrite is enough)
- A small per-rule helper op that only one lowering site uses (just
  an op in an existing dialect)
- A pure renderer variation (additional `@register_render` for the
  same op)

Applying these to the four built-in pragmas:

| Pragma | Sub-dialect? | Rationale |
|---|---|---|
| `dedup_hash` | **No** — keep `DedupTryInsert` in `relation.sorted_array` | One COMPOUND op. External re-use unlikely until a second relation dialect needs dedup; refactor to `relation.dedup` then. |
| `block_group` | **Yes — `parallel.block_group`** | Multiple ops (BgRootCJ, BgWorkBalance), distinct codegen scaffolding, runner integration. |
| `work_stealing` | **Yes — `parallel.atomic_ws`** | Multiple ops + WCOJTask runner integration. Pragma + scheduling impact. |
| `tiled_cartesian` | **No** — `SaTiledCartesian2D` stays in `relation.sorted_array` | Tightly coupled to sorted-array index access. Factor out only when a second data-structure dialect needs it. |

So we end up with **two new sub-dialects** for built-in pragmas:
`parallel.block_group`, `parallel.atomic_ws`. Not four.

See [`ir_derivation_topology.md`](ir_derivation_topology.md) for the
full dialect graph and per-dialect tables.

## 8. Migration policy

Big-bang risks byte-equivalence catastrophically. Per-MIR-op migration
with a feature flag is the safe path.

```python
# In compile_kernel_body, during migration:
USE_DECLARATIVE: frozenset[type] = frozenset({mir.Scan, mir.InsertInto})
# Ratchets up over time. CI enforces monotonic growth.

def lower_via_dispatcher(op, ctx):
    if type(op) in USE_DECLARATIVE:
        return ctx.lower(op)              # framework path
    return _legacy_imperative_lower(op, ctx)  # legacy fallback
```

Each PR migrates exactly **one** MIR op type:

1. New file `dialects/relation/sorted_array/lowerings/lower_mir_<op>.py`
   contains exactly one `@lowering(target=IIR, source=mir.<Op>)`.
2. Single-line addition to `USE_DECLARATIVE` set.
3. Per-op byte-equivalence test (golden against legacy fallback).
4. Full suite stays green.

After all 60 MIR ops are migrated: delete `_legacy_imperative_lower`,
`lower_scan_pipeline`, `LoweringCtx`, `USE_DECLARATIVE` flag.

This is the only way to keep the test suite green throughout. See
[`phase_b_lowering_dispatcher.md`](phase_b_lowering_dispatcher.md) for
the per-op migration order and per-op acceptance gate.

## 9. What this redesign does NOT promise

- **It does not promise a faster compiler.** The point is extensibility,
  not throughput. The dispatch overhead of `ctx.lower(op)` is one dict
  lookup per op — irrelevant compared to file I/O at the codegen edge.
- **It does not promise byte-equivalence in every case.** The
  per-MIR-op migration aims to preserve byte-equivalence by reproducing
  the legacy text exactly, but some emission quirks may not survive
  decomposition. Where they don't, golden updates with reviewer sign-off
  are explicit.
- **It does not deliver a working second target.** It enables one (a
  CPU/WASM target plugin can ship as a separate package), but actually
  building one is post-redesign work.
- **It does not retire any feature.** Every pragma, every relation
  dialect, every lowering quirk currently in the codebase has an
  equivalent in the new shape; the migration plans demonstrate this
  per-feature.

## 10. Phase order + parallelism

| Phase | Scope | Agents | PRs |
|---|---|---|---|
| **0: Design docs** | This package | 1 | 1 (this PR) |
| **1: Foundation (sequential)** | Pass kinds, MIR onto Op, LowerCtx, plugin discovery, declarative pipeline shim | 1 | 5 |
| **2A: Per-MIR-op lowering** | Each PR migrates one MIR op type | up to 5 in parallel | ~10 |
| **2B: Per-HIR-pass migration** | Each PR converts one HIR pass to ProgramPass | up to 3 in parallel | 7 |
| **2C: Per-pragma materialization** | Each PR adds one pragma + (if criteria met) sub-dialect | up to 4 in parallel | 4 |
| **2D: Plugin extensibility validation** | Re-ship one or two built-in features (e.g. an aggregation, a semiring) as plugin registrations to validate the extension model | 2 | 2 |
| **3: Cleanup (sequential)** | Delete monolith, legacy LoweringCtx, all `if ctx.X` branches | 1 | 1 |

**Approval gates** at every phase boundary. The user signs off on a
phase's design before its execution starts, and on its execution before
the next phase begins.

## 11. Anti-patterns this redesign forbids

The redesign is only worth doing if these are made structurally
impossible. See [`code_discipline.md`](code_discipline.md) for the full
list and CI enforcement.

| Anti-pattern | Forbidden because |
|---|---|
| `if ctx.<pragma>:` outside the pragma's own materialization rewrite | Pragma flags are partial-eval triggers, not branch keys. |
| Direct call to `lower_scan_pipeline(...)` (or any monolith) from production | Production goes through `Compiler.run(pipeline=...)`. No bypass. |
| `isinstance(op, mir.X)` dispatch chain | Cross-dialect dispatch goes through `@lowering` registry. |
| Module-import side effects in dialect modules | Import = no mutation. Registration = explicit. |
| Hardcoded pragma name strings in core | Pragmas are plugin-discovered; core has no list. |
| New `if`-branch in an existing lowering function | Every new lowering is a NEW file. Editing an existing branch is the "moving code around" anti-pattern. |
| Registration with no production-path consumer | Dead infra. Discipline test requires every `@lowering` / `@rewrite` / `@pragma` to be exercised. |

## 12. Approval

This document represents an architectural commitment. Sign-off includes:

- [ ] The Pass kinds in §4 are the only Pass kinds.
- [ ] `LowerCtx` is fixed at 5 fields per §5 (extensions require a
  doc amendment).
- [ ] Pragmas are partial evaluation per §6; pragma flags vanish after
  `MirPragmaPass`.
- [ ] Sub-dialect criteria from §7 are the only justification for
  factoring; no more sub-dialects than necessary.
- [ ] Migration is per-MIR-op with monotonic `USE_DECLARATIVE` flag
  per §8.
- [ ] Phase order from §10 is respected; approval gate at each phase.
- [ ] Anti-patterns from §11 are CI-enforced; PR fails if violated.

Reviewer initials and date below.
