---
orphan: true
---

# Phase C — Pragma materialization (`@pragma_handler` + sub-dialects)

The phase that deletes `if ctx.<pragma>:` branches. Each pragma
becomes a typed op insertion at MIR time; the lowering for the
inserted op handles the codegen specialization. Pragma instances
vanish after `MirPragmaPass`; downstream code never reads them.

> **AMENDMENT — pragmas are typed objects, not strings.** This doc
> originally described `@pragma(name="dedup_hash", value_type=bool)`
> with a `pragmas: tuple[tuple[str, Any], ...]` field. That sketch
> has been superseded by [`pragma_as_typed_object.md`](pragma_as_typed_object.md):
> pragmas are `Pragma` subclass instances (typed Python classes);
> registration is `@pragma_handler(PragmaCls, on=MirOpCls)`; the MIR
> field is `pragmas: tuple[Pragma, ...]`. **Read `pragma_as_typed_object.md`
> first.** Sections below have been kept for the per-pragma sub-dialect
> design, ordering decisions, and Wave 2C plan; the API specifics use
> the typed-object form.

Companion to [`compiler_redesign.md`](compiler_redesign.md) §6
(partial-eval framing) and §7 (sub-dialect criteria),
[`pragma_as_typed_object.md`](pragma_as_typed_object.md) (the typed-
object contract), and [`ir_derivation_topology.md`](ir_derivation_topology.md)
§3.2.2 (pragma wrap ops).

## 1. Goal

After Phase C:

- Every built-in pragma (`dedup_hash`, `block_group`, `work_stealing`,
  `tiled_cartesian`, `count`, `fanout`) lives in a single file under
  `pragmas/<pragma>.py`.
- Each pragma module exports a `@pragma`-decorated function that
  reads `op.pragmas[<name>]` and inserts the appropriate wrap op
  into the MIR tree.
- `MirPragmaPass` is in `DEFAULT_PIPELINE` after `HirToMirLowering`
  and before `MirOptPass` / `MirToIirLowering`.
- After `MirPragmaPass` runs, `op.pragmas` is empty for every op in
  the tree. Discipline test enforces this.
- Two new sub-dialects exist: `parallel.block_group`,
  `parallel.atomic_ws`. Each ships its own ops + lowerings + render.
- `LoweringCtx` has no pragma fields. `lower_scan_pipeline` (about
  to be deleted in Layer 3) has no `if ctx.<pragma>:` branches.

Phase C does NOT yet:

- Delete the legacy `lower_scan_pipeline` (Layer 3).
- Migrate HIR passes (Phase D).

## 2. The `@pragma` decorator contract

```python
def pragma(
    *,
    name: str,
    on: type,                                  # MIR op type the pragma reads from
    value_type: type = bool,                   # expected type of pragmas[name]
    before: tuple[str, ...] = (),
    after: tuple[str, ...] = (),
) -> Callable:
    """Register a pragma materialization rewrite.

    The decorated function fires during MirPragmaPass on every op
    of type `on` where `op.pragmas.get(name)` is truthy. It returns
    the transformed op (typed wrap-op insertion) with the pragma
    cleared, OR None to skip.

    `before` / `after` declare ordering against other registered
    pragmas. MirPragmaPass topo-sorts.

    `value_type` is checked at DSL construction time
    (`Rule.with_pragma(name, value)`) — fail-loud on type mismatch.
    """
```

### 2.1 Worked example: `pragmas/dedup_hash.py`

```python
"""Pragma: dedup_hash — atomically deduplicate emitted tuples via a
GPU hash table.

Trigger: rule pragma {"dedup_hash": True}.
Materialization: wrap each InsertInto in a DedupGate at MIR time;
   the lowering for DedupGate emits the dedup-table try_insert + if_p
   gate (which decomposes via the existing sorted_array.DedupTryInsert
   COMPOUND op rewrite).
"""
from srdatalog.core import pragma
from srdatalog.dialects import mir
from srdatalog.dialects.mir.pragma_ops import DedupGate

@pragma(
    name="dedup_hash",
    on=mir.ExecutePipeline,
    value_type=bool,
    before=("count_as_product",),  # R1 must consider DedupGate as InsertInto-equivalent
    after=(),
)
def materialize_dedup_hash(op, ctx):
    if not op.pragmas.get("dedup_hash"):
        return None
    new_pipeline = tuple(
        DedupGate(inner=child) if isinstance(child, mir.InsertInto)
        else child
        for child in op.pipeline
    )
    new_pragmas = tuple(
        (k, v) for (k, v) in op.pragmas if k != "dedup_hash"
    )
    return op.replace(pipeline=new_pipeline, pragmas=new_pragmas)
```

The corresponding `mir.pragma_ops.DedupGate`:

```python
# File: dialects/mir/pragma_ops/dedup_gate.py

from dataclasses import dataclass
from typing import final
from srdatalog.core import Op

@final
@dataclass(frozen=True, slots=True)
class DedupGate(Op):
    """MIR wrap op: dedup-hash gate around an emission op.

    Inserted by pragmas/dedup_hash.py during MirPragmaPass. Lowered
    by dialects/relation/sorted_array/lowerings/lower_mir_dedup_gate.py
    to the existing sorted_array.DedupTryInsert COMPOUND op.
    """
    inner: Op
```

The lowering for `DedupGate`:

```python
# File: dialects/relation/sorted_array/lowerings/lower_mir_dedup_gate.py

from srdatalog.core import lowering
from srdatalog.dialects.iir.cf import IIR_CF
from srdatalog.dialects.mir.pragma_ops import DedupGate
from srdatalog.dialects.relation.sorted_array.ops import DedupTryInsert

@lowering(target=IIR_CF, source=DedupGate)
def lower_dedup_gate(op: DedupGate, ctx, scope):
    inner_iir = ctx.lower(op.inner, scope)
    # Build the try_insert args from the inner InsertInto's vars
    # (extracted from op.inner before lowering).
    args = tuple(VarRef(name=v) for v in op.inner.vars)
    return DedupTryInsert(args=args, then_body=inner_iir)
```

The chain: `Rule(pragmas=("dedup_hash",True))` →
HIR → MIR(`ExecutePipeline.pragmas=("dedup_hash",True), pipeline=[...,InsertInto]`) →
`MirPragmaPass(materialize_dedup_hash)` → MIR(`pragmas=(), pipeline=[...,DedupGate(InsertInto)]`) →
`MirToIirLowering(lower_dedup_gate)` → IIR(`...,DedupTryInsert(args, InsertInto-IIR)`) →
`IirCanonicalizePass` (existing rewrite) → IIR(`...,BracedBlock(Bind, If(MemberCall, ...))`) →
`CudaRenderPass` → C++ text.

No `ctx.dedup_hash` anywhere.

## 3. `MirPragmaPass` algorithm

```python
class MirPragmaPass(RewritePass):
    """One-shot RewritePass: materializes every registered @pragma.

    Walks the registry of @pragma registrations, topo-sorts by
    declared before/after constraints, applies each registration's
    rewrite to the MIR tree.

    After this pass, op.pragmas is empty for every op in the tree.
    Discipline test test_pragmas_empty_after_materialization
    enforces this.
    """
    dialect = MIR_DIALECT
    until_fixpoint = False  # one-shot

    def apply(self, prog, compiler):
        registrations = compiler.pragma_registrations()
        ordered = topo_sort(registrations,
                            key=lambda r: (r.before, r.after))
        for reg in ordered:
            prog = self._apply_one(prog, reg, compiler)
        # Discipline: after all materialize, no pragmas survive
        for op in walk(prog):
            if hasattr(op, 'pragmas') and op.pragmas:
                raise UnconsumedPragmaError(op, op.pragmas)
        return prog

    def _apply_one(self, prog, reg, compiler):
        # Walk prog; for each op of type reg.on with pragma key
        # reg.name truthy, apply reg.fn.
        ...
```

`UnconsumedPragmaError` fires if a pragma name appeared in the DSL
but no `@pragma` registration consumed it. Catches typos
(`Rule(pragmas={"didup_hash": True})` ≠ `dedup_hash`) loudly at
compile time.

## 4. Per-built-in-pragma design

| Pragma | Module | Wrap op | Sub-dialect for IIR? | Lowering target |
|---|---|---|---|---|
| `dedup_hash` | `pragmas/dedup_hash.py` | `mir.pragma_ops.DedupGate` | No (uses existing `sorted_array.DedupTryInsert`) | `lowerings/lower_mir_dedup_gate.py` in `relation.sorted_array` |
| `block_group` | `pragmas/block_group.py` | `mir.pragma_ops.BlockGroupRoot` | **Yes — `parallel.block_group`** (new) | `lowerings/lower_mir_block_group_root.py` in `parallel.block_group` |
| `work_stealing` | `pragmas/work_stealing.py` | `mir.pragma_ops.WSScope` | **Yes — `parallel.atomic_ws`** (new) | `lowerings/lower_mir_ws_scope.py` in `parallel.atomic_ws` |
| `tiled_cartesian` | `pragmas/tiled_cartesian.py` | (no MIR wrap op — handled by MIR rewrite that picks alternative `@lowering` for `mir.CartesianJoin` based on a tag) | No (uses existing `sorted_array.SaTiledCartesian2D`) | augments existing `lowerings/lower_mir_cart.py` |
| `count` | `pragmas/count.py` | (currently a phase flag; becomes `mir.pragma_ops.CountPhase(body)` wrap) | No — `iir.cf.Phase(C, body)` already exists | augments existing lowerings |
| `fanout` | `pragmas/fanout.py` | (TBD; investigate existing usage) | No or new — decide during PR | TBD |

### 4.1 `parallel.block_group` (new sub-dialect)

```
src/srdatalog/dialects/parallel/block_group/
  ops.py              BgRootCJ, BgWorkBalance, BgRootIteration ops
  print.py            Print forms
  __init__.py         DIALECT registration
  lowerings/
    lower_mir_block_group_root.py   # @lowering(IIR_CF, BlockGroupRoot)

src/srdatalog/codegen/cuda/render/
  parallel_block_group.py           # @register_render(Op) per op
```

Per `compiler_redesign.md` §7 sub-dialect criteria, `parallel.block_group`
qualifies because:

- Multiple ops (BgRootCJ, BgWorkBalance, ...).
- Distinct codegen scaffolding (per-warp work-balance setup outside
  the kernel body — visible in runner-level emit).
- External re-use plausible (a future runner library could consume
  these ops for scheduling decisions).

### 4.2 `parallel.atomic_ws` (new sub-dialect)

```
src/srdatalog/dialects/parallel/atomic_ws/
  ops.py              WSCount, WSEmit, WSScope, WSCartesianValid
  print.py
  __init__.py
  lowerings/
    lower_mir_ws_scope.py           # @lowering(IIR_CF, WSScope)

src/srdatalog/codegen/cuda/render/
  parallel_atomic_ws.py
```

Same justification: multiple ops, distinct codegen + runner
integration (WCOJTask queue), external re-use plausible.

### 4.3 Pragmas that are NOT new sub-dialects

`dedup_hash` — uses existing `sorted_array.DedupTryInsert`. Stays
where it is until a non-sorted-array dialect needs dedup; refactor
to `relation.dedup` then.

`tiled_cartesian` — uses existing `sorted_array.SaTiledCartesian2D`.
Tightly coupled to sorted-array index access; factor out only when a
second data-structure dialect needs it.

`count` — phase flag becomes `iir.cf.Phase(C, body)` wrap (already
defined per the spec).

## 5. PR breakdown

Phase C is 4-6 PRs in parallel (Wave 2C from the partition).

| PR | Branch | Adds |
|---|---|---|
| **C1** | `feat/pragma-decorator-and-pass` | `core/pragma.py` — `@pragma` decorator + `MirPragmaPass`. Discipline test `test_pragmas_empty_after_materialization`. No pragmas registered yet — pass is a no-op. |
| **C2** | `feat/pragma-dedup-hash` | `mir/pragma_ops/dedup_gate.py`, `pragmas/dedup_hash.py`, `lowerings/lower_mir_dedup_gate.py`. Removes `if ctx.dedup_hash:` branches from sorted_array lowerings. Byte-equivalence preserved. |
| **C3** | `feat/pragma-block-group` | New `parallel.block_group` sub-dialect (full package), `pragmas/block_group.py`. Removes `if ctx.bg_enabled:` branches. Byte-equivalence preserved. |
| **C4** | `feat/pragma-work-stealing` | New `parallel.atomic_ws` sub-dialect, `pragmas/work_stealing.py`. Removes `if ctx.ws_enabled:` branches. WS runner-level changes deferred (not in scope here). |
| **C5** | `feat/pragma-tiled-cartesian` | `pragmas/tiled_cartesian.py`, augmented `lower_mir_cart.py`. Removes `if ctx.tiled_cartesian:` branches. |
| **C6** | `feat/pragma-count-and-fanout` | `pragmas/count.py`, `pragmas/fanout.py`. Removes remaining `is_counting` / `fanout` branches. |

C1 must land first (the framework). C2–C6 can land in parallel.

### 5.1 Per-PR acceptance gate (Wave 2C)

Beyond the global per-PR Done definition:

- The migrated pragma's old `if ctx.<pragma>:` branches are
  **deleted** from the codebase (or marked `# DEAD CODE — to remove
  in Layer 3` if they're inside `_legacy_imperative_lower` and
  Phase B hasn't migrated the host op yet).
- A new test `test_pragma_<name>_end_to_end` exercises the pragma via
  `Compiler.run` on a fixture that triggers it; asserts the wrap op
  appears in MIR-after-`MirPragmaPass`, asserts the lowered IIR is
  correct, asserts the rendered CUDA is byte-equivalent to legacy.
- `test_pragmas_empty_after_materialization` still passes (no leak).
- Discipline test `test_no_pragma_flags_outside_pragma_modules` (D1)
  runs against the migrated pragma name — fails if any
  `ctx.<pragma>` reference exists outside `pragmas/<pragma>.py`.

## 6. DSL surface change

The DSL needs to accept open-form pragmas without breaking back-compat.

### 6.1 Current DSL

```python
PlanEntry(
    delta=-1,
    var_order=("x", "y"),
    fanout=False,
    work_stealing=True,
    block_group=False,
    dedup_hash=True,
    ...
)
```

### 6.2 Phase A intermediate (named fields still accepted, sugar over
pragmas dict)

```python
# Same call still works (back-compat sugar):
PlanEntry(dedup_hash=True, work_stealing=True)

# Internally constructs:
PlanEntry(pragmas=tuple([("dedup_hash", True), ("work_stealing", True)]))
```

### 6.3 Phase C target

```python
# Generic call (the new canonical form):
PlanEntry(pragmas=tuple([("dedup_hash", True)]))

# Or via the .with_pragma builder (sugar, type-checked):
PlanEntry().with_pragma("dedup_hash", True).with_pragma("work_stealing", True)

# External pragmas (zero changes to DSL or core):
PlanEntry().with_pragma("my_custom_optimization", {"strategy": "hybrid"})
```

`with_pragma(name, value)` looks up the registered pragma's
`value_type` and validates `value`. Unknown name → loud error
("no pragma 'my_custom_optimization' registered; did you forget to
install some-pragma-package?").

## 7. Discipline tests added in Phase C

| Test | Enforces | PR |
|---|---|---|
| `test_pragmas_empty_after_materialization` | `op.pragmas` is empty for every op after `MirPragmaPass` | C1 |
| `test_pragma_registry_completeness` (R5) | Every pragma name the DSL can produce has a `@pragma` registration | C1 |
| `test_no_pragma_flags_outside_pragma_modules` (D1) | `ctx.<pragma>` not referenced outside `pragmas/<pragma>.py` | C1 (per pragma, ratchets per PR) |
| `test_pragma_value_type_validated_at_dsl` | `Rule.with_pragma("dedup_hash", "yes")` raises (string ≠ bool) | C1 |
| `test_pragma_<name>_end_to_end` | Per-pragma round-trip from DSL through `Compiler.run` to byte-equivalent CUDA | C2–C6 (one per pragma) |

## 8. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Pragma ordering matters and topo-sort is wrong | Each pragma declares `before` / `after` explicitly; conflicts surface as topo-sort errors at registration time, not at runtime |
| `parallel.block_group` lowering misses some scaffolding the legacy emit had | Per-PR byte-equivalence test on every fixture; if any divergence, debug or document with sign-off |
| Removing the `if ctx.bg_enabled:` branch breaks fixtures that were running through it | Migration order: land C1 first (no behavior change), then C2 (single pragma) and verify byte-equiv before C3+ |
| `count` pragma is currently a `is_counting` *phase flag* (not a per-rule pragma) — might not fit the `@pragma` shape | Investigate during C6 design; might need a separate `PhasePass` mechanism. If so, document and defer count migration to a Phase D follow-up |

## 9. Sign-off

Phase C is complete iff:

- [ ] All Wave 2C PRs (C1–C6) merged.
- [ ] No `ctx.<pragma>` reference exists in `src/` outside
  `pragmas/<pragma>.py` for any pragma name.
- [ ] `test_pragmas_empty_after_materialization` passes on the full
  fixture set.
- [ ] `test_pragma_registry_completeness` (R5) passes — every DSL
  pragma name has an `@pragma` handler.
- [ ] Two new sub-dialects (`parallel.block_group`,
  `parallel.atomic_ws`) registered; their op vocabulary documented in
  `ir_derivation_topology.md` §3.3.3.
- [ ] Byte-equivalence preserved across the entire fixture set (or
  documented divergences with owner sign-off).

After Phase C sign-off, the `if ctx.<pragma>:` branch class of code
is structurally extinct.
