---
orphan: true
---

# IR Dialect Contract: Per-Op Renderability and Rewrite-to-Fixpoint

This document specifies the per-op contract that governs how IIR ops
reach (or don't reach) the codegen renderer. It complements
[`ir_lowering_semantics.md`](ir_lowering_semantics.md) — that doc
defines the IR layer stack (HIR / MIR / IIR / codegen) and the
per-dialect ABI; this doc defines the discipline *inside* the IIR
layer that determines what the codegen actually sees.

It supersedes the implicit "everything in IIR is renderable" assumption
the codebase currently runs on, and replaces the rewrite-driver sketch
in §21 of `ir_lowering_semantics.md` with a concrete, testable
contract.

The motivating diagnosis is in
[`stage4_iir_vocabulary.md`](stage4_iir_vocabulary.md): of the 46
original `RawString` sites in `sorted_array/lowerings.py`, the majority
encoded high-level intent (lookup, intersect, dedup-try-insert) that
*should* have been first-class ops with rewrites — not text dumped
into `iir.cf.Block`. The contract below makes that the only legal
pattern.

## 1. The contract

Every concrete `Op` subclass that can appear in an IIR program (i.e.
reachable by codegen tree-walk) carries exactly one role for each
registered target:

- **LEAF** — has a `@register_render(Op, mode=...)` for the active
  target. Codegen renders it directly.
- **COMPOUND** — has a `@rewrite(DIALECT, Op)` registered. The pass
  driver decomposes it into other ops (LEAF or COMPOUND) before
  codegen sees it.

An op MAY be LEAF for one target and COMPOUND for another (this is
how target-specific decompositions ship without polluting other
targets). An op MUST NOT be NEITHER for any target a program targets:
this is the closure violation that `verify_renderability` catches.

Diagrammatically, after `PassDriver.run`:

```
IIR (mixed-level — LEAF and COMPOUND ops freely interleaved)
    │
    ▼  apply @rewrite to fixpoint
IIR' (LEAF only, modulo per-target renderer registrations)
    │
    ▼  codegen tree-walk: emit(op) → handler[type(op)](op, ctx)
target text (CUDA C++)
```

### 1.1 Why per-op, not per-dialect

The classification is per-op rather than per-dialect because real
dialects mix levels. Today's map:

| Dialect | LEAF ops | COMPOUND ops (planned) |
|---|---|---|
| `iir.cf` | `Block`, `If`, `Bind`, `BracedBlock`, `Assign`, … | (none today) |
| `iir.expr` | `BinOp`, `IndexExpr`, `MemberCall`, `Parens`, … | (none today) |
| `relation.sorted_array` | direct member-call primitives (today inlined via `RawString`) | `SortedArrayLookup`, `Intersect`, `DedupTryInsert` |
| `relation.d2l` | `Materialize` | (none today) |
| `parallel.data` | `ParallelFor`, `GridStrideLoop` | (none today) |

A "semantic dialect / structural dialect" split was rejected during
design (see §6.2). It conflates op role with dialect identity and
forbids useful patterns like a `sorted_array.SortedArrayLookup`
containing an `iir.cf.Block` as a child subtree.

### 1.2 Why one IR, not High-IIR / Low-IIR

An earlier proposal split IIR into a High layer (semantic dialects)
and a Low layer (structural dialects) with an explicit
`lower_high_to_low()` transition. Rejected because:

1. It doubles the print/render surface — every semantic op needs a
   "low-form" twin.
2. It hides the contract in the directory layout instead of in the
   type/registry system.
3. It forbids mixed-level subtrees (a high-level op containing
   structural children at the same nesting depth), which is exactly
   the shape `DedupTryInsert(then_block=BracedBlock(...))` wants.
4. MLIR — the closest existing precedent — uses mixed-level dialects
   with dialect conversion, not High/Low IR layers.

The per-op contract gives the same enforcement as a layer boundary
without the structural cost.

## 2. PassDriver.run semantics

Today `PassDriver.run` validates dependencies and runs verifiers
(`src/srdatalog/ir/core/passes.py`). Under this contract it gains
rewrite-to-fixpoint dispatch and renderability verification:

```python
def run(self, prog, *, target):
    self.validate_dependencies()
    prog = self.apply_rewrites_to_fixpoint(prog)
    errors = self.verify_renderability(prog, target=target)
    if errors:
        raise UnrenderableOpError(errors)
    errors = self.verify_all(prog)
    if errors:
        raise RuntimeError(f'verification failed: {errors}')
    return prog
```

### 2.1 Rewrite-to-fixpoint

`apply_rewrites_to_fixpoint` walks `prog`, looks up `Rewrite` instances
matching each op's type, and applies them — recursing on the
replacement. Termination is guaranteed by requiring that every
registered rewrite is *strictly decreasing* in some well-founded
measure. In practice the measure is: "the replacement contains no
op of the same class as the source." A debug-mode invariant counts
iterations and raises if a fixed iteration limit is exceeded.

A single rewrite may emit a tree containing further COMPOUND ops;
they are picked up on the next walk pass. The driver iterates until
a full walk finds no applicable rewrite.

### 2.2 At most one rewrite per op type

If two `@rewrite` decorators target the same op type, registration is
an error (raised at `Compiler.register_dialect` time). Rationale:
ambiguity in the rewrite system is a worse failure mode than missing
optimizations — the alternative (priority lists, declarative ordering)
adds machinery whose only consumer would be the optimizer, which we
don't yet have.

When the optimizer arrives, this restriction loosens to "exactly one
*normalizing* rewrite per op type; *optimizing* rewrites tagged as
such may stack." Until then: one per type.

### 2.3 Renderer-trumps-rewrite

If both a `@register_render` and a `@rewrite` are registered for an
op type for the active target, `register_render` wins — the renderer
is applied; the rewrite is dead code for this target. This is the
hook that lets a target ship a direct renderer for an op the framework
considers compound by default (e.g. a future `target.cpp_simd` that
renders `DedupTryInsert` via a SIMD intrinsic instead of decomposing
to atomic CAS).

`PassDriver.run` warns if it sees a `(renderer, rewrite)` pair both
registered for the same `(op, target)`. The warning is informational —
the configuration is legal but uncommon enough to be worth surfacing.

## 3. verify_renderability

```python
def verify_renderability(prog, *, target):
    errors = []
    for op in walk(prog):
        if has_renderer(type(op), target):
            continue
        errors.append(UnrenderableOp(op_type=type(op), target=target))
    return errors
```

This runs *after* `apply_rewrites_to_fixpoint`. By then, every
COMPOUND op should have decomposed; any survivor is a missing
rewrite or a missing renderer. The error names both the op type and
the target so the fix is mechanical.

This is the loud-failure replacement for the current implicit
"everything in IIR has a renderer" assumption — and in particular it
is what makes A4 / A8 (no silent fallback) hold at the dialect
boundary.

## 4. RawString reclassification

`iir.cf.RawString` becomes a renderer-registered escape hatch reserved
for **Category J** (user-supplied code, per
[`stage4_iir_vocabulary.md`](stage4_iir_vocabulary.md)): expressions
and predicate fragments that arrive from `Filter` / `ConstantBind`
and are by-construction opaque to the IR.

All other current uses are framework debt to be cleared by Stage 4
under the per-op contract:

| Current pattern | Reclassification |
|---|---|
| `RawString('handle.member()')` | `MemberCall` (LEAF) |
| `RawString('a + b')` | `BinOp` (LEAF) |
| `RawString('arr[i]')` | `IndexExpr` (LEAF) |
| `RawString('{ ... multi-line dedup ... }')` | `DedupTryInsert` (COMPOUND, rewrite to `BracedBlock + Bind + If + MemberCall`) |

The S4.7 ratchet test (`tests/test_iir_no_raw_string_growth.py`) is
the mechanical enforcement: the count drops monotonically as
compound ops land; new `RawString` sites must be Category J or
motivate an exception with documented categorization.

## 5. Migration plan

This contract maps onto the open Stage 4 tasks as:

- **S4.9 (this contract's first deliverable).** Implement
  `apply_rewrites_to_fixpoint` and `verify_renderability` in
  `src/srdatalog/ir/core/passes.py`. No behavior change for current
  pipelines: no rewrites are registered yet, so the fixpoint
  converges in one pass and
  `verify_renderability` is satisfied because every op reaching
  codegen today is LEAF (the `RawString` sites bypass the IR
  abstraction precisely *because* there's no rewrite-to-fixpoint).

- **S4.6b.** First COMPOUND op: `sorted_array.DedupTryInsert`. Lands
  with its `@rewrite` and a removal of the ~12 dedup-table
  `RawString` sites. The cap drops from 17 to ~5.

- **S4.8 (R1–R5).** The five `sorted_array` rewrites from §11 of
  `ir_lowering_semantics.md` become real `@rewrite` decorations
  now that the driver can run them.

- **Subsequent S4.x.** `SortedArrayLookup`, `Intersect`, etc. each
  become COMPOUND ops with rewrites — the remaining `RawString`
  sites outside Category J disappear.

## 6. Rejected alternatives

### 6.1 Two-IR split (High-IIR / Low-IIR)

See §1.2. The summary: enforcement-equivalent to the per-op
contract, but with double the print/render surface and a directory
layout that forbids mixed-level subtrees.

### 6.2 Per-dialect classification (semantic vs structural)

Forces a dialect to be all-LEAF or all-COMPOUND. Real dialects have
both — `relation.sorted_array` has direct member-call primitives
*and* semantic combinators. A per-dialect classification would force
either splitting `sorted_array` into two artificial dialects or
keeping the current "everything inlines via `RawString`" status quo.

### 6.3 Implicit renderer fallback

I.e. "if no renderer is registered, emit `{repr(op)}` with a
warning." Violates A4 / A8 (no silent fallback). The
`UnrenderableOpError` from §3 is the loud-failure replacement.

### 6.4 Single global rewrite list

I.e. one flat `[Rewrite, ...]` not partitioned by dialect. Loses
Property P3 (`ir_lowering_semantics.md §4.3`): rewrites must remain
per-dialect so adding a dialect cannot break another dialect's
passes. The current decorator API (`@rewrite(DIALECT, OpClass)`)
already enforces this.

## 7. Out of scope for this contract

- **Rewrite ordering / phases.** Today: rewrites are applied wherever
  they match, no phase ordering. If an optimizer phase arrives, it
  ships its own ordering metadata — this contract does not preempt
  that design.
- **Cross-target rewrites.** Today: rewrites are target-agnostic
  (decompose for *any* target). Target-specific rewrites are a future
  extension; the registration API would gain an optional `target=`
  parameter when a real consumer arrives.
- **MIR / HIR rewrites.** This contract speaks to IIR. MIR and HIR
  have their own (pre-existing) rewrite semantics. Stage 3B may
  unify them under the same `@rewrite` decorator; not in scope here.
