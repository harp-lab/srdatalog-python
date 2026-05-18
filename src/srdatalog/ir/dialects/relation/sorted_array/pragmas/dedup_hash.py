'''Pragma: `dedup_hash` — atomically deduplicate emitted tuples via a
GPU hash table.

Trigger: `Rule(...).with_pragma(DedupHash())` (preferred typed form)
   or the legacy `with_plan(..., dedup_hash=True)` keyword (back-compat,
   slated for removal in A3).

Materialization (this module): wrap each `mir.InsertInto` at the tail
   of an `ExecutePipeline.pipeline` with `mir.DedupGate(inner=...)`
   during `MirPragmaPass`, then strip the `DedupHash` instance from
   the EP's `pragmas` tuple.

Lowering (this module): a `@lowering(target=iir.cf, source=mir.
   DedupGate)` rule (registered via the parent dialect's
   `_register_passes`) emits the same IIR shape (`try_insert(thread_id,
   vars...) + if (_p) { write }`) that the legacy `if ctx.dedup_hash:`
   branch inside `_lower_insert_into` produces — so byte-equivalence
   holds across the migration. The lowering body delegates to that
   legacy helper with `ctx.dedup_hash` flipped True for the duration of
   the call, so the emission is literally the same code path the
   legacy bool-field flow produces (byte-equivalence by construction).

   The spec (`docs/phase_c_pragma_materialization.md` §2.1) sketches
   the lowering at `lowerings/lower_mir_dedup_gate.py`; that exact
   path collides with the existing `lowerings.py` monolith on the
   same dialect, so the C2 PR co-locates the lowering function here
   with its pragma class. Phase B will rename `lowerings.py` to
   something like `imperative_lowerings.py` and split per-op rules
   into a sibling `lowerings/` package; at that point this lowering
   moves to `lowerings/lower_mir_dedup_gate.py` per spec.

Spec: `docs/phase_c_pragma_materialization.md` §2.1, §4.3, §5 (C2 PR
row), §5.1 (per-PR acceptance gate); `docs/pragma_as_typed_object.md`
§2 (Pragma base class), §3 (`@pragma_handler` decorator).
'''

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, final

from srdatalog.ir.core import Op, Pragma, pragma_handler
from srdatalog.ir.dialects.iir.cf import Block
from srdatalog.ir.mir.types import DedupGate, ExecutePipeline, InsertInto


@final
@dataclass(frozen=True, slots=True)
class DedupHash(Pragma):
  '''Atomically dedup emitted tuples via a GPU hash table.

  Triggers `MirPragmaPass` to wrap each `InsertInto` at the tail of
  the carrying `ExecutePipeline.pipeline` in a `DedupGate` MIR op;
  the gate's `@lowering` rule emits the `dedup_table.try_insert(...)
  + if (_p) {...}` shape that the legacy `if ctx.dedup_hash:` branch
  produced.

  No fields today — the existing legacy emitter treats `dedup_hash`
  as a flat bool. Structured config (hash table capacity, custom
  hash functions, etc.) is plausible later but out of scope for C2;
  per `docs/pragma_as_typed_object.md` §2, future fields land via
  back-compat-preserving dataclass defaults.
  '''


# -----------------------------------------------------------------------------
# Materialization handler (registered at import time via @pragma_handler)
# -----------------------------------------------------------------------------


def _wrap_inserts_in_dedup_gate(
  pipeline: list[Any],
) -> list[Any]:
  '''Replace every `InsertInto` in `pipeline` with
  `DedupGate(inner=that_insert)`.

  Mirrors the spec §2.1 worked example: walk `pipeline`, swap each
  `InsertInto` for `DedupGate(inner=that_insert)`, leave non-insert
  ops unchanged. `DedupGate` only wraps `InsertInto` (not nested
  structures like Cartesian/CJ); this matches the legacy semantics
  where `ctx.dedup_hash` only gated the leaf emission, not the
  intermediate scan / join structure.
  '''
  wrapped: list[Any] = []
  for child in pipeline:
    if isinstance(child, InsertInto):
      wrapped.append(DedupGate(inner=child))
    else:
      wrapped.append(child)
  return wrapped


@pragma_handler(DedupHash, on=ExecutePipeline)
def materialize_dedup_hash(
  op: Any,  # ExecutePipeline at runtime — typed as Any to match the registry's
  # Callable[[Any, Pragma, PragmaCtx], Any] signature (see
  # core/pragma.py:PragmaRegistration). Body re-narrows via the typed
  # `op.pragmas` / `op.dedup_hash` accesses below.
  pragma: Any,  # DedupHash at runtime; same Any reason.
  ctx: Any,  # PragmaCtx, unused for this no-config pragma.
) -> Any:
  '''Materialize `DedupHash` into a `DedupGate` wrap around each
  trailing `InsertInto` in `op.pipeline`.

  Returns a new `ExecutePipeline` with:
    - `pipeline` rewritten so every `InsertInto` is replaced by
      `DedupGate(inner=that_insert)` — **except** in the C2
      dual-write transition state (see below),
    - `pragmas` filtered to drop the consumed `DedupHash` instance
      (per the `MirPragmaPass` post-flight invariant; see
      `docs/pragma_as_typed_object.md` §3).

  C2 dual-write transition: the DSL `Rule.with_pragma(DedupHash())`
  sets BOTH the typed pragma AND the legacy `dedup_hash: bool`
  field on the resulting `ExecutePipeline`, so the legacy emitter
  (`compile_kernel_body` -> `_lower_insert_into`'s
  `if ctx.dedup_hash:` branch) keeps producing byte-equivalent
  CUDA without seeing the new `DedupGate` wrap op (which the
  monolith doesn't know how to traverse). When `ep.dedup_hash` is
  True we therefore SKIP the wrap step: stripping the pragma is
  enough to satisfy `MirPragmaPass`'s post-flight invariant, and
  the bool field continues to drive emission via the legacy path.

  When `ep.dedup_hash` is False (the post-A3 pure-typed state, or
  test fixtures that exercise only the typed surface), the handler
  produces the wrap op and the registered
  `@lowering(target=iir.cf, source=DedupGate)` rule lowers it. The
  resulting IIR is byte-equivalent to the dedup branch of
  `_lower_insert_into` by construction (the lowering delegates to
  that helper with `ctx.dedup_hash` flipped True for the call).

  Idempotency: the EP carries at most one `DedupHash` (the DSL
  constructs at most one per `with_pragma(DedupHash())` call; future
  multi-instance support is out of scope). We filter by
  `isinstance(p, DedupHash)` to be defensive — any stray duplicates
  get removed together.

  Per spec §2.1, this handler does NOT recurse into nested CJ /
  Cartesian bodies — those contain no `InsertInto` directly; the
  legacy `ctx.dedup_hash` flag only gated leaf emission and the
  trailing `InsertInto` run is exactly the leaf position.
  '''
  new_pragmas = tuple(p for p in op.pragmas if not isinstance(p, DedupHash))
  # Dual-write transition: skip the wrap if the legacy bool is set
  # (the legacy lowering will handle emission). See docstring above.
  if op.dedup_hash:
    return dataclasses.replace(op, pragmas=new_pragmas)
  new_pipeline = _wrap_inserts_in_dedup_gate(list(op.pipeline))
  return dataclasses.replace(op, pipeline=new_pipeline, pragmas=new_pragmas)


# -----------------------------------------------------------------------------
# DedupGate lowering (registered by sorted_array `_register_passes`)
# -----------------------------------------------------------------------------


def lower_dedup_gate(op: DedupGate, ctx: Any) -> Op:
  '''Emit the IIR for `DedupGate(inner=InsertInto)`.

  Returns a `Block` wrapping the inner `InsertInto`'s emission
  rendered under `ctx.dedup_hash=True`. The legacy
  `_lower_insert_into` (in the dialect's `lowerings.py` monolith)
  already produces the full `try_insert + if (_p) { write }` gate
  when its `ctx.dedup_hash` is set; we reuse it verbatim so the
  new lowering and the legacy path emit byte-identical text for
  the same `InsertInto`.

  The save/restore pattern around `ctx.dedup_hash` is defensive:
  if a future call site invokes this lowering from a partial-ctx
  scope where `dedup_hash` is False (e.g. a refactored
  `compile_kernel_body` that no longer pre-sets the field), the
  flag still flips on for the duration of the gate.

  `LoweringCtx` is a non-frozen dataclass (it carries `name_counter`
  + per-pass mutable state), so attribute assignment is the
  supported mutation surface — no `object.__setattr__` shim needed
  and the D18 ratchet does not apply.
  '''
  # Deferred import: the monolith module imports nothing from this
  # subpackage at top level, but co-located ops + pragmas + lowerings
  # in `pragmas/dedup_hash.py` (this file) get loaded by the dialect's
  # `_register_passes` AFTER the monolith. Importing at function-call
  # time keeps the dialect import graph linear.
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _lower_insert_into,
  )

  prev = getattr(ctx, 'dedup_hash', False)
  try:
    ctx.dedup_hash = True
    stmts = _lower_insert_into(op.inner, ctx)
  finally:
    ctx.dedup_hash = prev
  return Block(stmts=tuple(stmts))


__all__ = [
  'DedupHash',
  'lower_dedup_gate',
  'materialize_dedup_hash',
]
