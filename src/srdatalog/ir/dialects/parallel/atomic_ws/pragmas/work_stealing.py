'''Pragma: `work_stealing` — atomic work-stealing kernel-functor emit.

Trigger: `Rule(...).with_pragma(WorkStealing())` (preferred typed
   form) or the legacy `with_plan(..., work_stealing=True)` keyword
   (back-compat, slated for removal in A3).

Materialization (this module): wrap each `mir.InsertInto` at the
   tail of an `ExecutePipeline.pipeline` with `mir.WSScope(inner=
   ...)` during `MirPragmaPass`, then strip the `WorkStealing`
   instance from the EP's `pragmas` tuple.

Lowering (this module): a `@lowering(target=iir.cf, source=mir.
   WSScope)` rule (registered via the parent dialect's
   `_register_passes`) emits the same IIR shape that the legacy
   `if ctx.ws_enabled:` branch inside `_lower_insert_into` produces
   — count-phase `<out>++` increments to a per-thread `local_count`
   instead of `<out>.emit_direct()`; materialize-phase passes
   through unchanged (the runner-level WS scaffolding that drives
   the warp-coalesced batched writes lives in
   `ws_cartesian_valid_var`, which is a separate code path threaded
   by `_lower_nested_cart` — not in C4 scope).

   The lowering body delegates to the legacy `_lower_insert_into`
   with `ctx.ws_enabled` flipped True for the duration of the call,
   so the emission is literally the same code path the legacy
   bool-field flow produces (byte-equivalence by construction).

C4 scope (per `docs/phase_c_pragma_materialization.md` §5, PR row
C4 + §4.2 sub-dialect description): only the kernel-functor-level
WS emit variants migrate. The runner-level WS scaffolding (WCOJTask
queue, steal-loop emission, etc.) stays on the legacy
`ExecutePipeline.work_stealing: bool` field path; A3 drops the bool
and the wrap op becomes the sole driver.

Spec: `docs/phase_c_pragma_materialization.md` §4.2, §5 (C4 PR
row), §5.1 (per-PR acceptance gate); `docs/pragma_as_typed_object.
md` §2 (Pragma base class), §3 (`@pragma_handler` decorator).
'''

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, final

from srdatalog.ir.core import Op, Pragma, pragma_handler
from srdatalog.ir.dialects.iir.cf import Block
from srdatalog.ir.mir.types import ExecutePipeline, InsertInto, WSScope


@final
@dataclass(frozen=True, slots=True)
class WorkStealing(Pragma):
  '''Atomic work-stealing kernel-functor emit variant.

  Triggers `MirPragmaPass` to wrap each `InsertInto` at the tail of
  the carrying `ExecutePipeline.pipeline` in a `WSScope` MIR op; the
  scope's `@lowering` rule emits the per-thread `<out>++` (count
  phase) shape that the legacy `if ctx.ws_enabled:` branch produced.

  No fields today — the existing legacy emitter treats
  `work_stealing` as a flat bool. Structured config (custom queue
  capacity, per-warp slice size, steal-victim policy, etc.) is
  plausible later but out of scope for C4; per
  `docs/pragma_as_typed_object.md` §2, future fields land via
  back-compat-preserving dataclass defaults.
  '''


# -----------------------------------------------------------------------------
# Materialization handler (registered at import time via @pragma_handler)
# -----------------------------------------------------------------------------


def _wrap_inserts_in_ws_scope(
  pipeline: list[Any],
) -> list[Any]:
  '''Replace every `InsertInto` in `pipeline` with
  `WSScope(inner=that_insert)`.

  Mirrors the C2 `_wrap_inserts_in_dedup_gate` pattern: walk
  `pipeline`, swap each `InsertInto` for `WSScope(inner=that_insert)`,
  leave non-insert ops unchanged. `WSScope` only wraps `InsertInto`
  (not nested structures like Cartesian/CJ); this matches the legacy
  semantics where `ctx.ws_enabled` only gated the leaf emission, not
  the intermediate scan / join structure.
  '''
  wrapped: list[Any] = []
  for child in pipeline:
    if isinstance(child, InsertInto):
      wrapped.append(WSScope(inner=child))
    else:
      wrapped.append(child)
  return wrapped


@pragma_handler(WorkStealing, on=ExecutePipeline)
def materialize_work_stealing(
  op: Any,  # ExecutePipeline at runtime — typed as Any to match the registry's
  # Callable[[Any, Pragma, PragmaCtx], Any] signature (see
  # core/pragma.py:PragmaRegistration). Body re-narrows via the typed
  # `op.pragmas` / `op.work_stealing` accesses below.
  pragma: Any,  # WorkStealing at runtime; same Any reason.
  ctx: Any,  # PragmaCtx, unused for this no-config pragma.
) -> Any:
  '''Materialize `WorkStealing` into a `WSScope` wrap around each
  trailing `InsertInto` in `op.pipeline`.

  Returns a new `ExecutePipeline` with:
    - `pipeline` rewritten so every `InsertInto` is replaced by
      `WSScope(inner=that_insert)` — **except** in the C4
      dual-write transition state (see below),
    - `pragmas` filtered to drop the consumed `WorkStealing`
      instance (per the `MirPragmaPass` post-flight invariant; see
      `docs/pragma_as_typed_object.md` §3).

  C4 dual-write transition: the DSL `Rule.with_pragma(WorkStealing
  ())` sets BOTH the typed pragma AND the legacy `work_stealing:
  bool` field on the resulting `ExecutePipeline`, so the legacy
  emitter (`compile_kernel_body` -> `_lower_insert_into`'s
  `if ctx.ws_enabled:` branch, plus the runner-side WS scaffolding
  in `codegen/cuda/batchfile.py` / `orchestrator.py`) keeps
  producing byte-equivalent output without seeing the new `WSScope`
  wrap op (which the monolith doesn't know how to traverse). When
  `ep.work_stealing` is True we therefore SKIP the wrap step:
  stripping the pragma is enough to satisfy `MirPragmaPass`'s
  post-flight invariant, and the bool field continues to drive
  emission via the legacy path.

  When `ep.work_stealing` is False (the post-A3 pure-typed state,
  or test fixtures that exercise only the typed surface), the
  handler produces the wrap op and the registered
  `@lowering(target=iir.cf, source=WSScope)` rule lowers it. The
  resulting IIR is byte-equivalent to the WS branch of
  `_lower_insert_into` by construction (the lowering delegates to
  that helper with `ctx.ws_enabled` flipped True for the call).

  Idempotency: the EP carries at most one `WorkStealing` (the DSL
  constructs at most one per `with_pragma(WorkStealing())` call;
  future multi-instance support is out of scope). We filter by
  `isinstance(p, WorkStealing)` to be defensive — any stray
  duplicates get removed together.

  Per spec §4.2, this handler does NOT recurse into nested CJ /
  Cartesian bodies — those contain no `InsertInto` directly; the
  legacy `ctx.ws_enabled` flag only gated leaf emission and the
  trailing `InsertInto` run is exactly the leaf position.
  '''
  new_pragmas = tuple(p for p in op.pragmas if not isinstance(p, WorkStealing))
  # Dual-write transition: skip the wrap if the legacy bool is set
  # (the legacy lowering will handle emission). See docstring above.
  if op.work_stealing:
    return dataclasses.replace(op, pragmas=new_pragmas)
  new_pipeline = _wrap_inserts_in_ws_scope(list(op.pipeline))
  return dataclasses.replace(op, pipeline=new_pipeline, pragmas=new_pragmas)


# -----------------------------------------------------------------------------
# WSScope lowering (registered by parallel.atomic_ws `_register_passes`)
# -----------------------------------------------------------------------------


def lower_ws_scope(op: WSScope, ctx: Any) -> Op:
  '''Emit the IIR for `WSScope(inner=InsertInto)`.

  Returns a `Block` wrapping the inner `InsertInto`'s emission
  rendered under `ctx.ws_enabled=True`. The legacy
  `_lower_insert_into` (in the sorted_array dialect's `lowerings.py`
  monolith) already produces the WS-flavoured emit (count-phase
  `<out>++` instead of `<out>.emit_direct()`) when its
  `ctx.ws_enabled` is set; we reuse it verbatim so the new lowering
  and the legacy path emit byte-identical text for the same
  `InsertInto`.

  The save/restore pattern around `ctx.ws_enabled` is defensive:
  if a future call site invokes this lowering from a partial-ctx
  scope where `ws_enabled` is False (e.g. a refactored
  `compile_kernel_body` that no longer pre-sets the field), the
  flag still flips on for the duration of the gate.

  `LoweringCtx` is a non-frozen dataclass (it carries
  `name_counter` + per-pass mutable state), so attribute assignment
  is the supported mutation surface — no `object.__setattr__` shim
  needed and the D18 ratchet does not apply.

  Implementation parallels `lower_dedup_gate` (C2 template) at
  `srdatalog.ir.dialects.relation.sorted_array.pragmas.dedup_hash.
  lower_dedup_gate`.
  '''
  # Deferred import: the sorted_array monolith module imports nothing
  # from this subpackage at top level. Importing at function-call
  # time keeps the dialect import graph linear (parallel.atomic_ws
  # does not need to load sorted_array eagerly just to register a
  # lowering whose body delegates back into it).
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _lower_insert_into,
  )

  prev = getattr(ctx, 'ws_enabled', False)
  try:
    ctx.ws_enabled = True
    stmts = _lower_insert_into(op.inner, ctx)
  finally:
    ctx.ws_enabled = prev
  return Block(stmts=tuple(stmts))


__all__ = [
  'WorkStealing',
  'lower_ws_scope',
  'materialize_work_stealing',
]
