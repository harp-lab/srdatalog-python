'''Pragma: `fanout` — fan-out work-stealing for Cartesian products.

Trigger: `Rule(...).with_pragma(FanOut())` (preferred typed form)
   or the legacy `with_plan(..., fanout=True)` keyword (back-compat,
   slated for removal in A3).

Materialization (this module): wrap each `mir.InsertInto` at the tail
   of an `ExecutePipeline.pipeline` with `mir.FanOut(inner=...)`
   during `MirPragmaPass`, then strip the `FanOut` instance from
   the EP's `pragmas` tuple.

Lowering (this module): a `@lowering(target=iir.cf, source=mir.
   FanOut)` rule (registered via the parent dialect's
   `_register_passes`) emits the IIR shape that the legacy
   `if ep.use_fan_out:` runner branch consumes. The legacy fan-out
   runtime path remains the byte-equivalence anchor for the C6
   transition: the runner-side scheduling (FanOutTaskQueue,
   `jit_fanout_executor.h`) is orthogonal and stays in
   `complete_runner.py` (per the C6 task constraint that the runner
   is out of scope).

Spec: `docs/phase_c_pragma_materialization.md` §4.3, §5 (C6 PR row),
§5.1 (per-PR acceptance gate); `docs/pragma_as_typed_object.md` §2
(Pragma base class), §3 (`@pragma_handler` decorator).
'''

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, final

from srdatalog.ir.core import Op, Pragma, pragma_handler
from srdatalog.ir.dialects.iir.cf import Block
from srdatalog.ir.mir.types import ExecutePipeline, InsertInto
from srdatalog.ir.mir.types import FanOut as MirFanOut


@final
@dataclass(frozen=True, slots=True)
class FanOut(Pragma):
  '''Fan-out work-stealing for Cartesian products.

  Triggers `MirPragmaPass` to wrap each `InsertInto` at the tail of
  the carrying `ExecutePipeline.pipeline` in a `mir.FanOut` MIR
  wrap op; the gate's `@lowering` rule emits the IIR shape the
  legacy `ep.use_fan_out` runner branch produces.

  No fields today — the legacy `use_fan_out` is a flat bool
  consumed by `runtime/.../jit_fanout_executor.h` for queue setup +
  task chunking. Structured config (chunk size, queue capacity,
  etc.) is plausible later but out of scope for C6; per
  `docs/pragma_as_typed_object.md` §2, future fields land via
  back-compat-preserving dataclass defaults.
  '''


# -----------------------------------------------------------------------------
# Materialization handler (registered at import time via @pragma_handler)
# -----------------------------------------------------------------------------


def _wrap_inserts_in_fan_out(
  pipeline: list[Any],
) -> list[Any]:
  '''Replace every `InsertInto` in `pipeline` with
  `mir.FanOut(inner=that_insert)`.

  Mirrors the C2 `_wrap_inserts_in_dedup_gate` shape: walk
  `pipeline`, swap each `InsertInto` for `FanOut(inner=that_insert)`,
  leave non-insert ops unchanged. `FanOut` only wraps `InsertInto`
  (not nested structures like Cartesian/CJ); this matches the legacy
  semantics where `ep.use_fan_out` gated the leaf emission shape
  (the runner chunks the Cartesian product into FanOutTasks; each
  task's worker pops a chunk and emits leaf rows).
  '''
  wrapped: list[Any] = []
  for child in pipeline:
    if isinstance(child, InsertInto):
      wrapped.append(MirFanOut(inner=child))
    else:
      wrapped.append(child)
  return wrapped


@pragma_handler(FanOut, on=ExecutePipeline)
def materialize_fanout(
  op: Any,  # ExecutePipeline at runtime — typed as Any to match the registry's
  # Callable[[Any, Pragma, PragmaCtx], Any] signature (see
  # core/pragma.py:PragmaRegistration). Body re-narrows via the typed
  # `op.pragmas` / `op.use_fan_out` accesses below.
  pragma: Any,  # FanOut at runtime; same Any reason.
  ctx: Any,  # PragmaCtx, unused for this no-config pragma.
) -> Any:
  '''Materialize `FanOut` into a `mir.FanOut` wrap around each
  trailing `InsertInto` in `op.pipeline`.

  Returns a new `ExecutePipeline` with:
    - `pipeline` rewritten so every `InsertInto` is replaced by
      `mir.FanOut(inner=that_insert)` — **except** in the C6
      dual-write transition state (see below),
    - `pragmas` filtered to drop the consumed `FanOut` instance
      (per the `MirPragmaPass` post-flight invariant; see
      `docs/pragma_as_typed_object.md` §3).

  C6 dual-write transition: the DSL `Rule.with_pragma(FanOut())`
  sets BOTH the typed pragma AND the legacy `use_fan_out: bool`
  field on the resulting `ExecutePipeline`, so the legacy emitter
  (`compile_kernel_body` -> runner consumption via
  `complete_runner.py`) keeps producing byte-equivalent CUDA
  without seeing the new `mir.FanOut` wrap op (which the monolith
  doesn't know how to traverse). When `ep.use_fan_out` is True we
  therefore SKIP the wrap step: stripping the pragma is enough to
  satisfy `MirPragmaPass`'s post-flight invariant, and the bool
  field continues to drive emission via the legacy path.

  When `ep.use_fan_out` is False (the post-A3 pure-typed state, or
  test fixtures that exercise only the typed surface), the handler
  produces the wrap op and the registered `@lowering(target=iir.cf,
  source=mir.FanOut)` rule lowers it.

  Idempotency: the EP carries at most one `FanOut` (the DSL
  constructs at most one per `with_pragma(FanOut())` call; future
  multi-instance support is out of scope). We filter by
  `isinstance(p, FanOut)` to be defensive — any stray duplicates
  get removed together.

  Per the C2 template, this handler does NOT recurse into nested CJ
  / Cartesian bodies — those contain no `InsertInto` directly; the
  legacy `ep.use_fan_out` flag only gated leaf emission and the
  trailing `InsertInto` run is exactly the leaf position.
  '''
  new_pragmas = tuple(p for p in op.pragmas if not isinstance(p, FanOut))
  # Dual-write transition: skip the wrap if the legacy bool is set
  # (the legacy lowering will handle emission). See docstring above.
  if op.use_fan_out:
    return dataclasses.replace(op, pragmas=new_pragmas)
  new_pipeline = _wrap_inserts_in_fan_out(list(op.pipeline))
  return dataclasses.replace(op, pipeline=new_pipeline, pragmas=new_pragmas)


# -----------------------------------------------------------------------------
# FanOut lowering (registered by sorted_array `_register_passes`)
# -----------------------------------------------------------------------------


def lower_fan_out(op: MirFanOut, ctx: Any) -> Op:
  '''Emit the IIR for `mir.FanOut(inner=InsertInto)`.

  Returns a `Block` wrapping the inner `InsertInto`'s emission. In
  the C6 transition, the legacy runner-side fan-out scheduling
  (FanOutTaskQueue, `jit_fanout_executor.h`) is what actually
  effects the work-stealing behavior — the kernel-body IIR is
  identical to the non-fanout shape. So this lowering simply
  delegates to the existing `_lower_insert_into` helper without
  touching the ctx.

  The legacy code path (when `ep.use_fan_out` is True) doesn't see
  this wrap op (the materialize handler's dual-write short-circuit
  keeps the typed wrap out of the pipeline). When only the typed
  pragma is set (post-A3 state), this lowering keeps the kernel-
  body emission byte-equivalent to the non-fanout shape, leaving
  the runner-side scheduling to a future migration (A3 / Phase D).
  '''
  # Deferred import: the monolith module imports nothing from this
  # subpackage at top level, but co-located ops + pragmas + lowerings
  # in `pragmas/fanout.py` (this file) get loaded by the dialect's
  # `_register_passes` AFTER the monolith. Importing at function-call
  # time keeps the dialect import graph linear.
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _lower_insert_into,
  )

  stmts = _lower_insert_into(op.inner, ctx)
  return Block(stmts=tuple(stmts))


__all__ = [
  'FanOut',
  'lower_fan_out',
  'materialize_fanout',
]
