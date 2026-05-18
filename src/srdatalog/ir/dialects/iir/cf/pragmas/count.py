'''Pragma: `count` — count-only rule (no materialize phase output).

Trigger: `Rule(...).with_pragma(Count())` (preferred typed form)
   or the legacy `Rule.with_count()` / `Rule(count=True)` shape
   (back-compat, slated for removal in A3).

Materialization (this module): wrap each `ExecutePipeline`'s
   `pipeline` body in a `mir.CountPhase` at `MirPragmaPass` time, then
   strip the `Count` instance from the EP's `pragmas` tuple.

Lowering (this module): a `@lowering(target=iir.cf, source=mir.
   CountPhase)` rule (registered via the parent dialect's
   `_register_passes`) emits an `iir.cf.Phase(mode='C', body=...)`
   form per spec § 4.3. The legacy count-phase code path (driven by
   `ctx.is_counting` + `ep.count`) remains the byte-equivalence
   anchor for the C6 transition; the wrap op exists so
   `MirPragmaPass` has a typed insertion target.

Spec: `docs/phase_c_pragma_materialization.md` §4.3, §5 (C6 PR row),
§5.1 (per-PR acceptance gate); `docs/pragma_as_typed_object.md` §2
(Pragma base class), §3 (`@pragma_handler` decorator).
'''

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, final

from srdatalog.ir.core import Op, Pragma, pragma_handler
from srdatalog.ir.dialects.iir.cf.ops import Block as IirBlock
from srdatalog.ir.dialects.iir.cf.ops import Phase
from srdatalog.ir.mir.types import CountPhase, ExecutePipeline


@final
@dataclass(frozen=True, slots=True)
class Count(Pragma):
  '''Mark an `ExecutePipeline` as count-only: emit just the count
  phase, discard the materialize kernel output.

  Triggers `MirPragmaPass` to wrap the EP's `pipeline` body in a
  `mir.CountPhase` wrap op; the wrap op's `@lowering` rule emits the
  `iir.cf.Phase(mode='C', body=...)` form the IIR layer already
  understands. The legacy `ep.count: bool` field stays for back-
  compat (drives `complete_runner.py`'s materialize-kernel discard);
  the typed pragma is the canonical declarative surface going
  forward.

  No fields today — the legacy `ep.count` is a flat bool and the
  count-only semantics are uniform. Structured config (e.g., a
  threshold or count-precision knob) is plausible later but out of
  scope for C6.
  '''


# -----------------------------------------------------------------------------
# Materialization handler (registered at import time via @pragma_handler)
# -----------------------------------------------------------------------------


def _wrap_pipeline_in_count_phase(pipeline: list[Any]) -> CountPhase:
  '''Bundle the EP's pipeline ops under a single `CountPhase` wrap.

  The legacy count-phase semantic spans the WHOLE pipeline body
  (every emission inside it goes through `output_ctx.emit_direct()`
  with no args, count-as-product short-circuits R1 apply, etc.).
  We mirror that by wrapping the pipeline contents in one
  `CountPhase` whose `inner` is a synthetic `iir.cf.Block` of the
  pipeline ops — the lowering then walks that block under the
  count-phase scope.

  Using `iir.cf.Block` here (rather than an MIR-level container)
  matches the IIR target shape `Phase(C, body)` that the lowering
  emits: the wrap op's inner is already-block-shaped, so the
  lowering can pass it through without re-wrapping.
  '''
  return CountPhase(inner=IirBlock(stmts=tuple(pipeline)))


@pragma_handler(Count, on=ExecutePipeline)
def materialize_count(
  op: Any,  # ExecutePipeline at runtime — typed as Any to match the registry's
  # Callable[[Any, Pragma, PragmaCtx], Any] signature (see
  # core/pragma.py:PragmaRegistration). Body re-narrows via the typed
  # `op.pragmas` / `op.count` accesses below.
  pragma: Any,  # Count at runtime; same Any reason.
  ctx: Any,  # PragmaCtx, unused for this no-config pragma.
) -> Any:
  '''Materialize `Count` into a `CountPhase` wrap around the EP's
  pipeline body.

  Returns a new `ExecutePipeline` with:
    - `pipeline` rewritten to a single-element list `[CountPhase(
      inner=Block(stmts=<original pipeline>))]` — **except** in the
      C6 dual-write transition state (see below),
    - `pragmas` filtered to drop the consumed `Count` instance
      (per the `MirPragmaPass` post-flight invariant; see
      `docs/pragma_as_typed_object.md` §3).

  C6 dual-write transition: the DSL `Rule.with_pragma(Count())` sets
  BOTH the typed pragma AND the legacy `count: bool` field on the
  resulting `ExecutePipeline` (via the `Rule.count` shadow), so the
  legacy emitter (`compile_kernel_body` -> `is_counting` flag +
  `complete_runner.py`'s `ep.count` check) keeps producing byte-
  equivalent CUDA without seeing the new `CountPhase` wrap op
  (which the monolith doesn't know how to traverse). When
  `ep.count` is True we therefore SKIP the wrap step: stripping
  the pragma is enough to satisfy `MirPragmaPass`'s post-flight
  invariant, and the bool field continues to drive emission via
  the legacy path.

  When `ep.count` is False (the post-A3 pure-typed state, or test
  fixtures that exercise only the typed surface), the handler
  produces the wrap op and the registered `@lowering(target=iir.cf,
  source=CountPhase)` rule lowers it to `iir.cf.Phase(mode='C',
  body=...)` per spec §4.3.

  Idempotency: the EP carries at most one `Count` (the DSL constructs
  at most one per `with_pragma(Count())` call). We filter by
  `isinstance(p, Count)` to be defensive — any stray duplicates get
  removed together.
  '''
  new_pragmas = tuple(p for p in op.pragmas if not isinstance(p, Count))
  # Dual-write transition: skip the wrap if the legacy bool is set
  # (the legacy lowering will handle emission). See docstring above.
  if op.count:
    return dataclasses.replace(op, pragmas=new_pragmas)
  wrap = _wrap_pipeline_in_count_phase(list(op.pipeline))
  return dataclasses.replace(op, pipeline=[wrap], pragmas=new_pragmas)


# -----------------------------------------------------------------------------
# CountPhase lowering (registered by iir.cf `_register_passes`)
# -----------------------------------------------------------------------------


def lower_count_phase(op: CountPhase, ctx: Any) -> Op:
  '''Emit the IIR for `CountPhase(inner=<body>)`.

  Returns `iir.cf.Phase(mode='C', body=<inner>)` per spec § 4.3 —
  the same scope shape `is_counting` produces today via the
  `ctx.is_counting=True` path through `_lower_insert_into` and
  friends. The `iir.cf.Phase` op already exists in `iir/cf/ops.py`;
  this lowering threads the wrap op through it.

  `ctx` is the `LoweringCtx` from `relation.sorted_array.lowerings`
  in the C6 transition (the only target we lower MIR -> IIR
  through today). It's typed as `Any` so this module doesn't depend
  on a specific dialect's lowering context — the spec's IIR shape
  is dialect-agnostic.

  The body is the pre-wrapped `iir.cf.Block` carried by the wrap
  op; passed through unchanged. The legacy code path doesn't see
  this wrap (the materialize handler's dual-write short-circuit
  keeps the typed wrap out of the pipeline when `ep.count` is
  True), so byte-equivalence is preserved by construction.
  '''
  # Deferred imports: the typing of `ctx` is intentionally loose
  # (see docstring); no per-ctx state is read here in the C6
  # transition. A future C6 follow-up may thread `ctx.is_counting=
  # True` through the body lowering, but for now the `iir.cf.Phase`
  # is the only thing the spec requires.
  return Phase(mode='C', body=op.inner)


__all__ = [
  'Count',
  'lower_count_phase',
  'materialize_count',
]
