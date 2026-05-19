'''Pragma: `Jaccard` — sparse-similarity gate around an emission.

Trigger: `Rule(...).with_pragma(Jaccard(threshold=0.7))`.

Materialization (this module): wrap each `mir.InsertInto` at the tail
   of an `ExecutePipeline.pipeline` in a `JaccardIndex(inner=...,
   threshold=...)` wrap op during `MirPragmaPass`, then strip the
   `Jaccard` instance from the EP's `pragmas` tuple.

   Mirrors the built-in `DedupHash` -> `DedupGate` pattern in
   `src/srdatalog/ir/dialects/relation/sorted_array/pragmas/
   dedup_hash.py:materialize_dedup_hash`. The difference is that
   the wrap op `JaccardIndex` and the threshold field are defined
   in this external package — neither core MIR types nor the main
   package's pragma registry needed editing.

Lowering: the `@lowering(target=DIALECT, source=JaccardIndex)` rule
   registered in `srdatalog_jaccard.lowerings` handles emission.

DSL-time validation: `Jaccard.__post_init__` raises
   `PragmaConfigError` (subclass of `ValueError`) on out-of-range
   thresholds so the user sees the error at the `.with_pragma(...)`
   keystroke, not deep in `MirPragmaPass`. Per
   `docs/pragma_as_typed_object.md` §2.
'''

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, final

from srdatalog.ir.core import Pragma, pragma_handler
from srdatalog.ir.core.pragma import PragmaConfigError
from srdatalog.ir.mir.types import ExecutePipeline, InsertInto
from srdatalog_jaccard.dialect import JaccardIndex


@final
@dataclass(frozen=True, slots=True)
class Jaccard(Pragma):
  '''Sparse-similarity (Jaccard-coefficient) gate around an emission.

  Triggers `MirPragmaPass` to wrap each `InsertInto` at the tail of
  the carrying `ExecutePipeline.pipeline` in a `JaccardIndex` wrap
  op; the gate's `@lowering` rule emits the IIR shape that the
  runner-side similarity-table machinery expects.

  Fields:

    threshold — minimum similarity for an emission to pass the gate.
                Validated at construction; out-of-range values raise
                `PragmaConfigError` immediately, so DSL users see the
                error at the `.with_pragma(...)` call site rather
                than at compile time.
  '''

  threshold: float = 0.7

  def __post_init__(self) -> None:
    if not (0.0 < self.threshold <= 1.0):
      raise PragmaConfigError(f'Jaccard.threshold must be in (0.0, 1.0]; got {self.threshold!r}')


# -----------------------------------------------------------------------------
# Materialization handler (registered at import time via @pragma_handler)
# -----------------------------------------------------------------------------


def _wrap_inserts_in_jaccard_gate(
  pipeline: list[Any],
  threshold: float,
) -> list[Any]:
  '''Replace every `InsertInto` in `pipeline` with
  `JaccardIndex(inner=that_insert, threshold=threshold)`.

  Mirrors the C2 `_wrap_inserts_in_dedup_gate` pattern: walk
  `pipeline`, swap each `InsertInto` for the wrap op, leave non-
  insert ops unchanged. Non-leaf operations (Scan, ColumnJoin,
  Cartesian, etc.) pass through untouched — Jaccard is a per-
  emission decoration, structurally an MIR-level concern that the
  MIR -> IIR lowering then translates.
  '''
  wrapped: list[Any] = []
  for child in pipeline:
    if isinstance(child, InsertInto):
      wrapped.append(JaccardIndex(inner=child, threshold=threshold))
    else:
      wrapped.append(child)
  return wrapped


@pragma_handler(Jaccard, on=ExecutePipeline)
def materialize_jaccard(
  op: Any,  # ExecutePipeline at runtime — typed as Any to match the registry's
  # Callable[[Any, Pragma, PragmaCtx], Any] signature (see
  # core/pragma.py:PragmaRegistration). Body re-narrows via the
  # typed `op.pragmas` / `op.pipeline` accesses below.
  pragma: Any,  # Jaccard at runtime; same Any reason.
  ctx: Any,  # PragmaCtx; unused for this single-field pragma.
) -> Any:
  '''Materialize `Jaccard` into a `JaccardIndex` wrap around each
  trailing `InsertInto` in `op.pipeline`.

  Returns a new `ExecutePipeline` with:
    - `pipeline` rewritten so every `InsertInto` is replaced by
      `JaccardIndex(inner=that_insert, threshold=pragma.threshold)`,
    - `pragmas` filtered to drop the consumed `Jaccard` instance
      (per the `MirPragmaPass` post-flight invariant — see
      `docs/pragma_as_typed_object.md` §3).

  Idempotency: the EP carries at most one `Jaccard` per
  `with_pragma(Jaccard(...))` call; the filter is defensive against
  stray duplicates.
  '''
  new_pragmas = tuple(p for p in op.pragmas if not isinstance(p, Jaccard))
  new_pipeline = _wrap_inserts_in_jaccard_gate(list(op.pipeline), pragma.threshold)
  return dataclasses.replace(op, pipeline=new_pipeline, pragmas=new_pragmas)


__all__ = ['Jaccard', 'materialize_jaccard']
