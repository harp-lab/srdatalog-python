'''parallel.atomic_ws — atomic work-stealing parallelism strategy.

Per `docs/phase_c_pragma_materialization.md` §4.2, this sub-dialect
houses the work-stealing-specific ops + lowerings + render. It is
introduced by Phase C4 to host the `WorkStealing` typed pragma's
materialization surface (see `pragmas/work_stealing.py`).

The atomic-work-stealing strategy targets unbalanced joins where the
default warp-strided dispatch leaves most warps idle waiting on a few
hot root keys. The runner queues work items in a WCOJTask board;
warps that drain their initial slice steal from neighbours via
atomics on the board's head/tail counters. Kernel functors emit a
per-thread `local_count` increment (count phase) and warp-coalesced
writes (materialize phase) instead of the standard
`<out>.emit_direct()` shape, so the runner's aggregation step can
fold per-thread counts and re-batch writes.

C4 scope (per spec §5, PR row C4): only the kernel-functor-level
WS emit variants migrate to typed pragma + wrap op. The runner-side
scaffolding (WCOJTask queue construction, steal-loop emission) stays
on the legacy `ExecutePipeline.work_stealing: bool` path; A3 will
drop the bool field and the C4 lowering will become the sole driver.

Ops currently registered: none — the only WS-specific op today is
the MIR wrap op `mir.WSScope`, which lives next to `DedupGate` in
`mir/types.py` (matching the C2 placement convention). The dialect's
`lowerings` list carries the `@lowering(target=iir.cf, source=
mir.WSScope)` rule registered in `_register_passes()` below.
'''

from __future__ import annotations

from srdatalog.ir.core import Dialect

DIALECT = Dialect(
  name='parallel.atomic_ws',
  ops=[],
)


def _register_passes() -> None:
  '''Register the `WSScope` lowering + verifier on this dialect.

  Mirrors the `sorted_array._register_passes` pattern (per C2). The
  `pragmas/work_stealing.py` module owns the lowering body and the
  `@pragma_handler(WorkStealing, on=ExecutePipeline)` materialization
  callback; importing it here runs both registrations as side
  effects.
  '''
  from typing import Any

  import srdatalog.ir.mir.types as mir
  from srdatalog.ir.core.passes import lowering, verifier
  from srdatalog.ir.dialects.parallel.atomic_ws.pragmas.work_stealing import (
    lower_ws_scope,
  )

  @lowering(
    DIALECT,
    mir.WSScope,
    consumes=('mir',),
    produces=('iir.cf', 'relation.sorted_array'),
  )
  def lower_mir_ws_scope(op: Any, ctx: Any) -> Any:
    return lower_ws_scope(op, ctx)

  # Verifier scaffolding — WS-specific invariants land incrementally
  # as we encode them (e.g. WSScope.inner must be an InsertInto).
  @verifier(DIALECT)
  def _verify(_prog):
    return []


_register_passes()


__all__ = ['DIALECT']
