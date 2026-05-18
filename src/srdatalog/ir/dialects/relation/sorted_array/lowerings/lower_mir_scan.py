'''Lowering: `mir.Scan` -> iir.cf — third Wave 2A per-op migration.

Per `docs/phase_b_lowering_dispatcher.md` §4 (per-MIR-op work-unit
table, row B-Scan, difficulty `medium`) and §4.1 (per-PR template):
each MIR op type gets one `@lowering(target=iir.cf, source=mir.X)`
rule in its own file under
`dialects/relation/sorted_array/lowerings/`. The registry
registration lives alongside the dialect's other lowerings in
`__init__.py:_register_passes`.

Byte-equivalence contract (§4.2): the migrated op produces the same
IIR tree as the legacy `if isinstance(head, mir.Scan):` branch
inside `lower_scan_pipeline` on every fixture that exercises it.

Root-position split (Scan is the FIRST head op of the pipeline, not
a middle chain op like B-Filter / B-ConstantBind). The legacy
dispatch lives in `lower_scan_pipeline` (not `_lower_inner_chain`):

    if isinstance(head, mir.Scan):
      return _lower_root_scan(head, rest, ctx)

`mir.Scan` never appears mid-chain — neither `_supported_pipeline`
nor `_lower_inner_chain` accepts a Scan in any non-head position.
We therefore expose two entry points whose names mirror the
B-Filter / B-ConstantBind split but reflect the root nature of the
op:

  - `lower_mir_scan_in_chain(op, rest, ctx)`: the real work,
    called from `lower_scan_pipeline` when
    `type(head) in USE_DECLARATIVE`. `rest` is the trailing
    pipeline (middle ops + InsertIntos), same shape that the
    legacy `_lower_root_scan` received.
  - `lower_mir_scan(op, ctx)`: the `@lowering`-registered stub.
    Asserts on direct invocation — the framework dispatch path is
    reserved for a future MIR-IIR walker that no longer routes
    through `lower_scan_pipeline` and can plumb `rest` through.

We deliberately kept the `_in_chain` suffix (rather than coining a
new `_root` / `_pipeline` name) so all Wave 2A per-op files share
one naming convention. The shape of the trailing argument is
op-dependent (`tail` for middle ops, `rest` for root ops); the
suffix only signals "there is one".

The split lets the registry pin dialect ownership (`Scan` belongs
to `relation.sorted_array`) without forcing the pipeline dispatcher
through the registry today.
'''

from __future__ import annotations

from typing import Any

import srdatalog.ir.mir.types as mir
from srdatalog.ir.core import Op


def lower_mir_scan_in_chain(
  op: mir.Scan,
  rest: list[Any],
  ctx: Any,
) -> Op:
  '''Emit the IIR for a root `mir.Scan` with trailing pipeline `rest`.

  Mirrors the legacy `if isinstance(head, mir.Scan): return
  _lower_root_scan(head, rest, ctx)` dispatch byte-for-byte by
  delegating to the legacy helper. The goal of this Wave 2A PR is
  byte-equivalence with the legacy emitter, not a from-scratch
  rewrite of the (M1+M2) scan-root scaffolding — the helper stays
  LIVE until Layer 3 cleanup deletes both the legacy branch AND
  the `USE_DECLARATIVE` ratchet.

  `rest` is the trailing pipeline AFTER the Scan: middle ops
  (Filter / ConstantBind / Negation / CartesianJoin /
  TiledCartesian) followed by one-or-more trailing InsertIntos.
  The legacy `_lower_root_scan` walks it via `_lower_inner_chain`
  for the body and emits the root-scan scaffold (handle bind,
  validity check, degree bind, ParallelFor grid-stride loop).
  '''
  # Deferred import: the pipeline dispatcher + `_lower_root_scan`
  # live in the package `__init__.py` (the legacy monolith), which
  # imports this module via `_register_passes`. Import-at-call-time
  # keeps the package import graph linear (avoids a circular import
  # between `__init__.py` and this sibling module).
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _lower_root_scan,
  )

  return _lower_root_scan(op, rest, ctx)


def lower_mir_scan(op: mir.Scan, ctx: Any) -> Op:
  '''Framework-registry stub for `@lowering(target=iir.cf,
  source=mir.Scan)`. The actual dispatch lives in
  `lowerings.lower_scan_pipeline` (via the `USE_DECLARATIVE`
  ratchet), which calls `lower_mir_scan_in_chain` with the
  trailing pipeline in scope.

  This stub exists so the dialect's `lowerings` list pins the
  (consumes, produces) contract for `mir.Scan` — the discipline
  test consults the dialect to verify ownership.

  Calling this stub directly raises a structural assertion — the
  framework path is reserved for future readers who can plumb in
  the missing `rest` (e.g. a refactored MIR-IIR walker that no
  longer routes through `lower_scan_pipeline` and can construct
  the body chain from the enclosing ExecutePipeline).
  '''
  raise AssertionError(
    'lower_mir_scan: dispatch goes through '
    '`lowerings.lower_scan_pipeline` -> `lower_mir_scan_in_chain` '
    'so the trailing pipeline is in scope. Direct invocation '
    'indicates a refactor that bypassed the pipeline dispatch — '
    'plumb the `rest` through and call `lower_mir_scan_in_chain` '
    'instead.'
  )


__all__ = [
  'lower_mir_scan',
  'lower_mir_scan_in_chain',
]
