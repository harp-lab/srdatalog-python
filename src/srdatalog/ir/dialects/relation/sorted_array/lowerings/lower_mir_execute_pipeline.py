'''Lowering: `mir.ExecutePipeline` -> iir.cf — Wave 2A B-ExecutePipeline
migration (per `docs/phase_b_lowering_dispatcher.md` §4, row
B-ExecutePipeline, order 10, difficulty `medium`).

Per `docs/phase_b_lowering_dispatcher.md` §4 (per-MIR-op work-unit
table) and §4.1 (per-PR template): each MIR op type gets one
`@lowering(target=iir.cf, source=mir.X)` rule in its own file under
`dialects/relation/sorted_array/lowerings/`. The registry registration
lives alongside the dialect's other lowerings in
`__init__.py:_register_passes`.

Why ExecutePipeline is "medium" (per the migration plan):

  - It is the TOP-level structural wrap that opens every kernel body.
    Unlike Filter / ConstantBind / Scan / Negation (which live mid-
    chain in `_lower_inner_chain` and need the trailing `tail` from
    the chain dispatcher), `mir.ExecutePipeline` is a self-contained
    op: its `.pipeline` field carries the full op sequence to lower.
    No chain-aware split is needed — direct invocation is the
    canonical entry point.
  - The body delegates straight into the legacy `lower_scan_pipeline`
    helper so byte-equivalence holds by construction. The migration
    moves dispatch ownership (the `@lowering` registration) into its
    own file without duplicating the body. This mirrors the
    delegation pattern used by B-InsertInto / B-Negation /
    B-Aggregate and the C2 / C4 / C6 wrap-op lowerings.

Byte-equivalence contract (§4.2): the migrated entry produces the
same IIR tree as the inline closure that previously lived in
`__init__.py:_register_passes` on every fixture that exercises it —
which is every fixture, because EP is the top-level wrap of every
kernel body.

USE_DECLARATIVE: per the spec note in the work-unit table for
B-ExecutePipeline, EP is NOT added to `USE_DECLARATIVE`. That set
governs CHAIN dispatch inside `_lower_inner_chain` / root dispatch
inside `lower_scan_pipeline`; EP sits OUTSIDE both — it wraps the
whole pipeline. The `@lowering` registration is sufficient to pin
dialect ownership (the discipline test consults the dialect to
verify `mir.ExecutePipeline` has exactly one registered lowering).

Caller posture during the migration:

  - `LowerKernelBodyShim` (in `srdatalog.ir.default_pipelines`)
    still calls `lower_scan_pipeline(state.ep.pipeline, lower_ctx)`
    directly. That direct call is byte-equivalent to going through
    the registered `@lowering` (this module's entry is a 1-line
    delegate to the same function). The registry-driven dispatch
    path is a follow-up — see the TODO at
    `LowerKernelBodyShim._fn` — that requires plumbing the
    Compiler reference through `KernelCtx` (F3's `LowerCtx`
    already carries one; `KernelCtx` does not yet).
  - `compile_pipeline(ep)` and `compile_kernel_body(ep, ...)` route
    through `LowerKernelBodyShim` (post-F5.3) so they inherit the
    same direct-call path. The new `@lowering` entry here is the
    discoverable canonical surface; direct callers and the shim are
    free to invoke it OR the underlying `lower_scan_pipeline`
    helper interchangeably.
'''

from __future__ import annotations

from typing import Any

import srdatalog.ir.mir.types as mir
from srdatalog.ir.core import Op


def lower_mir_execute_pipeline(op: mir.ExecutePipeline, ctx: Any) -> Op:
  '''Lower a `mir.ExecutePipeline` to IIR.

  Canonical entry for the dialect's MIR->IIR mapping of the top-level
  structural wrap. Delegates to `lower_scan_pipeline` so byte-
  equivalence holds against the legacy closure that previously lived
  in `__init__.py:_register_passes`. The delegation pattern matches
  B-InsertInto / B-Negation / B-Aggregate.

  Pragma coexistence: `op.dedup_hash`, `op.work_stealing`, etc. (and
  the new typed `op.pragmas` tuple) are consumed by `MirPragmaPass`
  BEFORE this entry is reached — by the time EP gets here, any
  pragma-materialized wrap ops (`DedupGate`, `WSScope`, `mir.FanOut`,
  `BlockGroupRoot`, `mir.TiledCartesian`) live INSIDE `op.pipeline`
  and dispatch through their own `@lowering` registrations from
  inside `lower_scan_pipeline`.
  '''
  # Deferred import: the body helper lives in the package
  # `__init__.py` (the legacy monolith), which imports this module
  # via `_register_passes`. Import-at-call-time keeps the package
  # import graph linear (avoids a circular import between
  # `__init__.py` and this sibling module).
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    lower_scan_pipeline,
  )

  return lower_scan_pipeline(op.pipeline, ctx)


__all__ = [
  'lower_mir_execute_pipeline',
]
