'''Lowering: `mir.InsertInto` -> iir.cf — fourth Wave 2A per-op
migration.

Per `docs/phase_b_lowering_dispatcher.md` §4 (per-MIR-op work-unit
table, row B-InsertInto) and §4.1 (per-PR template): each MIR op
type gets one `@lowering(target=iir.cf, source=mir.X)` rule in its
own file under `dialects/relation/sorted_array/lowerings/`. The
registry registration lives alongside the dialect's other lowerings
in `__init__.py:_register_passes`.

Byte-equivalence contract (§4.2): the migrated op produces the same
IIR tree as the legacy `if isinstance(head, mir.InsertInto):` head
branch inside `_lower_inner_chain` on every fixture that exercises
it. Because `mir.InsertInto` is the terminal op in every
`ExecutePipeline.pipeline`, this lowering is exercised by every
emitting kernel in the test fixture set — the byte-equivalence
harness is the strongest signal.

Chain-aware split (mirrors C5's `lower_tiled_cartesian_in_chain`
pattern + B-Filter / B-ConstantBind): `mir.InsertInto` appears as
the chain head ONLY when the entire trailing slice is a run of
sibling InsertIntos (multi-head rules emit several outputs from the
same body). We therefore expose two entry points:

  - `lower_mir_insert_into_in_chain(op, tail, ctx)`: the real work,
    called from `_lower_inner_chain` (legacy) when
    `type(head) in USE_DECLARATIVE`. `tail` is the rest of the
    pipeline AFTER `op`; combined they form the trailing
    InsertInto-only run that the legacy `_trailing_inserts` /
    `_lower_insert_into` pair already knows how to emit.

  - `lower_mir_insert_into(op, ctx)`: the `@lowering`-registered
    stub. Asserts on direct invocation — the framework dispatch
    path is reserved for a future MIR-IIR walker that no longer
    routes through `_lower_inner_chain` and can plumb `tail`
    through.

Coexistence with typed-pragma wrap ops (C2 / C4 / C6):

`mir.DedupGate(inner=InsertInto)`, `mir.WSScope(inner=InsertInto)`,
and `mir.FanOut(inner=InsertInto)` are typed pragma wrap ops that
each wrap a single `InsertInto`. Their `@lowering` rules
(registered in `pragmas/dedup_hash.py`, `pragmas/work_stealing.py`,
`pragmas/fanout.py`) call the legacy `_lower_insert_into` helper
with mutated context flags. Those wrap ops are NOT in
`USE_DECLARATIVE` and they live OUTSIDE the chain dispatcher (the
wrap op replaces the InsertInto in `pipeline` during
`MirPragmaPass`; the chain dispatcher sees the wrap op, not the
inner InsertInto). The B-InsertInto migration is therefore
orthogonal to C2/C4/C6 — both paths delegate to the same
`_lower_insert_into` helper, so byte-equivalence holds by
construction.
'''

from __future__ import annotations

from typing import Any

import srdatalog.ir.mir.types as mir
from srdatalog.ir.core import Op
from srdatalog.ir.dialects.iir.cf import Block


def lower_mir_insert_into_in_chain(
  op: mir.InsertInto,
  tail: list[Any],
  ctx: Any,
) -> Op:
  '''Emit the IIR for a `mir.InsertInto` chain head with trailing
  `tail`.

  Mirrors the legacy `if isinstance(head, mir.InsertInto):` branch
  in `_lower_inner_chain` byte-for-byte:

    - All ops in `[op] + tail` must be `InsertInto` (the trailing
      multi-head run); a non-InsertInto in the tail is a structural
      error — same `ValueError` that the legacy branch raises.
    - When `ctx.tiled_cartesian_valid_var` is set, the full run is
      rendered as a single `TiledBallotBlock` (the ballot-coalesced
      write variant used by the tiled-Cartesian 2D body); delegates
      to the existing `_lower_tiled_ballot_block` helper.
    - Otherwise each InsertInto is lowered via the legacy
      `_lower_insert_into` helper (which encodes the
      `is_counting` / `dedup_hash` / `ws_enabled` /
      `ws_cartesian_valid_var` branches) and the resulting statement
      runs are concatenated into a single `Block`.

  Delegation back into the legacy helper is what guarantees
  byte-equivalence with the C2/C4/C6 typed-pragma wrap op lowerings
  (`lower_dedup_gate`, `lower_ws_scope`, `lower_fan_out`) — all
  paths funnel through `_lower_insert_into` so the rendered text
  matches across the four entry points.
  '''
  # Deferred import: the chain dispatcher + helpers live in the
  # package `__init__.py` (the legacy monolith), which imports this
  # module via `_register_passes`. Import-at-call-time keeps the
  # package import graph linear (avoids a circular import between
  # `__init__.py` and this sibling module).
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _lower_insert_into,
    _lower_tiled_ballot_block,
  )

  inserts: list[mir.InsertInto] = [op]
  for child in tail:
    if not isinstance(child, mir.InsertInto):
      raise ValueError(
        f'lower_mir_insert_into_in_chain: expected pure InsertInto tail at '
        f'this point, got {[type(o).__name__ for o in [op, *tail]]}'
      )
    inserts.append(child)

  # Tiled-Cartesian ballot variant: render the trailing InsertInto
  # run as a single TiledBallotBlock that owns the ballot setup +
  # per-output write at once. Replaces the legacy stateful
  # `tiled_cartesian_ballot_done` flag. Same branch as the legacy
  # head==InsertInto path in `_lower_inner_chain`.
  if ctx.tiled_cartesian_valid_var:
    return _lower_tiled_ballot_block(inserts, ctx)

  stmts: list[Op] = []
  for ins in inserts:
    stmts.extend(_lower_insert_into(ins, ctx))
  return Block(stmts=tuple(stmts))


def lower_mir_insert_into(op: mir.InsertInto, ctx: Any) -> Op:
  '''Framework-registry stub for `@lowering(target=iir.cf,
  source=mir.InsertInto)`. The actual dispatch lives in
  `lowerings._lower_inner_chain` (via the `USE_DECLARATIVE`
  ratchet), which calls `lower_mir_insert_into_in_chain` with the
  trailing chain in scope.

  This stub exists so the dialect's `lowerings` list pins the
  (consumes, produces) contract for `mir.InsertInto` — the
  discipline test consults the dialect to verify ownership.

  Calling this stub directly raises a structural assertion — the
  framework path is reserved for future readers who can plumb in
  the missing `tail` (e.g. a refactored MIR-IIR walker that no
  longer routes through `_lower_inner_chain`).
  '''
  raise AssertionError(
    'lower_mir_insert_into: dispatch goes through '
    '`lowerings._lower_inner_chain` -> `lower_mir_insert_into_in_chain` '
    'so the trailing multi-head InsertInto run is in scope. Direct '
    'invocation indicates a refactor that bypassed the chain dispatch '
    '— plumb the `tail` through and call '
    '`lower_mir_insert_into_in_chain` instead.'
  )


__all__ = [
  'lower_mir_insert_into',
  'lower_mir_insert_into_in_chain',
]
