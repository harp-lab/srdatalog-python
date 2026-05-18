'''Lowering: `mir.ColumnJoin` (single-source) -> iir.cf — fifth
Wave 2A per-op migration.

Per `docs/phase_b_lowering_dispatcher.md` §4 (per-MIR-op work-unit
table, row B-CJ-single, difficulty `medium`) and §4.1 (per-PR
template): each MIR op type gets one `@lowering(target=iir.cf,
source=mir.X)` rule in its own file under
`dialects/relation/sorted_array/lowerings/`. The registry
registration lives alongside the dialect's other lowerings in
`__init__.py:_register_passes`.

Source-shape split (B-CJ-single vs B-CJ-multi):

`mir.ColumnJoin` carries a `sources: list[ColumnSource]` field
whose length determines the structural shape:

  - `len(sources) == 1` (THIS PR): a single per-source iteration,
    structurally similar to a nested scan. Lowers via
    `_lower_nested_cj_single` in `lowerings/__init__.py` —
    cooperative lane-strided `for` over the source's degree, binds
    the join var via `SaGetValAt`, registers a `SaChildRange` for
    deeper-CJ aliasing, recurses on the body.

  - `len(sources) >= 2` (NEXT PR, B-CJ-multi): prefix-cooperative
    intersection across multiple narrowed handles, plus the
    BG-eligible (`BlockGroup` pragma) root variant. Stays on the
    legacy `_lower_nested_cj_multi` / `_lower_root_cj_multi`
    branches in `lowerings/__init__.py` until B-CJ-multi lands.

The `_should_use_declarative` helper in `lowerings/__init__.py`
encodes the "type AND structural shape" gate so the chain
dispatcher routes single-source through this file and lets
multi-source fall through to the legacy branch — a single
decision point, not two cascading isinstance trees.

Byte-equivalence contract (§4.2): the migrated path produces the
same IIR tree as the legacy `if isinstance(head, mir.ColumnJoin)
and len(head.sources) == 1:` branch in `_lower_inner_chain` on
every fixture that exercises it. The legacy branch delegates to
the same `_lower_nested_cj_single` helper that this file's chain
entry calls — so the two paths are byte-identical by construction.

Chain-aware split (mirrors C5's `lower_tiled_cartesian_in_chain`
pattern + every other Wave 2A row): single-source ColumnJoin
appears as a MIDDLE chain op (no fixture roots one yet —
`_supported_pipeline` requires `len(sources) >= 2` for root CJ).
We therefore expose two entry points:

  - `lower_mir_cj_single_in_chain(op, tail, ctx)`: the real work,
    called from `_lower_inner_chain` (legacy) when
    `_should_use_declarative(head)` returns True. `tail` is the
    rest of the pipeline AFTER `op`.

  - `lower_mir_cj_single(op, ctx)`: the `@lowering`-registered
    stub. Asserts on direct invocation — the framework dispatch
    path is reserved for a future MIR-IIR walker that no longer
    routes through `_lower_inner_chain` and can plumb `tail`
    through.

The split lets the registry pin dialect ownership (`ColumnJoin`
belongs to `relation.sorted_array`) without forcing the chain
dispatcher through the registry today.
'''

from __future__ import annotations

from typing import Any

import srdatalog.ir.mir.types as mir
from srdatalog.ir.core import Op


def lower_mir_cj_single_in_chain(
  op: mir.ColumnJoin,
  tail: list[Any],
  ctx: Any,
) -> Op:
  '''Emit the IIR for a single-source `mir.ColumnJoin` chain head
  with trailing `tail`.

  Mirrors the legacy `if isinstance(head, mir.ColumnJoin) and
  len(head.sources) == 1:` branch in `_lower_inner_chain` byte-for-
  byte by delegating to the legacy `_lower_nested_cj_single`
  helper. The goal of this Wave 2A PR is byte-equivalence with the
  legacy emitter, not a from-scratch rewrite of the single-source
  CJ scaffolding — the helper stays LIVE until Layer 3 cleanup
  deletes both the legacy branch AND the `USE_DECLARATIVE` ratchet.

  Caller-side guard: `_should_use_declarative` in `_lower_inner_chain`
  already enforced `len(op.sources) == 1` before calling here. We
  re-assert below as a structural contract — a future refactor
  that bypasses the helper and calls this entry directly with a
  multi-source CJ would silently misroute, and the assertion is
  the load-bearing safety net.
  '''
  # Deferred import: the chain dispatcher + `_lower_nested_cj_single`
  # live in the package `__init__.py` (the legacy monolith), which
  # imports this module via `_register_passes`. Import-at-call-time
  # keeps the package import graph linear (avoids a circular import
  # between `__init__.py` and this sibling module).
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _lower_nested_cj_single,
  )

  if len(op.sources) != 1:
    raise AssertionError(
      f'lower_mir_cj_single_in_chain: expected exactly one ColumnSource, '
      f'got {len(op.sources)}. Multi-source ColumnJoin is owned by the '
      f'next PR (B-CJ-multi); `_should_use_declarative` should have '
      f'routed it through the legacy `_lower_nested_cj_multi` branch.'
    )

  return _lower_nested_cj_single(op, tail, ctx)


def lower_mir_cj_single(op: mir.ColumnJoin, ctx: Any) -> Op:
  '''Framework-registry stub for `@lowering(target=iir.cf,
  source=mir.ColumnJoin)`. The actual dispatch lives in
  `lowerings._lower_inner_chain` (gated by `_should_use_declarative`),
  which calls `lower_mir_cj_single_in_chain` with the trailing
  chain in scope.

  This stub exists so the dialect's `lowerings` list pins the
  (consumes, produces) contract for `mir.ColumnJoin` — the
  discipline test consults the dialect to verify ownership. Note
  that this single registration covers BOTH the single-source
  (this PR) and multi-source (next PR, B-CJ-multi) shapes; the
  shape-aware dispatch lives in `_should_use_declarative`.

  Calling this stub directly raises a structural assertion — the
  framework path is reserved for future readers who can plumb in
  the missing `tail` (e.g. a refactored MIR-IIR walker that no
  longer routes through `_lower_inner_chain`).
  '''
  raise AssertionError(
    'lower_mir_cj_single: dispatch goes through '
    '`lowerings._lower_inner_chain` -> `lower_mir_cj_single_in_chain` '
    'so the trailing chain is in scope. Direct invocation indicates '
    'a refactor that bypassed the chain dispatch — plumb the `tail` '
    'through and call `lower_mir_cj_single_in_chain` instead.'
  )


__all__ = [
  'lower_mir_cj_single',
  'lower_mir_cj_single_in_chain',
]
