'''Lowering: `mir.ColumnJoin` (multi-source) -> iir.cf — sixth
Wave 2A per-op migration.

Per `docs/phase_b_lowering_dispatcher.md` §4 (per-MIR-op work-unit
table, row B-CJ-multi, difficulty `hard`, migration order 6) and
§4.1 (per-PR template): each MIR op type gets one `@lowering(target=
iir.cf, source=mir.X)` rule in its own file under
`dialects/relation/sorted_array/lowerings/`. The dialect-level
`@lowering(...)` registration for `mir.ColumnJoin` lives alongside
the other lowerings in `__init__.py:_register_passes`; this file is
referenced from the chain dispatcher (`_lower_inner_chain`) and the
root dispatcher (`lower_scan_pipeline`) in `lowerings/__init__.py`.

Source-shape split (B-CJ-single vs B-CJ-multi):

`mir.ColumnJoin` carries a `sources: list[ColumnSource]` field
whose length determines the structural shape:

  - `len(sources) == 1` (B-CJ-single, PR #59): handled by
    `lower_mir_cj_single_in_chain`. Always a middle-of-chain op
    (`_supported_pipeline` does not accept it as a root).

  - `len(sources) >= 2` (THIS PR, B-CJ-multi): both root-position
    and middle-of-chain. Root-position is the production-load-
    bearing case (every TC-like fixture roots a multi-source
    intersection); middle-of-chain handles nested intersections
    (e.g. CJ-under-CJ in the deepest TC shape).

The `_should_use_declarative` helper in `lowerings/__init__.py`,
after this PR, no longer gates by source count for `mir.ColumnJoin`:
ALL ColumnJoin variants route through the new per-op path. The
single dialect-level `@lowering(mir.ColumnJoin, ...)` registration
covers both shapes for ownership / discipline purposes.

Byte-equivalence contract (§4.2): the migrated path produces the
same IIR tree as the legacy `if isinstance(head, mir.ColumnJoin) and
len(head.sources) >= 2:` branches (root + nested) by delegating to
the same `_lower_root_cj_multi` / `_lower_nested_cj_multi` helpers
the legacy branches called directly. Layer 3 cleanup deletes those
helpers once the chain-aware variants here own the emission outright.

BlockGroup coexistence — the load-bearing constraint of this PR:

When the EP carries `ep.block_group=True` (legacy dual-write path)
or its `pipeline[0]` is a `BlockGroupRoot` wrap op (typed-pragma
path, post C3 materialization), the BG variant of the lowering
(`_lower_root_cj_bg`) is the correct emission target — NOT the
standard `_lower_root_cj_multi`. The dispatch ordering in
`lower_scan_pipeline` keeps the BG path intact:

  1. `lower_scan_pipeline` first checks for `BlockGroupRoot` head and
     dispatches to `_lower_root_cj_bg` directly. The check sits
     BEFORE the `_should_use_declarative` branch — wrap ops never
     reach the new path.
  2. For non-wrap pipelines (the `ep.block_group=True` dual-write
     case), `_should_use_declarative` accepts the bare multi-source
     `ColumnJoin` and routes through `lower_mir_cj_multi_root`. The
     latter delegates to `_lower_root_cj_multi`, whose first
     statement is `if ctx.bg_enabled: return _lower_root_cj_bg(...)`.
     `compile_kernel_body` threads `bg_enabled` from `ep.block_group`,
     so the BG variant fires before any non-BG scaffolding runs.

Both delegation paths preserve byte-equivalence by construction:
the new entries call the same legacy helpers the legacy `if
isinstance` branches called, with the same `ctx` and `rest`.

Chain-aware split (mirrors B-CJ-single's `lower_mir_cj_single_in_chain`
pattern + every other Wave 2A row that needs the trailing chain):

  - `lower_mir_cj_multi_root(op, rest, ctx)`: root-position entry,
    called from `lower_scan_pipeline`. Delegates to
    `_lower_root_cj_multi`.

  - `lower_mir_cj_multi_in_chain(op, tail, ctx)`: middle-of-chain
    entry, called from `_lower_inner_chain`. Delegates to
    `_lower_nested_cj_multi`.

  - `lower_mir_cj_multi(op, ctx)`: the `@lowering`-registered stub
    (sibling of `lower_mir_cj_single` in the registry; both share
    the single dialect-level `mir.ColumnJoin` registration via the
    shape-aware dispatcher `_dispatch_mir_cj_registered` in
    `__init__.py`). Asserts on direct invocation because the chain-
    aware variants need either `rest` or `tail` from the enclosing
    dispatcher — same split rationale as every other chain-aware
    Wave 2A entry.
'''

from __future__ import annotations

from typing import Any

import srdatalog.ir.mir.types as mir
from srdatalog.ir.core import Op


def lower_mir_cj_multi_root(
  op: mir.ColumnJoin,
  rest: list[Any],
  ctx: Any,
) -> Op:
  '''Emit the IIR for a root-position multi-source `mir.ColumnJoin`
  with trailing `rest`.

  Mirrors the legacy `if isinstance(head, mir.ColumnJoin):` branch
  in `lower_scan_pipeline` byte-for-byte by delegating to
  `_lower_root_cj_multi`. The helper internally re-dispatches to
  `_lower_root_cj_bg` when `ctx.bg_enabled` is set — that BG entry
  is the load-bearing dual-write anchor (see module docstring).

  Caller-side guard: `_should_use_declarative` in `lower_scan_pipeline`
  already filtered through `USE_DECLARATIVE` membership. We re-assert
  `len(op.sources) >= 2` below as the structural contract: a future
  refactor that bypassed the helper and called this entry with a
  single-source CJ would silently misroute (single-source CJ has its
  own root-emission scaffolding — though no fixture exercises it).
  '''
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _lower_root_cj_multi,
  )

  if len(op.sources) < 2:
    raise AssertionError(
      f'lower_mir_cj_multi_root: expected at least two ColumnSources, '
      f'got {len(op.sources)}. Single-source ColumnJoin is owned by '
      f'B-CJ-single (lower_mir_cj_single) — `_should_use_declarative` '
      f'should have routed it through the single-source entry.'
    )

  return _lower_root_cj_multi(op, rest, ctx)


def lower_mir_cj_multi_in_chain(
  op: mir.ColumnJoin,
  tail: list[Any],
  ctx: Any,
) -> Op:
  '''Emit the IIR for a middle-of-chain multi-source `mir.ColumnJoin`
  with trailing `tail`.

  Mirrors the legacy `if isinstance(head, mir.ColumnJoin) and
  len(head.sources) >= 2:` branch in `_lower_inner_chain` byte-for-
  byte by delegating to `_lower_nested_cj_multi`. The legacy helper
  stays LIVE until Layer 3 cleanup deletes both the legacy branch
  AND `USE_DECLARATIVE`.

  Caller-side guard: `_should_use_declarative` in `_lower_inner_chain`
  already filtered through `USE_DECLARATIVE` membership. We re-assert
  `len(op.sources) >= 2` below as a structural contract — see
  `lower_mir_cj_multi_root` for the same rationale.
  '''
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _lower_nested_cj_multi,
  )

  if len(op.sources) < 2:
    raise AssertionError(
      f'lower_mir_cj_multi_in_chain: expected at least two ColumnSources, '
      f'got {len(op.sources)}. Single-source ColumnJoin is owned by '
      f'B-CJ-single (lower_mir_cj_single_in_chain) — '
      f'`_should_use_declarative` should have routed it through the '
      f'single-source entry.'
    )

  return _lower_nested_cj_multi(op, tail, ctx)


def lower_mir_cj_multi(op: mir.ColumnJoin, ctx: Any) -> Op:
  '''Framework-registry stub for the multi-source half of
  `@lowering(target=iir.cf, source=mir.ColumnJoin)`. The actual
  dispatch lives in `lowerings._lower_inner_chain` (mid-chain) and
  `lowerings.lower_scan_pipeline` (root) — both call the
  corresponding chain-aware variant with the surrounding pipeline
  slice in scope.

  Note: the dialect-level `@lowering(mir.ColumnJoin, ...)`
  registration in `relation.sorted_array.__init__._register_passes`
  is shared between this file and `lower_mir_cj_single.py`. The
  registered dispatcher inspects `len(op.sources)` and forwards to
  this stub OR `lower_mir_cj_single` accordingly. Both stubs assert
  because direct registry invocation has no access to the trailing
  pipeline `rest` / `tail` that the chain-aware variants need.

  Calling this stub directly raises a structural assertion — the
  framework path is reserved for future readers who can plumb in
  the missing `tail` / `rest` (e.g. a refactored MIR-IIR walker that
  no longer routes through `_lower_inner_chain`).
  '''
  raise AssertionError(
    'lower_mir_cj_multi: dispatch goes through '
    '`lowerings._lower_inner_chain` -> `lower_mir_cj_multi_in_chain` '
    '(mid-chain) or `lowerings.lower_scan_pipeline` -> '
    '`lower_mir_cj_multi_root` (root) so the trailing pipeline is '
    'in scope. Direct invocation indicates a refactor that bypassed '
    'the dispatcher — plumb the `tail` / `rest` through and call the '
    'chain-aware variant instead.'
  )


__all__ = [
  'lower_mir_cj_multi',
  'lower_mir_cj_multi_in_chain',
  'lower_mir_cj_multi_root',
]
