'''Lowering: `mir.Filter` -> iir.cf — first Wave 2A per-op migration.

Per `docs/phase_b_lowering_dispatcher.md` §4 (per-MIR-op work-unit
table, row B-Filter) and §4.1 (per-PR template): each MIR op type
gets one `@lowering(target=iir.cf, source=mir.X)` rule in its own
file under `dialects/relation/sorted_array/lowerings/`. The registry
registration lives alongside the dialect's other lowerings in
`__init__.py:_register_passes`.

Byte-equivalence contract (§4.2): the migrated op produces the same
IIR tree as the legacy `if isinstance(head, mir.Filter):` branch
inside `_lower_inner_chain` on every fixture that exercises it.

Chain-aware split (mirrors C5's `lower_tiled_cartesian_in_chain`
pattern): `mir.Filter` only appears as a middle op in the inner
chain — its emission depends on the trailing `tail` (which the
chain dispatcher provides) for the body. We therefore expose two
entry points:

  - `lower_mir_filter_in_chain(op, tail, ctx)`: the real work,
    called from `_lower_inner_chain` (legacy) when
    `type(head) in USE_DECLARATIVE`.
  - `lower_mir_filter(op, ctx)`: the `@lowering`-registered stub.
    Asserts on direct invocation — the framework dispatch path is
    reserved for a future MIR-IIR walker that no longer routes
    through `_lower_inner_chain` and can plumb `tail` through.

The split lets the registry pin dialect ownership (`Filter` belongs
to `relation.sorted_array`) without forcing the chain dispatcher
through the registry today.
'''

from __future__ import annotations

from typing import Any

import srdatalog.ir.mir.types as mir
from srdatalog.ir.core import Op
from srdatalog.ir.dialects.iir.cf import Block, If, RawString


def lower_mir_filter_in_chain(
  op: mir.Filter,
  tail: list[Any],
  ctx: Any,
) -> Op:
  '''Emit the IIR for a `mir.Filter` chain head with trailing `tail`.

  Mirrors the legacy `if isinstance(head, mir.Filter):` branch in
  `_lower_inner_chain` byte-for-byte:

    - The filter's `code` is normalized via `_filter_expr` (strip
      `return ` prefix and trailing `;`).
    - The body is whatever the rest of the chain lowers to.
    - When `ctx.ws_cartesian_valid_var` or
      `ctx.tiled_cartesian_valid_var` is set, the filter folds its
      condition into the active valid flag (`<v> = <v> && (<cond>);`)
      instead of emitting an `if (cond) {...}` block — this avoids
      divergence around cooperative warp writes and matches legacy
      `jit_filter` ws / tiled-Cartesian branches.
    - Otherwise emits a plain `if (cond) { body }`.
  '''
  # Deferred import: the chain dispatcher + helpers live in the
  # package `__init__.py` (the legacy monolith), which imports this
  # module via `_register_passes`. Import-at-call-time keeps the
  # package import graph linear (avoids a circular import between
  # `__init__.py` and this sibling module).
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _filter_expr,
    _lower_inner_chain,
  )

  cond_expr = _filter_expr(op.code)
  body_op = _lower_inner_chain(tail, ctx)
  fold_var = ctx.ws_cartesian_valid_var or ctx.tiled_cartesian_valid_var
  if fold_var:
    return Block(
      stmts=(
        RawString(text=f'{fold_var} = {fold_var} && ({cond_expr});'),
        body_op,
      )
    )
  return If(cond=RawString(text=cond_expr), body=body_op)


def lower_mir_filter(op: mir.Filter, ctx: Any) -> Op:
  '''Framework-registry stub for `@lowering(target=iir.cf,
  source=mir.Filter)`. The actual dispatch lives in
  `lowerings._lower_inner_chain` (via the `USE_DECLARATIVE`
  ratchet), which calls `lower_mir_filter_in_chain` with the
  trailing chain in scope.

  This stub exists so the dialect's `lowerings` list pins the
  (consumes, produces) contract for `mir.Filter` — the discipline
  test consults the dialect to verify ownership.

  Calling this stub directly raises a structural assertion — the
  framework path is reserved for future readers who can plumb in
  the missing `tail` (e.g. a refactored MIR-IIR walker that no
  longer routes through `_lower_inner_chain`).
  '''
  raise AssertionError(
    'lower_mir_filter: dispatch goes through '
    '`lowerings._lower_inner_chain` -> `lower_mir_filter_in_chain` '
    'so the trailing chain is in scope. Direct invocation '
    'indicates a refactor that bypassed the chain dispatch — '
    'plumb the `tail` through and call `lower_mir_filter_in_chain` '
    'instead.'
  )


__all__ = [
  'lower_mir_filter',
  'lower_mir_filter_in_chain',
]
