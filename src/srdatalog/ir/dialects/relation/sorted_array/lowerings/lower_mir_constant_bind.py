'''Lowering: `mir.ConstantBind` -> iir.cf — second Wave 2A per-op
migration.

Per `docs/phase_b_lowering_dispatcher.md` §4 (per-MIR-op work-unit
table, row B-ConstantBind) and §4.1 (per-PR template): each MIR op
type gets one `@lowering(target=iir.cf, source=mir.X)` rule in its
own file under `dialects/relation/sorted_array/lowerings/`. The
registry registration lives alongside the dialect's other lowerings
in `__init__.py:_register_passes`.

Byte-equivalence contract (§4.2): the migrated op produces the same
IIR tree as the legacy `if isinstance(head, mir.ConstantBind):`
branch inside `_lower_inner_chain` on every fixture that exercises
it.

Chain-aware split (mirrors C5's `lower_tiled_cartesian_in_chain`
pattern): `mir.ConstantBind` only appears as a middle op in the
inner chain — its emission depends on the trailing `tail` (which
the chain dispatcher provides) for the body. We therefore expose
two entry points:

  - `lower_mir_constant_bind_in_chain(op, tail, ctx)`: the real
    work, called from `_lower_inner_chain` (legacy) when
    `type(head) in USE_DECLARATIVE`.
  - `lower_mir_constant_bind(op, ctx)`: the `@lowering`-registered
    stub. Asserts on direct invocation — the framework dispatch path
    is reserved for a future MIR-IIR walker that no longer routes
    through `_lower_inner_chain` and can plumb `tail` through.

The split lets the registry pin dialect ownership (`ConstantBind`
belongs to `relation.sorted_array`) without forcing the chain
dispatcher through the registry today.
'''

from __future__ import annotations

from typing import Any

import srdatalog.ir.mir.types as mir
from srdatalog.ir.core import Op
from srdatalog.ir.dialects.iir.cf import Bind, Block, RawString


def lower_mir_constant_bind_in_chain(
  op: mir.ConstantBind,
  tail: list[Any],
  ctx: Any,
) -> Op:
  '''Emit the IIR for a `mir.ConstantBind` chain head with trailing
  `tail`.

  Mirrors the legacy `if isinstance(head, mir.ConstantBind):` branch
  in `_lower_inner_chain` byte-for-byte:

    - The var name is sanitized (C++ keyword collisions get a
      `_val` suffix).
    - An IIR `Bind(name=<var>, expr=RawString(code))` is emitted.
    - The rest of the chain is lowered; if the result is already a
      `Block`, the bind is prepended to its statements (flat block);
      otherwise the bind and the single op are wrapped in a fresh
      `Block`.

  The flatten-when-Block trick mirrors the legacy emit so the IIR
  tree has the same shape and the rendered CUDA is byte-identical.
  '''
  # Deferred import: the chain dispatcher + helpers live in the
  # package `__init__.py` (the legacy monolith), which imports this
  # module via `_register_passes`. Import-at-call-time keeps the
  # package import graph linear (avoids a circular import between
  # `__init__.py` and this sibling module).
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _lower_inner_chain,
    _sanitize_var_name,
  )

  var = _sanitize_var_name(op.var_name)
  bind_stmt = Bind(name=var, expr=RawString(text=op.code))
  rest_op = _lower_inner_chain(tail, ctx)
  if isinstance(rest_op, Block):
    return Block(stmts=(bind_stmt, *rest_op.stmts))
  return Block(stmts=(bind_stmt, rest_op))


def lower_mir_constant_bind(op: mir.ConstantBind, ctx: Any) -> Op:
  '''Framework-registry stub for `@lowering(target=iir.cf,
  source=mir.ConstantBind)`. The actual dispatch lives in
  `lowerings._lower_inner_chain` (via the `USE_DECLARATIVE`
  ratchet), which calls `lower_mir_constant_bind_in_chain` with the
  trailing chain in scope.

  This stub exists so the dialect's `lowerings` list pins the
  (consumes, produces) contract for `mir.ConstantBind` — the
  discipline test consults the dialect to verify ownership.

  Calling this stub directly raises a structural assertion — the
  framework path is reserved for future readers who can plumb in
  the missing `tail` (e.g. a refactored MIR-IIR walker that no
  longer routes through `_lower_inner_chain`).
  '''
  raise AssertionError(
    'lower_mir_constant_bind: dispatch goes through '
    '`lowerings._lower_inner_chain` -> `lower_mir_constant_bind_in_chain` '
    'so the trailing chain is in scope. Direct invocation '
    'indicates a refactor that bypassed the chain dispatch — '
    'plumb the `tail` through and call '
    '`lower_mir_constant_bind_in_chain` instead.'
  )


__all__ = [
  'lower_mir_constant_bind',
  'lower_mir_constant_bind_in_chain',
]
