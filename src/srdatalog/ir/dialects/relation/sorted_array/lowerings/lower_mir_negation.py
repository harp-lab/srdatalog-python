'''Lowering: `mir.Negation` -> iir.cf — Wave 2A B-Negation migration
(per `docs/phase_b_lowering_dispatcher.md` §4, row B-Negation, order 9,
difficulty `hard`).

Per `docs/phase_b_lowering_dispatcher.md` §4 (per-MIR-op work-unit
table) and §4.1 (per-PR template): each MIR op type gets one
`@lowering(target=iir.cf, source=mir.X)` rule in its own file under
`dialects/relation/sorted_array/lowerings/`. The registry registration
lives alongside the dialect's other lowerings in
`__init__.py:_register_passes`.

Why Negation is "hard" (per the migration plan):

  - It can be reached via TWO paths: a "standard" path (just-a-root
    Negation) that builds a fresh chained-prefix handle from root, and
    a "pre-narrow" path (Negation after a Cartesian) that reuses the
    handle the surrounding `_lower_nested_cart` already pre-allocated
    via `_register_neg_pre_narrow` + `ctx.neg_pre_narrow`. Which path
    is taken is decided at lowering time by checking
    `ctx.neg_pre_narrow.get(src_idx)`.
  - The pre-narrow path additionally handles `const_args` (pre-bound
    constant prefix columns) and `in_cartesian_vars` (vars bound by
    the surrounding Cart, applied via SaPrefSeq inside the loop).
  - N5.4 (per `docs/milestones.md`): standard-path Negation over a
    D2L FULL_VER source needs a segment-loop wrap that is NOT
    implemented (both we AND Nim are broken; deferred under F5). The
    legacy raise `NotImplementedError` is preserved verbatim here —
    the migration is byte-equivalent to legacy, NOT a fix.
  - WS / tiled-Cartesian valid-var folding mirrors the Filter path:
    when `ctx.ws_cartesian_valid_var` or
    `ctx.tiled_cartesian_valid_var` is set, the `if (!handle.valid())`
    branch folds into `<v> = <v> && (!handle.valid());`.

Byte-equivalence contract (§4.2): the migrated op produces the same
IIR tree as the legacy `if isinstance(head, mir.Negation):` branch
inside `_lower_inner_chain` on every fixture that exercises it —
including the N5.4 Nim-broken raise. We delegate straight back into
the legacy `_lower_negation` helper to make byte-equivalence hold by
construction; the migration moves dispatch ownership without
duplicating the body. This is the same delegation pattern used by
B-InsertInto (delegates to `_lower_insert_into`) and the C2/C4/C6
wrap-op lowerings.

Chain-aware split (mirrors C5's `lower_tiled_cartesian_in_chain` +
B-Filter / B-ConstantBind): `mir.Negation` only appears as a middle
op in the inner chain — its emission depends on the trailing `tail`
(which the chain dispatcher provides) for the body. We therefore
expose two entry points:

  - `lower_mir_negation_in_chain(op, tail, ctx)`: the real work,
    called from `_lower_inner_chain` (legacy) when
    `type(head) in USE_DECLARATIVE`. Delegates to `_lower_negation`.
  - `lower_mir_negation(op, ctx)`: the `@lowering`-registered stub.
    Asserts on direct invocation — the framework dispatch path is
    reserved for a future MIR-IIR walker that no longer routes
    through `_lower_inner_chain` and can plumb `tail` (and the
    surrounding `ctx.neg_pre_narrow` registrations) through.

The split lets the registry pin dialect ownership (`Negation` belongs
to `relation.sorted_array`) without forcing the chain dispatcher
through the registry today.
'''

from __future__ import annotations

from typing import Any

import srdatalog.ir.mir.types as mir
from srdatalog.ir.core import Op


def lower_mir_negation_in_chain(
  op: mir.Negation,
  tail: list[Any],
  ctx: Any,
) -> Op:
  '''Emit the IIR for a `mir.Negation` chain head with trailing `tail`.

  Mirrors the legacy `if isinstance(head, mir.Negation):` branch in
  `_lower_inner_chain` byte-for-byte by delegating directly into the
  existing `_lower_negation` helper. The helper handles:

    - Pre-narrow path (Negation inside a Cartesian, with
      `ctx.neg_pre_narrow[src_idx]` registered by the surrounding
      `_lower_nested_cart`): reuses the pre-allocated handle, applies
      any in-Cartesian vars via SaPrefSeq, and emits the
      `if (!handle.valid())` / fold-into-ws-or-tiled-valid branch.
    - Standard path: allocates a fresh `h_<rel>_neg_<idx>` handle,
      walks the prefix vars cooperatively (SaPrefCoop) outside a
      Cart or sequentially (SaPrefSeq) inside, then emits the
      `if (!handle.valid())` / fold branch.
    - N5.4 guard (Nim-broken; deferred under F5 per
      `docs/milestones.md`): standard-path Negation over a D2L
      FULL_VER source raises `NotImplementedError`. Preserved by
      delegation — this migration is byte-equivalent to legacy, not
      a fix.
    - `const_args` on the standard path is similarly unsupported
      and raises (legacy behavior preserved).

  All counter-bump trajectories, debug-comment emissions, and
  ws/tiled valid-var folding are inherited from `_lower_negation`
  unchanged.
  '''
  # Deferred import: the chain dispatcher + helpers live in the
  # package `__init__.py` (the legacy monolith), which imports this
  # module via `_register_passes`. Import-at-call-time keeps the
  # package import graph linear (avoids a circular import between
  # `__init__.py` and this sibling module).
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _lower_negation,
  )

  return _lower_negation(op, tail, ctx)


def lower_mir_negation(op: mir.Negation, ctx: Any) -> Op:
  '''Framework-registry stub for `@lowering(target=iir.cf,
  source=mir.Negation)`. The actual dispatch lives in
  `lowerings._lower_inner_chain` (via the `USE_DECLARATIVE`
  ratchet), which calls `lower_mir_negation_in_chain` with the
  trailing chain in scope.

  This stub exists so the dialect's `lowerings` list pins the
  (consumes, produces) contract for `mir.Negation` — the discipline
  test consults the dialect to verify ownership.

  Calling this stub directly raises a structural assertion — the
  framework path is reserved for future readers who can plumb in
  the missing `tail` (and the surrounding `ctx.neg_pre_narrow`
  registration) through (e.g. a refactored MIR-IIR walker that no
  longer routes through `_lower_inner_chain`).
  '''
  raise AssertionError(
    'lower_mir_negation: dispatch goes through '
    '`lowerings._lower_inner_chain` -> `lower_mir_negation_in_chain` '
    'so the trailing chain (and surrounding `ctx.neg_pre_narrow` '
    'registration from `_lower_nested_cart`) is in scope. Direct '
    'invocation indicates a refactor that bypassed the chain '
    'dispatch — plumb the `tail` through and call '
    '`lower_mir_negation_in_chain` instead.'
  )


__all__ = [
  'lower_mir_negation',
  'lower_mir_negation_in_chain',
]
