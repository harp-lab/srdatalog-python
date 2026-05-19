'''Lowering: `mir.CartesianJoin` -> iir.cf — Wave 2A B-Cart migration
(per `docs/phase_b_lowering_dispatcher.md` §4, row B-Cart, order 7,
difficulty `hard`).

Per `docs/phase_b_lowering_dispatcher.md` §4 (per-MIR-op work-unit
table) and §4.1 (per-PR template): each MIR op type gets one
`@lowering(target=iir.cf, source=mir.X)` rule in its own file under
`dialects/relation/sorted_array/lowerings/`. The registry registration
lives alongside the dialect's other lowerings in
`__init__.py:_register_passes`.

Structural-position split (root vs nested), mirroring `mir.Scan`'s
root-only situation and the B-CJ-multi root + chain dual-entry pattern:

`mir.CartesianJoin` is structurally allowed in TWO positions per
`_supported_pipeline`:

  - ROOT position (`lower_scan_pipeline` head): a Cartesian as the
    pipeline's first op, followed by a trailing run of InsertIntos.
    Legacy dispatch: `_lower_root_cart(head, rest, ctx)`. No middle
    ops between Cart and the InsertInto run today (see
    `_supported_pipeline`'s `len(middle) == 0` guard on the
    Cart-root branch).

  - NESTED / MID-CHAIN position (`_lower_inner_chain` head): a
    Cartesian following a Scan or multi-source ColumnJoin root, with
    arbitrary intervening Filter / ConstantBind / Negation /
    CartesianJoin / `mir.TiledCartesian` between it and the trailing
    InsertIntos. Legacy dispatch: `_lower_nested_cart(head, tail,
    ctx)`. This is the path that interacts with the typed
    `TiledCartesian` pragma + the `ctx.neg_pre_narrow` registration
    consumed by mid-chain `mir.Negation`.

Both positions share the same MIR op type (`mir.CartesianJoin`), so a
single `@lowering` registration covers both — the dispatching site
decides which legacy helper to forward to. We therefore expose THREE
entry points:

  - `lower_mir_cart_root(op, rest, ctx)`: called from
    `lower_scan_pipeline` when `_should_use_declarative(head)` and
    `isinstance(head, mir.CartesianJoin)`. Delegates to
    `_lower_root_cart`.
  - `lower_mir_cart_in_chain(op, tail, ctx)`: called from
    `_lower_inner_chain` when `_should_use_declarative(head)` and
    `isinstance(head, mir.CartesianJoin)`. Delegates to
    `_lower_nested_cart`.
  - `lower_mir_cart(op, ctx)`: the `@lowering`-registered stub.
    Asserts on direct invocation — the framework dispatch path is
    reserved for a future MIR-IIR walker that no longer routes
    through `lower_scan_pipeline` / `_lower_inner_chain` and can
    plumb the positional trailing argument through.

Coexistence with the C5 `TiledCartesian` typed pragma + wrap op
(per `docs/phase_c_pragma_materialization.md` §4.3):

  `mir.TiledCartesian(inner=mir.CartesianJoin)` is a DIFFERENT MIR
  op type — the C5 pragma's materialization handler wraps shape-
  eligible nested Cartesians in `TiledCartesian` BEFORE lowering
  reaches `_lower_inner_chain`. The wrap op has its own `@lowering`
  registration (registered by `_register_passes` from
  `pragmas/tiled_cartesian.py:lower_tiled_cartesian`) and its own
  chain-aware variant (`lower_tiled_cartesian_in_chain`) that calls
  `_lower_nested_cart_tiled` directly. Because the wrap op is a
  distinct type, the `_lower_inner_chain` dispatch sees
  `TiledCartesian` (not `CartesianJoin`) for the wrapped case — the
  B-Cart `mir.CartesianJoin` entry only fires for non-wrapped /
  non-eligible Cartesians.

  In the C5 dual-write transition (where the legacy runner-driven
  `ep.tiled_cartesian=True` short-circuits the wrap step in
  `materialize_tiled_cartesian`), the eligible Cartesian remains a
  bare `mir.CartesianJoin` and reaches `lower_mir_cart_in_chain`.
  The delegated `_lower_nested_cart` then checks
  `_tiled_cart_eligible(cart_op, ctx)` and forwards to
  `_lower_nested_cart_tiled` itself — so the tiled path still fires
  via the legacy `ctx.tiled_cartesian` bool. Either way the wrap-op
  dispatch (when present) and the bare-Cart dispatch (when not)
  reach the same `_lower_nested_cart_tiled` body, and the C5
  end-to-end test pins the cross-path byte-equivalence.

Negation interaction (per `_register_neg_pre_narrow`):

  `_lower_nested_cart` sets up `ctx.neg_pre_narrow` BEFORE rendering
  the body, so any `mir.Negation` in `tail` (already routed through
  `lower_mir_negation_in_chain` under USE_DECLARATIVE) finds the
  pre-allocated handle. This delegation preserves that contract
  byte-for-byte — the Cart's chain-aware variant just hands the
  whole `(cart_op, tail, ctx)` triple to `_lower_nested_cart`, which
  performs the pre-narrow registration + body render + scaffold
  emission atomically.

Byte-equivalence contract (§4.2): the migrated op produces the same
IIR tree as the legacy `if isinstance(head, mir.CartesianJoin):`
branches in both `lower_scan_pipeline` (root) and `_lower_inner_chain`
(mid-chain) on every fixture that exercises them. We delegate
straight back into the legacy `_lower_root_cart` / `_lower_nested_cart`
helpers to make byte-equivalence hold by construction; the migration
moves dispatch ownership without duplicating the body. This is the
same delegation pattern used by B-Filter / B-ConstantBind /
B-InsertInto / B-Scan / B-CJ-single / B-Negation / B-Aggregate and
the C2 / C4 / C5 / C6 wrap-op lowerings.
'''

from __future__ import annotations

from typing import Any

import srdatalog.ir.mir.types as mir
from srdatalog.ir.core import Op


def lower_mir_cart_root(
  op: mir.CartesianJoin,
  rest: list[Any],
  ctx: Any,
) -> Op:
  '''Emit the IIR for a ROOT `mir.CartesianJoin` with trailing pipeline `rest`.

  Mirrors the legacy `if isinstance(head, mir.CartesianJoin): return
  _lower_root_cart(head, rest, ctx)` dispatch in
  `lower_scan_pipeline` byte-for-byte by delegating to the legacy
  helper. The goal of this Wave 2A PR is byte-equivalence with the
  legacy emitter, not a from-scratch rewrite of the (M7.x) root
  Cartesian scaffolding — the helper stays LIVE until Layer 3
  cleanup deletes both the legacy branches AND the
  `USE_DECLARATIVE` ratchet.

  `rest` is the trailing pipeline AFTER the Cart at root position:
  per `_supported_pipeline` this is a pure run of InsertIntos today
  (no middle ops between root Cart and the InsertInto tail). The
  legacy `_lower_root_cart` walks it via `_lower_inner_chain` for
  the body and emits the root-Cart scaffold (per-source handle bind
  off SaRoot, combined validity `return`, per-source degree, total,
  total-zero `return`, ParallelFor grid-stride loop with per-source
  decompose + SaGetValAt var binds).

  Per the module docstring's coexistence note: the C5
  `TiledCartesian` wrap op only fires for NESTED Cartesians (per
  `_wrap_eligible_carts`'s `pipeline[1:]` slice), so the root
  position never sees a `TiledCartesian` head — only bare
  `mir.CartesianJoin`. This entry is therefore unaffected by C5.
  '''
  # Deferred import: the pipeline dispatcher + `_lower_root_cart`
  # live in the package `__init__.py` (the legacy monolith), which
  # imports this module via `_register_passes`. Import-at-call-time
  # keeps the package import graph linear (avoids a circular import
  # between `__init__.py` and this sibling module).
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _lower_root_cart,
  )

  return _lower_root_cart(op, rest, ctx)


def lower_mir_cart_in_chain(
  op: mir.CartesianJoin,
  tail: list[Any],
  ctx: Any,
) -> Op:
  '''Emit the IIR for a NESTED `mir.CartesianJoin` chain head with
  trailing `tail`.

  Mirrors the legacy `if isinstance(head, mir.CartesianJoin):
  return _lower_nested_cart(head, tail, ctx)` branch in
  `_lower_inner_chain` byte-for-byte by delegating to the legacy
  helper. The helper handles:

    - 1, 2, or N>=3 source counts (3+ uses the CartesianNDecompose
      countdown remainder).
    - State-key handle reuse for prefix-bearing sources; fresh root
      construction for prefix-empty sources; chained `.prefix(...)`
      walks for Scan-bound prefix vars.
    - `neg_pre_narrow` registration for any `mir.Negation` in
      `tail` (pre-allocates the negation handle BEFORE body rendering
      so the counter trajectory matches legacy).
    - The R1 count-as-product short-circuit (closed-form
      `add_count(lane_share)` in count phase, no per-thread idx).
    - Tiled-Cartesian dispatch (N7): when `_tiled_cart_eligible(...)`
      returns True (legacy `ctx.tiled_cartesian` bool path), the
      helper forwards to `_lower_nested_cart_tiled` itself —
      preserving the C5 dual-write transition without this entry
      needing to know about it.

  Per the module docstring's coexistence note: the C5
  `TiledCartesian` wrap op is a DIFFERENT MIR op type
  (`mir.TiledCartesian`) — its dispatch lives below this branch in
  `_lower_inner_chain` and forwards to
  `lower_tiled_cartesian_in_chain`. The bare `mir.CartesianJoin`
  entry here only fires for non-wrapped Cartesians (which includes
  the legacy `ctx.tiled_cartesian` bool path for shape-eligible
  Cartesians that haven't been wrapped — `_lower_nested_cart`'s own
  `_tiled_cart_eligible` check forwards to the tiled helper in
  that case). The dispatch order in `_lower_inner_chain` is
  load-bearing: the `TiledCartesian` branch sits AFTER this Cart
  branch because the wrap op type doesn't match `mir.CartesianJoin`
  isinstance — both branches coexist independently.
  '''
  # Deferred import: the chain dispatcher + `_lower_nested_cart`
  # live in the package `__init__.py` (the legacy monolith), which
  # imports this module via `_register_passes`. Import-at-call-time
  # keeps the package import graph linear (avoids a circular import
  # between `__init__.py` and this sibling module).
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _lower_nested_cart,
  )

  return _lower_nested_cart(op, tail, ctx)


def lower_mir_cart(op: mir.CartesianJoin, ctx: Any) -> Op:
  '''Framework-registry stub for `@lowering(target=iir.cf,
  source=mir.CartesianJoin)`. The actual dispatch lives in
  `lowerings.lower_scan_pipeline` (root position) and
  `lowerings._lower_inner_chain` (mid-chain position), both gated
  by the `USE_DECLARATIVE` ratchet. They call `lower_mir_cart_root`
  / `lower_mir_cart_in_chain` with the trailing pipeline / chain
  in scope.

  This stub exists so the dialect's `lowerings` list pins the
  (consumes, produces) contract for `mir.CartesianJoin` — the
  discipline test consults the dialect to verify ownership. Note
  that a single registration covers BOTH structural positions; the
  position-aware dispatch lives in the two callers above.

  Calling this stub directly raises a structural assertion — the
  framework path is reserved for future readers who can plumb in
  the missing trailing argument (`rest` at root, `tail` mid-chain)
  through (e.g. a refactored MIR-IIR walker that no longer routes
  through `lower_scan_pipeline` / `_lower_inner_chain`).
  '''
  raise AssertionError(
    'lower_mir_cart: dispatch goes through '
    '`lowerings.lower_scan_pipeline` -> `lower_mir_cart_root` (root '
    'position) or `lowerings._lower_inner_chain` -> '
    '`lower_mir_cart_in_chain` (mid-chain position) so the trailing '
    'pipeline / chain is in scope. Direct invocation indicates a '
    'refactor that bypassed the positional dispatch — plumb the '
    'trailing argument through and call `lower_mir_cart_root` or '
    '`lower_mir_cart_in_chain` instead.'
  )


__all__ = [
  'lower_mir_cart',
  'lower_mir_cart_in_chain',
  'lower_mir_cart_root',
]
