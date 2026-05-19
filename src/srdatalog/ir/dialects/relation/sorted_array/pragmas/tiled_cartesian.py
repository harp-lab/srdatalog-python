'''Pragma: `tiled_cartesian` — atomic-free coalesced writes for 2-source
1-var-per-source nested Cartesian joins on sorted-array sources.

Trigger: `Rule(...).with_pragma(TiledCartesian())` (preferred typed form).
   The legacy bool `ExecutePipeline.tiled_cartesian` is driven from
   `complete_runner.py:has_tiled_cartesian_eligible(pipeline)` and
   threaded through `compile_kernel_body(tiled_cartesian=...)` /
   `LoweringCtx.tiled_cartesian`; that runner-side detection stays
   intact through C5, slated for removal in A3.

Materialization (this module): walk `ExecutePipeline.pipeline` and
   replace every eligible nested `mir.CartesianJoin` with
   `mir.TiledCartesian(inner=that_cj)` during `MirPragmaPass`, then
   strip the `TiledCartesian` instance from the EP's `pragmas`
   tuple. Eligibility mirrors legacy
   `_tiled_cart_eligible`: 2 sources, 1 var per source, not counting.
   The handler short-circuits when `ep.tiled_cartesian` is True so
   the legacy runner-driven path remains the sole emission driver
   (the dual-write transition; see `dedup_hash.py` for the parallel
   pattern).

Lowering (this module): a `@lowering(target=iir.cf, source=mir.
   TiledCartesian)` rule (registered via the parent dialect's
   `_register_passes`) delegates to the existing
   `_lower_nested_cart_tiled` helper with `ctx.tiled_cartesian`
   flipped True for the duration of the call, so the new lowering
   and the legacy path emit byte-identical IIR. Because
   `_lower_nested_cart_tiled` needs the trailing chain after the
   Cartesian (the `rest` parameter — InsertIntos plus any
   intervening Filter / ConstantBind), the dispatch lives inside
   `_lower_inner_chain` rather than as a standalone lowering: that
   site has both `head` and `tail` in scope. The
   `@lowering`-registered rule serves as the dialect's contract that
   `TiledCartesian` is owned here — actual emission happens via the
   `_lower_inner_chain` branch added in `lowerings.py`.

Spec: `docs/phase_c_pragma_materialization.md` §4.3 (`tiled_cartesian`
stays in sorted_array — no new sub-dialect needed), §5 (C5 PR row),
§5.1 (per-PR acceptance gate); `docs/pragma_as_typed_object.md` §2
(Pragma base class), §3 (`@pragma_handler` decorator).
'''

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, final

from srdatalog.ir.core import Op, Pragma, pragma_handler
from srdatalog.ir.mir.types import (
  CartesianJoin,
  ExecutePipeline,
)
from srdatalog.ir.mir.types import (
  TiledCartesian as TiledCartesianOp,
)


@final
@dataclass(frozen=True, slots=True)
class TiledCartesian(Pragma):
  '''Atomic-free coalesced writes for eligible nested Cartesian joins.

  Triggers `MirPragmaPass` to wrap every eligible nested
  `CartesianJoin` in `ExecutePipeline.pipeline` with a
  `mir.TiledCartesian` wrap op; the wrap's `@lowering` rule emits
  the same `SaTiledCartesian2D` + `TiledBallotBlock` IIR shape that
  the legacy `if ctx.tiled_cartesian:` branch inside
  `_lower_nested_cart` produces.

  Eligibility (mirrors legacy `_tiled_cart_eligible`):
    - exactly 2 sources,
    - 1 var bound per source,
    - not in the count phase (handled at lowering time via
      `ctx.is_counting`; the wrap op is inert under counting).

  No fields today — the existing legacy emitter treats
  `tiled_cartesian` as a flat bool. Structured config (custom tile
  sizes, smem footprint hints, etc.) is plausible later but out of
  scope for C5; per `docs/pragma_as_typed_object.md` §2, future
  fields land via back-compat-preserving dataclass defaults.
  '''


# -----------------------------------------------------------------------------
# Materialization handler (registered at import time via @pragma_handler)
# -----------------------------------------------------------------------------


def _is_eligible_cart(op: Any) -> bool:
  '''True iff `op` is a 2-source / 1-var-per-source CartesianJoin.

  Pure shape predicate — does NOT consult `ctx.is_counting` (the
  pragma pass has no count-phase notion). The wrap op's lowering
  defers to the legacy emitter's count-phase short-circuit, so a
  wrap that's emitted unconditionally still produces the right IIR
  in the count phase (the legacy `_lower_nested_cart` body fires
  instead of `_lower_nested_cart_tiled`).
  '''
  return (
    isinstance(op, CartesianJoin)
    and len(op.sources) == 2
    and len(op.var_from_source) == 2
    and len(op.var_from_source[0]) == 1
    and len(op.var_from_source[1]) == 1
  )


def _wrap_eligible_carts(
  pipeline: list[Any],
) -> list[Any]:
  '''Replace every eligible nested `CartesianJoin` in `pipeline` with
  `TiledCartesian(inner=that_cj)`.

  Walks the top-level pipeline list. The pipeline shape supported by
  `_supported_pipeline` allows Cartesians as middle ops (after a
  Scan or CJ root, or as the root itself). Root Cartesians are NOT
  wrapped here — the tiled path is reserved for nested Cartesians
  (`_lower_root_cart` doesn't dispatch to `_lower_nested_cart_tiled`,
  per the legacy emitter shape). Per-source / per-prefix details
  (state-key reuse, view var) are resolved later by
  `_lower_nested_cart_tiled` from the wrapped `inner.sources`.

  Non-eligible Cartesians pass through unchanged so a pipeline with
  both 2-source and N-source Carts (rare but possible) still
  type-checks. Non-Cartesian ops always pass through.
  '''
  if not pipeline:
    return pipeline
  wrapped: list[Any] = []
  # The first op is the pipeline root (Scan / CJ / Cart). Root
  # Cartesians use `_lower_root_cart` which never dispatches tiled,
  # so we leave index 0 alone even if shape-eligible.
  wrapped.append(pipeline[0])
  for child in pipeline[1:]:
    if _is_eligible_cart(child):
      wrapped.append(TiledCartesianOp(inner=child))
    else:
      wrapped.append(child)
  return wrapped


@pragma_handler(TiledCartesian, on=ExecutePipeline)
def materialize_tiled_cartesian(
  op: Any,  # ExecutePipeline at runtime — typed as Any to match the registry's
  # Callable[[Any, Pragma, PragmaCtx], Any] signature (see
  # core/pragma.py:PragmaRegistration). Body re-narrows via the typed
  # `op.pragmas` / `op.tiled_cartesian` accesses below.
  pragma: Any,  # TiledCartesian at runtime; same Any reason.
  ctx: Any,  # PragmaCtx, unused for this no-config pragma.
) -> Any:
  '''Materialize `TiledCartesian` into a `TiledCartesian` wrap op
  around each eligible nested `CartesianJoin` in `op.pipeline`.

  Returns a new `ExecutePipeline` with:
    - `pipeline` rewritten so every shape-eligible nested
      `CartesianJoin` (2 sources, 1 var per source) is replaced by
      `mir.TiledCartesian(inner=that_cj)` — **except** in the C5
      dual-write transition state (see below),
    - `pragmas` filtered to drop the consumed `TiledCartesian`
      instance (per the `MirPragmaPass` post-flight invariant; see
      `docs/pragma_as_typed_object.md` §3).

  C5 dual-write transition: the legacy runner-driven path
  (`complete_runner.py:has_tiled_cartesian_eligible(pipeline)` ->
  `compile_kernel_body(tiled_cartesian=True)` ->
  `LoweringCtx.tiled_cartesian=True` ->
  `_lower_nested_cart` dispatches to `_lower_nested_cart_tiled`)
  unconditionally enables tiled emission on shape-eligible
  pipelines, independent of any rule pragma. When `ep.tiled_cartesian`
  is True we therefore SKIP the wrap step: stripping the pragma is
  enough to satisfy `MirPragmaPass`'s post-flight invariant, and
  the legacy ctx-driven dispatch continues to emit byte-equivalent
  CUDA.

  When `ep.tiled_cartesian` is False (the post-A3 pure-typed state,
  or test fixtures that exercise only the typed surface), the
  handler produces the wrap op and the dialect's
  `_lower_inner_chain` branch (added in C5 to detect
  `TiledCartesian` heads) delegates to `_lower_nested_cart_tiled`
  with `ctx.tiled_cartesian` flipped True for the duration — so the
  resulting IIR is byte-equivalent to the legacy branch by
  construction.

  Idempotency: the EP carries at most one `TiledCartesian` (the
  DSL constructs at most one per `with_pragma(TiledCartesian())`
  call). We filter by `isinstance(p, TiledCartesian)` to be
  defensive — any stray duplicates get removed together. Re-running
  the pass on already-wrapped pipelines is also safe because
  `_is_eligible_cart` returns False for `TiledCartesianOp` (it's
  not a `CartesianJoin`).
  '''
  new_pragmas = tuple(p for p in op.pragmas if not isinstance(p, TiledCartesian))
  # Dual-write transition: skip the wrap if the legacy bool is set
  # (the legacy lowering will handle emission). See docstring above.
  if op.tiled_cartesian:
    return dataclasses.replace(op, pragmas=new_pragmas)
  new_pipeline = _wrap_eligible_carts(list(op.pipeline))
  return dataclasses.replace(op, pipeline=new_pipeline, pragmas=new_pragmas)


# -----------------------------------------------------------------------------
# TiledCartesian lowering (registered by sorted_array `_register_passes`)
# -----------------------------------------------------------------------------


def lower_tiled_cartesian_in_chain(
  op: TiledCartesianOp,
  tail: list[Any],
  ctx: Any,
) -> Op:
  '''Emit the IIR for `TiledCartesian(inner=CartesianJoin)` with
  trailing chain `tail` (InsertIntos plus any intervening
  Filter / ConstantBind from the surrounding `_lower_inner_chain`).

  Called from `lowerings._lower_inner_chain` when a `TiledCartesian`
  wrap op appears as a chain head — that site has both `head` and
  `tail` in scope, which the legacy `_lower_nested_cart_tiled`
  helper needs to render the inner body. The helper takes the
  tiled path directly (it IS the tiled variant); the inner chain
  rendering picks up `ctx.tiled_cartesian_valid_var` via the
  helper's own save/restore.

  Byte-equivalence by construction: the legacy
  `if _tiled_cart_eligible:` -> `_lower_nested_cart_tiled` dispatch
  inside `_lower_nested_cart` and this wrap-op dispatch call the
  SAME helper with the SAME args, so the produced IIR is identical
  for the same input `(cart_op, rest)`.

  PR-1 refactor (per `docs/phase_decomposition_redesign.md` §
  3.2.1): this helper used to flip `ctx.tiled_cartesian = True` for
  the duration of the `_lower_nested_cart_tiled` call. PR-1 removed
  the flip because `_lower_nested_cart_tiled` IS the tiled variant
  — it doesn't need a ctx flag to dispatch to itself. The flag was
  only consulted by `_tiled_cart_eligible` (called from
  `_lower_nested_cart`) on the legacy `ep.tiled_cartesian: bool`
  path; the wrap-op path bypasses `_lower_nested_cart` entirely.
  '''
  # Deferred import: the monolith module imports nothing from this
  # subpackage at top level, but co-located ops + pragmas + lowerings
  # in `pragmas/tiled_cartesian.py` (this file) get loaded by the
  # dialect's `_register_passes` AFTER the monolith. Importing at
  # function-call time keeps the dialect import graph linear.
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _lower_nested_cart_tiled,
  )

  return _lower_nested_cart_tiled(op.inner, list(tail), ctx)


def lower_tiled_cartesian(op: TiledCartesianOp, ctx: Any) -> Op:
  '''Framework-registry stub for `@lowering(target=iir.cf,
  source=TiledCartesian)`. The actual dispatch lives in
  `lowerings._lower_inner_chain`, which calls
  `lower_tiled_cartesian_in_chain` with the trailing chain in scope.

  This stub exists so the dialect's `lowerings` list pins the
  (consumes, produces) contract for `TiledCartesian` — the
  discipline test `test_tiled_cartesian_lowering_is_registered_on_
  sorted_array_dialect` consults the dialect to verify ownership.
  The C2 sibling `lower_dedup_gate` doesn't need this split because
  `DedupGate(inner=InsertInto)` is a leaf with no `tail` to thread;
  `TiledCartesian` wraps a non-leaf and needs the surrounding
  chain, so the chain-aware variant gets a separate entry point.

  Calling this stub directly raises a structural assertion — the
  framework path is reserved for future readers who can plumb in
  the missing `tail` (e.g. a refactored MIR-IIR walker that no
  longer routes through `_lower_inner_chain`).
  '''
  raise AssertionError(
    'lower_tiled_cartesian: dispatch goes through '
    '`lowerings._lower_inner_chain` -> `lower_tiled_cartesian_in_chain` '
    'so the trailing chain is in scope. Direct invocation '
    'indicates a refactor that bypassed the chain dispatch — '
    'plumb the `tail` through and call `lower_tiled_cartesian_in_chain` '
    'instead.'
  )


__all__ = [
  'TiledCartesian',
  'lower_tiled_cartesian',
  'lower_tiled_cartesian_in_chain',
  'materialize_tiled_cartesian',
]
