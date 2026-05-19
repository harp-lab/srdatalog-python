'''Pragma: `block_group` — block-group work-balanced dispatch around
the root multi-source ColumnJoin of a pipeline.

Trigger: `Rule(...).with_pragma(BlockGroup())` (preferred typed form)
   or the legacy `with_plan(..., block_group=True)` keyword
   (back-compat, slated for removal in A3).

Materialization (this module): wrap the root op of an
   `ExecutePipeline.pipeline` with `mir.BlockGroupRoot(inner=...)`
   during `MirPragmaPass`, then strip the `BlockGroup` instance from
   the EP's `pragmas` tuple. Only fires when the root op is a
   multi-source `mir.ColumnJoin` — the only legacy BG-supported root
   shape (legacy `jit_root_column_join_block_group` only dispatches
   on root CJ multi; root Scan + root Cartesian never went through
   the BG path).

Lowering (this module): a `@lowering(target=iir.cf, source=mir.
   BlockGroupRoot)` rule (registered via the parent dialect's
   `_register_passes`) is exposed for registry-discovery (R5 / PR
   acceptance gate), but the actual dispatch hook lives in
   `lower_scan_pipeline` (sorted_array dialect) — that function
   recognizes a `BlockGroupRoot` head op, unwraps it, and re-enters
   the standard root-CJ-multi flow with `ctx.bg_enabled=True`. The
   resulting IIR is identical to the legacy
   `if ctx.bg_enabled: return _lower_root_cj_bg(...)` branch by
   construction.

C3 dual-write contract: the DSL `Rule.with_pragma(BlockGroup())`
sets BOTH the typed pragma AND the legacy `block_group: bool` field
on the resulting `ExecutePipeline`, so the legacy runner-side
plumbing (`complete_runner.py:_gen_kernel_bg_count` / `_gen_kernel_
bg_materialize` and the runner-level `is_block_group=True` branches
that emit `kernel_bg_*` definitions + the histogram + BG launch
glue) keeps producing byte-equivalent CUDA without seeing the new
`BlockGroupRoot` wrap op. When `ep.block_group` is True we therefore
SKIP the wrap step: stripping the pragma is enough to satisfy
`MirPragmaPass`'s post-flight invariant, and the bool field
continues to drive emission via the legacy path.

Spec: `docs/phase_c_pragma_materialization.md` §4.1 (sub-dialect
choice), §5 (C3 PR row), §5.1 (per-PR acceptance gate);
`docs/pragma_as_typed_object.md` §2 (Pragma base class), §3
(`@pragma_handler` decorator).
'''

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, final

from srdatalog.ir.core import Op, Pragma, pragma_handler
from srdatalog.ir.mir.types import (
  BlockGroupRoot,
  ColumnJoin,
  ExecutePipeline,
)


@final
@dataclass(frozen=True, slots=True)
class BlockGroup(Pragma):
  '''Block-group work-balanced parallelism around a root multi-source
  ColumnJoin.

  Triggers `MirPragmaPass` to wrap the root op of the carrying
  `ExecutePipeline.pipeline` in a `BlockGroupRoot` MIR op; the wrap
  op's lowering re-dispatches the root-CJ-multi flow with
  `ctx.bg_enabled=True`, emitting the same scaffolding (per-block
  cumulative-work binary search, warp-row redistribution, optional
  D2L segment loops) that the legacy `ctx.bg_enabled` branch
  produced.

  No fields today — the legacy `block_group` flag is a flat bool. The
  per-rule decision of whether BG is profitable is made at runtime
  by the runner (the `bg_total_work > 0` check in `_gen_execute`);
  the pragma only signals "emit the BG variant alongside the
  baseline kernel". Structured config (work-balance heuristic
  weights, custom histogram kernel) is plausible later but out of
  scope for C3; per `docs/pragma_as_typed_object.md` §2, future
  fields land via back-compat-preserving dataclass defaults.
  '''


# -----------------------------------------------------------------------------
# Materialization handler (registered at import time via @pragma_handler)
# -----------------------------------------------------------------------------


def _wrap_root_in_block_group(
  pipeline: list[Any],
) -> list[Any]:
  '''Replace `pipeline[0]` with `BlockGroupRoot(inner=pipeline[0])`
  when the root op is a multi-source `ColumnJoin`.

  Mirrors the legacy `ctx.bg_enabled` dispatch in
  `_lower_root_cj_multi`: BG only fires on root CJ multi, NOT on
  root Scan or root Cartesian (`jit_root_column_join_block_group`
  is the only legacy BG entry point). When the root op doesn't
  qualify, returns the pipeline unchanged — the pragma is still
  consumed (the legacy bool field would have been a no-op too on a
  non-CJ root, since `_lower_root_scan` / `_lower_root_cart` never
  check `ctx.bg_enabled`).
  '''
  if not pipeline:
    return pipeline
  head = pipeline[0]
  if not isinstance(head, ColumnJoin) or len(head.sources) < 2:
    return list(pipeline)
  new_pipeline = list(pipeline)
  new_pipeline[0] = BlockGroupRoot(inner=head)
  return new_pipeline


@pragma_handler(BlockGroup, on=ExecutePipeline)
def materialize_block_group(
  op: Any,  # ExecutePipeline at runtime — typed as Any to match the registry's
  # Callable[[Any, Pragma, PragmaCtx], Any] signature (see
  # core/pragma.py:PragmaRegistration). Body re-narrows via the typed
  # `op.pragmas` / `op.block_group` accesses below.
  pragma: Any,  # BlockGroup at runtime; same Any reason.
  ctx: Any,  # PragmaCtx, unused for this no-config pragma.
) -> Any:
  '''Materialize `BlockGroup` into a `BlockGroupRoot` wrap around the
  root multi-source `ColumnJoin` in `op.pipeline`.

  Returns a new `ExecutePipeline` with:
    - `pipeline` rewritten so `pipeline[0]` is replaced by
      `BlockGroupRoot(inner=that_root_cj)` when the root qualifies —
      **except** in the C3 dual-write transition state (see below),
    - `pragmas` filtered to drop the consumed `BlockGroup` instance
      (per the `MirPragmaPass` post-flight invariant; see
      `docs/pragma_as_typed_object.md` §3).

  C3 dual-write transition: the DSL `Rule.with_pragma(BlockGroup())`
  sets BOTH the typed pragma AND the legacy `block_group: bool`
  field on the resulting `ExecutePipeline`, so the legacy runner-
  side emit (`complete_runner.py:_gen_kernel_bg_*` and the
  associated launch / setup branches gated on `node.block_group`)
  keeps producing byte-equivalent CUDA without seeing the new
  `BlockGroupRoot` wrap op. When `ep.block_group` is True we
  therefore SKIP the wrap step: stripping the pragma is enough to
  satisfy `MirPragmaPass`'s post-flight invariant, and the bool
  field continues to drive emission via the legacy path.

  When `ep.block_group` is False (the post-A3 pure-typed state, or
  test fixtures that exercise only the typed surface), the handler
  produces the wrap op and the BlockGroupRoot dispatch hook in
  `lower_scan_pipeline` (sorted_array dialect) re-enters the root-
  CJ-multi flow with `ctx.bg_enabled=True`. The resulting IIR is
  byte-equivalent to the BG branch of `_lower_root_cj_multi` by
  construction.

  Idempotency: the EP carries at most one `BlockGroup` (the DSL
  constructs at most one per `with_pragma(BlockGroup())` call;
  future multi-instance support is out of scope). We filter by
  `isinstance(p, BlockGroup)` to be defensive — any stray
  duplicates get removed together.

  Per spec §4.1, the wrap only covers the root op (not the trailing
  body chain) — the legacy `ctx.bg_enabled` dispatch fires inside
  `_lower_root_cj_multi`, with the body lowered under `bg_enabled=
  False`. Wrapping deeper would change the lowering call site and
  break byte-equivalence.
  '''
  new_pragmas = tuple(p for p in op.pragmas if not isinstance(p, BlockGroup))
  # Dual-write transition: skip the wrap if the legacy bool is set
  # (the legacy lowering will handle emission). See docstring above.
  if op.block_group:
    return dataclasses.replace(op, pragmas=new_pragmas)
  new_pipeline = _wrap_root_in_block_group(list(op.pipeline))
  return dataclasses.replace(op, pipeline=new_pipeline, pragmas=new_pragmas)


# -----------------------------------------------------------------------------
# BlockGroupRoot lowering (registered by parallel.block_group `_register_passes`)
# -----------------------------------------------------------------------------


def lower_block_group_root(op: Any, ctx: Any) -> Op:
  '''Emit the IIR for `BlockGroupRoot(inner=ColumnJoin)`.

  Returns the same `Block` that the legacy `ctx.bg_enabled` branch
  inside `_lower_root_cj_multi` would have produced for the wrapped
  root ColumnJoin, by calling `_lower_root_cj_bg` directly. The
  inner op carries the original root ColumnJoin; the post-root
  `rest` is NOT carried on the wrap op — the caller
  (`lower_scan_pipeline`) is responsible for slicing it off the
  enclosing pipeline and passing it through.

  Two call sites:

    1. `lower_scan_pipeline` (sorted_array dialect): detects a
       `BlockGroupRoot` head, slices `rest = pipeline[1:]`, and
       calls `_lower_root_cj_bg` directly (this function exists as
       the `@lowering`-registered entry so registry-completeness
       discipline checks pass).

    2. Tests that invoke this function directly: pass the wrap op
       and a ctx whose `rest_for_bg_root` attribute carries the
       trailing pipeline ops (typically a single `InsertInto`).

  PR-1 refactor (per `docs/phase_decomposition_redesign.md` §
  3.2.1): this helper used to flip `ctx.bg_enabled = True` for the
  duration of the `_lower_root_cj_bg` call. PR-1 removed the flip
  because `_lower_root_cj_bg` IS the BG function — it doesn't need
  a ctx flag to dispatch to itself. The flag was only ever
  consulted by `_lower_root_cj_multi` to RE-ROUTE to the BG variant
  on the legacy `ep.block_group: bool` path; the wrap-op path
  bypasses that re-route entirely, so no flag flip is needed.
  '''
  # Deferred import: the sorted_array monolith imports nothing from
  # this subpackage at top level, but co-located ops + pragmas +
  # lowerings in this dialect get loaded AFTER the monolith.
  # Importing at function-call time keeps the dialect import graph
  # linear.
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _lower_root_cj_bg,
  )

  rest = getattr(ctx, 'rest_for_bg_root', None)
  if rest is None:
    rest = []

  return _lower_root_cj_bg(op.inner, list(rest), ctx)


__all__ = [
  'BlockGroup',
  'lower_block_group_root',
  'materialize_block_group',
]
