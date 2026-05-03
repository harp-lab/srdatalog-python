'''Tests for the kernel-functor-level work-stealing emit variants in
the dialect (N8).

The legacy emitter has only two pieces of WS support:
  - `ws_enabled` (count phase): InsertInto emits `<out>++` instead of
    `<out>.emit_direct()` — the WS count uses a per-thread local
    counter that the runner aggregates.
  - `ws_cartesian_valid_var` (materialize, batched-Cartesian): Filter
    / Negation fold their guard into the valid flag, and InsertInto
    emits `<out>.emit_warp_coalesced(tile, valid, args)` instead of
    a lane-zero-guarded emit_direct.

The full WS scaffolding (WCOJTask queue, stealing across warps) was
never implemented in the legacy runner — these tests cover only what
the kernel-functor-level emit produces. Compare dialect output against
the legacy `emit_helpers.jit_insert_into` / `jit_filter` / `jit_negation`
on synthetic contexts.
'''

from __future__ import annotations

import srdatalog.ir.mir.types as mir
from srdatalog.ir.dialects.iir.cf import Block as IirBlock
from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
  LoweringCtx,
  _lower_inner_chain,
  _lower_insert_into,
)
from srdatalog.ir.dialects.target.cuda.emit import EmitCtx, emit
from srdatalog.ir.hir.types import Version


def _emit_one(op_or_list) -> str:
  ctx = EmitCtx(indent_level=2)
  if isinstance(op_or_list, list):
    return ''.join(emit(o, ctx) for o in op_or_list)
  return emit(op_or_list, ctx)


# -----------------------------------------------------------------------------
# ws_enabled count-phase InsertInto -> `<out>++`
# -----------------------------------------------------------------------------


def test_insert_count_ws_enabled_increments_local_counter():
  '''Mirrors `test_jit_emit_helpers.test_insert_count_ws_enabled_
  increments_local_counter` but exercises the dialect lowering.'''
  node = mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['x', 'y'],
    index=[0, 1],
  )
  ctx = LoweringCtx(
    is_counting=True,
    ws_enabled=True,
    output_var='local_count',
  )
  ops = _lower_insert_into(node, ctx)
  out = _emit_one(ops)
  assert 'if (tile.thread_rank() == 0) local_count++;' in out
  assert 'emit_direct' not in out


def test_insert_count_ws_enabled_inside_cart_drops_lane0():
  '''Inside a Cartesian, no lane-0 guard — `local_count++` directly.'''
  node = mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['x'],
    index=[0],
  )
  ctx = LoweringCtx(
    is_counting=True,
    ws_enabled=True,
    inside_cartesian=True,
    output_var='local_count',
  )
  ops = _lower_insert_into(node, ctx)
  out = _emit_one(ops)
  assert 'local_count++;' in out
  assert 'tile.thread_rank()' not in out


# -----------------------------------------------------------------------------
# ws_cartesian_valid_var materialize InsertInto -> emit_warp_coalesced
# -----------------------------------------------------------------------------


def test_insert_materialize_ws_cart_emits_warp_coalesced():
  node = mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['y', 'x'],
    index=[0, 1],
  )
  ctx = LoweringCtx(
    is_counting=False,
    ws_cartesian_valid_var='ws_valid',
    output_var='ctx0',
  )
  ops = _lower_insert_into(node, ctx)
  out = _emit_one(ops)
  assert 'ctx0.emit_warp_coalesced(tile, ws_valid, y, x);' in out
  assert 'emit_direct' not in out


# -----------------------------------------------------------------------------
# Filter folds into ws_cartesian_valid_var
# -----------------------------------------------------------------------------


def test_filter_ws_cart_folds_into_valid():
  '''Filter with `ws_cartesian_valid_var` set folds into
  `<v> = <v> && (<cond>);` instead of `if (<cond>) {...}` — matches
  legacy `jit_filter` ws branch.'''
  filt = mir.Filter(vars=['x', 'y'], code='return x != y;')
  ins = mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['x', 'y'],
    index=[0, 1],
  )
  ctx = LoweringCtx(
    ws_cartesian_valid_var='ws_valid',
    output_var='ctx0',
  )
  op = _lower_inner_chain([filt, ins], ctx)
  out = _emit_one(op)
  assert 'ws_valid = ws_valid && (x != y);' in out
  assert 'if (' not in out


# -----------------------------------------------------------------------------
# Negation folds into ws_cartesian_valid_var (standard non-pre-narrow path)
# -----------------------------------------------------------------------------


def test_negation_ws_cart_folds_into_valid():
  '''Standard-path Negation in WS mode emits
  `<v> = <v> && (!<handle>.valid());` then body, no `if`.'''
  neg = mir.Negation(
    rel_name='Bad',
    version=Version.FULL,
    prefix_vars=['x'],
    index=[0],
    handle_start=1,
  )
  ins = mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['x'],
    index=[0],
  )
  ctx = LoweringCtx(
    ws_cartesian_valid_var='ws_valid',
    view_var_names={'1': 'view_Bad_0_FULL_VER'},
    output_var='ctx0',
    bound_vars=['x'],
  )
  op = _lower_inner_chain([neg, ins], ctx)
  out = _emit_one(op)
  assert 'ws_valid = ws_valid && (!' in out
  # No `if (!handle.valid()) {` — folded.
  assert 'if (!' not in out


def test_iir_block_imported():
  '''Sanity check that IirBlock is the right symbol (no shadowing issues
  if the Block name is reused in this test module).'''
  assert IirBlock is not None
