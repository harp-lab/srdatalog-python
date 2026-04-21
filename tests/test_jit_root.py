'''Tests for codegen/jit/root.py — baseline root op emitters.'''

import sys

import srdatalog.mir.types as m
from srdatalog.codegen.jit.context import new_code_gen_context
from srdatalog.codegen.jit.root import (
  gen_grid_stride_loop,
  jit_root_cartesian_join,
  jit_root_column_join,
  jit_root_scan,
)
from srdatalog.hir.types import Version


def _cs(rel, ver, idx, prefix=(), handle_start=0):
  return m.ColumnSource(
    rel_name=rel,
    version=ver,
    index=idx,
    prefix_vars=list(prefix),
    handle_start=handle_start,
  )


# -----------------------------------------------------------------------------
# gen_grid_stride_loop
# -----------------------------------------------------------------------------


def test_grid_stride_warp_mode():
  ctx = new_code_gen_context()
  out = gen_grid_stride_loop("  ", "idx", "total", ctx)
  assert "// WARP MODE" in out
  assert "for (uint32_t idx = warp_id; idx < total; idx += num_warps)" in out


def test_grid_stride_scalar_mode():
  ctx = new_code_gen_context()
  ctx.scalar_mode = True
  out = gen_grid_stride_loop("  ", "idx", "bound", ctx)
  assert "// SCALAR MODE" in out


def test_grid_stride_atomic_ws():
  ctx = new_code_gen_context()
  out = gen_grid_stride_loop("  ", "idx", "bound", ctx, use_atomic_ws=True)
  assert "// ATOMIC WS" in out
  assert "while (true) {" in out
  assert "atomicAdd(&_d_aws_counter, 1)" in out
  assert "tile.shfl(idx, 0)" in out
  assert "if (idx >= bound) break;" in out


# -----------------------------------------------------------------------------
# jit_root_scan
# -----------------------------------------------------------------------------


def test_root_scan_single_var_binds_column_0():
  sc = m.Scan(vars=["x"], rel_name="Edge", version=Version.FULL, index=[0], handle_start=0)
  ctx = new_code_gen_context()
  out = jit_root_scan(sc, ctx, "        body();\n")

  assert "// Root Scan: Edge binding x" in out
  # View and root handle emitted
  assert "auto root_view_" in out
  assert "= views[0];" in out
  assert "auto root_handle_" in out
  assert "HandleType(0, root_view_" in out
  # Validity return
  assert "return;" in out
  # Degree + loop
  assert "uint32_t degree_" in out
  assert "for (uint32_t idx_" in out and "< degree_" in out
  # Var binding via get_value(col=0, ...)
  assert ".get_value(0, " in out
  assert "body();" in out


def test_root_scan_two_vars_bind_cols_0_and_1():
  sc = m.Scan(
    vars=["y", "x"], rel_name="AddressOf", version=Version.FULL, index=[0, 1], handle_start=0
  )
  ctx = new_code_gen_context()
  out = jit_root_scan(sc, ctx, "body();\n")

  assert ".get_value(0, " in out  # y from col 0
  assert ".get_value(1, " in out  # x from col 1


def test_root_scan_reuses_preexisting_view():
  sc = m.Scan(vars=["x"], rel_name="R", version=Version.FULL, index=[0], handle_start=5)
  ctx = new_code_gen_context()
  ctx.view_vars["5"] = "view_R_pre"
  out = jit_root_scan(sc, ctx, "")
  assert "auto root_view_" not in out  # no new view decl
  assert "view_R_pre" in out


def test_root_scan_counting_skips_unused_var_fetch():
  sc = m.Scan(vars=["x"], rel_name="R", version=Version.FULL, index=[0], handle_start=0)
  ctx = new_code_gen_context()
  ctx.is_counting = True
  # body doesn't reference x -> skip get_value
  out = jit_root_scan(sc, ctx, "        output.emit_direct();\n")
  assert "auto x = " not in out


def test_root_scan_restores_bound_vars_and_indent():
  sc = m.Scan(vars=["x", "y"], rel_name="R", version=Version.FULL, index=[0, 1], handle_start=0)
  ctx = new_code_gen_context()
  initial_indent = ctx.indent
  initial_bound = list(ctx.bound_vars)
  jit_root_scan(sc, ctx, "body();\n")
  assert ctx.indent == initial_indent
  assert ctx.bound_vars == initial_bound


# -----------------------------------------------------------------------------
# jit_root_cartesian_join
# -----------------------------------------------------------------------------


def test_root_cart_two_sources_div_mod_decomposition():
  cart = m.CartesianJoin(
    vars=["x", "y"],
    sources=[
      _cs("R", Version.FULL, [0], handle_start=0),
      _cs("S", Version.FULL, [0], handle_start=1),
    ],
    var_from_source=[["x"], ["y"]],
  )
  ctx = new_code_gen_context()
  out = jit_root_cartesian_join(cart, ctx, "body();\n")

  # Root handles + validity check
  assert "auto h_R_0_" in out
  assert "auto h_S_1_" in out
  assert "if (!" in out and " || !" in out  # combined validity
  # Degrees + total
  assert "uint32_t degree_" in out
  assert "uint32_t total_" in out
  assert "if (total_" in out
  # Grid-stride loop
  assert "for (uint32_t flat_idx_" in out
  # 2-source div/mod decomposition
  assert " / " in out  # idx0 = flat / deg1
  assert " % " in out  # idx1 = flat % deg1
  # Var binding via get_value_at
  assert ".get_value_at(" in out
  assert "body();" in out


def test_root_cart_single_source_identity_decomposition():
  cart = m.CartesianJoin(
    vars=["x"],
    sources=[_cs("R", Version.FULL, [0], handle_start=0)],
    var_from_source=[["x"]],
  )
  ctx = new_code_gen_context()
  out = jit_root_cartesian_join(cart, ctx, "")
  # Single-source case: uint32_t idx0 = flat_idx (no mod/div)
  assert "uint32_t idx0_" in out
  lines = out.splitlines()
  # Find the idx0 decl line and confirm it's identity
  for ln in lines:
    if "uint32_t idx0_" in ln:
      assert " / " not in ln
      assert " % " not in ln
      break


def test_root_cart_three_sources_sequential_decomposition():
  cart = m.CartesianJoin(
    vars=["x", "y", "z"],
    sources=[
      _cs("R", Version.FULL, [0], handle_start=0),
      _cs("S", Version.FULL, [0], handle_start=1),
      _cs("T", Version.FULL, [0], handle_start=2),
    ],
    var_from_source=[["x"], ["y"], ["z"]],
  )
  ctx = new_code_gen_context()
  out = jit_root_cartesian_join(cart, ctx, "")
  # N-source path uses `remaining` carry
  assert "uint32_t remaining = flat_idx_" in out
  assert "remaining %" in out
  assert "remaining /=" in out


def test_root_cart_toggles_inside_cartesian_and_restores():
  cart = m.CartesianJoin(
    vars=["x"],
    sources=[_cs("R", Version.FULL, [0], handle_start=0)],
    var_from_source=[["x"]],
  )
  ctx = new_code_gen_context()
  assert ctx.inside_cartesian is False
  out = jit_root_cartesian_join(cart, ctx, "body();\n")
  assert ctx.inside_cartesian is False
  assert ctx.cartesian_bound_vars == []


def test_root_cart_counting_skips_unused_var():
  cart = m.CartesianJoin(
    vars=["x", "y"],
    sources=[
      _cs("R", Version.FULL, [0], handle_start=0),
      _cs("S", Version.FULL, [0], handle_start=1),
    ],
    var_from_source=[["x"], ["y"]],
  )
  ctx = new_code_gen_context()
  ctx.is_counting = True
  # Body doesn't mention x or y -> skip bindings
  out = jit_root_cartesian_join(cart, ctx, "        output.emit_direct();\n")
  assert "auto x = " not in out
  assert "auto y = " not in out


# -----------------------------------------------------------------------------
# jit_root_column_join — single source
# -----------------------------------------------------------------------------


def test_root_cj_single_source_baseline():
  cj = m.ColumnJoin(
    var_name="z",
    sources=[
      _cs("Edge", Version.FULL, [0, 1], handle_start=0),
    ],
  )
  ctx = new_code_gen_context()
  out = jit_root_column_join(cj, ctx, "        body();\n")

  assert "// Root ColumnJoin (single source): bind 'z' from Edge" in out
  # View + root handle + validity check + degree
  assert "auto view_Edge_0 = views[0];" in out
  assert "HandleType(0, view_Edge_0.num_rows_, 0)" in out
  assert "return;" in out
  assert "uint32_t degree_" in out
  # Grid-stride loop
  assert "for (uint32_t idx_" in out
  # Var bind + child handle registered
  assert ".get_value_at(" in out
  assert "auto ch_Edge_" in out
  assert "body();" in out


def test_root_cj_single_counting_skips_var_fetch():
  cj = m.ColumnJoin(
    var_name="z",
    sources=[
      _cs("Edge", Version.FULL, [0, 1], handle_start=0),
    ],
  )
  ctx = new_code_gen_context()
  ctx.is_counting = True
  out = jit_root_column_join(cj, ctx, "        output.emit_direct();\n")
  # z isn't referenced in body — no `auto z = ...` line
  assert "auto z = " not in out


def test_root_cj_single_registers_handle_in_state_keys():
  cj = m.ColumnJoin(
    var_name="z",
    sources=[
      _cs("Edge", Version.FULL, [0, 1], handle_start=0),
    ],
  )
  ctx = new_code_gen_context()
  # Capture handle_vars during body via a sentinel body that peeks at ctx
  body_state: dict = {}

  def capture_body() -> str:
    # This actually runs before jit_root_column_join; we only have the
    # snapshot ctx AFTER handle_vars is set. Instead, check structural
    # shape: after return, handle_vars should be cleaned back up.
    return "body();\n"

  jit_root_column_join(cj, ctx, capture_body())
  # After emit: ctx.handle_vars has NO leftover state for this CJ.
  assert "0" not in ctx.handle_vars
  assert all(not k.startswith("Edge_") for k in ctx.handle_vars)


# -----------------------------------------------------------------------------
# jit_root_column_join — multi source
# -----------------------------------------------------------------------------


def test_root_cj_multi_source_uses_root_unique_values_pattern():
  cj = m.ColumnJoin(
    var_name="y",
    sources=[
      _cs("R", Version.FULL, [0, 1], handle_start=0),
      _cs("S", Version.FULL, [1, 0], handle_start=1),
    ],
  )
  ctx = new_code_gen_context()
  out = jit_root_column_join(cj, ctx, "        body();\n")

  assert "// Root ColumnJoin (multi-source intersection): bind 'y' from 2 sources" in out
  # root_unique_values iteration
  assert "root_unique_values[" in out
  assert "num_unique_root_keys" in out
  # Deterministic handle names `h_<rel>_<idx>_root`
  assert "h_R_0_root" in out
  assert "h_S_1_root" in out
  # First source uses hinted range + .prefix
  assert "uint32_t hint_lo_" in out
  assert "uint32_t hint_hi_" in out
  assert "HandleType(hint_lo_" in out and ".prefix(" in out
  # Subsequent sources use plain root handle + prefix
  assert ".prefix(root_val_" in out
  # Early-continue on invalid handle
  assert "continue;" in out
  # Root var bound to root_val
  assert "auto y = root_val_" in out
  # Body emitted
  assert "body();" in out


def test_root_cj_multi_restores_handle_vars_on_return():
  cj = m.ColumnJoin(
    var_name="y",
    sources=[
      _cs("R", Version.FULL, [0, 1], handle_start=0),
      _cs("S", Version.FULL, [1, 0], handle_start=1),
    ],
  )
  ctx = new_code_gen_context()
  jit_root_column_join(cj, ctx, "body();\n")
  # No numeric keys leaked
  assert "0" not in ctx.handle_vars
  assert "1" not in ctx.handle_vars


# -----------------------------------------------------------------------------
# Feature-flag guards
# -----------------------------------------------------------------------------


def test_root_cj_rejects_ws_flag():
  cj = m.ColumnJoin(var_name="z", sources=[_cs("R", Version.FULL, [0, 1])])
  ctx = new_code_gen_context()
  ctx.ws_enabled = True
  try:
    jit_root_column_join(cj, ctx, "")
  except NotImplementedError as e:
    assert "work-stealing" in str(e)
  else:
    raise AssertionError("expected NotImplementedError")


def test_root_cj_bg_flag_routes_to_block_group_variant():
  '''BG root CJ is now ported. A multi-source root CJ with bg_enabled
  dispatches through `jit_root_column_join_block_group` (via pipeline.py)
  and emits the BG partitioning preamble (bg_work_per_block, binary
  search on bg_cumulative_work, per-key loop). Single-source BG falls
  back to baseline.'''
  from srdatalog.codegen.jit.root import jit_root_column_join_block_group

  cj = m.ColumnJoin(
    var_name="z",
    sources=[
      _cs("R", Version.FULL, [0, 1], handle_start=0),
      _cs("S", Version.FULL, [0, 1], handle_start=1),
    ],
  )
  ctx = new_code_gen_context()
  out = jit_root_column_join_block_group(cj, ctx, "body();\n")
  assert "BLOCK-GROUP" in out
  assert "bg_work_per_block" in out
  assert "bg_cumulative_work" in out
  assert "bg_key_idx" in out


def test_root_cj_rejects_fan_out_explore_flag():
  cj = m.ColumnJoin(var_name="z", sources=[_cs("R", Version.FULL, [0, 1])])
  ctx = new_code_gen_context()
  ctx.is_fan_out_explore = True
  try:
    jit_root_column_join(cj, ctx, "")
  except NotImplementedError as e:
    assert "fan-out" in str(e)
  else:
    raise AssertionError("expected NotImplementedError")


if __name__ == "__main__":
  import inspect

  this = sys.modules[__name__]
  passed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if name.startswith("test_"):
      fn()
      passed += 1
  print(f"{passed} passed")
