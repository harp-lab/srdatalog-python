'''Tests for codegen/jit/instructions.py (baseline non-WS / non-BG).

Covers the structural shapes emitted:
  - single-source ColumnJoin: view decl, handle narrow, degree, index loop,
    get_value_at, child + register in handle_vars
  - multi-source ColumnJoin: per-source handles, intersect_handles(...),
    iteration, child_range registration
  - multi-source ColumnJoin handle reuse via handle_state_key
  - CartesianJoin baseline: per-source degrees, flat loop, position-index
    decomposition, bind vars, inside_cartesian state toggle
  - PositionedExtract baseline: per-source probe, valid guard, bind_vars
  - NotImplementedError guards on WS / BG / tiled / multi-view flags
'''
import sys
from pathlib import Path


import srdatalog.mir.types as m
from srdatalog.hir_types import Version
from srdatalog.codegen.jit.context import new_code_gen_context
from srdatalog.codegen.jit.instructions import (
  jit_nested_column_join,
  jit_nested_cartesian_join,
  jit_positioned_extract,
)


def _cs(rel, ver, idx, prefix=(), handle_start=0):
  return m.ColumnSource(
    rel_name=rel, version=ver, index=idx,
    prefix_vars=list(prefix), handle_start=handle_start,
  )


# ---------------------------------------------------------------------------
# jit_nested_column_join — single source
# ---------------------------------------------------------------------------

def test_nested_cj_single_source_baseline():
  cj = m.ColumnJoin(var_name="z", sources=[_cs("R", Version.FULL, [0, 1],
                                                  handle_start=0)])
  ctx = new_code_gen_context()
  out = jit_nested_column_join(cj, ctx, "        body();\n")
  assert "// Nested ColumnJoin (single): bind 'z' from R" in out
  # Emits view decl, root handle, if(!valid) continue, degree, index loop
  assert "auto view_R_0 = views[0];" in out
  assert "HandleType(0, view_R_0.num_rows_, 0)" in out
  assert "if (!" in out and ".valid()) continue;" in out
  assert "uint32_t degree_" in out
  assert "for (uint32_t idx_" in out
  # get_value_at for z + child_range handle registration
  assert ".get_value_at(" in out
  # child var has deterministic name ch_<Rel>_<handleIdx>_<var>
  assert "auto ch_R_0_z = " in out
  assert "body();" in out
  assert out.rstrip().endswith("}")


def test_nested_cj_single_source_with_prefix_narrows_from_parent():
  cj = m.ColumnJoin(var_name="y", sources=[_cs("R", Version.FULL, [0, 1],
                                                  prefix=("x",), handle_start=0)])
  ctx = new_code_gen_context()
  ctx.handle_vars["0"] = "parent_h"
  ctx.view_vars["0"] = "view_R_0"
  out = jit_nested_column_join(cj, ctx, "")
  # Uses parent_h.prefix(x, tile, view_R_0)
  assert "parent_h.prefix(x, tile, view_R_0)" in out


def test_nested_cj_single_source_counting_skips_unused_value():
  cj = m.ColumnJoin(var_name="z", sources=[_cs("R", Version.FULL, [0, 1])])
  ctx = new_code_gen_context()
  ctx.is_counting = True
  # body doesn't reference z -> skip get_value_at for z
  body = "        output.emit_direct();\n"
  out = jit_nested_column_join(cj, ctx, body)
  # z is unused in body; no `auto z = ...` line
  assert "auto z = " not in out
  # But child_range is still emitted
  assert "auto ch_R_0_z = " in out


# ---------------------------------------------------------------------------
# jit_nested_column_join — multi source
# ---------------------------------------------------------------------------

def test_nested_cj_multi_source_intersection():
  cj = m.ColumnJoin(var_name="z", sources=[
    _cs("R", Version.DELTA, [0, 1], handle_start=0),
    _cs("S", Version.FULL, [1, 0], handle_start=1),
  ])
  ctx = new_code_gen_context()
  out = jit_nested_column_join(cj, ctx, "        body();\n")
  assert "// Nested ColumnJoin (intersection): bind 'z' from 2 sources" in out
  # Views declared
  assert "auto view_R_0 = views[0];" in out
  assert "auto view_S_1 = views[1];" in out
  # intersect_handles(tile, iterators...)
  assert "intersect_handles(tile, " in out
  # Iterator using .begin() / .valid() / .next()
  assert ".begin();" in out
  assert ".valid();" in out
  assert ".next())" in out
  # z bound from the iterator value
  assert "auto z = it_" in out and ".value();" in out
  # positions[] array used for per-source child_range
  assert "auto positions = it_" in out
  # Child names deterministic
  assert "auto ch_R_0_z = " in out
  assert "auto ch_S_1_z = " in out
  assert "body();" in out


def test_nested_cj_multi_source_handle_reuse_full_match():
  cj = m.ColumnJoin(var_name="y", sources=[
    _cs("R", Version.FULL, [0, 1], prefix=("x",), handle_start=0),
    _cs("S", Version.FULL, [0, 1], handle_start=1),
  ])
  ctx = new_code_gen_context()
  # Pre-register a fully-prefixed handle for source 0
  from srdatalog.codegen.jit.context import gen_handle_state_key
  key = gen_handle_state_key("R", [0, 1], ["x"], "FULL_VER")
  ctx.handle_vars[key] = "existing_full_h"
  out = jit_nested_column_join(cj, ctx, "")
  # Source 0 reuses the pre-registered handle directly
  assert "= existing_full_h;" in out


# ---------------------------------------------------------------------------
# jit_nested_column_join — guards
# ---------------------------------------------------------------------------

def test_nested_cj_ws_flag_raises_not_implemented():
  cj = m.ColumnJoin(var_name="z", sources=[_cs("R", Version.FULL, [0, 1])])
  ctx = new_code_gen_context()
  ctx.ws_enabled = True
  try:
    jit_nested_column_join(cj, ctx, "")
  except NotImplementedError as e:
    assert "work-stealing" in str(e)
  else:
    raise AssertionError("expected NotImplementedError")


def test_nested_cj_bg_flag_raises_not_implemented():
  cj = m.ColumnJoin(var_name="z", sources=[_cs("R", Version.FULL, [0, 1])])
  ctx = new_code_gen_context()
  ctx.bg_enabled = True
  try:
    jit_nested_column_join(cj, ctx, "")
  except NotImplementedError as e:
    assert "block-group" in str(e)
  else:
    raise AssertionError("expected NotImplementedError")


# ---------------------------------------------------------------------------
# jit_nested_cartesian_join
# ---------------------------------------------------------------------------

def test_cart_two_sources_flat_loop_and_positions():
  cart = m.CartesianJoin(
    vars=["x", "y"],
    sources=[
      _cs("R", Version.FULL, [0], handle_start=0),
      _cs("S", Version.FULL, [0], handle_start=1),
    ],
    var_from_source=[["x"], ["y"]],
  )
  ctx = new_code_gen_context()
  out = jit_nested_cartesian_join(cart, ctx, "body();\n")
  # lane + group_size preamble
  assert "uint32_t lane_" in out and ".thread_rank();" in out
  assert "uint32_t group_size_" in out and ".size();" in out
  # Per-source view + degree
  assert "auto view_R_0 = views[0];" in out
  assert "auto view_S_1 = views[1];" in out
  assert "uint32_t degree_" in out  # two degree_N decls
  assert out.count("uint32_t degree_") >= 2
  # Validity check with `continue`
  assert "!" in out and ".valid()" in out
  assert "continue;" in out
  # Total + parallel flat loop (lane + group_size stride)
  assert "uint32_t total_" in out
  assert "for (uint32_t flat_idx_" in out
  assert "< total_" in out
  assert "+= group_size_" in out
  # 2-source major_is_1 decomposition
  assert "const bool major_is_1_" in out
  # Bound vars via get_value(col, handle.begin() + idx)
  assert "auto x = view_R_0.get_value(0, " in out and ".begin() + idx" in out
  assert "auto y = view_S_1.get_value(0, " in out
  assert "body();" in out


def test_cart_toggles_inside_cartesian_state_and_restores():
  cart = m.CartesianJoin(
    vars=["x"],
    sources=[_cs("R", Version.FULL, [0], handle_start=0)],
    var_from_source=[["x"]],
  )
  ctx = new_code_gen_context()
  assert ctx.inside_cartesian is False
  jit_nested_cartesian_join(cart, ctx, "body();\n")
  # Restored to False after emit
  assert ctx.inside_cartesian is False
  assert ctx.cartesian_bound_vars == []


def test_cart_ws_valid_flag_raises_not_implemented():
  cart = m.CartesianJoin(
    vars=["x"],
    sources=[_cs("R", Version.FULL, [0], handle_start=0)],
    var_from_source=[["x"]],
  )
  ctx = new_code_gen_context()
  ctx.ws_cartesian_valid_var = "ws_valid"
  try:
    jit_nested_cartesian_join(cart, ctx, "")
  except NotImplementedError as e:
    assert "ws Cartesian" in str(e)
  else:
    raise AssertionError("expected NotImplementedError")


# ---------------------------------------------------------------------------
# jit_positioned_extract
# ---------------------------------------------------------------------------

def test_positioned_extract_baseline():
  pe = m.PositionedExtract(
    sources=[
      _cs("R", Version.FULL, [0, 1], handle_start=0),
      _cs("S", Version.FULL, [0, 1], handle_start=1),
    ],
    var_name="z",
    bind_vars=["w"],
  )
  ctx = new_code_gen_context()
  # Pretend we're inside a Cartesian loop so extract_var probe uses prefix_seq
  ctx.cartesian_bound_vars = ["root_var"]
  out = jit_positioned_extract(pe, ctx, "        body();\n")
  assert "// PositionedExtract: extract z then bind w" in out
  assert "auto view_R_0 = views[0];" in out
  assert "auto view_S_1 = views[1];" in out
  # Probes with prefix_seq since cartesian_bound_vars includes extract_var
  assert ".prefix_seq(z," in out
  # Valid flag
  assert "bool pe_valid_" in out
  # if (pe_valid) wraps the binding + body
  assert "if (pe_valid_" in out
  # `w` binding
  assert "auto w = view_R_0.get_value" in out
  assert "body();" in out


if __name__ == "__main__":
  import inspect
  this = sys.modules[__name__]
  passed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if name.startswith("test_"):
      fn()
      passed += 1
  print(f"{passed} passed")
