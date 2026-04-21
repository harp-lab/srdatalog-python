'''Tests for codegen/jit/context.py — CodeGenContext + indentation / name /
key / plugin-dispatched expression helpers.'''

import sys

from srdatalog.codegen.jit.context import (
  CPP_KEYWORDS,
  NegPreNarrowInfo,
  RunnerGenState,
  dec_indent,
  default_hooks,
  gen_chained_prefix_calls,
  gen_chained_prefix_calls_seq,
  gen_chained_prefix_with_last_lower_bound,
  gen_child,
  gen_child_range,
  gen_degree,
  gen_get_value,
  gen_get_value_at,
  gen_handle_state_key,
  gen_handle_var_name,
  gen_index_spec_key,
  gen_iterators,
  gen_root_handle,
  gen_root_handle_from_view_idx,
  gen_unique_name,
  gen_valid,
  gen_view_access,
  gen_view_var_name,
  get_rel_index_type,
  get_view_slot_base,
  inc_indent,
  ind,
  is_var_bound,
  new_code_gen_context,
  sanitize_var_name,
  with_bound_var,
)

# -----------------------------------------------------------------------------
# Sanitization
# -----------------------------------------------------------------------------


def test_sanitize_keyword():
  assert sanitize_var_name("int") == "int_val"
  assert sanitize_var_name("class") == "class_val"
  assert sanitize_var_name("template") == "template_val"


def test_sanitize_non_keyword_passthrough():
  assert sanitize_var_name("x") == "x"
  assert sanitize_var_name("my_var") == "my_var"
  assert sanitize_var_name("PointsTo") == "PointsTo"


def test_cpp_keywords_cover_common_primitives():
  for kw in ("int", "bool", "double", "void", "return", "this", "template"):
    assert kw in CPP_KEYWORDS


# -----------------------------------------------------------------------------
# Context defaults
# -----------------------------------------------------------------------------


def test_new_code_gen_context_defaults():
  ctx = new_code_gen_context()
  assert ctx.bound_vars == []
  assert ctx.handle_vars == {}
  assert ctx.view_vars == {}
  assert ctx.indent == 2
  assert ctx.name_counter == 0
  assert ctx.debug is True
  assert ctx.output_var_name == "output"
  assert ctx.group_size == 32
  assert ctx.tile_var == "tile"
  assert ctx.parent_tile_var == "tile"
  assert ctx.is_leaf_level is False
  assert ctx.is_counting is False
  assert ctx.scalar_mode is False
  assert ctx.ws_enabled is False
  assert ctx.bg_enabled is False
  assert ctx.dedup_hash_enabled is False


def test_default_hooks_are_no_ops():
  h = default_hooks()
  # wrap_emit is identity
  ctx = new_code_gen_context()
  assert h.wrap_emit("foo();", ctx) == "foo();"
  # pre/post column join are no-ops (don't raise)
  h.pre_column_join(None, ctx)
  h.post_column_join(None, ctx)
  h.pre_cartesian_join(None, [], ctx)
  # Extra-emit hooks return empty strings
  state = RunnerGenState()
  assert h.emit_extra_types(state) == ""
  assert h.emit_extra_kernels(state) == ""
  assert h.emit_phase_methods(state) == ""


# -----------------------------------------------------------------------------
# Indentation
# -----------------------------------------------------------------------------


def test_indent_default_is_two_levels():
  ctx = new_code_gen_context()
  assert ind(ctx) == "    "  # indent=2 * 2 spaces each


def test_inc_dec_indent():
  ctx = new_code_gen_context()
  inc_indent(ctx)
  assert ind(ctx) == "      "
  dec_indent(ctx)
  assert ind(ctx) == "    "


def test_dec_indent_clamps_at_zero():
  ctx = new_code_gen_context()
  for _ in range(10):
    dec_indent(ctx)
  assert ctx.indent == 0
  assert ind(ctx) == ""


# -----------------------------------------------------------------------------
# Unique name + bound var tracking
# -----------------------------------------------------------------------------


def test_gen_unique_name_increments():
  ctx = new_code_gen_context()
  assert gen_unique_name(ctx, "h") == "h_1"
  assert gen_unique_name(ctx, "h") == "h_2"
  assert gen_unique_name(ctx, "x") == "x_3"


def test_with_bound_var_returns_copy():
  ctx = new_code_gen_context()
  ctx2 = with_bound_var(ctx, "x")
  assert ctx.bound_vars == []
  assert ctx2.bound_vars == ["x"]
  # Original is unchanged
  assert "x" not in ctx.bound_vars


def test_is_var_bound():
  ctx = new_code_gen_context()
  ctx.bound_vars = ["x", "y"]
  assert is_var_bound(ctx, "x")
  assert is_var_bound(ctx, "y")
  assert not is_var_bound(ctx, "z")


def test_get_rel_index_type_falls_back_to_empty():
  ctx = new_code_gen_context()
  assert get_rel_index_type(ctx, "PointsTo") == ""
  ctx.rel_index_types["subset"] = "SRDatalog::GPU::Device2LevelIndex"
  assert get_rel_index_type(ctx, "subset") == ("SRDatalog::GPU::Device2LevelIndex")


def test_get_view_slot_base_falls_back_to_handle_idx():
  ctx = new_code_gen_context()
  assert get_view_slot_base(ctx, 5) == 5
  ctx.view_slot_offsets[5] = 12
  assert get_view_slot_base(ctx, 5) == 12


# -----------------------------------------------------------------------------
# Name / key generators
# -----------------------------------------------------------------------------


def test_gen_view_access():
  assert gen_view_access(0) == "views[0]"
  assert gen_view_access(42) == "views[42]"


def test_gen_view_var_name():
  assert gen_view_var_name("PointsTo", 3) == "view_PointsTo_3"


def test_gen_handle_var_name_unique():
  ctx = new_code_gen_context()
  a = gen_handle_var_name("PointsTo", 0, ctx)
  b = gen_handle_var_name("PointsTo", 0, ctx)
  assert a != b
  assert a.startswith("h_PointsTo_0_")
  assert b.startswith("h_PointsTo_0_")


def test_gen_index_spec_key_without_version():
  assert gen_index_spec_key("PointsTo", [0, 1]) == "PointsTo_0_1"
  assert gen_index_spec_key("R", [1, 0, 2]) == "R_1_0_2"


def test_gen_index_spec_key_with_version():
  assert gen_index_spec_key("PointsTo", [0, 1], "DELTA") == "PointsTo_0_1_DELTA"


def test_gen_handle_state_key_no_prefixes():
  assert gen_handle_state_key("R", [0, 1], []) == "R_0_1"
  assert gen_handle_state_key("R", [0, 1], [], "FULL") == "R_0_1_FULL"


def test_gen_handle_state_key_with_prefixes():
  got = gen_handle_state_key(
    "MethodLookup",
    [2, 0, 1, 3],
    ["heaptype", "simplename"],
  )
  assert got == "MethodLookup_2_0_1_3_heaptype_simplename"


# -----------------------------------------------------------------------------
# Plugin-dispatched C++ expression helpers
# -----------------------------------------------------------------------------


def test_gen_root_handle_dsai_default():
  assert gen_root_handle("view_Edge_0") == ("HandleType(0, view_Edge_0.num_rows_, 0)")


def test_gen_root_handle_from_view_idx():
  assert gen_root_handle_from_view_idx(2) == ("HandleType(0, views[2].num_rows_, 0)")


def test_gen_degree_and_valid():
  assert gen_degree("h_1") == "h_1.degree()"
  assert gen_valid("h_1") == "h_1.valid()"


def test_gen_get_value_and_get_value_at():
  assert gen_get_value("view_R_0", 0, "idx") == "view_R_0.get_value(0, idx)"
  assert gen_get_value_at("h", "v", "i") == "h.get_value_at(v, i)"


def test_gen_child_and_child_range():
  assert gen_child("h", "i") == "h.child(i)"
  assert gen_child_range("h", "p", "k", "tile", "v") == ("h.child_range(p, k, tile, v)")


def test_gen_iterators():
  assert gen_iterators("h", "v") == "h.iterators(v)"


def test_gen_chained_prefix_calls_sanitizes_keywords():
  # 'int' is a C++ keyword — should become int_val
  got = gen_chained_prefix_calls("root", ["int"], "view_R_0")
  assert "int_val" in got
  assert "tile" in got  # cooperative mode


def test_gen_chained_prefix_calls_sequential_scalar_mode():
  got = gen_chained_prefix_calls(
    "root",
    ["x", "y"],
    "view_R_0",
    scalar_mode=True,
  )
  assert got == "root.prefix_seq(x, view_R_0).prefix_seq(y, view_R_0)"


def test_gen_chained_prefix_with_last_lower_bound():
  got = gen_chained_prefix_with_last_lower_bound(
    "root",
    ["x", "y"],
    "view_R_0",
  )
  assert got == ("root.prefix(x, tile, view_R_0).prefix_lower_bound(y, tile, view_R_0)")


def test_gen_chained_prefix_calls_seq_all_sequential():
  got = gen_chained_prefix_calls_seq("root", ["x", "y"], "view_R_0")
  assert got == "root.prefix_seq(x, view_R_0).prefix_seq(y, view_R_0)"


# -----------------------------------------------------------------------------
# NegPreNarrowInfo + RunnerGenState defaults
# -----------------------------------------------------------------------------


def test_neg_pre_narrow_info_defaults():
  n = NegPreNarrowInfo()
  assert n.var_name == ""
  assert n.pre_vars == []
  assert n.in_cartesian_vars == []
  assert n.pre_consts == []


def test_runner_gen_state_defaults():
  r = RunnerGenState()
  assert r.db_type_name == ""
  assert r.dest_arities == []
  assert r.is_balanced is False
  assert r.is_work_stealing is False


if __name__ == "__main__":
  import inspect

  this = sys.modules[__name__]
  passed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if name.startswith("test_"):
      fn()
      passed += 1
  print(f"{passed} passed")
