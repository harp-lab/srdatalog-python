'''Tests for codegen/jit/plugin.py — DSAI default hooks + registry dispatch.'''
import sys
from pathlib import Path


from srdatalog.codegen.jit.plugin import (
  PrefixMode,
  IndexPlugin,
  new_default_plugin,
  resolve_plugin,
  register_index_plugin,
  get_extra_headers_for_types,
  plugin_gen_root_handle,
  plugin_gen_prefix,
  plugin_gen_prefix_lower_bound,
  plugin_gen_degree,
  plugin_gen_valid,
  plugin_gen_get_value_at,
  plugin_gen_get_value,
  plugin_gen_child,
  plugin_gen_child_range,
  plugin_gen_iterators,
  plugin_view_count,
  plugin_gen_host_view_setup,
  plugin_chained_prefix_calls,
  plugin_chained_prefix_with_last_lower_bound,
)


# -----------------------------------------------------------------------------
# DSAI default expressions
# -----------------------------------------------------------------------------

def test_dsai_root_handle():
  assert plugin_gen_root_handle("view_Edge_0") == (
    "HandleType(0, view_Edge_0.num_rows_, 0)"
  )


def test_dsai_prefix_cooperative():
  assert plugin_gen_prefix("h", "x", "view_R_0", PrefixMode.COOPERATIVE) == (
    "h.prefix(x, tile, view_R_0)"
  )


def test_dsai_prefix_sequential():
  assert plugin_gen_prefix("h", "x", "view_R_0", PrefixMode.SEQUENTIAL) == (
    "h.prefix_seq(x, view_R_0)"
  )


def test_dsai_prefix_lower_bound_cooperative():
  assert plugin_gen_prefix_lower_bound("h", "x", "v", PrefixMode.COOPERATIVE) == (
    "h.prefix_lower_bound(x, tile, v)"
  )


def test_dsai_prefix_lower_bound_sequential():
  assert plugin_gen_prefix_lower_bound("h", "x", "v", PrefixMode.SEQUENTIAL) == (
    "h.prefix_lower_bound_seq(x, v)"
  )


def test_dsai_degree_and_valid():
  assert plugin_gen_degree("h_42") == "h_42.degree()"
  assert plugin_gen_valid("h_42") == "h_42.valid()"


def test_dsai_get_value_at_and_get_value():
  assert plugin_gen_get_value_at("h", "view_R_0", "idx") == (
    "h.get_value_at(view_R_0, idx)"
  )
  assert plugin_gen_get_value("view_R_0", 1, "pos") == "view_R_0.get_value(1, pos)"


def test_dsai_child_and_child_range():
  assert plugin_gen_child("h", "i") == "h.child(i)"
  assert plugin_gen_child_range("h", "pos", "k", "tile", "view") == (
    "h.child_range(pos, k, tile, view)"
  )


def test_dsai_iterators():
  assert plugin_gen_iterators("h", "view_R_0") == "h.iterators(view_R_0)"


def test_dsai_view_count_and_host_setup():
  assert plugin_view_count("FULL_VER") == 1
  assert plugin_gen_host_view_setup("idx", "FULL_VER") == ["idx.view()"]


# -----------------------------------------------------------------------------
# Registry + resolution
# -----------------------------------------------------------------------------

def test_resolve_empty_string_returns_default():
  p = resolve_plugin("")
  assert p.name == "DeviceSortedArrayIndex"


def test_resolve_unknown_falls_back_to_default():
  assert resolve_plugin("SomeWeirdIndex").name == "DeviceSortedArrayIndex"


def test_resolve_registered_plugin_exact_match():
  custom = IndexPlugin(
    name="TwoLevel",
    cpp_type="SRDatalog::GPU::Device2LevelIndex",
    cpp_headers=["gpu/two_level.h"],
    gen_root_handle=lambda v: f"TwoLevelHandle(0, {v}.num_rows_, 0)",
  )
  register_index_plugin(custom)
  try:
    p = resolve_plugin("SRDatalog::GPU::Device2LevelIndex")
    assert p.name == "TwoLevel"
    # dispatch uses the overridden hook
    assert plugin_gen_root_handle("v", "SRDatalog::GPU::Device2LevelIndex") == (
      "TwoLevelHandle(0, v.num_rows_, 0)"
    )
  finally:
    # teardown so other tests don't see leaked state
    from srdatalog.codegen.jit import plugin as _p
    _p._PLUGIN_REGISTRY.pop("SRDatalog::GPU::Device2LevelIndex", None)


def test_get_extra_headers_dedupes_and_skips_empty():
  custom = IndexPlugin(
    name="Z",
    cpp_type="Z_Index",
    cpp_headers=["gpu/z.h", "gpu/shared.h"],
  )
  register_index_plugin(custom)
  try:
    got = get_extra_headers_for_types(["", "Z_Index", "Z_Index"])
    assert got == ["gpu/z.h", "gpu/shared.h"]
  finally:
    from srdatalog.codegen.jit import plugin as _p
    _p._PLUGIN_REGISTRY.pop("Z_Index", None)


# -----------------------------------------------------------------------------
# Chained prefix calls
# -----------------------------------------------------------------------------

def test_chained_prefix_empty_returns_parent():
  assert plugin_chained_prefix_calls("root_h", [], "view") == "root_h"


def test_chained_prefix_cooperative_default():
  got = plugin_chained_prefix_calls("root_h", ["x", "y"], "view_R")
  assert got == "root_h.prefix(x, tile, view_R).prefix(y, tile, view_R)"


def test_chained_prefix_scalar_mode_all_sequential():
  got = plugin_chained_prefix_calls(
    "root_h", ["x", "y"], "view_R", scalar_mode=True,
  )
  assert got == "root_h.prefix_seq(x, view_R).prefix_seq(y, view_R)"


def test_chained_prefix_cartesian_bound_var_goes_sequential():
  got = plugin_chained_prefix_calls(
    "root_h", ["x", "y"], "view_R", cartesian_bound_vars=["x"],
  )
  # x is in cartesian bound -> sequential; y isn't -> cooperative
  assert got == "root_h.prefix_seq(x, view_R).prefix(y, tile, view_R)"


def test_chained_prefix_with_last_lower_bound():
  got = plugin_chained_prefix_with_last_lower_bound(
    "root_h", ["x", "y", "z"], "view_R",
  )
  assert got == (
    "root_h.prefix(x, tile, view_R).prefix(y, tile, view_R)"
    ".prefix_lower_bound(z, tile, view_R)"
  )


if __name__ == "__main__":
  import inspect
  this = sys.modules[__name__]
  passed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if name.startswith("test_"):
      fn()
      passed += 1
  print(f"{passed} passed")
