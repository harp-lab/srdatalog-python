'''Tests for ir/dialects/target/cuda/plugin.py — DSAI default hooks + registry dispatch.'''

import sys

from srdatalog.ir.codegen.cuda.plugin import (
  IndexPlugin,
  PrefixMode,
  get_extra_headers_for_types,
  plugin_chained_prefix_calls,
  plugin_chained_prefix_with_last_lower_bound,
  plugin_gen_child,
  plugin_gen_child_range,
  plugin_gen_degree,
  plugin_gen_get_value,
  plugin_gen_get_value_at,
  plugin_gen_host_view_setup,
  plugin_gen_iterators,
  plugin_gen_prefix,
  plugin_gen_prefix_lower_bound,
  plugin_gen_root_handle,
  plugin_gen_valid,
  plugin_view_count,
  register_index_plugin,
  resolve_plugin,
)

# -----------------------------------------------------------------------------
# DSAI default expressions
# -----------------------------------------------------------------------------


def test_dsai_root_handle():
  assert plugin_gen_root_handle("view_Edge_0") == ("HandleType(0, view_Edge_0.num_rows_, 0)")


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
  assert plugin_gen_get_value_at("h", "view_R_0", "idx") == ("h.get_value_at(view_R_0, idx)")
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
  from srdatalog.ir.codegen.cuda import plugin as _p

  # Capture any pre-existing registration (e.g., the real two_level
  # plugin, registered at import time of ir.dialects.relation.d2l.cuda)
  # so teardown can restore it. Bare `.pop()` would leave the registry
  # empty and break downstream tests that rely on D2L plugin_view_count.
  prior = _p._PLUGIN_REGISTRY.get("SRDatalog::GPU::Device2LevelIndex")
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
    if prior is not None:
      _p._PLUGIN_REGISTRY["SRDatalog::GPU::Device2LevelIndex"] = prior
    else:
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
    from srdatalog.ir.codegen.cuda import plugin as _p

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
    "root_h",
    ["x", "y"],
    "view_R",
    scalar_mode=True,
  )
  assert got == "root_h.prefix_seq(x, view_R).prefix_seq(y, view_R)"


def test_chained_prefix_cartesian_bound_var_goes_sequential():
  got = plugin_chained_prefix_calls(
    "root_h",
    ["x", "y"],
    "view_R",
    cartesian_bound_vars=["x"],
  )
  # x is in cartesian bound -> sequential; y isn't -> cooperative
  assert got == "root_h.prefix_seq(x, view_R).prefix(y, tile, view_R)"


def test_chained_prefix_with_last_lower_bound():
  got = plugin_chained_prefix_with_last_lower_bound(
    "root_h",
    ["x", "y", "z"],
    "view_R",
  )
  assert got == (
    "root_h.prefix(x, tile, view_R).prefix(y, tile, view_R).prefix_lower_bound(z, tile, view_R)"
  )


# -----------------------------------------------------------------------------
# PluginRegistry instance form (Bundle C — encapsulated state)
# -----------------------------------------------------------------------------


def test_registry_fresh_instance_resolves_to_default():
  '''A fresh PluginRegistry has DSAI as its default and no registered
  plugins; resolve("") and resolve(unknown) both return DSAI.'''
  from srdatalog.ir.codegen.cuda.plugin import PluginRegistry

  reg = PluginRegistry()
  assert reg.resolve("").name == "DeviceSortedArrayIndex"
  assert reg.resolve("Some::Unknown::Index").name == "DeviceSortedArrayIndex"
  assert reg.registered_types() == frozenset()


def test_registry_register_then_resolve_exact():
  from srdatalog.ir.codegen.cuda.plugin import PluginRegistry

  reg = PluginRegistry()
  custom = IndexPlugin(
    name="MyIndex",
    cpp_type="My::Index",
    cpp_headers=["my_index.h"],
    gen_root_handle=lambda v: f"MyHandle({v})",
  )
  reg.register(custom)
  assert reg.resolve("My::Index").name == "MyIndex"
  assert reg.gen_root_handle("v", "My::Index") == "MyHandle(v)"
  assert "My::Index" in reg
  assert reg.registered_types() == frozenset({"My::Index"})


def test_registry_substring_match_resolves_partial_type():
  '''resolve("My::Index<int, 4>") matches the registered "My::Index"
  because of the substring fallback (mirrors Nim's resolvePlugin).'''
  from srdatalog.ir.codegen.cuda.plugin import PluginRegistry

  reg = PluginRegistry()
  reg.register(
    IndexPlugin(name="Mine", cpp_type="My::Index", gen_degree=lambda h: f"{h}.deg_mine()")
  )
  assert reg.resolve("My::Index<int, 4>").name == "Mine"
  assert reg.gen_degree("h", "My::Index<int, 4>") == "h.deg_mine()"


def test_registry_register_is_idempotent_overwrite():
  '''Re-registering the same cpp_type overwrites the previous entry.'''
  from srdatalog.ir.codegen.cuda.plugin import PluginRegistry

  reg = PluginRegistry()
  reg.register(IndexPlugin(name="V1", cpp_type="X"))
  reg.register(IndexPlugin(name="V2", cpp_type="X"))
  assert reg.resolve("X").name == "V2"
  assert reg.registered_types() == frozenset({"X"})


def test_registry_extra_headers_for_types_dedupes_and_skips_empty():
  from srdatalog.ir.codegen.cuda.plugin import PluginRegistry

  reg = PluginRegistry()
  reg.register(IndexPlugin(name="A", cpp_type="A_T", cpp_headers=["a.h", "shared.h"]))
  reg.register(IndexPlugin(name="B", cpp_type="B_T", cpp_headers=["b.h", "shared.h"]))
  got = reg.extra_headers_for_types(["", "A_T", "B_T", "A_T"])
  assert got == ["a.h", "shared.h", "b.h"]


def test_registry_instances_are_isolated():
  '''Two PluginRegistry instances do not share state.'''
  from srdatalog.ir.codegen.cuda.plugin import PluginRegistry

  reg_a = PluginRegistry()
  reg_b = PluginRegistry()
  reg_a.register(IndexPlugin(name="OnlyInA", cpp_type="A::Only"))
  assert "A::Only" in reg_a
  assert "A::Only" not in reg_b
  assert reg_b.resolve("A::Only").name == "DeviceSortedArrayIndex"


def test_get_default_registry_returns_singleton():
  '''The module-level default registry is a singleton; back-compat
  module functions all delegate to it.'''
  from srdatalog.ir.codegen.cuda.plugin import get_default_registry

  assert get_default_registry() is get_default_registry()


if __name__ == "__main__":
  import inspect

  this = sys.modules[__name__]
  passed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if name.startswith("test_"):
      fn()
      passed += 1
  print(f"{passed} passed")
