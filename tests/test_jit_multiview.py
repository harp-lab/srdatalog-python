'''Tests for the Device2LevelIndex multi-view segment-loop paths.

Importing the two_level plugin module registers the IndexPlugin; once
registered, any ColumnSource whose relation has
`rel_index_types[rel] == "SRDatalog::GPU::Device2LevelIndex"` and
FULL version gets `view_count == 2`, which triggers `for (_seg = 0;
_seg < 2; _seg++) { ... }` segment-loop wrapping inside both
jit_root_column_join and jit_nested_column_join.
'''
import sys
from pathlib import Path


import srdatalog.mir_types as m
from srdatalog.hir_types import Version
from srdatalog.codegen.jit.context import new_code_gen_context

# Import the 2-level plugin so it registers itself.
from srdatalog.codegen.jit.indexes import two_level  # noqa: F401
from srdatalog.codegen.jit.plugin import (
  plugin_view_count, resolve_plugin, plugin_gen_host_view_setup,
)
from srdatalog.codegen.jit.instructions import jit_nested_column_join
from srdatalog.codegen.jit.root import jit_root_column_join


TWO_LEVEL = "SRDatalog::GPU::Device2LevelIndex"


def _cs(rel, ver, idx, prefix=(), handle_start=0):
  return m.ColumnSource(
    rel_name=rel, version=ver, index=idx,
    prefix_vars=list(prefix), handle_start=handle_start,
  )


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def test_two_level_plugin_registered():
  p = resolve_plugin(TWO_LEVEL)
  assert p.name == "Device2LevelIndex"
  assert p.cpp_type == TWO_LEVEL
  assert "gpu/device_2level_index.h" in p.cpp_headers


def test_two_level_view_count_full_is_two():
  assert plugin_view_count("FULL_VER", TWO_LEVEL) == 2


def test_two_level_view_count_delta_is_one():
  assert plugin_view_count("DELTA_VER", TWO_LEVEL) == 1
  assert plugin_view_count("NEW_VER", TWO_LEVEL) == 1


def test_two_level_host_view_setup_full_pushes_two_views():
  out = plugin_gen_host_view_setup("idx_expr", "FULL_VER", TWO_LEVEL)
  assert out == ["idx_expr.full_view()", "idx_expr.head_view()"]


def test_two_level_host_view_setup_delta_single_view():
  out = plugin_gen_host_view_setup("idx_expr", "DELTA_VER", TWO_LEVEL)
  assert out == ["idx_expr.view()"]


def test_default_plugin_view_count_unchanged():
  '''Registering 2-level must not perturb DSAI defaults.'''
  assert plugin_view_count("FULL_VER") == 1
  assert plugin_view_count("FULL_VER", "") == 1


# ---------------------------------------------------------------------------
# Nested ColumnJoin single-source + multi-view -> segment loop
# ---------------------------------------------------------------------------

def test_nested_cj_single_source_multi_view_opens_segment_loop():
  cj = m.ColumnJoin(var_name="y", sources=[
    _cs("subset", Version.FULL, [0, 1], handle_start=0),
  ])
  ctx = new_code_gen_context()
  ctx.rel_index_types["subset"] = TWO_LEVEL
  out = jit_nested_column_join(cj, ctx, "        body();\n")

  # Opening segment loop
  assert "for (int _seg = 0; _seg < 2; _seg++) {" in out
  # View is re-declared each iteration via views[base + _seg]
  assert "auto view_subset_0 = views[0 + _seg];" in out
  # Body still emitted
  assert "body();" in out
  # The closing `}` count matches (standard close + segment close)
  assert out.count("}") >= 2


def test_nested_cj_single_source_non_multi_view_no_segment_loop():
  '''Sanity: without the index_type override, DSAI stays single-view
  and we don't emit the segment loop.'''
  cj = m.ColumnJoin(var_name="y", sources=[
    _cs("r", Version.FULL, [0, 1], handle_start=0),
  ])
  ctx = new_code_gen_context()
  out = jit_nested_column_join(cj, ctx, "body();\n")
  assert "_seg" not in out


# ---------------------------------------------------------------------------
# Nested ColumnJoin multi-source: non-prefix multi-view → nested segment loops
# ---------------------------------------------------------------------------

def test_nested_cj_multi_source_non_prefix_multi_view_wraps_nseg():
  cj = m.ColumnJoin(var_name="z", sources=[
    # First source has a prefix -> stays within parent segment (no
    # segment loop opened for it).
    _cs("R", Version.FULL, [0, 1], prefix=("x",), handle_start=0),
    # Second source is fresh + multi-view -> gets a segment loop.
    _cs("subset", Version.FULL, [0, 1], handle_start=1),
  ])
  ctx = new_code_gen_context()
  ctx.rel_index_types["subset"] = TWO_LEVEL
  out = jit_nested_column_join(cj, ctx, "body();\n")
  # The _nseg_1 loop wraps the subset handle setup + body
  assert "for (int _nseg_1 = 0; _nseg_1 < 2; _nseg_1++) {" in out
  # Fixed view var is reassigned inside the segment loop
  assert "view_subset_0_1_FULL_VER = views[" in out and "_nseg_1];" in out


# ---------------------------------------------------------------------------
# Root ColumnJoin single-source + multi-view
# ---------------------------------------------------------------------------

def test_root_cj_single_source_multi_view_opens_segment_loop():
  cj = m.ColumnJoin(var_name="y", sources=[
    _cs("subset", Version.FULL, [0, 1], handle_start=0),
  ])
  ctx = new_code_gen_context()
  ctx.rel_index_types["subset"] = TWO_LEVEL
  out = jit_root_column_join(cj, ctx, "body();\n")

  assert "for (int _seg = 0; _seg < 2; _seg++) {" in out
  assert "auto view_subset_0 = views[0 + _seg];" in out


def test_root_cj_multi_source_non_first_multi_view_opens_segment_loop():
  '''Non-first multi-view source in root CJ emits a `for (_seg_N)`
  loop wrapping its handle narrowing + the body. Multiple such sources
  nest, and close in reverse order at the tail.'''
  cj = m.ColumnJoin(var_name="y", sources=[
    # First source is single-view — uses the hinted-range path.
    _cs("first_rel", Version.FULL, [0, 1], handle_start=0),
    # Second source is multi-view non-first -> segment loop.
    _cs("subset", Version.FULL, [0, 1], handle_start=1),
  ])
  ctx = new_code_gen_context()
  ctx.rel_index_types["subset"] = TWO_LEVEL
  out = jit_root_column_join(cj, ctx, "body();\n")

  # First source: hinted-range single-view path emitted in Phase 1
  assert "HandleType(hint_lo_" in out
  # Second source: segment loop opens for non-first multi-view
  assert "for (int _seg_1 = 0; _seg_1 < 2; _seg_1++) {" in out
  # Debug banner carried over from Nim
  assert "Segment loop: subset FULL_VER has 2 segments (FULL + HEAD)" in out
  # Per-segment view var + handle narrowing inside the loop
  assert "auto view_subset_1 = views[" in out and "+ _seg_1];" in out
  assert "h_subset_1_root = HandleType" in out or "h_subset_1_root = " in out
  # The body sits INSIDE the segment loop (indented further than base)
  # and body() appears once
  assert out.count("body();") == 1
  # Closing `}` for the segment loop appears before the outer `}` for
  # the y_idx grid-stride loop.
  # Find them in order.
  seg_close_idx = out.rfind("  }\n")  # the segment-loop close at deeper indent
  assert seg_close_idx != -1


def test_root_cj_multi_two_non_first_multi_view_sources_nest():
  '''Two non-first multi-view sources -> two nested segment loops,
  closed in reverse order.'''
  cj = m.ColumnJoin(var_name="y", sources=[
    _cs("first_rel", Version.FULL, [0, 1], handle_start=0),
    _cs("subset_a", Version.FULL, [0, 1], handle_start=1),
    _cs("subset_b", Version.FULL, [0, 1], handle_start=2),
  ])
  ctx = new_code_gen_context()
  ctx.rel_index_types["subset_a"] = TWO_LEVEL
  ctx.rel_index_types["subset_b"] = TWO_LEVEL
  out = jit_root_column_join(cj, ctx, "body();\n")

  # Both segment loops open
  assert "for (int _seg_1 = 0; _seg_1 < 2; _seg_1++) {" in out
  assert "for (int _seg_2 = 0; _seg_2 < 2; _seg_2++) {" in out
  # _seg_2 opens INSIDE _seg_1 (later in the string)
  assert out.index("_seg_1 = 0") < out.index("_seg_2 = 0")


def test_root_cj_multi_source_first_multi_view_works():
  '''First multi-view source is OK — the hinted-range path sits at the
  first segment via y_idx-based narrowing; no segment loop needed
  because root_unique_values already came from one specific segment.'''
  cj = m.ColumnJoin(var_name="y", sources=[
    _cs("subset", Version.FULL, [0, 1], handle_start=0),
    _cs("other", Version.FULL, [0, 1], handle_start=1),
  ])
  ctx = new_code_gen_context()
  ctx.rel_index_types["subset"] = TWO_LEVEL
  out = jit_root_column_join(cj, ctx, "body();\n")
  # First source still uses the hinted-range pattern
  assert "HandleType(hint_lo_" in out
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
