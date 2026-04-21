'''Tests for codegen/jit/view_management.py.'''

import sys

import srdatalog.mir.types as m
from srdatalog.codegen.jit.context import new_code_gen_context
from srdatalog.codegen.jit.view_management import (
  ViewSpec,
  build_root_slot_map,
  collect_unique_view_specs,
  compute_total_view_count,
  compute_view_slot_offsets,
  get_source_index,
  jit_emit_view_declarations,
  register_pipeline_handles,
  source_spec_key,
  spec_key,
)
from srdatalog.hir.types import Version


def _cs(rel, ver, idx, handle_start=-1, prefix=()):
  return m.ColumnSource(
    rel_name=rel,
    version=ver,
    index=idx,
    prefix_vars=list(prefix),
    handle_start=handle_start,
  )


# -----------------------------------------------------------------------------
# Source helpers
# -----------------------------------------------------------------------------


def test_get_source_index_covers_all_source_kinds():
  assert get_source_index(_cs("R", Version.FULL, [0, 1])) == [0, 1]
  assert get_source_index(m.Scan(vars=["x"], rel_name="R", version=Version.FULL, index=[1, 0])) == [
    1,
    0,
  ]
  assert get_source_index(m.Negation(rel_name="R", version=Version.FULL, index=[0])) == [0]
  assert get_source_index(
    m.Aggregate(result_var="c", agg_func="AggCount", rel_name="R", version=Version.FULL, index=[0])
  ) == [0]
  # Not a source spec -> empty
  assert (
    get_source_index(m.InsertInto(rel_name="R", version=Version.NEW, vars=["x"], index=[0])) == []
  )


def test_source_spec_key_scan_vs_column_source_equal():
  # Same (rel, ver, idx) from Scan or ColumnSource yield identical keys
  scan = m.Scan(vars=["x"], rel_name="R", version=Version.FULL, index=[0, 1])
  cs = _cs("R", Version.FULL, [0, 1])
  assert source_spec_key(scan) == source_spec_key(cs) == "R_FULL_VER_0_1"


def test_source_spec_key_distinguishes_delta_vs_full():
  a = _cs("R", Version.DELTA, [0, 1])
  b = _cs("R", Version.FULL, [0, 1])
  assert source_spec_key(a) != source_spec_key(b)


def test_source_spec_key_distinguishes_index_orderings():
  a = _cs("R", Version.FULL, [0, 1])
  b = _cs("R", Version.FULL, [1, 0])
  assert source_spec_key(a) != source_spec_key(b)


# -----------------------------------------------------------------------------
# compute_total_view_count
# -----------------------------------------------------------------------------


def test_total_view_count_single_source():
  assert compute_total_view_count([_cs("R", Version.FULL, [0, 1])], {}) == 1


def test_total_view_count_dedupes_same_spec():
  specs = [
    _cs("R", Version.FULL, [0, 1]),
    _cs("R", Version.FULL, [0, 1]),  # duplicate
    _cs("S", Version.FULL, [0, 1]),
  ]
  assert compute_total_view_count(specs, {}) == 2


def test_total_view_count_distinct_versions_count_twice():
  specs = [
    _cs("R", Version.DELTA, [0, 1]),
    _cs("R", Version.FULL, [0, 1]),
  ]
  assert compute_total_view_count(specs, {}) == 2


# -----------------------------------------------------------------------------
# compute_view_slot_offsets
# -----------------------------------------------------------------------------


def test_view_slot_offsets_assigns_distinct_slots_per_unique_spec():
  specs = [
    _cs("R", Version.DELTA, [0, 1], handle_start=0),
    _cs("S", Version.FULL, [0, 1], handle_start=1),
    _cs("T", Version.FULL, [0, 1], handle_start=2),
  ]
  offsets = compute_view_slot_offsets(specs, {})
  assert offsets == {0: 0, 1: 1, 2: 2}


def test_view_slot_offsets_dedupes_handles_sharing_spec():
  '''Two distinct handles referring to the same (rel, ver, idx) share
  one view slot.'''
  specs = [
    _cs("R", Version.DELTA, [0, 1], handle_start=0),
    _cs("R", Version.DELTA, [0, 1], handle_start=5),  # same spec
    _cs("S", Version.FULL, [0, 1], handle_start=2),
  ]
  offsets = compute_view_slot_offsets(specs, {})
  # handles 0 and 5 share slot 0; S gets slot 1
  assert offsets == {0: 0, 5: 0, 2: 1}


def test_view_slot_offsets_skips_negative_handle_start():
  specs = [_cs("R", Version.FULL, [0, 1])]  # handle_start=-1
  assert compute_view_slot_offsets(specs, {}) == {}


# -----------------------------------------------------------------------------
# build_root_slot_map + register_pipeline_handles
# -----------------------------------------------------------------------------


def test_build_root_slot_map_first_occurrence_wins():
  specs = [
    _cs("R", Version.DELTA, [0, 1], handle_start=0),
    _cs("R", Version.DELTA, [0, 1], handle_start=3),  # dup spec
    _cs("S", Version.FULL, [0, 1], handle_start=1),
  ]
  m_ = build_root_slot_map(specs, {})
  assert m_["R_DELTA_VER_0_1"] == 0
  assert m_["S_FULL_VER_0_1"] == 2  # slot 2 because R advanced slot to 1


def test_register_pipeline_handles_maps_nested_cj_handle_to_root_slot():
  offsets: dict[int, int] = {}
  root_slots = {"R_DELTA_VER_0_1": 0, "S_FULL_VER_0_1": 1}
  pipeline = [
    m.ColumnJoin(
      var_name="z",
      sources=[
        _cs("R", Version.DELTA, [0, 1], handle_start=7),
        _cs("S", Version.FULL, [0, 1], handle_start=8),
      ],
    ),
  ]
  register_pipeline_handles(offsets, pipeline, {}, root_slots)
  assert offsets == {7: 0, 8: 1}


def test_register_pipeline_handles_does_not_overwrite_existing():
  offsets = {5: 99}
  root_slots = {"R_DELTA_VER_0_1": 0}
  pipeline = [m.ColumnSource(rel_name="R", version=Version.DELTA, index=[0, 1], handle_start=5)]
  register_pipeline_handles(offsets, pipeline, {}, root_slots)
  assert offsets[5] == 99  # untouched


# -----------------------------------------------------------------------------
# spec_key
# -----------------------------------------------------------------------------


def test_spec_key_with_and_without_version():
  assert spec_key("R", [0, 1]) == "R_0_1"
  assert spec_key("R", [0, 1], "DELTA_VER") == "R_0_1_DELTA_VER"
  assert spec_key("PointsTo", [1, 0], "FULL_VER") == "PointsTo_1_0_FULL_VER"


# -----------------------------------------------------------------------------
# collect_unique_view_specs
# -----------------------------------------------------------------------------


def test_collect_unique_view_specs_column_join():
  cj = m.ColumnJoin(
    var_name="z",
    sources=[
      _cs("R", Version.DELTA, [0, 1], handle_start=0),
      _cs("S", Version.FULL, [1, 0], handle_start=1),
    ],
  )
  specs = collect_unique_view_specs([cj])
  assert len(specs) == 2
  assert specs[0].rel_name == "R" and specs[0].version == "DELTA_VER"
  assert specs[1].rel_name == "S" and specs[1].version == "FULL_VER"


def test_collect_unique_view_specs_dedupes_across_joins():
  shared = _cs("R", Version.DELTA, [0, 1], handle_start=0)
  cj1 = m.ColumnJoin(
    var_name="z",
    sources=[
      shared,
      _cs("A", Version.FULL, [0, 1], handle_start=1),
    ],
  )
  cj2 = m.ColumnJoin(
    var_name="w",
    sources=[
      _cs("R", Version.DELTA, [0, 1], handle_start=4),
      _cs("B", Version.FULL, [0, 1], handle_start=5),
    ],
  )
  specs = collect_unique_view_specs([cj1, cj2])
  names = [(s.rel_name, s.version) for s in specs]
  assert ("R", "DELTA_VER") in names
  assert ("A", "FULL_VER") in names
  assert ("B", "FULL_VER") in names
  assert len(specs) == 3


def test_collect_unique_view_specs_scan_negation_aggregate():
  scan = m.Scan(vars=["x"], rel_name="R", version=Version.FULL, index=[0, 1])
  neg = m.Negation(rel_name="N", version=Version.FULL, index=[0])
  agg = m.Aggregate(
    result_var="c", agg_func="AggCount", rel_name="A", version=Version.FULL, index=[0]
  )
  specs = collect_unique_view_specs([scan, neg, agg])
  rels = sorted(s.rel_name for s in specs)
  assert rels == ["A", "N", "R"]


def test_collect_unique_view_specs_balanced_scan_has_two_sources():
  bs = m.BalancedScan(
    group_var="z",
    source1=_cs("V", Version.DELTA, [0, 1], handle_start=0),
    source2=_cs("A", Version.FULL, [1, 0], handle_start=1),
    vars1=["x"],
    vars2=["y"],
  )
  specs = collect_unique_view_specs([bs])
  assert len(specs) == 2


# -----------------------------------------------------------------------------
# jit_emit_view_declarations
# -----------------------------------------------------------------------------


def test_jit_emit_view_declarations_empty_returns_empty():
  ctx = new_code_gen_context()
  assert jit_emit_view_declarations([], [], [], ctx) == ""


def test_jit_emit_view_declarations_basic_two_views():
  specs = [
    ViewSpec(rel_name="R", index=[0, 1], version="DELTA_VER", handle_idx=0),
    ViewSpec(rel_name="S", index=[1, 0], version="FULL_VER", handle_idx=1),
  ]
  cj = m.ColumnJoin(
    var_name="z",
    sources=[
      _cs("R", Version.DELTA, [0, 1], handle_start=0),
      _cs("S", Version.FULL, [1, 0], handle_start=1),
    ],
  )
  ctx = new_code_gen_context()
  out = jit_emit_view_declarations(specs, [cj], [], ctx)

  # Headers
  assert "using ViewType = std::remove_cvref_t<decltype(views[0])>;" in out
  assert "using HandleType = ViewType::NodeHandle;" in out
  # Debug comment
  assert "// View declarations (deduplicated by spec, 2 unique views)" in out
  # Declarations
  assert "auto view_R_0_1_DELTA_VER = views[0];" in out
  assert "auto view_S_1_0_FULL_VER = views[1];" in out
  # ctx.view_vars populated by both spec_key AND str(handle_start)
  assert ctx.view_vars["R_0_1_DELTA_VER"] == "view_R_0_1_DELTA_VER"
  assert ctx.view_vars["S_1_0_FULL_VER"] == "view_S_1_0_FULL_VER"
  assert ctx.view_vars["0"] == "view_R_0_1_DELTA_VER"
  assert ctx.view_vars["1"] == "view_S_1_0_FULL_VER"


def test_jit_emit_view_declarations_respects_view_slot_offsets():
  '''When the context has view_slot_offsets set (multi-view source case),
  declarations use the remapped slot, not the raw handle_idx.
  '''
  specs = [ViewSpec(rel_name="R", index=[0, 1], version="FULL_VER", handle_idx=5)]
  ctx = new_code_gen_context()
  ctx.view_slot_offsets[5] = 2  # handle 5 lives at slot 2 in views[]
  out = jit_emit_view_declarations(specs, [], [], ctx)
  assert "auto view_R_0_1_FULL_VER = views[2];" in out


if __name__ == "__main__":
  import inspect

  this = sys.modules[__name__]
  passed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if name.startswith("test_"):
      fn()
      passed += 1
  print(f"{passed} passed")
