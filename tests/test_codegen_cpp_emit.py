'''C++ emitters for scalar / maintenance MIR ops.

Adapted from `test_mir_commands_notemplate.py` on python-api-notemplate;
each fixture is now built with our `mir_types` nodes (not mhk's parallel
nt_commands dataclasses), so a single IR serves both the compiler
pipeline and the codegen backend.
'''
import sys
from pathlib import Path


import srdatalog.mir_types as m
from srdatalog.hir_types import Version
from srdatalog.codegen.cpp_emit import (
  fact_index_str,
  index_spec_t,
  index_create_call,
  emit,
  emit_orchestrator,
  emit_rebuild_index,
  emit_merge_index,
  emit_check_size,
  emit_compute_delta,
  emit_compute_delta_index,
  emit_clear_relation,
  emit_merge_relation,
  emit_rebuild_index_from_index,
  emit_inject_cpp_hook,
  emit_post_stratum_reconstruct,
)


def _nows(s: str) -> str:
  return s.replace(" ", "").replace("\n", "")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def test_fact_index_str():
  assert fact_index_str("PointsTo", [0, 1]) == "PointsTo, 0, 1"
  assert fact_index_str("Assign", [0]) == "Assign, 0"
  assert fact_index_str("Load", [2, 0, 1]) == "Load, 2, 0, 1"


def test_index_spec_t():
  assert index_spec_t("PointsTo", Version.NEW, [0, 1]) == (
    "SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, NEW_VER>"
  )
  assert index_spec_t("Assign", Version.DELTA, [1, 0]) == (
    "SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 1, 0>, DELTA_VER>"
  )
  assert index_spec_t("Load", Version.FULL, [2, 0, 1]) == (
    "SRDatalog::mir::IndexSpecT<Load, std::integer_sequence<int, 2, 0, 1>, FULL_VER>"
  )


def test_index_create_call():
  assert index_create_call("PointsTo", Version.NEW, [1, 0]) == (
    'mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<PointsTo, '
    'std::integer_sequence<int, 1, 0>, NEW_VER>>(db, 0);'
  )


# -----------------------------------------------------------------------------
# RebuildIndex
# -----------------------------------------------------------------------------

def test_rebuild_index_new():
  ri = m.RebuildIndex(rel_name="PointsTo", version=Version.NEW, index=[1, 0])
  assert emit_rebuild_index(ri) == (
    'SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<PointsTo, '
    'std::integer_sequence<int, 1, 0>, NEW_VER>>(db);'
  )


def test_rebuild_index_full():
  ri = m.RebuildIndex(rel_name="Assign", version=Version.FULL, index=[0, 1])
  assert emit_rebuild_index(ri) == (
    'SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<Assign, '
    'std::integer_sequence<int, 0, 1>, FULL_VER>>(db);'
  )


def test_rebuild_index_orchestrator_no_streams():
  ri = m.RebuildIndex(rel_name="PointsTo", version=Version.NEW, index=[1, 0])
  assert emit_orchestrator(ri, 9999, None) == emit_rebuild_index(ri)


def test_rebuild_index_orchestrator_with_stream_wait():
  ri = m.RebuildIndex(rel_name="PointsTo", version=Version.NEW, index=[1, 0])
  got = emit_orchestrator(ri, 9999, {"PointsTo": [0, 2]})
  assert "// Wait for streams that wrote to PointsTo" in got
  assert "_stream_pool.wait_event(0);" in got
  assert "_stream_pool.wait_event(2);" in got
  assert got.endswith(emit_rebuild_index(ri))


# -----------------------------------------------------------------------------
# MergeIndex (FULL-only in our MIR)
# -----------------------------------------------------------------------------

def test_merge_index_points_to():
  mi = m.MergeIndex(rel_name="PointsTo", index=[1, 0])
  assert emit_merge_index(mi) == (
    'SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<PointsTo, '
    'std::integer_sequence<int, 1, 0>, FULL_VER>>(db);'
  )


def test_merge_index_three_col():
  mi = m.MergeIndex(rel_name="Store", index=[0, 1, 2])
  assert emit_merge_index(mi) == (
    'SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<Store, '
    'std::integer_sequence<int, 0, 1, 2>, FULL_VER>>(db);'
  )


# -----------------------------------------------------------------------------
# CheckSize / ClearRelation / MergeRelation
# -----------------------------------------------------------------------------

def test_check_size_new():
  cs = m.CheckSize(rel_name="PointsTo", version=Version.NEW)
  assert emit_check_size(cs) == (
    'SRDatalog::GPU::mir_helpers::check_size_fn<PointsTo, NEW_VER, PointsTo_canonical_spec_t>(db);'
  )


def test_check_size_delta():
  cs = m.CheckSize(rel_name="Load", version=Version.DELTA)
  assert emit_check_size(cs) == (
    'SRDatalog::GPU::mir_helpers::check_size_fn<Load, DELTA_VER, Load_canonical_spec_t>(db);'
  )


def test_clear_relation_new():
  cr = m.ClearRelation(rel_name="PointsTo", version=Version.NEW)
  assert emit_clear_relation(cr) == (
    'SRDatalog::GPU::mir_helpers::clear_relation_fn<PointsTo, NEW_VER>(db);'
  )


def test_clear_relation_delta():
  cr = m.ClearRelation(rel_name="Assign", version=Version.DELTA)
  assert emit_clear_relation(cr) == (
    'SRDatalog::GPU::mir_helpers::clear_relation_fn<Assign, DELTA_VER>(db);'
  )


def test_merge_relation():
  mr = m.MergeRelation(rel_name="PointsTo")
  assert emit_merge_relation(mr) == (
    'SRDatalog::GPU::mir_helpers::merge_relation_fn<PointsTo>(db);'
  )


# -----------------------------------------------------------------------------
# ComputeDelta / ComputeDeltaIndex
# -----------------------------------------------------------------------------

def test_compute_delta():
  cd = m.ComputeDelta(rel_name="PointsTo", index=[0, 1])
  assert emit_compute_delta(cd) == (
    'SRDatalog::GPU::mir_helpers::compute_delta_fn<PointsTo, '
    'SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, NEW_VER>>(db);'
  )


def test_compute_delta_index_points_to():
  cdi = m.ComputeDeltaIndex(rel_name="PointsTo", canonical_index=[1, 0])
  got = _nows(emit_compute_delta_index(cdi))
  want = _nows(
    'SRDatalog::GPU::mir_helpers::compute_delta_index_fn<'
    'SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 1, 0>, NEW_VER>, '
    'SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>, '
    'SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);'
  )
  assert got == want


def test_compute_delta_index_other_rel():
  cdi = m.ComputeDeltaIndex(rel_name="Assign", canonical_index=[0, 1])
  got = _nows(emit_compute_delta_index(cdi))
  assert got == _nows(
    'SRDatalog::GPU::mir_helpers::compute_delta_index_fn<'
    'SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, NEW_VER>, '
    'SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, FULL_VER>, '
    'SRDatalog::mir::IndexSpecT<Assign, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);'
  )


# -----------------------------------------------------------------------------
# RebuildIndexFromIndex (same-rel same-version)
# -----------------------------------------------------------------------------

def test_rebuild_index_from_index_delta_swap():
  rifi = m.RebuildIndexFromIndex(
    rel_name="PointsTo",
    source_index=[1, 0],
    target_index=[0, 1],
    version=Version.DELTA,
  )
  got = _nows(emit_rebuild_index_from_index(rifi))
  want = _nows(
    'SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<'
    'SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>, '
    'SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);'
  )
  assert got == want


# -----------------------------------------------------------------------------
# InjectCppHook / PostStratumReconstructInternCols
# -----------------------------------------------------------------------------

def test_inject_cpp_hook():
  ich = m.InjectCppHook(code='std::cout << "hi";')
  assert emit_inject_cpp_hook(ich) == '{std::cout << "hi";}'


def test_post_stratum_reconstruct():
  r = m.PostStratumReconstructInternCols(rel_name="PointsTo", canonical_index=[1, 0])
  assert emit_post_stratum_reconstruct(r) == (
    'mir_helpers::reconstruct_fn<SRDatalog::mir::IndexSpecT<PointsTo, '
    'std::integer_sequence<int, 1, 0>, FULL_VER>>(db);'
  )


# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------

def test_emit_dispatches():
  assert emit(m.RebuildIndex(rel_name="R", version=Version.FULL, index=[0])) == (
    emit_rebuild_index(m.RebuildIndex(rel_name="R", version=Version.FULL, index=[0]))
  )
  assert emit(m.ClearRelation(rel_name="R", version=Version.NEW)) == (
    'SRDatalog::GPU::mir_helpers::clear_relation_fn<R, NEW_VER>(db);'
  )


def test_emit_raises_on_pipeline_node():
  try:
    emit(m.ColumnSource(rel_name="R", version=Version.FULL, index=[0]))
  except NotImplementedError as e:
    assert "ColumnSource" in str(e)
  else:
    raise AssertionError("expected NotImplementedError for ColumnSource")


if __name__ == "__main__":
  import inspect
  this = sys.modules[__name__]
  passed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if name.startswith("test_"):
      fn()
      passed += 1
  print(f"{passed} passed")
