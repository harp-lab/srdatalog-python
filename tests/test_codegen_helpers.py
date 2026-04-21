'''Tests for codegen helpers (CodeGenContext, spec_key, view-spec collection,
source-index lookup, view-declaration emission).

Mirrors the helper coverage implicit in mhk's `test_generate_batchfiles_no_template.py`
harness, but built on our MIR node shapes.
'''
import sys
from pathlib import Path


import srdatalog.mir.types as m
from srdatalog.hir_types import Version
from srdatalog.codegen.helpers import (
  ViewSpec,
  CodeGenContext,
  spec_key,
  collect_unique_view_specs,
  find_source_idx,
  emit_view_declarations,
)


# -----------------------------------------------------------------------------
# spec_key
# -----------------------------------------------------------------------------

def test_spec_key_single_col():
  assert spec_key("PointsTo", Version.NEW, [0]) == "PointsTo_0_NEW_VER"


def test_spec_key_two_cols():
  assert spec_key("PointsTo", Version.DELTA, [1, 0]) == "PointsTo_1_0_DELTA_VER"


def test_spec_key_three_cols():
  assert spec_key("Load", Version.FULL, [2, 0, 1]) == "Load_2_0_1_FULL_VER"


# -----------------------------------------------------------------------------
# CodeGenContext
# -----------------------------------------------------------------------------

def test_context_defaults():
  ctx = CodeGenContext(output_name="out", is_counting=True, is_jit_mode=True)
  assert ctx.output_vars == {}
  assert ctx.view_vars == {}
  assert ctx.inside_cartesian_join is False
  assert ctx.cartesian_bound_vars == []


def test_context_set_output_vars_single_dest():
  ctx = CodeGenContext(output_name="out", is_counting=True, is_jit_mode=True)
  dests = [m.InsertInto(rel_name="PointsTo", version=Version.NEW,
                         vars=["y", "x"], index=[0, 1])]
  ctx.set_output_vars(dests)
  assert ctx.output_vars == {"PointsTo": "output_ctx"}


def test_context_set_output_vars_multi_dest():
  ctx = CodeGenContext(output_name="out", is_counting=True, is_jit_mode=True)
  dests = [
    m.InsertInto(rel_name="PointsTo", version=Version.NEW, vars=["x"], index=[0]),
    m.InsertInto(rel_name="Store", version=Version.NEW, vars=["y"], index=[0]),
  ]
  ctx.set_output_vars(dests)
  assert ctx.output_vars == {
    "PointsTo": "output_ctx",
    "Store": "__skip_counting__",
  }


# -----------------------------------------------------------------------------
# collect_unique_view_specs
# -----------------------------------------------------------------------------

def _cs(rel, ver, idx):
  return m.ColumnSource(rel_name=rel, version=ver, index=idx)


def test_collect_view_specs_column_join():
  cj = m.ColumnJoin(var_name="z", sources=[
    _cs("PointsTo", Version.DELTA, [0, 1]),
    _cs("Assign", Version.FULL, [1, 0]),
  ])
  specs = collect_unique_view_specs([cj])
  assert len(specs) == 2
  assert specs[0].rel_name == "PointsTo" and specs[0].version == Version.DELTA
  assert specs[1].rel_name == "Assign" and specs[1].version == Version.FULL


def test_collect_view_specs_dedupes_across_ops():
  # Same (rel, ver, idx) appears in two ColumnJoins -> one ViewSpec
  shared = _cs("PointsTo", Version.DELTA, [0, 1])
  cj1 = m.ColumnJoin(var_name="z", sources=[shared, _cs("A", Version.FULL, [0, 1])])
  cj2 = m.ColumnJoin(var_name="w", sources=[_cs("PointsTo", Version.DELTA, [0, 1]),
                                              _cs("B", Version.FULL, [0, 1])])
  specs = collect_unique_view_specs([cj1, cj2])
  # PointsTo_DELTA_0_1 appears once; A_FULL and B_FULL each once
  rels = [(s.rel_name, s.version) for s in specs]
  assert ("PointsTo", Version.DELTA) in rels
  assert ("A", Version.FULL) in rels
  assert ("B", Version.FULL) in rels
  assert len(specs) == 3


def test_collect_view_specs_scan():
  sc = m.Scan(vars=["x", "y"], rel_name="Edge", version=Version.FULL, index=[0, 1])
  specs = collect_unique_view_specs([sc])
  assert len(specs) == 1
  assert specs[0].rel_name == "Edge"


def test_collect_view_specs_negation():
  neg = m.Negation(rel_name="R", version=Version.FULL, index=[0], prefix_vars=["x"])
  specs = collect_unique_view_specs([neg])
  assert len(specs) == 1
  assert specs[0].rel_name == "R"


def test_collect_view_specs_aggregate():
  agg = m.Aggregate(result_var="c", agg_func="AggCount", rel_name="R",
                    version=Version.FULL, index=[0, 1])
  specs = collect_unique_view_specs([agg])
  assert len(specs) == 1
  assert specs[0].rel_name == "R"


def test_collect_view_specs_balanced_scan():
  bs = m.BalancedScan(
    group_var="z",
    source1=_cs("VarPointsTo", Version.DELTA, [0, 1]),
    source2=_cs("Assign", Version.FULL, [1, 0]),
    vars1=["x"],
    vars2=["y"],
  )
  specs = collect_unique_view_specs([bs])
  assert len(specs) == 2


def test_collect_view_specs_cartesian_join():
  cart = m.CartesianJoin(vars=["x", "y"], sources=[
    _cs("R", Version.FULL, [0, 1]),
    _cs("S", Version.FULL, [0, 1]),
  ])
  specs = collect_unique_view_specs([cart])
  assert len(specs) == 2


# -----------------------------------------------------------------------------
# find_source_idx
# -----------------------------------------------------------------------------

def test_find_source_idx_exact_match():
  specs = [
    _cs("PointsTo", Version.DELTA, [0, 1]),
    _cs("Assign", Version.FULL, [1, 0]),
  ]
  assert find_source_idx(specs, "PointsTo", [0, 1], Version.DELTA) == 0
  assert find_source_idx(specs, "Assign", [1, 0], Version.FULL) == 1


def test_find_source_idx_version_none_ignores_version():
  specs = [_cs("R", Version.DELTA, [0, 1])]
  assert find_source_idx(specs, "R", [0, 1], None) == 0
  assert find_source_idx(specs, "R", [0, 1], Version.FULL) == -1


def test_find_source_idx_miss():
  specs = [_cs("R", Version.DELTA, [0, 1])]
  assert find_source_idx(specs, "S", [0, 1], Version.DELTA) == -1
  assert find_source_idx(specs, "R", [1, 0], Version.DELTA) == -1


def test_find_source_idx_skips_non_sources():
  specs = [
    m.InsertInto(rel_name="PointsTo", version=Version.NEW, vars=["y", "x"], index=[0, 1]),
    _cs("Assign", Version.FULL, [1, 0]),
  ]
  # InsertInto isn't a source spec — should be skipped
  assert find_source_idx(specs, "Assign", [1, 0], Version.FULL) == 1


# -----------------------------------------------------------------------------
# emit_view_declarations
# -----------------------------------------------------------------------------

def test_emit_view_declarations_registers_spec_keys_in_ctx():
  ep = m.ExecutePipeline(
    pipeline=[
      m.ColumnJoin(var_name="z", sources=[
        _cs("PointsTo", Version.DELTA, [0, 1]),
        _cs("Assign", Version.FULL, [1, 0]),
      ]),
      m.InsertInto(rel_name="PointsTo", version=Version.NEW,
                   vars=["y", "x"], index=[0, 1]),
    ],
    source_specs=[
      _cs("PointsTo", Version.DELTA, [0, 1]),
      _cs("Assign", Version.FULL, [1, 0]),
    ],
    dest_specs=[
      m.InsertInto(rel_name="PointsTo", version=Version.NEW,
                   vars=["y", "x"], index=[0, 1]),
    ],
    rule_name="Assign_D0",
  )
  ctx = CodeGenContext(output_name="output_ctx", is_counting=True, is_jit_mode=True)
  out = emit_view_declarations(ep, ctx)

  assert "using ViewType = std::remove_cvref_t<decltype(views[0])>;" in out
  assert "using HandleType = ViewType::NodeHandle;" in out
  assert "auto view_PointsTo_0_1_DELTA = views[0];" in out
  assert "auto view_Assign_1_0_FULL = views[1];" in out

  assert "PointsTo_0_1_DELTA_VER" in ctx.view_vars
  assert ctx.view_vars["PointsTo_0_1_DELTA_VER"] == "view_PointsTo_0_1_DELTA"
  assert ctx.view_vars["Assign_1_0_FULL_VER"] == "view_Assign_1_0_FULL"


def test_emit_view_declarations_dedupes_across_multiple_joins():
  shared = _cs("PointsTo", Version.DELTA, [0, 1])
  ep = m.ExecutePipeline(
    pipeline=[
      m.ColumnJoin(var_name="z", sources=[shared, _cs("A", Version.FULL, [0, 1])]),
      m.ColumnJoin(var_name="w", sources=[_cs("PointsTo", Version.DELTA, [0, 1]),
                                            _cs("B", Version.FULL, [0, 1])]),
    ],
    source_specs=[
      _cs("PointsTo", Version.DELTA, [0, 1]),
      _cs("A", Version.FULL, [0, 1]),
      _cs("B", Version.FULL, [0, 1]),
    ],
    dest_specs=[],
    rule_name="Dummy",
  )
  ctx = CodeGenContext(output_name="o", is_counting=True, is_jit_mode=True)
  out = emit_view_declarations(ep, ctx)

  # PointsTo delta should only be declared once
  assert out.count("auto view_PointsTo_0_1_DELTA = views[") == 1
  assert out.count("auto view_A_0_1_FULL = views[") == 1
  assert out.count("auto view_B_0_1_FULL = views[") == 1


if __name__ == "__main__":
  import inspect
  this = sys.modules[__name__]
  passed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if name.startswith("test_"):
      fn()
      passed += 1
  print(f"{passed} passed")
