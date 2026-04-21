'''Phase-1 HIR -> MIR lowering tests.

Runs the full HIR pipeline on tc, then lowers each single-clause variant
(EdgeLoad, TCBase) to an MIR pipeline (Scan + InsertInto), asserting:
  - The MIR node shapes
  - The S-expr round-trip format

Multi-clause variants (TCRec) raise NotImplementedError until Phase 2.
'''

import srdatalog.mir.types as mir
from srdatalog.dsl import Program, Relation, Var
from srdatalog.hir import compile_to_hir
from srdatalog.hir.lower import (
  generate_column_source,
  generate_insert_into,
  lower_variant_to_pipeline,
)
from srdatalog.hir.types import Version
from srdatalog.mir.emit import print_mir_sexpr


def build_tc() -> Program:
  X, Y, Z = Var("x"), Var("y"), Var("z")
  arc = Relation("ArcInput", 2)
  edge = Relation("Edge", 2)
  path = Relation("Path", 2)
  return Program(
    relations=[arc, edge, path],
    rules=[
      (edge(X, Y) <= arc(X, Y)).named("EdgeLoad"),
      (path(X, Y) <= edge(X, Y)).named("TCBase"),
      (path(X, Z) <= path(X, Y) & edge(Y, Z)).named("TCRec"),
    ],
  )


def test_edge_load_lowers_to_scan_plus_insert_into():
  hir = compile_to_hir(build_tc())
  stratum = hir.strata[0]
  variant = stratum.base_variants[0]  # EdgeLoad
  ops = lower_variant_to_pipeline(variant, stratum)

  assert len(ops) == 2
  scan, insert = ops
  assert isinstance(scan, mir.Scan)
  assert scan.rel_name == "ArcInput"
  assert scan.version is Version.FULL
  assert scan.vars == ["x", "y"]
  assert scan.index == [0, 1]
  assert scan.prefix_vars == []

  assert isinstance(insert, mir.InsertInto)
  assert insert.rel_name == "Edge"
  assert insert.version is Version.NEW
  assert insert.vars == ["x", "y"]
  # Stratum 0's canonical index for Edge fell through from globalIndexMap
  # (see test_hir_index_tc): [0, 1].
  assert insert.index == [0, 1]


def test_edge_load_sexpr_roundtrip():
  hir = compile_to_hir(build_tc())
  ops = lower_variant_to_pipeline(hir.strata[0].base_variants[0], hir.strata[0])
  sexpr = "\n".join(print_mir_sexpr(op) for op in ops)
  assert sexpr == (
    "(scan :vars (x y) :index (ArcInput 0 1) :ver FULL :prefix ())\n"
    "(insert-into :schema Edge :ver NEW :dedup-index (0 1) :terms (x y))"
  )


def test_tc_base_uses_path_canonical_index_one_zero():
  '''Stratum 1 canonical_index[Path] is [1,0] (the layout that stratum 2's
  DELTA consumer expects). InsertInto must use it.
  '''
  hir = compile_to_hir(build_tc())
  stratum = hir.strata[1]
  variant = stratum.base_variants[0]  # TCBase
  ops = lower_variant_to_pipeline(variant, stratum)

  insert = ops[1]
  assert isinstance(insert, mir.InsertInto)
  assert insert.rel_name == "Path"
  assert insert.vars == ["x", "y"]
  assert insert.index == [1, 0]


def test_tc_rec_multi_clause_lowers_no_longer_raises():
  '''Phase 2 superseded the Phase 1 NotImplementedError guard. Just confirm
  the lowering returns a non-empty pipeline; detailed assertions live in
  test_hir_lower_phase2.py.
  '''
  hir = compile_to_hir(build_tc())
  stratum = hir.strata[2]
  variant = stratum.recursive_variants[0]  # TCRec, 2 access patterns
  ops = lower_variant_to_pipeline(variant, stratum)
  assert len(ops) >= 3  # ColumnJoin + CartesianJoin + InsertInto at minimum


def test_generate_column_source_shape():
  '''generate_column_source is a Phase-2 helper but shape-testable now.
  A pattern with prefix_len=1, access_order=[y,x], index_cols=[1,0]
  produces a ColumnSource with prefix_vars=[y].
  '''
  from srdatalog.hir.types import AccessPattern

  ap = AccessPattern(
    rel_name="Path",
    version=Version.DELTA,
    access_order=["y", "x"],
    index_cols=[1, 0],
    prefix_len=1,
    clause_idx=0,
  )
  cs = generate_column_source(ap)
  assert cs.rel_name == "Path"
  assert cs.version is Version.DELTA
  assert cs.index == [1, 0]
  assert cs.prefix_vars == ["y"]
  assert cs.clause_idx == 0


def test_generate_insert_into_const_arg_not_in_vars():
  '''If the head contains a constant, only LVars become InsertInto vars.'''
  from srdatalog.dsl import ArgKind, Atom, ClauseArg

  head = Atom(
    rel="R",
    args=(
      ClauseArg(kind=ArgKind.LVAR, var_name="x"),
      ClauseArg(kind=ArgKind.CONST, const_value=42, const_cpp_expr="42"),
    ),
  )
  ins = generate_insert_into(head, canonical_index=[0, 1])
  assert ins.vars == ["x"]  # constant not in vars
  assert ins.index == [0, 1]
  assert ins.rel_name == "R"


if __name__ == "__main__":
  tests = [
    test_edge_load_lowers_to_scan_plus_insert_into,
    test_edge_load_sexpr_roundtrip,
    test_tc_base_uses_path_canonical_index_one_zero,
    test_tc_rec_multi_clause_lowers_no_longer_raises,
    test_generate_column_source_shape,
    test_generate_insert_into_const_arg_not_in_vars,
  ]
  for t in tests:
    t()
  print(f"OK ({len(tests)} tests)")
