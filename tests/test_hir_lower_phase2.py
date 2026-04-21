'''Phase-2 HIR -> MIR lowering tests.

Covers:
  - Multi-clause variant lowering (TCRec: ColumnJoin + CartesianJoin + InsertInto)
  - Dual-delta PathCompose variants
  - Fixpoint maintenance generators for tc (simple + loop)

The MIR tree is spot-checked structurally and then re-emitted through the
S-expr printer so the concatenated output is locked against regressions.
'''
import sys
from pathlib import Path


from srdatalog.dsl import Var, Relation, Program
from srdatalog.hir.types import Version
from srdatalog.hir import compile_to_hir
from srdatalog.hir.lower import (
  lower_variant_to_pipeline,
  generate_rebuild_indices,
  generate_merge_indices,
  generate_simple_maintenance,
  generate_loop_maintenance,
)
import srdatalog.mir.types as mir
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


def build_path_compose() -> Program:
  X, Y, Z = Var("x"), Var("y"), Var("z")
  seed = Relation("Seed", 2)
  path = Relation("Path", 2)
  return Program(
    relations=[seed, path],
    rules=[
      (path(X, Y) <= seed(X, Y)).named("PathSeed"),
      (path(X, Z) <= path(X, Y) & path(Y, Z)).named("PathCompose"),
    ],
  )


# -----------------------------------------------------------------------------
# Multi-clause lowering shape
# -----------------------------------------------------------------------------

def test_tc_rec_lowers_to_cj_cart_insert():
  hir = compile_to_hir(build_tc())
  variant = hir.strata[2].recursive_variants[0]  # TCRec
  ops = lower_variant_to_pipeline(variant, hir.strata[2])

  assert len(ops) == 3
  cj, cart, ins = ops

  # ColumnJoin on var y with two sources (Path DELTA, Edge FULL), empty prefixes.
  assert isinstance(cj, mir.ColumnJoin)
  assert cj.var_name == "y"
  assert len(cj.sources) == 2
  path_src, edge_src = cj.sources
  assert path_src.rel_name == "Path"
  assert path_src.version is Version.DELTA
  assert path_src.index == [1, 0]
  assert path_src.prefix_vars == []
  assert edge_src.rel_name == "Edge"
  assert edge_src.version is Version.FULL
  assert edge_src.index == [0, 1]
  assert edge_src.prefix_vars == []

  # CartesianJoin over independent vars x and z; same sources with prefix=[y]
  # and per-source var lists matching column positions.
  assert isinstance(cart, mir.CartesianJoin)
  assert cart.vars == ["x", "z"]
  assert cart.var_from_source == [["x"], ["z"]]
  assert len(cart.sources) == 2
  for src in cart.sources:
    assert src.prefix_vars == ["y"]

  # InsertInto uses Path canonical_index [1,0].
  assert isinstance(ins, mir.InsertInto)
  assert ins.rel_name == "Path"
  assert ins.vars == ["x", "z"]
  assert ins.index == [1, 0]


def test_tc_rec_sexpr_roundtrip():
  hir = compile_to_hir(build_tc())
  variant = hir.strata[2].recursive_variants[0]
  ops = lower_variant_to_pipeline(variant, hir.strata[2])
  sexpr = "\n".join(print_mir_sexpr(op) for op in ops)
  expected = (
    "(column-join :var y\n"
    "  :sources (\n"
    "    (column-source :index (Path 1 0) :ver DELTA :prefix ())\n"
    "    (column-source :index (Edge 0 1) :ver FULL :prefix ())\n"
    "  ))\n"
    "(cartesian-join :vars (x z) :var-from-source ((x) (z))\n"
    "  :sources (\n"
    "    (column-source :index (Path 1 0) :ver DELTA :prefix (y))\n"
    "    (column-source :index (Edge 0 1) :ver FULL :prefix (y))\n"
    "  ))\n"
    "(insert-into :schema Path :ver NEW :dedup-index (1 0) :terms (x z))"
  )
  assert sexpr == expected


def test_path_compose_both_delta_variants_lower():
  '''Dual-delta: delta_idx=0 drives on first Path (DELTA), delta_idx=1 drives
  on second (DELTA). Verify each produces a well-formed ColumnJoin + Cart + Insert.
  '''
  hir = compile_to_hir(build_path_compose())
  rec = hir.strata[1].recursive_variants
  assert len(rec) == 2
  for variant in rec:
    ops = lower_variant_to_pipeline(variant, hir.strata[1])
    assert len(ops) == 3
    cj, cart, ins = ops
    assert isinstance(cj, mir.ColumnJoin)
    assert cj.var_name == "y"  # join var is y in both variants
    assert len(cj.sources) == 2
    assert {s.version for s in cj.sources} == {Version.DELTA, Version.FULL}
    assert isinstance(cart, mir.CartesianJoin)
    assert isinstance(ins, mir.InsertInto)
    assert ins.rel_name == "Path"


# -----------------------------------------------------------------------------
# Maintenance generators
# -----------------------------------------------------------------------------

def test_generate_rebuild_indices_round_trip():
  ops = generate_rebuild_indices("Path", [[1, 0], [0, 1]], Version.NEW)
  assert len(ops) == 2
  assert all(isinstance(o, mir.RebuildIndex) for o in ops)
  assert ops[0].version is Version.NEW
  assert ops[0].index == [1, 0]
  assert ops[1].index == [0, 1]


def test_generate_merge_indices():
  ops = generate_merge_indices("Path", [[1, 0], [0, 1]])
  assert [o.index for o in ops] == [[1, 0], [0, 1]]
  assert all(isinstance(o, mir.MergeIndex) for o in ops)


def test_simple_maintenance_shape_for_edge():
  '''Edge in stratum 0 is non-recursive with canonical [0,1] and one index.
  Simple maintenance: rebuild NEW, size-check, delta, clear NEW, merge [0,1].
  '''
  ops = generate_simple_maintenance("Edge", [[0, 1]], [0, 1], arity=2)
  # Expected shape: RebuildIndex, CheckSize, ComputeDeltaIndex, ClearRelation,
  # MergeIndex (no RebuildIndexFromIndex because idx == canonical).
  kinds = [type(o).__name__ for o in ops]
  assert kinds == [
    "RebuildIndex", "CheckSize", "ComputeDeltaIndex", "ClearRelation", "MergeIndex",
  ]
  assert ops[0].version is Version.NEW
  assert ops[0].index == [0, 1]
  assert ops[3].version is Version.NEW  # ClearRelation NEW
  assert ops[4].index == [0, 1]


def test_simple_maintenance_with_non_canonical_index():
  '''If indices include a non-canonical entry, a RebuildIndexFromIndex is
  emitted for it before MergeIndex.
  '''
  ops = generate_simple_maintenance("Path", [[1, 0], [0, 1]], [1, 0], arity=2)
  # For [1,0] (canonical): just MergeIndex.
  # For [0,1]: RebuildIndexFromIndex + MergeIndex.
  kinds = [type(o).__name__ for o in ops]
  assert kinds == [
    "RebuildIndex", "CheckSize", "ComputeDeltaIndex", "ClearRelation",
    "MergeIndex",                 # [1,0] canonical
    "RebuildIndexFromIndex",      # [0,1] from [1,0]
    "MergeIndex",                 # [0,1]
  ]
  rfi = ops[5]
  assert isinstance(rfi, mir.RebuildIndexFromIndex)
  assert rfi.source_index == [1, 0]
  assert rfi.target_index == [0, 1]
  assert rfi.version is Version.DELTA


def test_loop_maintenance_shape_for_tc_path():
  '''Path in stratum 2 (recursive) has canonical [1,0], one index [1,0], and
  full_needed={} (TCRec reads Edge as FULL but never Path as FULL -- only as
  DELTA). Loop maintenance still emits MergeIndex([1,0]) because canonical
  always needs FULL.
  '''
  ops = generate_loop_maintenance(
    "Path", [[1, 0]], [1, 0], arity=2, full_needed=set(),
  )
  kinds = [type(o).__name__ for o in ops]
  assert kinds == [
    "RebuildIndex",          # canonical NEW
    "ClearRelation",         # clear DELTA
    "CheckSize",
    "ComputeDeltaIndex",
    "ClearRelation",         # clear NEW
    "MergeIndex",            # canonical [1,0] — always merged to FULL
  ]
  assert ops[0].version is Version.NEW
  assert ops[1].version is Version.DELTA
  assert ops[4].version is Version.NEW


def test_loop_maintenance_skips_full_merge_for_non_full_needed_non_canonical():
  '''For a non-canonical index whose FULL isn't read, we rebuild its DELTA
  but do NOT merge it into FULL (avoids maintaining a dead FULL index).
  '''
  ops = generate_loop_maintenance(
    "R", [[0, 1], [1, 0]], [0, 1], arity=2, full_needed=set(),
  )
  # Expected:
  #  RebuildIndex NEW(0,1), ClearRelation DELTA, CheckSize NEW,
  #  ComputeDeltaIndex (0,1), ClearRelation NEW,
  #  MergeIndex (0,1)                            <-- canonical -> FULL
  #  RebuildIndexFromIndex DELTA (0,1) -> (1,0) <-- non-canonical DELTA
  #  (no MergeIndex (1,0) because FULL not needed)
  kinds = [type(o).__name__ for o in ops]
  assert kinds == [
    "RebuildIndex", "ClearRelation", "CheckSize", "ComputeDeltaIndex",
    "ClearRelation", "MergeIndex", "RebuildIndexFromIndex",
  ]
  assert ops[5].index == [0, 1]        # canonical merged
  assert ops[6].target_index == [1, 0]


def test_loop_maintenance_includes_full_merge_when_needed():
  '''If a non-canonical index's FULL IS read (in full_needed), its FULL gets
  merged too.
  '''
  ops = generate_loop_maintenance(
    "R", [[0, 1], [1, 0]], [0, 1], arity=2, full_needed={(1, 0)},
  )
  kinds = [type(o).__name__ for o in ops]
  assert kinds == [
    "RebuildIndex", "ClearRelation", "CheckSize", "ComputeDeltaIndex",
    "ClearRelation",
    "MergeIndex",                   # canonical (0,1)
    "RebuildIndexFromIndex",        # DELTA rebuild for (1,0)
    "MergeIndex",                   # (1,0) FULL merge — required
  ]
  assert ops[7].index == [1, 0]


if __name__ == "__main__":
  tests = [
    test_tc_rec_lowers_to_cj_cart_insert,
    test_tc_rec_sexpr_roundtrip,
    test_path_compose_both_delta_variants_lower,
    test_generate_rebuild_indices_round_trip,
    test_generate_merge_indices,
    test_simple_maintenance_shape_for_edge,
    test_simple_maintenance_with_non_canonical_index,
    test_loop_maintenance_shape_for_tc_path,
    test_loop_maintenance_skips_full_merge_for_non_full_needed_non_canonical,
    test_loop_maintenance_includes_full_merge_when_needed,
  ]
  for t in tests:
    t()
  print(f"OK ({len(tests)} tests)")
