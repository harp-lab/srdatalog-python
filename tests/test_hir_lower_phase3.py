'''Phase-3 HIR -> MIR lowering tests.

Covers stratum wrapping (wrap_in_execute_pipeline + lower_hir_to_mir_steps
+ lower_hir_to_mir) on tc. End-to-end byte-diff against a Nim-generated
MIR S-expr fixture lives in test_hir_mir_tc_e2e.py — this file only
verifies structure.
'''
import sys
from pathlib import Path


from srdatalog.dsl import Var, Relation, Program
from srdatalog.hir import compile_to_hir, compile_to_mir
from srdatalog.hir.lower import (
  wrap_in_execute_pipeline,
  lower_hir_to_mir_steps,
  lower_hir_to_mir,
)
import srdatalog.mir.types as mir


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


def test_wrap_in_execute_pipeline_extracts_sources_and_dests():
  '''A pipeline with Scan + InsertInto: sources=[Scan], dests=[InsertInto].'''
  from srdatalog.hir.types import Version
  scan = mir.Scan(vars=["x", "y"], rel_name="Edge", version=Version.FULL,
                  index=[0, 1], prefix_vars=[])
  ins = mir.InsertInto(rel_name="Path", version=Version.NEW,
                       vars=["x", "y"], index=[1, 0])
  ep = wrap_in_execute_pipeline([scan, ins], clause_order=[0], rule_name="TCBase")
  assert ep.rule_name == "TCBase"
  assert ep.clause_order == [0]
  assert ep.source_specs == [scan]
  assert ep.dest_specs == [ins]


def test_wrap_in_execute_pipeline_flattens_join_sources():
  '''Sources under a ColumnJoin/CartesianJoin get extracted into the flat
  source_specs list so the scheduler sees every index-spec.
  '''
  from srdatalog.hir.types import Version
  src1 = mir.ColumnSource(rel_name="A", version=Version.DELTA, index=[0, 1],
                          prefix_vars=[])
  src2 = mir.ColumnSource(rel_name="B", version=Version.FULL, index=[0, 1],
                          prefix_vars=[])
  cj = mir.ColumnJoin(var_name="y", sources=[src1, src2])
  ins = mir.InsertInto(rel_name="R", version=mir.Version.NEW if hasattr(mir, "Version") else Version.NEW,
                       vars=["x"], index=[0])
  # (handle import quirk)
  ins = mir.InsertInto(rel_name="R", version=Version.NEW, vars=["x"], index=[0])
  ep = wrap_in_execute_pipeline([cj, ins], clause_order=[0, 1], rule_name="Test")
  assert ep.source_specs == [src1, src2]
  assert ep.dest_specs == [ins]


def test_lower_hir_to_mir_steps_tc_count_and_shape():
  '''tc has 3 strata -> 3 FixpointPlan + 3 PostStratumReconstructInternCols.
  Order: stratum 0 (Edge, simple), stratum 1 (Path, simple), stratum 2
  (Path, recursive). Each stratum emits (FixpointPlan, is_rec) + its
  PostStratumReconstructInternCols.
  '''
  hir = compile_to_hir(build_tc())
  steps = lower_hir_to_mir_steps(hir)

  # 3 strata × 2 steps (FixpointPlan + PostStratumReconstructInternCols) = 6
  assert len(steps) == 6
  kinds = [(type(n).__name__, r) for n, r in steps]
  assert kinds == [
    ("FixpointPlan", False), ("PostStratumReconstructInternCols", False),
    ("FixpointPlan", False), ("PostStratumReconstructInternCols", False),
    ("FixpointPlan", True),  ("PostStratumReconstructInternCols", False),
  ]

  # Stratum 2's FixpointPlan is the recursive one; first instruction should
  # be the ExecutePipeline (only one recursive variant in tc), then maintenance.
  fp_rec = steps[4][0]
  assert isinstance(fp_rec, mir.FixpointPlan)
  assert isinstance(fp_rec.instructions[0], mir.ExecutePipeline)
  assert fp_rec.instructions[0].rule_name == "TCRec_D0"

  # Post-stratum reconstruct for stratum 2 is Path with canonical [1,0].
  psr = steps[5][0]
  assert isinstance(psr, mir.PostStratumReconstructInternCols)
  assert psr.rel_name == "Path"
  assert psr.canonical_index == [1, 0]


def test_lower_hir_to_mir_wraps_in_program():
  hir = compile_to_hir(build_tc())
  prog = lower_hir_to_mir(hir)
  assert isinstance(prog, mir.Program)
  assert len(prog.steps) == 6
  # is_recursive flags mirror the steps output.
  flags = [is_rec for _, is_rec in prog.steps]
  assert flags == [False, False, False, False, True, False]


def test_compile_to_mir_end_to_end_returns_program():
  prog = compile_to_mir(build_tc())
  assert isinstance(prog, mir.Program)
  assert len(prog.steps) == 6


def test_path_compose_recursive_stratum_has_parallel_group():
  '''PathCompose has two delta variants in the recursive stratum -> wrapped
  in a ParallelGroup (rather than a single ExecutePipeline).
  '''
  X, Y, Z = Var("x"), Var("y"), Var("z")
  seed = Relation("Seed", 2)
  path = Relation("Path", 2)
  p = Program(
    relations=[seed, path],
    rules=[
      (path(X, Y) <= seed(X, Y)).named("PathSeed"),
      (path(X, Z) <= path(X, Y) & path(Y, Z)).named("PathCompose"),
    ],
  )
  mir_prog = compile_to_mir(p)
  # Stratum 0 = base, stratum 1 = recursive.
  rec_fp = mir_prog.steps[2][0]  # stratum 1's FixpointPlan
  assert isinstance(rec_fp, mir.FixpointPlan)
  # First instruction should be a ParallelGroup (2 variants)
  assert isinstance(rec_fp.instructions[0], mir.ParallelGroup)
  assert len(rec_fp.instructions[0].ops) == 2
  names = [op.rule_name for op in rec_fp.instructions[0].ops]
  assert names == ["PathCompose_D0", "PathCompose_D1"]


if __name__ == "__main__":
  tests = [
    test_wrap_in_execute_pipeline_extracts_sources_and_dests,
    test_wrap_in_execute_pipeline_flattens_join_sources,
    test_lower_hir_to_mir_steps_tc_count_and_shape,
    test_lower_hir_to_mir_wraps_in_program,
    test_compile_to_mir_end_to_end_returns_program,
    test_path_compose_recursive_stratum_has_parallel_group,
  ]
  for t in tests:
    t()
  print(f"OK ({len(tests)} tests)")
