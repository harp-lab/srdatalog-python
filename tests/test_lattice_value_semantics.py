import math

import pytest

import srdatalog.ir.mir.types as mir
from srdatalog import (
  Program,
  Relation,
  Var,
  compile_to_hir,
  compile_to_mir,
  float32_to_u32,
  interval_lattice,
  max_lower_lattice,
  u32_to_float32,
)
from srdatalog.ir.hir.emit import hir_to_obj
from srdatalog.ir.mir.print import print_mir_sexpr


def _analyst_program() -> Program:
  node, next_node = Var("node"), Var("next_node")
  time, next_time = Var("time"), Var("next_time")
  lower, upper = Var("lower"), Var("upper")

  seed = Relation("AnalystSeed", 4)
  step = Relation("StepFrom", 3)
  successor = Relation("Successor", 2)
  analyst_at = Relation(
    "AnalystAt",
    4,
    value_spec=interval_lattice(
      key_columns=(0, 1),
      lower_column=2,
      upper_column=3,
    ),
  )
  return Program(
    rules=[
      (
        analyst_at(node, time, lower, upper)
        <= seed(node, time, lower, upper)
      ).named("AnalystSeed"),
      (
        analyst_at(next_node, next_time, lower, upper)
        <= analyst_at(node, time, lower, upper)
        & step(node, next_node, time)
        & successor(time, next_time)
      ).named("AnalystDelay"),
    ]
  )


def test_interval_relation_validates_functional_partition():
  with pytest.raises(ValueError, match="partition"):
    Relation(
      "Bad",
      4,
      value_spec=interval_lattice(
        key_columns=(0,),
        lower_column=2,
        upper_column=3,
      ),
    )


def test_float32_bit_encoding_is_exact_and_order_preserving_for_bounds():
  values = [0.0, 0.25, 0.5, 0.75, 1.0]
  encoded = [float32_to_u32(v) for v in values]
  assert encoded == sorted(encoded)
  for value, bits in zip(values, encoded):
    assert math.isclose(u32_to_float32(bits), value, rel_tol=0.0, abs_tol=1e-7)


def test_max_lower_relation_makes_first_candidate_rank_explicit():
  spec = max_lower_lattice(
    key_columns=(0, 1), rank_column=2, lower_column=3, upper_column=4
  )
  spec.validate(5)
  assert spec.join.value == "max-lower-select"
  assert spec.value_columns == (2, 3, 4)


def test_lattice_value_semantics_survive_dsl_to_hir():
  hir = compile_to_hir(_analyst_program())
  decl = next(d for d in hir.relation_decls if d.rel_name == "AnalystAt")
  assert decl.value_spec is not None
  assert decl.value_spec.key_columns == (0, 1)
  assert decl.value_spec.value_columns == (2, 3)
  assert hir_to_obj(hir)["relations"]["AnalystAt"]["functionalValue"] == {
    "keyColumns": [0, 1],
    "valueColumns": [2, 3],
    "join": "interval-intersection",
    "encoding": "float32-bits",
  }


def test_lattice_relation_lowers_to_changed_value_delta_merge():
  program = _analyst_program()
  lowered = compile_to_mir(program, apply_mir_passes=False)
  maintenance = [
    op
    for step, _ in lowered.steps
    if isinstance(step, mir.FixpointPlan)
    for op in step.instructions
    if isinstance(op, mir.LatticeMergeDelta)
  ]
  assert len(maintenance) == 2  # base load and recursive delayed propagation
  assert all(op.key_columns == [0, 1] for op in maintenance)
  assert all(op.value_columns == [2, 3] for op in maintenance)

  analyst_ordinary_maintenance = [
    op
    for step, _ in lowered.steps
    if isinstance(step, mir.FixpointPlan)
    for op in step.instructions
    if isinstance(op, (mir.ComputeDeltaIndex, mir.MergeIndex)) and op.rel_name == "AnalystAt"
  ]
  assert analyst_ordinary_maintenance == []


def test_lattice_merge_delta_sexpr_is_storage_independent():
  op = mir.LatticeMergeDelta(
    rel_name="AnalystAt",
    key_columns=[0, 1],
    value_columns=[2, 3],
    join=interval_lattice(
      key_columns=(0, 1), lower_column=2, upper_column=3
    ).join,
    encoding=interval_lattice(
      key_columns=(0, 1), lower_column=2, upper_column=3
    ).encoding,
    canonical_index=[0, 1, 2, 3],
    delta_indices=[[0, 1, 2, 3]],
    full_indices=[[0, 1, 2, 3]],
  )
  assert print_mir_sexpr(op) == (
    "(lattice-merge-delta #:schema AnalystAt #:key-columns (0 1) "
    "#:value-columns (2 3) #:join interval-intersection #:encoding float32-bits "
    "#:canonical-index (0 1 2 3) #:delta-indices ((0 1 2 3)) "
    "#:full-indices ((0 1 2 3)))"
  )
