import srdatalog.ir.mir.types as mir
from srdatalog.ir.codegen.cuda.orchestrator import (
  collect_canonical_specs,
  gen_instruction_code,
)
from srdatalog.value_semantics import LatticeJoin, ValueEncoding


def _lattice_op() -> mir.LatticeMergeDelta:
  return mir.LatticeMergeDelta(
    rel_name="AnalystAt",
    key_columns=[0, 1],
    value_columns=[2, 3],
    join=LatticeJoin.INTERVAL_INTERSECTION,
    encoding=ValueEncoding.FLOAT32_BITS,
    canonical_index=[0, 1, 2, 3],
    delta_indices=[[0, 1, 2, 3], [1, 0, 2, 3]],
    full_indices=[[0, 1, 2, 3]],
  )


def test_lattice_merge_delta_codegen_calls_gpu_primitive_and_rebuilds_views():
  code = gen_instruction_code(_lattice_op(), "  ", "iter", {})
  assert "lattice_merge_delta_fn<" in code
  assert ", 2, false>(db);" in code
  assert "rebuild_index_from_index_fn<" in code
  assert "DELTA_VER" in code
  assert "lattice_merge_delta" in code


def test_lattice_merge_delta_supplies_fixpoint_canonical_spec():
  assert collect_canonical_specs([_lattice_op()]) == [
    ("AnalystAt", [0, 1, 2, 3])
  ]


def test_max_lower_codegen_selects_ranked_winner_mode():
  op = mir.LatticeMergeDelta(
    rel_name="Candidate",
    key_columns=[0, 1],
    value_columns=[2, 3, 4],
    join=LatticeJoin.MAX_LOWER_SELECT,
    encoding=ValueEncoding.UINT32_WORDS,
    canonical_index=[0, 1, 2, 3, 4],
    delta_indices=[[0, 1, 2, 3, 4]],
    full_indices=[[0, 1, 2, 3, 4]],
  )
  code = gen_instruction_code(op, "", "iter", {})
  assert ", 2, true>(db);" in code
