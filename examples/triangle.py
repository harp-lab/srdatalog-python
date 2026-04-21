"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/triangle/triangle.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/triangle/triangle.nim --out <this file>
"""
from __future__ import annotations

from srdatalog.dsl import Filter, Program, Relation, Var
from srdatalog.dataset_const import load_meta, resolve_program_consts

# ----- Relations ----------------------------------------------

RRel = Relation("RRel", 2, column_types=(int, int,))
SRel = Relation("SRel", 3, column_types=(int, int, int,))
TRel = Relation("TRel", 3, column_types=(int, int, int,))
ZRel = Relation("ZRel", 3, column_types=(int, int, int,))

# ----- dataset_const declarations -----------------------------

DATASET_CONST_DECLS = {
}

# ----- Rules: TriangleDB -----

def build_triangledb_program() -> Program:
  f = Var("f")
  h = Var("h")
  x = Var("x")
  y = Var("y")
  z = Var("z")

  return Program(
    relations=[
      RRel,
      SRel,
      TRel,
      ZRel,
    ],
    rules=[
      (ZRel(x, y, z) <= RRel(x, y) & SRel(y, z, h) & TRel(z, x, f)).named('Triangle').with_plan(var_order=['x', 'y', 'z']),
    ],
  )


def build_triangledb(meta_json_path: str) -> tuple[Program, dict[str, int]]:
  """Convenience: build the program, load dataset_consts, substitute."""
  consts = load_meta(meta_json_path, DATASET_CONST_DECLS)
  return resolve_program_consts(build_triangledb_program(), consts), consts
