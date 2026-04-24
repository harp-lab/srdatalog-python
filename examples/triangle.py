"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/triangle/triangle.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/triangle/triangle.nim --out <this file>
"""

from __future__ import annotations

from srdatalog.dsl import SPLIT, Filter, Program, Relation, Var

# ----- Relations ----------------------------------------------

RRel = Relation(
  "RRel",
  2,
  column_types=(
    int,
    int,
  ),
)
SRel = Relation(
  "SRel",
  3,
  column_types=(
    int,
    int,
    int,
  ),
)
TRel = Relation(
  "TRel",
  3,
  column_types=(
    int,
    int,
    int,
  ),
)
ZRel = Relation(
  "ZRel",
  3,
  column_types=(
    int,
    int,
    int,
  ),
)

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
      (ZRel(x, y, z) <= RRel(x, y) & SRel(y, z, h) & TRel(z, x, f))
      .named('Triangle')
      .with_plan(var_order=['x', 'y', 'z']),
    ],
  )
