"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/triangle/debug_triangle.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/triangle/debug_triangle.nim --out <this file>
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
  2,
  column_types=(
    int,
    int,
  ),
)
TRel = Relation(
  "TRel",
  2,
  column_types=(
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

# ----- Rules: TriangleDebugDB -----


def build_triangledebugdb_program() -> Program:
  x = Var("x")
  y = Var("y")
  z = Var("z")

  return Program(
    rules=[
      (ZRel(x, y, z) <= RRel(x, y) & SRel(y, z) & TRel(z, x))
      .named('TriangleDebug')
      .with_plan(var_order=['x', 'y', 'z']),
    ],
  )
