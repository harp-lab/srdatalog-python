"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/triangle/gen_jit_code.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/triangle/gen_jit_code.nim --out <this file>
"""

from __future__ import annotations

from srdatalog.dsl import SPLIT, Filter, Program, Relation, Var

# ----- Relations ----------------------------------------------

Edge = Relation(
  "Edge",
  2,
  column_types=(
    int,
    int,
  ),
)
Z = Relation(
  "Z",
  3,
  column_types=(
    int,
    int,
    int,
  ),
)

# ----- Rules: TriangleDB -----


def build_triangledb_program() -> Program:
  x = Var("x")
  y = Var("y")
  z = Var("z")

  return Program(
    rules=[
      (Z(x, y, z) <= Edge(x, y) & Edge(y, z) & Edge(z, x))
      .named('Triangle')
      .with_plan(var_order=['x', 'y', 'z']),
    ],
  )
