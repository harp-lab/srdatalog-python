"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/tc/tc.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/tc/tc.nim --out <this file>
"""

from __future__ import annotations

from srdatalog.dsl import SPLIT, Filter, Program, Relation, Var

# ----- Relations ----------------------------------------------

ArcInput = Relation(
  "ArcInput",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Arc.csv",
)
Edge = Relation(
  "Edge",
  2,
  column_types=(
    int,
    int,
  ),
)
Path = Relation(
  "Path",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)

# ----- Rules: TCDB -----


def build_tcdb_program() -> Program:
  x = Var("x")
  y = Var("y")
  z = Var("z")

  return Program(
    rules=[
      (Edge(x, y) <= ArcInput(x, y)).named('EdgeLoad'),
      (Path(x, y) <= Edge(x, y)).named('TCBase'),
      (Path(x, z) <= Path(x, y) & Edge(y, z)).named('TCRec'),
    ],
  )
