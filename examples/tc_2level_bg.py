"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/tc/tc_2level_bg.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/tc/tc_2level_bg.nim --out <this file>
"""

from __future__ import annotations

from srdatalog.dsl import Filter, Program, Relation, SPLIT, Var

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
  index_type="SRDatalog::GPU::Device2LevelIndex",
)

# ----- Rules: TC2LBGDB -----


def build_tc2lbgdb_program() -> Program:
  x = Var("x")
  y = Var("y")
  z = Var("z")

  return Program(
    relations=[
      ArcInput,
      Edge,
      Path,
    ],
    rules=[
      (Edge(x, y) <= ArcInput(x, y)).named('EdgeLoad'),
      (Path(x, y) <= Edge(x, y)).named('TCBase'),
      (Path(x, z) <= Path(x, y) & Edge(y, z))
      .named('TCRec')
      .with_plan(delta=0, var_order=['y', 'x', 'z'], block_group=True),
    ],
  )
