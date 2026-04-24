"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/sg/sg.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/sg/sg.nim --out <this file>
"""

from __future__ import annotations

from srdatalog.dsl import Filter, Program, Relation, SPLIT, Var

# ----- Relations ----------------------------------------------

Arc = Relation(
  "Arc",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Arc.csv",
)
Sg = Relation(
  "Sg",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
  index_type="SRDatalog::GPU::Device2LevelIndex",
)

# ----- Rules: SGDB -----


def build_sgdb_program() -> Program:
  p = Var("p")
  q = Var("q")
  x = Var("x")
  y = Var("y")

  return Program(
    relations=[
      Arc,
      Sg,
    ],
    rules=[
      (
        Sg(x, y)
        <= Arc(p, x)
        & Arc(p, y)
        & Filter(
          (
            'x',
            'y',
          ),
          "return x != y;",
        )
      )
      .named('SGBase')
      .with_plan(var_order=['p', 'x', 'y']),
      (Sg(x, y) <= Arc(p, x) & Sg(p, q) & Arc(q, y))
      .named('SGRec')
      .with_plan(var_order=['p', 'q', 'x', 'y']),
    ],
  )
