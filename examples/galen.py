"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/galen/galen.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/galen/galen.nim --out <this file>
"""

from __future__ import annotations

from srdatalog.dsl import SPLIT, Filter, Program, Relation, Var

# ----- Relations ----------------------------------------------

PInput = Relation(
  "PInput",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="P.csv",
)
QInput = Relation(
  "QInput",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="Q.csv",
)
RInput = Relation(
  "RInput",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="R.csv",
)
CInput = Relation(
  "CInput",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="C.csv",
)
UInput = Relation(
  "UInput",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="U.csv",
)
SInput = Relation(
  "SInput",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="S.csv",
)
OutP = Relation(
  "OutP",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
OutQ = Relation(
  "OutQ",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  print_size=True,
)

# ----- Rules: GalenDB -----


def build_galendb_program() -> Program:
  e = Var("e")
  o = Var("o")
  q = Var("q")
  r = Var("r")
  u = Var("u")
  w = Var("w")
  x = Var("x")
  y = Var("y")
  z = Var("z")

  return Program(
    rules=[
      (OutP(x, z) <= PInput(x, z)).named('LoadP'),
      (OutQ(x, r, z) <= QInput(x, r, z)).named('LoadQ'),
      (OutP(x, z) <= OutP(x, y) & OutP(y, z)).named('TC'),
      (OutQ(x, r, z) <= OutP(x, y) & OutQ(y, r, z)).named('PropQ'),
      (OutP(x, z) <= OutP(y, w) & UInput(w, r, z) & OutQ(x, r, y))
      .named('Join3a')
      .with_plan(delta=2, var_order=['y', 'w', 'r', 'x', 'z']),
      (OutP(x, z) <= CInput(y, w, z) & OutP(x, w) & OutP(x, y)).named('Join3b'),
      (OutQ(x, q, z) <= OutQ(x, r, z) & SInput(r, q)).named('PropQS'),
      (OutQ(x, e, o) <= OutQ(x, y, z) & RInput(y, u, e) & OutQ(z, u, o))
      .named('Join3c')
      .with_plan(delta=0, var_order=['z', 'y', 'u', 'x', 'e', 'o']),
    ],
  )
