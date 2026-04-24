"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/triangle/lsqb_q6_nosj.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/triangle/lsqb_q6_nosj.nim --out <this file>
"""

from __future__ import annotations

from srdatalog.dsl import SPLIT, Filter, Program, Relation, Var

# ----- Relations ----------------------------------------------

KnowsInput = Relation(
  "KnowsInput",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Person_knows_Person.csv",
)
KnowsInput2 = Relation(
  "KnowsInput2",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Person_knows_Person.csv",
)
HasInterestInput = Relation(
  "HasInterestInput",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Person_hasInterest_Tag.csv",
)
Knows = Relation(
  "Knows",
  2,
  column_types=(
    int,
    int,
  ),
)
Knows2 = Relation(
  "Knows2",
  2,
  column_types=(
    int,
    int,
  ),
)
HasInterest = Relation(
  "HasInterest",
  2,
  column_types=(
    int,
    int,
  ),
)
Path = Relation(
  "Path",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
)

# ----- Rules: LSQB_Q6_NoSJ_DB -----


def build_lsqb_q6_nosj_db_program() -> Program:
  p = Var("p")
  p1 = Var("p1")
  p2 = Var("p2")
  p3 = Var("p3")
  t = Var("t")
  x = Var("x")
  y = Var("y")

  return Program(
    rules=[
      (Knows(x, y) <= KnowsInput(x, y)).named('KnowsLoad'),
      (Knows2(x, y) <= KnowsInput2(x, y)).named('Knows2Load'),
      (HasInterest(p, t) <= HasInterestInput(p, t)).named('InterestLoad'),
      (
        Path(p1, p2, p3, t)
        <= Knows(p1, p2)
        & Knows2(p2, p3)
        & HasInterest(p3, t)
        & Filter(
          (
            'p1',
            'p3',
          ),
          "return p1 != p3;",
        )
      )
      .named('TwoHopPath')
      .with_plan(var_order=['p1', 'p2', 'p3', 't']),
    ],
  )
