"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/triangle/lsqb_q3_triangle.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/triangle/lsqb_q3_triangle.nim --out <this file>
"""

from __future__ import annotations

from srdatalog.dataset_const import load_meta, resolve_program_consts
from srdatalog.dsl import Program, Relation, Var

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
IsLocatedInInput = Relation(
  "IsLocatedInInput",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Person_isLocatedIn_City.csv",
)
IsPartOfInput = Relation(
  "IsPartOfInput",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="City_isPartOf_Country.csv",
)
Knows = Relation(
  "Knows",
  2,
  column_types=(
    int,
    int,
  ),
)
IsLocatedIn = Relation(
  "IsLocatedIn",
  2,
  column_types=(
    int,
    int,
  ),
)
IsPartOf = Relation(
  "IsPartOf",
  2,
  column_types=(
    int,
    int,
  ),
)
Triangle = Relation(
  "Triangle",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  print_size=True,
)

# ----- dataset_const declarations -----------------------------

DATASET_CONST_DECLS = {}

# ----- Rules: LSQB_Q3_DB -----


def build_lsqb_q3_db_program() -> Program:
  c = Var("c")
  c1 = Var("c1")
  c2 = Var("c2")
  c3 = Var("c3")
  co = Var("co")
  p = Var("p")
  p1 = Var("p1")
  p2 = Var("p2")
  p3 = Var("p3")
  x = Var("x")
  y = Var("y")

  return Program(
    relations=[
      KnowsInput,
      IsLocatedInInput,
      IsPartOfInput,
      Knows,
      IsLocatedIn,
      IsPartOf,
      Triangle,
    ],
    rules=[
      (Knows(x, y) <= KnowsInput(x, y)).named('KnowsLoad'),
      (IsLocatedIn(p, c) <= IsLocatedInInput(p, c)).named('LocLoad'),
      (IsPartOf(c, co) <= IsPartOfInput(c, co)).named('PartOfLoad'),
      (
        Triangle(p1, p2, p3)
        <= Knows(p1, p2)
        & Knows(p2, p3)
        & Knows(p3, p1)
        & IsLocatedIn(p1, c1)
        & IsLocatedIn(p2, c2)
        & IsLocatedIn(p3, c3)
        & IsPartOf(c1, co)
        & IsPartOf(c2, co)
        & IsPartOf(c3, co)
      )
      .named('LabeledTriangle')
      .with_plan(var_order=['p1', 'p2', 'p3', 'c1', 'c2', 'c3', 'co']),
    ],
  )


def build_lsqb_q3_db(meta_json_path: str) -> tuple[Program, dict[str, int]]:
  """Convenience: build the program, load dataset_consts, substitute."""
  consts = load_meta(meta_json_path, DATASET_CONST_DECLS)
  return resolve_program_consts(build_lsqb_q3_db_program(), consts), consts
