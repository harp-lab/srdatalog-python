"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/triangle/lsqb_triangle_count.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/triangle/lsqb_triangle_count.nim --out <this file>
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
Knows = Relation(
  "Knows",
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
)

# ----- dataset_const declarations -----------------------------

DATASET_CONST_DECLS = {}

# ----- Rules: Triangle_DB -----


def build_triangle_db_program() -> Program:
  a = Var("a")
  b = Var("b")
  c = Var("c")
  x = Var("x")
  y = Var("y")

  return Program(
    relations=[
      KnowsInput,
      Knows,
      Triangle,
    ],
    rules=[
      (Knows(x, y) <= KnowsInput(x, y)).named('KnowsLoad'),
      (Triangle(a, b, c) <= Knows(a, b) & Knows(b, c) & Knows(a, c))
      .named('TriangleJoin')
      .with_plan(var_order=['a', 'b', 'c'])
      .with_count(),
    ],
  )


def build_triangle_db(meta_json_path: str) -> tuple[Program, dict[str, int]]:
  """Convenience: build the program, load dataset_consts, substitute."""
  consts = load_meta(meta_json_path, DATASET_CONST_DECLS)
  return resolve_program_consts(build_triangle_db_program(), consts), consts
