"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/triangle/lsqb_q9_neg2hop.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/triangle/lsqb_q9_neg2hop.nim --out <this file>
"""

from __future__ import annotations

from srdatalog.dataset_const import load_meta, resolve_program_consts
from srdatalog.dsl import Filter, Program, Relation, Var

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
  print_size=True,
)

# ----- dataset_const declarations -----------------------------

DATASET_CONST_DECLS = {}

# ----- Rules: LSQB_Q9_DB -----


def build_lsqb_q9_db_program() -> Program:
  p = Var("p")
  p1 = Var("p1")
  p2 = Var("p2")
  p3 = Var("p3")
  t = Var("t")
  x = Var("x")
  y = Var("y")

  return Program(
    relations=[
      KnowsInput,
      HasInterestInput,
      Knows,
      HasInterest,
      Path,
    ],
    rules=[
      (Knows(x, y) <= KnowsInput(x, y)).named('KnowsLoad'),
      (HasInterest(p, t) <= HasInterestInput(p, t)).named('InterestLoad'),
      (
        Path(p1, p2, p3, t)
        <= Knows(p1, p2)
        & Knows(p2, p3)
        & HasInterest(p3, t)
        & ~Knows(p1, p3)
        & Filter(
          (
            'p1',
            'p3',
          ),
          "return p1 != p3;",
        )
      )
      .named('NegPath')
      .with_plan(var_order=['p2', 'p3', 'p1', 't'])
      .with_count(),
    ],
  )


def build_lsqb_q9_db(meta_json_path: str) -> tuple[Program, dict[str, int]]:
  """Convenience: build the program, load dataset_consts, substitute."""
  consts = load_meta(meta_json_path, DATASET_CONST_DECLS)
  return resolve_program_consts(build_lsqb_q9_db_program(), consts), consts
