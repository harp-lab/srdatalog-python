"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/tc/tc.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/tc/tc.nim --out <this file>
"""
from __future__ import annotations

from srdatalog.dsl import Filter, Program, Relation, Var
from srdatalog.dataset_const import load_meta, resolve_program_consts

# ----- Relations ----------------------------------------------

ArcInput = Relation("ArcInput", 2, column_types=(int, int,), input_file="Arc.csv")
Edge = Relation("Edge", 2, column_types=(int, int,))
Path = Relation("Path", 2, column_types=(int, int,), print_size=True)

# ----- dataset_const declarations -----------------------------

DATASET_CONST_DECLS = {
}

# ----- Rules: TCDB -----

def build_tcdb_program() -> Program:
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
      (Path(x, z) <= Path(x, y) & Edge(y, z)).named('TCRec'),
    ],
  )


def build_tcdb(meta_json_path: str) -> tuple[Program, dict[str, int]]:
  """Convenience: build the program, load dataset_consts, substitute."""
  consts = load_meta(meta_json_path, DATASET_CONST_DECLS)
  return resolve_program_consts(build_tcdb_program(), consts), consts
