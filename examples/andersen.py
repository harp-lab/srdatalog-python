"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/andersen_gpu/andersen.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/andersen_gpu/andersen.nim --out <this file>
"""

from __future__ import annotations

from srdatalog.dataset_const import load_meta, resolve_program_consts
from srdatalog.dsl import Program, Relation, Var

# ----- Relations ----------------------------------------------

AddressOf = Relation(
  "AddressOf",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="addressOf.csv",
)
Assign = Relation(
  "Assign",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="assign.csv",
)
Load = Relation(
  "Load",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="load.csv",
)
Store = Relation(
  "Store",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="store.csv",
)
PointsTo = Relation(
  "PointsTo",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)

# ----- dataset_const declarations -----------------------------

DATASET_CONST_DECLS = {}

# ----- Rules: AndersenDB -----


def build_andersendb_program() -> Program:
  w = Var("w")
  x = Var("x")
  y = Var("y")
  z = Var("z")

  return Program(
    relations=[
      AddressOf,
      Assign,
      Load,
      Store,
      PointsTo,
    ],
    rules=[
      (PointsTo(y, x) <= AddressOf(y, x)).named('Base').with_plan(var_order=['y', 'x']),
      (PointsTo(y, x) <= PointsTo(z, x) & Assign(y, z))
      .named('Assign')
      .with_plan(var_order=['z', 'x', 'y']),
      (PointsTo(y, w) <= PointsTo(x, z) & Load(y, x) & PointsTo(z, w))
      .named('Load')
      .with_plan(delta=0, var_order=['x', 'z', 'y', 'w'])
      .with_plan(delta=2, var_order=['z', 'x', 'y', 'w']),
      (PointsTo(z, w) <= PointsTo(y, z) & Store(y, x) & PointsTo(x, w))
      .named('Store')
      .with_plan(delta=0, var_order=['y', 'x', 'z', 'w'])
      .with_plan(delta=2, var_order=['x', 'y', 'w', 'z']),
    ],
  )


def build_andersendb(meta_json_path: str) -> tuple[Program, dict[str, int]]:
  """Convenience: build the program, load dataset_consts, substitute."""
  consts = load_meta(meta_json_path, DATASET_CONST_DECLS)
  return resolve_program_consts(build_andersendb_program(), consts), consts
