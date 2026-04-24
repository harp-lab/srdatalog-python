"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/andersen_gpu/print_mir.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/andersen_gpu/print_mir.nim --out <this file>
"""

from __future__ import annotations

from srdatalog.dsl import SPLIT, Filter, Program, Relation, Var

# ----- Relations ----------------------------------------------

AddressOf = Relation(
  "AddressOf",
  2,
  column_types=(
    int,
    int,
  ),
)
Assign = Relation(
  "Assign",
  2,
  column_types=(
    int,
    int,
  ),
)
Load = Relation(
  "Load",
  2,
  column_types=(
    int,
    int,
  ),
)
Store = Relation(
  "Store",
  2,
  column_types=(
    int,
    int,
  ),
)
PointsTo = Relation(
  "PointsTo",
  2,
  column_types=(
    int,
    int,
  ),
)

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
      .with_plan(delta=2, var_order=['z', 'w', 'x', 'y']),
      (PointsTo(z, w) <= PointsTo(y, z) & Store(y, x) & PointsTo(x, w))
      .named('Store')
      .with_plan(delta=0, var_order=['y', 'x', 'z', 'w'])
      .with_plan(delta=2, var_order=['x', 'y', 'w', 'z']),
    ],
  )
