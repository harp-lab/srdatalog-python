"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/andersen_gpu/andersen.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/andersen_gpu/andersen.nim --out <this file>
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
      .with_plan(delta=2, var_order=['x', 'y', 'w', 'z'])
      .with_inject_cpp("""
      auto& points_to_delta = get_relation_by_schema<PointsTo, DELTA_VER>(db);
      auto& points_to_delta_idx = points_to_delta.get_index({{0,1}});
      auto& points_to_full = get_relation_by_schema<PointsTo, FULL_VER>(db);
      auto& points_to_full_idx = points_to_full.get_index({{0,1}});
      std::cout << "  PointsTo delta: " << points_to_delta_idx.root().degree()
                << ", full: " << points_to_full_idx.root().degree() << std::endl;  
    """),
    ],
  )
