"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/andersen_gpu/cspa.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/andersen_gpu/cspa.nim --out <this file>
"""

from __future__ import annotations

from srdatalog.dataset_const import load_meta, resolve_program_consts
from srdatalog.dsl import Program, Relation, Var

# ----- Relations ----------------------------------------------

AssignInput = Relation(
  "AssignInput",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Assign.csv",
)
DerefInput = Relation(
  "DerefInput",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Dereference.csv",
)
Assign = Relation(
  "Assign",
  2,
  column_types=(
    int,
    int,
  ),
)
Deref = Relation(
  "Deref",
  2,
  column_types=(
    int,
    int,
  ),
)
ValueFlow = Relation(
  "ValueFlow",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
ValueAlias = Relation(
  "ValueAlias",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
MemoryAlias = Relation(
  "MemoryAlias",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)

# ----- dataset_const declarations -----------------------------

DATASET_CONST_DECLS = {}

# ----- Rules: CSPA_DB -----


def build_cspa_db_program() -> Program:
  w = Var("w")
  x = Var("x")
  y = Var("y")
  z = Var("z")

  return Program(
    relations=[
      AssignInput,
      DerefInput,
      Assign,
      Deref,
      ValueFlow,
      ValueAlias,
      MemoryAlias,
    ],
    rules=[
      (Assign(x, y) <= AssignInput(x, y)).named('LoadAssign'),
      (Deref(x, y) <= DerefInput(x, y)).named('LoadDeref'),
      (ValueFlow(y, x) <= Assign(y, x)).named('VF_Assign'),
      (ValueFlow(x, x) <= Assign(x, Var("_"))).named('VF_Refl1'),
      (ValueFlow(x, x) <= Assign(Var("_"), x)).named('VF_Refl2'),
      (MemoryAlias(x, x) <= Assign(Var("_"), x)).named('MA_Refl1'),
      (MemoryAlias(x, x) <= Assign(x, Var("_"))).named('MA_Refl2'),
      (ValueFlow(x, y) <= ValueFlow(x, z) & ValueFlow(z, y))
      .named('VF_Trans')
      .with_plan(var_order=['z', 'x', 'y']),
      (ValueFlow(x, y) <= Assign(x, z) & MemoryAlias(z, y))
      .named('VF_Assign_MA')
      .with_plan(var_order=['z', 'x', 'y']),
      (ValueAlias(x, y) <= ValueFlow(z, x) & ValueFlow(z, y))
      .named('VA_VF')
      .with_plan(var_order=['z', 'x', 'y']),
      (ValueAlias(x, y) <= ValueFlow(z, x) & MemoryAlias(z, w) & ValueFlow(w, y))
      .named('VA_VF_MA_VF')
      .with_plan(var_order=['z', 'w', 'x', 'y']),
      (MemoryAlias(x, w) <= Deref(y, x) & ValueAlias(y, z) & Deref(z, w))
      .named('MA_Deref_VA')
      .with_plan(var_order=['y', 'z', 'x', 'w']),
    ],
  )


def build_cspa_db(meta_json_path: str) -> tuple[Program, dict[str, int]]:
  """Convenience: build the program, load dataset_consts, substitute."""
  consts = load_meta(meta_json_path, DATASET_CONST_DECLS)
  return resolve_program_consts(build_cspa_db_program(), consts), consts
