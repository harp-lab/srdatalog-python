'''Triangle-counting example — mirrors `integration_tests/examples/triangle/triangle.nim`
in the parent SRDatalog repo so the emitted .cpp tree is byte-comparable.

Run:

    python examples/triangle.py

emits the same `jit_batch_0.cpp` Nim writes to its JIT cache, so the
*exact same* compile flags / external dependencies that work for Nim's
output work here too.
'''
from srdatalog import Var, Relation, Program, build_project


def program() -> Program:
  '''Z(x, y, z) :- R(x, y), S(y, z, h), T(z, x, f).
  Matches the Nim definition: T uses (z, x, f) — index [1, 0, 2].
  '''
  x, y, z = Var("x"), Var("y"), Var("z")
  h, f = Var("h"), Var("f")
  R = Relation("RRel", 2, column_types=(int, int))
  S = Relation("SRel", 3, column_types=(int, int, int))
  T = Relation("TRel", 3, column_types=(int, int, int))
  Z = Relation("ZRel", 3, column_types=(int, int, int))
  return Program(
    relations=[R, S, T, Z],
    rules=[
      (Z(x, y, z) <= R(x, y) & S(y, z, h) & T(z, x, f)).named("Triangle"),
    ],
  )


def main() -> None:
  result = build_project(
    program(),
    project_name="TrianglePlan",
    cache_base="./build",
  )
  print("Wrote:")
  print(f"  dir   : {result['dir']}")
  print(f"  main  : {result['main']}")
  for b in result["batches"]:
    print(f"  batch : {b}")
  print()
  print("Each batch file inlines the schema typedefs + DB blueprint, so it")
  print("compiles standalone with the same -I flags as Nim's JIT output.")


if __name__ == "__main__":
  main()
