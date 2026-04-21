'''gen_jit_code.nim -- trivial triangle from Edge'''
from integration_helpers import diff_hir, diff_mir
from srdatalog.dsl import Var, Relation, Program


def build_gen_jit_code() -> Program:
  X, Y, Z = Var("x"), Var("y"), Var("z")
  edge = Relation("Edge", 2)
  z = Relation("Z", 3)
  return Program(
    relations=[edge, z],
    rules=[
      (
        z(X, Y, Z) <= edge(X, Y) & edge(Y, Z) & edge(Z, X)
      ).named("Triangle").with_plan(var_order=["x", "y", "z"]),
    ],
  )


def test_gen_jit_code_hir():
  diff_hir(build_gen_jit_code(), "gen_jit_code")


def test_gen_jit_code_mir():
  diff_mir(build_gen_jit_code(), "gen_jit_code")


if __name__ == "__main__":
  test_gen_jit_code_hir()
  test_gen_jit_code_mir()
  print("gen_jit_code: OK")
