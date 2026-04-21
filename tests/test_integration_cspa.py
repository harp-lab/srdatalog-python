'''cspa.nim -- mutual-recursion + wildcard body args (_gen1..._gen4)'''
from integration_helpers import diff_hir, diff_mir
from srdatalog.dsl import Var, Relation, Program


def build_cspa() -> Program:
  X, Y, Z, W = Var("x"), Var("y"), Var("z"), Var("w")
  G1, G2, G3, G4 = Var("_gen1"), Var("_gen2"), Var("_gen3"), Var("_gen4")
  ai = Relation("AssignInput", 2)
  di = Relation("DerefInput", 2)
  asgn = Relation("Assign", 2)
  deref = Relation("Deref", 2)
  vf = Relation("ValueFlow", 2)
  va = Relation("ValueAlias", 2)
  ma = Relation("MemoryAlias", 2)
  return Program(
    relations=[ai, di, asgn, deref, vf, va, ma],
    rules=[
      (asgn(X, Y) <= ai(X, Y)).named("LoadAssign"),
      (deref(X, Y) <= di(X, Y)).named("LoadDeref"),
      (vf(Y, X) <= asgn(Y, X)).named("VF_Assign"),
      (vf(X, X) <= asgn(X, G1)).named("VF_Refl1"),
      (vf(X, X) <= asgn(G2, X)).named("VF_Refl2"),
      (ma(X, X) <= asgn(G3, X)).named("MA_Refl1"),
      (ma(X, X) <= asgn(X, G4)).named("MA_Refl2"),
      (
        vf(X, Y) <= vf(X, Z) & vf(Z, Y)
      ).named("VF_Trans").with_plan(var_order=["z", "x", "y"]),
      (
        vf(X, Y) <= asgn(X, Z) & ma(Z, Y)
      ).named("VF_Assign_MA").with_plan(var_order=["z", "x", "y"]),
      (
        va(X, Y) <= vf(Z, X) & vf(Z, Y)
      ).named("VA_VF").with_plan(var_order=["z", "x", "y"]),
      (
        va(X, Y) <= vf(Z, X) & ma(Z, W) & vf(W, Y)
      ).named("VA_VF_MA_VF").with_plan(var_order=["z", "w", "x", "y"]),
      (
        ma(X, W) <= deref(Y, X) & va(Y, Z) & deref(Z, W)
      ).named("MA_Deref_VA").with_plan(var_order=["y", "z", "x", "w"]),
    ],
  )


def test_cspa_hir():
  diff_hir(build_cspa(), "cspa")


def test_cspa_mir():
  diff_mir(build_cspa(), "cspa")


if __name__ == "__main__":
  test_cspa_hir()
  test_cspa_mir()
  print("cspa: OK")
