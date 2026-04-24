'''lsqb_q3_triangle.nim -- 9-clause labeled triangle'''

from integration_helpers import diff_hir, diff_mir

from srdatalog.dsl import Program, Relation, Var


def build_lsqb_q3() -> Program:
  X, Y, P, C, CO = Var("x"), Var("y"), Var("p"), Var("c"), Var("co")
  P1, P2, P3 = Var("p1"), Var("p2"), Var("p3")
  C1, C2, C3 = Var("c1"), Var("c2"), Var("c3")
  COv = Var("co")
  k_in = Relation("KnowsInput", 2)
  knows = Relation("Knows", 2)
  il_in = Relation("IsLocatedInInput", 2)
  il = Relation("IsLocatedIn", 2)
  po_in = Relation("IsPartOfInput", 2)
  po = Relation("IsPartOf", 2)
  tri = Relation("Triangle", 3)
  return Program(
    rules=[
      (knows(X, Y) <= k_in(X, Y)).named("KnowsLoad"),
      (il(P, C) <= il_in(P, C)).named("LocLoad"),
      (po(C, CO) <= po_in(C, CO)).named("PartOfLoad"),
      (
        tri(P1, P2, P3)
        <= knows(P1, P2)
        & knows(P2, P3)
        & knows(P3, P1)
        & il(P1, C1)
        & il(P2, C2)
        & il(P3, C3)
        & po(C1, COv)
        & po(C2, COv)
        & po(C3, COv)
      )
      .named("LabeledTriangle")
      .with_plan(
        var_order=["p1", "p2", "p3", "c1", "c2", "c3", "co"],
      ),
    ],
  )


def test_lsqb_q3_hir():
  diff_hir(build_lsqb_q3(), "lsqb_q3_triangle")


def test_lsqb_q3_mir():
  diff_mir(build_lsqb_q3(), "lsqb_q3_triangle")


if __name__ == "__main__":
  test_lsqb_q3_hir()
  test_lsqb_q3_mir()
  print("lsqb_q3: OK")
