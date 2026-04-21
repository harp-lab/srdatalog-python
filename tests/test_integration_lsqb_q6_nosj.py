'''lsqb_q6_nosj.nim -- 2-hop with work_stealing'''

from integration_helpers import diff_hir, diff_mir

from srdatalog.dsl import Filter, Program, Relation, Var


def build_lsqb_q6_nosj() -> Program:
  X, Y = Var("x"), Var("y")
  P, T_ = Var("p"), Var("t")
  P1, P2, P3, T = Var("p1"), Var("p2"), Var("p3"), Var("t")
  k_in = Relation("KnowsInput", 2)
  k2_in = Relation("KnowsInput2", 2)
  hi_in = Relation("HasInterestInput", 2)
  knows = Relation("Knows", 2)
  knows2 = Relation("Knows2", 2)
  hi = Relation("HasInterest", 2)
  path = Relation("Path", 4)
  return Program(
    relations=[k_in, k2_in, hi_in, knows, knows2, hi, path],
    rules=[
      (knows(X, Y) <= k_in(X, Y)).named("KnowsLoad"),
      (knows2(X, Y) <= k2_in(X, Y)).named("Knows2Load"),
      (hi(P, T_) <= hi_in(P, T_)).named("InterestLoad"),
      (
        path(P1, P2, P3, T)
        <= knows(P1, P2)
        & knows2(P2, P3)
        & hi(P3, T)
        & Filter(vars=("p1", "p3"), code="return p1 != p3;")
      )
      .named("TwoHopPath")
      .with_plan(
        var_order=["p1", "p2", "p3", "t"],
        work_stealing=True,
      ),
    ],
  )


def test_lsqb_q6_nosj_hir():
  diff_hir(build_lsqb_q6_nosj(), "lsqb_q6_nosj")


def test_lsqb_q6_nosj_mir():
  diff_mir(build_lsqb_q6_nosj(), "lsqb_q6_nosj")


if __name__ == "__main__":
  test_lsqb_q6_nosj_hir()
  test_lsqb_q6_nosj_mir()
  print("lsqb_q6_nosj: OK")
