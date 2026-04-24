'''lsqb_q9_neg2hop.nim -- 2-hop + negation + count'''

from integration_helpers import diff_hir, diff_mir

from srdatalog.dsl import Filter, Program, Relation, Var


def build_lsqb_q9_neg2hop() -> Program:
  X, Y, P, T_ = Var("x"), Var("y"), Var("p"), Var("t")
  P1, P2, P3, T = Var("p1"), Var("p2"), Var("p3"), Var("t")
  k_in = Relation("KnowsInput", 2)
  hi_in = Relation("HasInterestInput", 2)
  knows = Relation("Knows", 2)
  hi = Relation("HasInterest", 2)
  path = Relation("Path", 4)
  return Program(
    rules=[
      (knows(X, Y) <= k_in(X, Y)).named("KnowsLoad"),
      (hi(P, T_) <= hi_in(P, T_)).named("InterestLoad"),
      (
        path(P1, P2, P3, T)
        <= knows(P1, P2)
        & knows(P2, P3)
        & hi(P3, T)
        & ~knows(P1, P3)
        & Filter(vars=("p1", "p3"), code="return p1 != p3;")
      )
      .named("NegPath")
      .with_plan(
        var_order=["p2", "p3", "p1", "t"],
      )
      .with_count(),
    ],
  )


def test_lsqb_q9_neg2hop_hir():
  diff_hir(build_lsqb_q9_neg2hop(), "lsqb_q9_neg2hop")


def test_lsqb_q9_neg2hop_mir():
  diff_mir(build_lsqb_q9_neg2hop(), "lsqb_q9_neg2hop")


if __name__ == "__main__":
  test_lsqb_q9_neg2hop_hir()
  test_lsqb_q9_neg2hop_mir()
  print("lsqb_q9_neg2hop: OK")
