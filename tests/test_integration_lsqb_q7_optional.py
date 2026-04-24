'''lsqb_q7_optional.nim -- 4 cases via negation + wildcards _gen1, _gen2'''

from integration_helpers import diff_hir, diff_mir

from srdatalog.dsl import Program, Relation, Var


def build_lsqb_q7_optional() -> Program:
  G1, G2 = Var("_gen1"), Var("_gen2")
  M, T_, C, CMT, LKR = (
    Var("m"),
    Var("t"),
    Var("c"),
    Var("comment"),
    Var("liker"),
  )
  has_tag = Relation("HasTag", 2)
  has_creator = Relation("HasCreator", 2)
  reply_of = Relation("ReplyOf", 2)
  likes = Relation("Likes", 2)
  has_reply = Relation("HasReply", 1)
  has_liker = Relation("HasLiker", 1)
  case1 = Relation("Case1", 5)
  case2 = Relation("Case2", 4)
  case3 = Relation("Case3", 4)
  case4 = Relation("Case4", 3)
  return Program(
    rules=[
      (has_reply(M) <= reply_of(G1, M)).named("MarkHasReply"),
      (has_liker(M) <= likes(G2, M)).named("MarkHasLiker"),
      (
        case1(M, T_, C, CMT, LKR)
        <= has_tag(M, T_) & has_creator(M, C) & reply_of(CMT, M) & likes(LKR, M)
      )
      .named("Case1Both")
      .with_plan(
        var_order=["m", "comment", "liker", "t", "c"],
      )
      .with_count(),
      (
        case2(M, T_, C, CMT)
        <= has_tag(M, T_) & has_creator(M, C) & reply_of(CMT, M) & ~has_liker(M)
      )
      .named("Case2ReplyOnly")
      .with_plan(
        var_order=["m", "comment", "t", "c"],
      )
      .with_count(),
      (case3(M, T_, C, LKR) <= has_tag(M, T_) & has_creator(M, C) & likes(LKR, M) & ~has_reply(M))
      .named("Case3LikeOnly")
      .with_plan(
        var_order=["m", "liker", "t", "c"],
      )
      .with_count(),
      (case4(M, T_, C) <= has_tag(M, T_) & has_creator(M, C) & ~has_reply(M) & ~has_liker(M))
      .named("Case4Neither")
      .with_plan(
        var_order=["m", "t", "c"],
      )
      .with_count(),
    ],
  )


def test_lsqb_q7_optional_hir():
  diff_hir(build_lsqb_q7_optional(), "lsqb_q7_optional")


def test_lsqb_q7_optional_mir():
  diff_mir(build_lsqb_q7_optional(), "lsqb_q7_optional")


if __name__ == "__main__":
  test_lsqb_q7_optional_hir()
  test_lsqb_q7_optional_mir()
  print("lsqb_q7_optional: OK")
