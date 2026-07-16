'''lsqb_q7_optional.nim -- 4 cases via negation + wildcards _gen1, _gen2'''

import csv
from collections import Counter
from pathlib import Path

from integration_helpers import diff_hir, diff_mir, diff_orchestrator_exact

from srdatalog.dsl import Program, Relation, Var
from srdatalog.ir.codegen.cuda.orchestrator import gen_step_body
from srdatalog.ir.hir import compile_to_mir

SMALL_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "lsqb_q7_small"


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


def test_lsqb_q7_optional_orchestrator_exactly_matches_nim():
  mir = compile_to_mir(build_lsqb_q7_optional())
  actual = "".join(
    gen_step_body(step, "LSQB_Q7_Plan_DB_DeviceDB", is_recursive, i)
    for i, (step, is_recursive) in enumerate(mir.steps)
  )
  diff_orchestrator_exact("lsqb_q7_optional", actual)


def test_lsqb_q7_small_fixture_has_expected_case_counts():
  def message_counts(filename: str, message_column: int) -> Counter[int]:
    with (SMALL_FIXTURE / filename).open(newline="") as f:
      return Counter(int(row[message_column]) for row in csv.reader(f))

  tags = message_counts("Message_hasTag_Tag.csv", 0)
  creators = message_counts("Message_hasCreator_Person.csv", 0)
  replies = message_counts("ReplyOf_Comment_Message.csv", 1)
  likes = message_counts("Person_likes_Message.csv", 1)
  cases = [0, 0, 0, 0]
  for message in tags.keys() & creators.keys():
    base = tags[message] * creators[message]
    reply_count = replies[message]
    like_count = likes[message]
    if reply_count and like_count:
      cases[0] += base * reply_count * like_count
    elif reply_count:
      cases[1] += base * reply_count
    elif like_count:
      cases[2] += base * like_count
    else:
      cases[3] += base
  assert cases == [6, 2, 3, 1]


if __name__ == "__main__":
  test_lsqb_q7_optional_hir()
  test_lsqb_q7_optional_mir()
  test_lsqb_q7_optional_orchestrator_exactly_matches_nim()
  test_lsqb_q7_small_fixture_has_expected_case_counts()
  print("lsqb_q7_optional: OK")
