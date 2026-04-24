"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/triangle/lsqb_q7_optional.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/triangle/lsqb_q7_optional.nim --out <this file>
"""

from __future__ import annotations

from srdatalog.dsl import SPLIT, Filter, Program, Relation, Var

# ----- Relations ----------------------------------------------

HasTag = Relation(
  "HasTag",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Message_hasTag_Tag.csv",
)
HasCreator = Relation(
  "HasCreator",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Message_hasCreator_Person.csv",
)
ReplyOf = Relation(
  "ReplyOf",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="ReplyOf_Comment_Message.csv",
)
Likes = Relation(
  "Likes",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Person_likes_Message.csv",
)
HasReply = Relation("HasReply", 1, column_types=(int,))
HasLiker = Relation("HasLiker", 1, column_types=(int,))
Case1 = Relation(
  "Case1",
  5,
  column_types=(
    int,
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
Case2 = Relation(
  "Case2",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
Case3 = Relation(
  "Case3",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
Case4 = Relation(
  "Case4",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  print_size=True,
)

# ----- Rules: LSQB_Q7_DB -----


def build_lsqb_q7_db_program() -> Program:
  c = Var("c")
  comment = Var("comment")
  liker = Var("liker")
  m = Var("m")
  t = Var("t")

  return Program(
    relations=[
      HasTag,
      HasCreator,
      ReplyOf,
      Likes,
      HasReply,
      HasLiker,
      Case1,
      Case2,
      Case3,
      Case4,
    ],
    rules=[
      (HasReply(m) <= ReplyOf(Var("_"), m)).named('MarkHasReply'),
      (HasLiker(m) <= Likes(Var("_"), m)).named('MarkHasLiker'),
      (
        Case1(m, t, c, comment, liker)
        <= HasTag(m, t) & HasCreator(m, c) & ReplyOf(comment, m) & Likes(liker, m)
      )
      .named('Case1Both')
      .with_plan(var_order=['m', 'comment', 'liker', 't', 'c'])
      .with_count(),
      (
        Case2(m, t, c, comment)
        <= HasTag(m, t) & HasCreator(m, c) & ReplyOf(comment, m) & ~HasLiker(m)
      )
      .named('Case2ReplyOnly')
      .with_plan(var_order=['m', 'comment', 't', 'c'])
      .with_count(),
      (Case3(m, t, c, liker) <= HasTag(m, t) & HasCreator(m, c) & Likes(liker, m) & ~HasReply(m))
      .named('Case3LikeOnly')
      .with_plan(var_order=['m', 'liker', 't', 'c'])
      .with_count(),
      (Case4(m, t, c) <= HasTag(m, t) & HasCreator(m, c) & ~HasReply(m) & ~HasLiker(m))
      .named('Case4Neither')
      .with_plan(var_order=['m', 't', 'c'])
      .with_count(),
    ],
  )
