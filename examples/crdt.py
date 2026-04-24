"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/crdt/crdt.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/crdt/crdt.nim --out <this file>
"""

from __future__ import annotations

from srdatalog.dsl import SPLIT, Filter, Program, Relation, Var

# ----- Relations ----------------------------------------------

InsertInput = Relation(
  "InsertInput",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  input_file="Insert_input.csv",
)
RemoveInput = Relation(
  "RemoveInput",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Remove_input.csv",
)
Insert = Relation(
  "Insert",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
Remove = Relation(
  "Remove",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
Assign = Relation(
  "Assign",
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
HasChild = Relation(
  "HasChild",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
LaterChild = Relation(
  "LaterChild",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
FirstChild = Relation(
  "FirstChild",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
Sibling = Relation(
  "Sibling",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
LaterSibling = Relation(
  "LaterSibling",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
LaterSibling2 = Relation(
  "LaterSibling2",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
NextSibling = Relation(
  "NextSibling",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
HasNextSibling = Relation(
  "HasNextSibling",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
NotSiblingInsert = Relation(
  "NotSiblingInsert",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
NextSiblingAnc = Relation(
  "NextSiblingAnc",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
NextElem = Relation(
  "NextElem",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
NotHasValueNextElem = Relation(
  "NotHasValueNextElem",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
CurrentValue = Relation(
  "CurrentValue",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  print_size=True,
)
HasValue = Relation(
  "HasValue",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
SkipBlank = Relation(
  "SkipBlank",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
NextVisible = Relation(
  "NextVisible",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
Result = Relation(
  "Result",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  print_size=True,
)

# ----- Rules: CRDTDB -----


def build_crdtdb_program() -> Program:
  ChildCtr = Var("ChildCtr")
  ChildCtr1 = Var("ChildCtr1")
  ChildCtr2 = Var("ChildCtr2")
  ChildN = Var("ChildN")
  ChildN1 = Var("ChildN1")
  ChildN2 = Var("ChildN2")
  Ctr1 = Var("Ctr1")
  Ctr2 = Var("Ctr2")
  Ctr3 = Var("Ctr3")
  ElemCtr = Var("ElemCtr")
  ElemN = Var("ElemN")
  FromCtr = Var("FromCtr")
  FromN = Var("FromN")
  IDCtr = Var("IDCtr")
  IDN = Var("IDN")
  N1 = Var("N1")
  N2 = Var("N2")
  N3 = Var("N3")
  NextCtr = Var("NextCtr")
  NextN = Var("NextN")
  ParentCtr = Var("ParentCtr")
  ParentN = Var("ParentN")
  PrevCtr = Var("PrevCtr")
  PrevN = Var("PrevN")
  SibCtr1 = Var("SibCtr1")
  SibN1 = Var("SibN1")
  StartCtr = Var("StartCtr")
  StartN = Var("StartN")
  ToCtr = Var("ToCtr")
  ToN = Var("ToN")
  Value = Var("Value")
  ViaCtr = Var("ViaCtr")
  ViaN = Var("ViaN")
  a = Var("a")
  b = Var("b")
  c = Var("c")
  ctr = Var("ctr")
  ctr1 = Var("ctr1")
  ctr2 = Var("ctr2")
  d = Var("d")
  n = Var("n")
  value = Var("value")

  return Program(
    rules=[
      (Insert(a, b, c, d) <= InsertInput(a, b, c, d)).named('Insert'),
      (Remove(a, b) <= RemoveInput(a, b)).named('Remove'),
      (
        LaterChild(ParentCtr, ParentN, Ctr2, N2)
        <= Insert(Ctr1, N1, ParentCtr, ParentN)
        & Insert(Ctr2, N2, ParentCtr, ParentN)
        & Filter(
          (
            'Ctr1',
            'N1',
            'Ctr2',
            'N2',
          ),
          "return (Ctr1 * 10 + N1) > (Ctr2 * 10 + N2);",
        )
      ).named('LaterChild'),
      (
        FirstChild(ParentCtr, ParentN, ChildCtr, ChildN)
        <= Insert(ChildCtr, ChildN, ParentCtr, ParentN)
        & ~LaterChild(ParentCtr, ParentN, ChildCtr, ChildN)
      ).named('FirstChild'),
      (
        Sibling(ChildCtr1, ChildN1, ChildCtr2, ChildN2)
        <= Insert(ChildCtr1, ChildN1, ParentCtr, ParentN)
        & Insert(ChildCtr2, ChildN2, ParentCtr, ParentN)
      ).named('Sibling'),
      (
        LaterSibling(Ctr1, N1, Ctr2, N2)
        <= Sibling(Ctr1, N1, Ctr2, N2)
        & Filter(
          (
            'Ctr1',
            'N1',
            'Ctr2',
            'N2',
          ),
          "return (Ctr1 * 10 + N1) > (Ctr2 * 10 + N2);",
        )
      ).named('LaterSibling'),
      (
        LaterSibling2(Ctr1, N1, Ctr3, N3)
        <= Sibling(Ctr1, N1, Ctr2, N2)
        & Sibling(Ctr1, N1, Ctr3, N3)
        & Filter(
          (
            'Ctr1',
            'N1',
            'Ctr2',
            'N2',
          ),
          "return (Ctr1 * 10 + N1) > (Ctr2 * 10 + N2);",
        )
        & Filter(
          (
            'Ctr2',
            'N2',
            'Ctr3',
            'N3',
          ),
          "return (Ctr2 * 10 + N2) > (Ctr3 * 10 + N3);",
        )
      ).named('LaterSibling2'),
      (
        NextSibling(Ctr1, N1, Ctr2, N2)
        <= LaterSibling(Ctr1, N1, Ctr2, N2) & ~LaterSibling2(Ctr1, N1, Ctr2, N2)
      ).named('NextSibling'),
      (HasNextSibling(SibCtr1, SibN1) <= LaterSibling(SibCtr1, SibN1, Var("_"), Var("_"))).named(
        'HasNextSibling'
      ),
      (
        NotSiblingInsert(Ctr1, N1, Ctr2, N2)
        <= Insert(Ctr1, N1, Ctr2, N2) & ~HasNextSibling(Ctr1, N1)
      ).named('NotSiblingInsert'),
      (
        NextSiblingAnc(StartCtr, StartN, NextCtr, NextN)
        <= NextSibling(StartCtr, StartN, NextCtr, NextN)
      ).named('NextSiblingAnc1'),
      (
        NextSiblingAnc(StartCtr, StartN, NextCtr, NextN)
        <= NextSiblingAnc(ParentCtr, ParentN, NextCtr, NextN)
        & NotSiblingInsert(StartCtr, StartN, ParentCtr, ParentN)
      ).named('NextSiblingAnc2'),
      (HasChild(ParentCtr, ParentN) <= Insert(Var("_"), Var("_"), ParentCtr, ParentN)).named(
        'HasChild'
      ),
      (
        NextElem(PrevCtr, PrevN, NextCtr, NextN) <= FirstChild(PrevCtr, PrevN, NextCtr, NextN)
      ).named('NextElem1'),
      (
        NextElem(PrevCtr, PrevN, NextCtr, NextN)
        <= NextSiblingAnc(PrevCtr, PrevN, NextCtr, NextN) & ~HasChild(PrevCtr, PrevN)
      ).named('NextElem2'),
      (Assign(ctr, n, ctr, n, n) <= Insert(ctr, n, Var("_"), Var("_"))).named('Assign'),
      (
        CurrentValue(ElemCtr, ElemN, Value)
        <= Assign(IDCtr, IDN, ElemCtr, ElemN, Value) & ~Remove(IDCtr, IDN)
      ).named('CurrentValue'),
      (HasValue(ElemCtr, ElemN) <= CurrentValue(ElemCtr, ElemN, Var("_"))).named('HasValue'),
      (
        NotHasValueNextElem(Ctr1, N1, Ctr2, N2)
        <= NextElem(Ctr1, N1, Ctr2, N2) & ~HasValue(Ctr2, N2)
      ).named('NotHasValueNextElem'),
      (SkipBlank(FromCtr, FromN, ToCtr, ToN) <= NextElem(FromCtr, FromN, ToCtr, ToN)).named(
        'SkipBlank1'
      ),
      (
        SkipBlank(FromCtr, FromN, ToCtr, ToN)
        <= SkipBlank(ViaCtr, ViaN, ToCtr, ToN) & NotHasValueNextElem(FromCtr, FromN, ViaCtr, ViaN)
      )
      .named('SkipBlank2')
      .with_inject_cpp("""
      auto& sb_new = get_relation_by_schema<SkipBlank, DELTA_VER>(db);
      auto& sb_new_idx = sb_new.get_index({{0,1,2,3}});
      auto& sb_full = get_relation_by_schema<SkipBlank, FULL_VER>(db);
      auto& sb_full_idx = sb_full.get_index({{0,1,2,3}});
      std::cout << "  SkipBlank delta: " << sb_new_idx.root().degree() 
                << ", full: " << sb_full_idx.root().degree() << std::endl;
      """),
      (
        NextVisible(PrevCtr, PrevN, NextCtr, NextN)
        <= HasValue(PrevCtr, PrevN)
        & SkipBlank(PrevCtr, PrevN, NextCtr, NextN)
        & HasValue(NextCtr, NextN)
      ).named('NextVisible1'),
      (
        NextVisible(PrevCtr, PrevN, NextCtr, NextN)
        <= HasValue(PrevCtr, PrevN)
        & SkipBlank(PrevCtr, PrevN, NextCtr, NextN)
        & HasValue(NextCtr, NextN)
      ).named('NextVisible2'),
      (
        Result(ctr1, ctr2, value)
        <= NextVisible(ctr1, Var("_"), ctr2, N2) & CurrentValue(ctr2, N2, value)
      ).named('Result'),
    ],
  )
