## Exercises body-constant + head-constant rewriting.
##
## Body rule: P(x) :- In(x, 42, 99)
##   -> body has two Const args -> rewrite to P(x) :- In(x, _c0, _c1), Filter(_c0==42 && _c1==99)
## Head rule: P(x, 7) :- In2(x)
##   -> head has one Const -> rewrite to P(x, _hc0) :- In2(x), Let _hc0 = 7

import srdatalog/lang

srdatalog_prelude()
srdatalog_gpu_prelude()

schema CDB:
  In:
    Relation[int, int, int] {.input: "in.csv".}
  In2:
    Relation[int] {.input: "in2.csv".}
  P:
    Relation[int] {.print_size.}
  Q:
    Relation[int, int] {.print_size.}

rules_def CRules:
  [{(P x)} <-- {(In x 42 99)}] {.name: BodyConst.}
  [{(Q x 7)} <-- {(In2 x)}] {.name: HeadConst.}
