## Semi-join optimization fixture.
##
## Main(x, z) :- R(x, y, z), S(y), T(x, z)  with {.semi_join: true.}
##   R's target vars {x, y, z}; S's filter vars {y} ⊂ target.
##   y is NOT used elsewhere (head has {x, z}, T has {x, z}).
##   -> generate _SJ_R_S_0_2(x, z) := R(v0, v1, v2), S(v1)
##   -> rewrite to Main(x, z) :- _SJ_R_S_0_2(x, z), T(x, z)

import srdatalog/lang

srdatalog_prelude()
srdatalog_gpu_prelude()

schema SJDB:
  R:
    Relation[int, int, int] {.input: "R.csv".}
  S:
    Relation[int] {.input: "S.csv".}
  T:
    Relation[int, int] {.input: "T.csv".}
  Main:
    Relation[int, int] {.print_size.}

rules_def SJRules:
  [{(Main x z)} <-- {(R x y z), (S y), (T x z)}] {.name: SJTest, semi_join: true.}
