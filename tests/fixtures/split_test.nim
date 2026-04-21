## Split-rule fixture (negation pushdown pattern).
##
## Rule: Q(x, z) :- A(x, y), !B(y), ---, C(x, z)
##   above split: A(x, y), !B(y)     → temp(x)   (y only above)
##   below split: C(x, z)            uses temp(x) to probe C
##
## tempVars = [x]  (bound above, used below body + head)

import srdatalog/lang

srdatalog_prelude()
srdatalog_gpu_prelude()

schema SplitDB:
  A:
    Relation[int, int] {.input: "A.csv".}
  B:
    Relation[int] {.input: "B.csv".}
  C:
    Relation[int, int] {.input: "C.csv".}
  Q:
    Relation[int, int] {.print_size.}

rules_def SplitRules:
  [{(Q x z)} <-- {(A x y), (!B y), (split), (C x z)}] {.name: SplitTest.}
