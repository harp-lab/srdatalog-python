## Aggregate DSL fixture.
##
## Rule: Counts(c) :- cnt = count(R x y).
## Exercises AggClause emission in HIR JSON (kind: "aggregation").
##
## Note: Nim's HIR pipeline doesn't lower AggClause to moAggregate MIR
## nodes (there are no such constructions in src/srdatalog). So the MIR
## output is equivalent to "aggregate drops out" — both sides match.

import srdatalog/lang

srdatalog_prelude()
srdatalog_gpu_prelude()

schema AggDB:
  R:
    Relation[int, int] {.input: "R.csv".}
  Counts:
    Relation[int] {.print_size.}

rules_def AggRules:
  [{(Counts cnt)} <-- {(cnt = count(R x y))}] {.name: CountR.}
