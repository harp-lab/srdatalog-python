## PathCompose — stress test for the Python planner.
##
## Differs from tc: the recursive rule has TWO SCC-member body clauses, so
## semi-naive generates TWO delta variants (one per body clause). Each
## variant exercises the clause-ordering + var-ordering heuristics with
## different delta positions.
##
## This file is processed by srdatalog_plan.nim to produce the golden
## HIR JSON; it is not built or run.

import srdatalog/lang

srdatalog_prelude()
srdatalog_gpu_prelude()

schema PathDB:
  Seed:
    Relation[int, int] {.input: "seed.csv".}
  Path:
    Relation[int, int] {.print_size.}

rules_def PathRules:
  [{(Path x y)} <-- {(Seed x y)}] {.name: PathSeed.}
  [{(Path x z)} <-- {(Path x y), (Path y z)}] {.name: PathCompose.}
