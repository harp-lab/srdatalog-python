## User-plan test program — exercises plan pragmas.
## - Base rule uses explicit var_order = [y, x]; planner should override
##   the default [x, y].
## - Recursive rule uses explicit delta+clause_order+var_order; planner
##   should consume the plan verbatim.

import srdatalog/lang

srdatalog_prelude()
srdatalog_gpu_prelude()

schema PDB:
  Seed:
    Relation[int, int] {.input: "seed.csv".}
  P:
    Relation[int, int] {.print_size.}

rules_def PRules:
  # Base: custom var_order only (clause_order derived).
  [{(P x y)} <-- {(Seed x y)}] {.name: Base, var_order: [y, x].}

  # Recursive: full plan — specific delta + clause_order + var_order that
  # differs from the default heuristic (forcing x first rather than y).
  [{(P x z)} <-- {(P x y), (P y z)}] {.
    name: Compose,
    plan: [
      (delta: 0, clause_order: [0, 1], var_order: [x, y, z]),
      (delta: 1, clause_order: [1, 0], var_order: [z, y, x]),
    ]
  .}
