## Recursive split-rule fixture.
##
## RBase:  R(x, y) :- Seed(x, y)                               (base, no split)
## RRec:   R(x, z) :- R(x, y), !Filter(y), split, Edge(y, z)   (recursive, split)
##
## The recursive variant has a single SCC-member body clause (R at index 0)
## so semi-naive generates ONE delta variant with delta_idx=0. The split
## emits Pipeline A (Scan ΔR + Negation + InsertInto _temp NEW) + CreateFlatView
## + Pipeline B (Scan _temp NEW + CartesianJoin Edge + InsertInto R NEW).
## Recursive splits also emit a ClearRelation(_temp, NEW) at the start of
## each iteration.

import srdatalog/lang

srdatalog_prelude()
srdatalog_gpu_prelude()

schema RSDB:
  Seed:
    Relation[int, int] {.input: "Seed.csv".}
  Filter:
    Relation[int] {.input: "Filter.csv".}
  Edge:
    Relation[int, int] {.input: "Edge.csv".}
  R:
    Relation[int, int] {.print_size.}

rules_def RSRules:
  [{(R x y)} <-- {(Seed x y)}] {.name: RBase.}
  [{(R x z)} <-- {(R x y), (!Filter y), (split), (Edge y z)}] {.name: RRec.}
