"""KG `labeled` query on GPU srdatalog (same rule as the PyReason / Ascent benchmark).

    labeled(x) <- function(x), hasLabel(x, y), concept(y)

Manual join plan (no cardinality estimation, so the order is chosen by hand):

  Cardinalities on kg_small:
    function : 42895   (unary)
    concept  :    25   (unary)        <- tiny, but heavily skewed as a join key
    hasLabel : 111647  (binary)

  `concept(y)` is the target of ~111647/25 ≈ 4466 hasLabel edges *each*, so binding
  `y` first (var_order=[y, x]) would spawn ~25 tasks each fanning out ~4466 ways —
  catastrophic load imbalance on a GPU. `function(x)` is the source of only
  ~111647/42895 ≈ 2.6 edges each, so binding `x` first is uniform.

  => var_order = ['x', 'y']: scan function(x) [42895, balanced], probe hasLabel by x,
     filter concept(y) by membership. This is a semijoin (one labeled(x) per function),
     no dedup, no skew. Body order function, hasLabel, concept makes function the driver.
"""

from __future__ import annotations

from srdatalog.dsl import Program, Relation, Var

Function = Relation("Function", 1, column_types=(int,), input_file="function.csv")
Concept = Relation("Concept", 1, column_types=(int,), input_file="concept.csv")
HasLabel = Relation("HasLabel", 2, column_types=(int, int), input_file="haslabel.csv")
Labeled = Relation("Labeled", 1, column_types=(int,), print_size=True)


def build_kg_program() -> Program:
  x = Var("x")
  y = Var("y")
  return Program(
    rules=[
      (Labeled(x) <= Function(x) & HasLabel(x, y) & Concept(y))
      .named("LabelRule")
      .with_plan(var_order=["x", "y"]),
    ],
  )
