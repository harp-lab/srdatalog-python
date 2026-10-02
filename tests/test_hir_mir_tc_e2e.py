'''End-to-end MIR lifetime checks for nonrecursive seeds and recursive strata.'''

from collections import Counter

from integration_helpers import check_mir_lifetimes

from srdatalog.dsl import Program, Relation, Var
from srdatalog.ir.hir import compile_to_mir
from srdatalog.ir.mir import types as mir


def build_tc() -> Program:
  X, Y, Z = Var("x"), Var("y"), Var("z")
  arc = Relation("ArcInput", 2)
  edge = Relation("Edge", 2)
  path = Relation("Path", 2)
  return Program(
    rules=[
      (edge(X, Y) <= arc(X, Y)).named("EdgeLoad"),
      (path(X, Y) <= edge(X, Y)).named("TCBase"),
      (path(X, Z) <= path(X, Y) & edge(Y, Z)).named("TCRec"),
    ],
  )


def test_tc_seed_and_recursive_delta_lifetimes():
  check_mir_lifetimes(build_tc())


def test_shared_heads_finalize_once_after_all_producers():
  x, y = Var("x"), Var("y")
  first, second = Relation("First", 2), Relation("Second", 2)
  combined = Relation("Combined", 2)
  program = Program(
    rules=[
      (combined(x, y) <= first(x, y)).named("FirstSeed"),
      (combined(x, y) <= second(x, y)).named("SecondSeed"),
    ]
  )
  check_mir_lifetimes(program)
  plans = [
    step for step, _ in compile_to_mir(program).steps if isinstance(step, mir.FixpointPlan)
  ]
  assert len(plans) == 1
  producers = set()
  finalized = Counter()
  for op in plans[0].instructions:
    if isinstance(op, mir.ParallelGroup):
      for pipeline in op.ops:
        if isinstance(pipeline, mir.ExecutePipeline):
          producers.add(pipeline.rule_name)
    elif isinstance(op, mir.ExecutePipeline):
      producers.add(op.rule_name)
    elif isinstance(op, mir.ComputeDeltaIndex):
      assert producers == {"FirstSeed", "SecondSeed"}
      finalized[op.rel_name] += 1
  assert finalized == {"Combined": 1}


if __name__ == "__main__":
  test_tc_seed_and_recursive_delta_lifetimes()
  test_shared_heads_finalize_once_after_all_producers()
  print("OK")
