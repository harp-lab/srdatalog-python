'''Test: HIR Pass 5 (index selection) on tc and path_compose.

Verifies required_indices / canonical_index / global_index_map shape.
These fields are NOT emitted by the current Nim JSON emitter, so we
test them directly on the HirProgram object rather than via byte-diff.
'''

from srdatalog.dsl import Program, Relation, Var
from srdatalog.hir import compile_to_hir


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


def test_tc_global_index_map():
  hir = compile_to_hir(build_tc())
  # TCRec delta of Path uses accessOrder [y,x] -> index [1,0].
  # Edge and ArcInput are both accessed as [0,1].
  assert hir.global_index_map == {
    "ArcInput": [[0, 1]],
    "Edge": [[0, 1]],
    "Path": [[1, 0]],
  }


def test_tc_stratum_0_edge_required_and_canonical():
  hir = compile_to_hir(build_tc())
  s = hir.strata[0]
  # Edge is the head of EdgeLoad; no local patterns for Edge (ArcInput is
  # the only accessed body). Falls through to globalIndexMap -> [0,1].
  assert s.required_indices == {"Edge": [[0, 1]]}
  assert s.canonical_index == {"Edge": [0, 1]}


def test_tc_stratum_1_path_base_uses_global_path_index():
  hir = compile_to_hir(build_tc())
  s = hir.strata[1]
  # Path is the head; no local patterns for Path here. Must fall through
  # to globalIndexMap so the produced Path uses the [1,0] index that
  # stratum 2's DELTA consumer expects.
  assert s.required_indices == {"Path": [[1, 0]]}
  assert s.canonical_index == {"Path": [1, 0]}


def test_tc_stratum_2_recursive_path_delta_index():
  hir = compile_to_hir(build_tc())
  s = hir.strata[2]
  assert s.required_indices == {"Path": [[1, 0]]}
  assert s.canonical_index == {"Path": [1, 0]}


def test_path_compose_delta_union_of_indices():
  '''PathCompose has TWO delta variants: delta_idx=0 yields Path[1,0] access,
  delta_idx=1 yields Path[0,1] access. Recursive-stratum required_indices
  for Path should union both (sorted by tuple order).
  '''
  X, Y, Z = Var("x"), Var("y"), Var("z")
  seed = Relation("Seed", 2)
  path = Relation("Path", 2)
  prog = Program(
    rules=[
      (path(X, Y) <= seed(X, Y)).named("PathSeed"),
      (path(X, Z) <= path(X, Y) & path(Y, Z)).named("PathCompose"),
    ],
  )
  hir = compile_to_hir(prog)
  rec = hir.strata[1]
  # Both [0,1] and [1,0] accessed as DELTA across the two variants.
  # Sorted insertion: (0,1) then (1,0).
  assert rec.required_indices == {"Path": [[0, 1], [1, 0]]}
  # Canonical prefers a full-arity idx whose FULL version is used.
  # FULL access patterns across the two variants: Path[0,1] (variant 0)
  # and Path[1,0] (variant 1). Either qualifies; the first-appearing in
  # `required_indices` that is in full_indices wins: [0,1].
  assert rec.canonical_index["Path"] == [0, 1]


if __name__ == "__main__":
  tests = [
    test_tc_global_index_map,
    test_tc_stratum_0_edge_required_and_canonical,
    test_tc_stratum_1_path_base_uses_global_path_index,
    test_tc_stratum_2_recursive_path_delta_index,
    test_path_compose_delta_union_of_indices,
  ]
  for t in tests:
    t()
  print(f"OK ({len(tests)} tests)")
