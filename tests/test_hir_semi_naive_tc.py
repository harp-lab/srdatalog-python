'''Test: semi-naive variant generation (HIR Pass 3).

Verifies the base/delta variant layout on tc and on rules with multiple
SCC-member body clauses.
'''

from srdatalog.dsl import Program, Relation, Var
from srdatalog.hir import compile_to_hir
from srdatalog.hir.types import Version


def build_tc_program() -> Program:
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


def test_tc_stratum_0_one_base_variant():
  hir = compile_to_hir(build_tc_program())
  s = hir.strata[0]
  assert len(s.base_variants) == 1
  assert len(s.recursive_variants) == 0
  v = s.base_variants[0]
  assert v.original_rule.name == "EdgeLoad"
  assert v.delta_idx == -1
  assert v.clause_versions == [Version.FULL]


def test_tc_stratum_1_path_base_one_base_variant():
  hir = compile_to_hir(build_tc_program())
  s = hir.strata[1]
  assert len(s.base_variants) == 1
  assert len(s.recursive_variants) == 0
  v = s.base_variants[0]
  assert v.original_rule.name == "TCBase"
  assert v.delta_idx == -1


def test_tc_stratum_2_recursive_one_delta_variant():
  '''TCRec body = [Path, Edge]; only Path is in SCC {Path}, so exactly
  one delta variant with delta_idx=0, clause_versions=[DELTA, FULL].
  '''
  hir = compile_to_hir(build_tc_program())
  s = hir.strata[2]
  assert len(s.base_variants) == 0
  assert len(s.recursive_variants) == 1
  v = s.recursive_variants[0]
  assert v.original_rule.name == "TCRec"
  assert v.delta_idx == 0
  assert v.clause_versions == [Version.DELTA, Version.FULL]


def test_rule_with_two_scc_clauses_gets_two_delta_variants():
  '''Path(x,z) :- Path(x,y), Path(y,z) -- both body clauses in SCC {Path}.
  Should produce TWO delta variants: one with delta_idx=0 (Δ,F),
  one with delta_idx=1 (F,Δ).
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
  # 2 strata: base {Path} with [PathSeed], recursive {Path} with PathCompose variants
  assert len(hir.strata) == 2
  rec = hir.strata[1]
  assert len(rec.recursive_variants) == 2
  d0, d1 = rec.recursive_variants
  assert d0.delta_idx == 0
  assert d0.clause_versions == [Version.DELTA, Version.FULL]
  assert d1.delta_idx == 1
  assert d1.clause_versions == [Version.FULL, Version.DELTA]
  # Both variants share the same source rule
  assert d0.original_rule is d1.original_rule


def test_negation_body_clause_is_not_delta_candidate():
  '''R(x) :- R(x), ~S(x)  -- body clauses [R (in SCC), ~S (negation)].
  Only the positive R should produce a delta variant. The negation is
  never incrementalized.
  '''
  X = Var("x")
  seed = Relation("Seed", 1)
  s = Relation("S", 1)
  r = Relation("R", 1)
  prog = Program(
    rules=[
      (r(X) <= seed(X)).named("RSeed"),
      (s(X) <= seed(X)).named("SSeed"),
      (r(X) <= r(X) & ~s(X)).named("RSelf"),
    ],
  )
  hir = compile_to_hir(prog)
  # Find the recursive {R} stratum (S is in its own non-rec stratum)
  rec_stratum = next(st for st in hir.strata if st.is_recursive)
  assert rec_stratum.scc_members == {"R"}
  assert len(rec_stratum.recursive_variants) == 1
  v = rec_stratum.recursive_variants[0]
  assert v.delta_idx == 0
  assert v.clause_versions == [Version.DELTA, Version.FULL]


def test_mutual_recursion_variant_layout():
  X = Var("x")
  seed = Relation("Seed", 1)
  a = Relation("A", 1)
  b = Relation("B", 1)
  prog = Program(
    rules=[
      (a(X) <= seed(X)).named("ASeed"),
      (a(X) <= b(X)).named("AFromB"),
      (b(X) <= a(X)).named("BFromA"),
    ],
  )
  hir = compile_to_hir(prog)
  # Stratum 0: base stratum of {A,B} with ASeed (non-recursive body)
  # Stratum 1: recursive stratum with AFromB and BFromA, each one delta variant
  base, rec = hir.strata
  assert len(base.base_variants) == 1
  assert base.base_variants[0].original_rule.name == "ASeed"
  assert len(rec.recursive_variants) == 2
  assert [v.original_rule.name for v in rec.recursive_variants] == ["AFromB", "BFromA"]
  assert all(v.delta_idx == 0 for v in rec.recursive_variants)


if __name__ == "__main__":
  tests = [
    test_tc_stratum_0_one_base_variant,
    test_tc_stratum_1_path_base_one_base_variant,
    test_tc_stratum_2_recursive_one_delta_variant,
    test_rule_with_two_scc_clauses_gets_two_delta_variants,
    test_negation_body_clause_is_not_delta_candidate,
    test_mutual_recursion_variant_layout,
  ]
  for t in tests:
    t()
  print(f"OK ({len(tests)} tests)")
