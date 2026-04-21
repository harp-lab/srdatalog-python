'''Test: stratification on tc/tc.nim produces the 3-strata split matching
the Nim golden (Edge | Path-base | Path-recursive).

This test does NOT use the full HIR emitter diff because variant population
(access patterns, var_order, etc.) is a later pass. It verifies the stratum
structure directly.
'''
import sys
from pathlib import Path


from srdatalog.dsl import Var, Relation, Program
from srdatalog.hir import compile_to_hir
from srdatalog.hir_stratify import stratify


def build_tc_program() -> Program:
  X, Y, Z = Var("x"), Var("y"), Var("z")
  arc = Relation("ArcInput", 2)
  edge = Relation("Edge", 2)
  path = Relation("Path", 2)
  return Program(
    relations=[arc, edge, path],
    rules=[
      (edge(X, Y) <= arc(X, Y)).named("EdgeLoad"),
      (path(X, Y) <= edge(X, Y)).named("TCBase"),
      (path(X, Z) <= path(X, Y) & edge(Y, Z)).named("TCRec"),
    ],
  )


def test_stratify_tc_produces_3_strata():
  hir = compile_to_hir(build_tc_program())
  assert len(hir.strata) == 3, f"expected 3 strata, got {len(hir.strata)}"


def test_stratify_tc_stratum_0_edge_simple():
  hir = compile_to_hir(build_tc_program())
  s = hir.strata[0]
  assert s.scc_members == {"Edge"}
  assert s.is_recursive is False
  assert [r.name for r in s.stratum_rules] == ["EdgeLoad"]


def test_stratify_tc_stratum_1_path_base():
  hir = compile_to_hir(build_tc_program())
  s = hir.strata[1]
  assert s.scc_members == {"Path"}
  assert s.is_recursive is False
  assert [r.name for r in s.stratum_rules] == ["TCBase"]


def test_stratify_tc_stratum_2_path_recursive():
  hir = compile_to_hir(build_tc_program())
  s = hir.strata[2]
  assert s.scc_members == {"Path"}
  assert s.is_recursive is True
  assert [r.name for r in s.stratum_rules] == ["TCRec"]


def test_stratify_tc_relation_decls_in_source_order():
  hir = compile_to_hir(build_tc_program())
  names = [d.rel_name for d in hir.relation_decls]
  assert names == ["ArcInput", "Edge", "Path"]


def test_stratify_idempotent_across_runs():
  '''Same program, two stratify calls, identical stratum structure.
  Guards against set-iteration order bleeding into output.
  '''
  h1 = compile_to_hir(build_tc_program())
  h2 = compile_to_hir(build_tc_program())
  shape = lambda h: [
    (sorted(s.scc_members), s.is_recursive, [r.name for r in s.stratum_rules])
    for s in h.strata
  ]
  assert shape(h1) == shape(h2)


def test_stratify_mutual_recursion_two_relations():
  '''A(x) :- B(x); B(x) :- A(x); A(x) :- Seed(x)
  Expects: base stratum {A,B} with [ASeed], recursive stratum {A,B} with the cycle rules.
  Stresses multi-member SCC discovery and base/recursive split for a 2-node SCC.
  '''
  X = Var("x")
  seed = Relation("Seed", 1)
  a = Relation("A", 1)
  b = Relation("B", 1)
  prog = Program(
    relations=[seed, a, b],
    rules=[
      (a(X) <= seed(X)).named("ASeed"),
      (a(X) <= b(X)).named("AFromB"),
      (b(X) <= a(X)).named("BFromA"),
    ],
  )
  hir = compile_to_hir(prog)
  assert len(hir.strata) == 2

  s_base, s_rec = hir.strata
  assert s_base.scc_members == {"A", "B"}
  assert s_base.is_recursive is False
  assert [r.name for r in s_base.stratum_rules] == ["ASeed"]

  assert s_rec.scc_members == {"A", "B"}
  assert s_rec.is_recursive is True
  assert [r.name for r in s_rec.stratum_rules] == ["AFromB", "BFromA"]


def test_stratify_empty_program():
  hir = compile_to_hir(Program(relations=[], rules=[]))
  assert hir.strata == []
  assert hir.relation_decls == []


def test_stratify_bare_function_takes_rules_and_decls():
  '''The bare stratify() function takes (rules, decls) directly, matching
  the contract the Pipeline uses internally and that other HIR passes will
  plug into after rule-rewrite passes run.
  '''
  from srdatalog.hir_pass import program_to_decls
  prog = build_tc_program()
  rules = list(prog.rules)
  decls = program_to_decls(prog)
  hir = stratify(rules, decls)
  assert len(hir.strata) == 3


if __name__ == "__main__":
  tests = [
    test_stratify_tc_produces_3_strata,
    test_stratify_tc_stratum_0_edge_simple,
    test_stratify_tc_stratum_1_path_base,
    test_stratify_tc_stratum_2_path_recursive,
    test_stratify_tc_relation_decls_in_source_order,
    test_stratify_idempotent_across_runs,
    test_stratify_mutual_recursion_two_relations,
    test_stratify_empty_program,
    test_stratify_bare_function_takes_rules_and_decls,
  ]
  for t in tests:
    t()
  print(f"OK ({len(tests)} tests)")
