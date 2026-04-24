'''Smoke tests for the Python migration scaffold.

These verify the DSL and HIR types can be constructed and refer to each other.
They do NOT verify the pipeline (stratification, planning, lowering, emission) —
those tests live alongside each pass as it's ported.
'''

from srdatalog.dsl import ArgKind, Atom, Negation, Program, Relation, Var
from srdatalog.hir.types import (
  AccessPattern,
  HirProgram,
  HirRuleVariant,
  HirStratum,
  RelationDecl,
  Version,
)


def test_tc_dsl_construction():
  '''Build the tc program in the Python DSL and verify AST shape.'''
  X, Y, Z = Var("X"), Var("Y"), Var("Z")
  edge = Relation("Edge", 2)
  path = Relation("Path", 2)

  base = (path(X, Y) <= edge(X, Y)).named("TCBase")
  rec = (path(X, Z) <= path(X, Y) & edge(Y, Z)).named("TCRec")

  assert base.name == "TCBase"
  assert base.head.rel == "Path"
  assert len(base.body) == 1
  assert isinstance(base.body[0], Atom)
  assert base.body[0].rel == "Edge"

  assert rec.name == "TCRec"
  assert rec.head.rel == "Path"
  assert len(rec.body) == 2
  assert all(isinstance(b, Atom) for b in rec.body)
  assert [b.rel for b in rec.body] == ["Path", "Edge"]

  # All three variables should be LVAR
  assert all(a.kind is ArgKind.LVAR for a in rec.head.args)


def test_negation_via_invert():
  '''~atom should wrap in Negation; & should still compose.'''
  (X,) = (Var("X"),)
  r = Relation("R", 1)
  s = Relation("S", 1)
  # S(X) :- R(X), ~R(X)  -- nonsense but tests the operators
  rule = s(X) <= r(X) & ~r(X)
  assert len(rule.body) == 2
  assert isinstance(rule.body[0], Atom)
  assert isinstance(rule.body[1], Negation)
  assert rule.body[1].atom.rel == "R"


def test_program_add():
  X, Y = Var("X"), Var("Y")
  edge = Relation("Edge", 2)
  path = Relation("Path", 2)
  prog = Program().add(path(X, Y) <= edge(X, Y))
  # relations auto-derived from rule atoms: head first (Path), then body (Edge)
  assert [r.name for r in prog.relations] == ["Path", "Edge"]
  assert len(prog.rules) == 1


def test_hir_types_default_construct():
  '''HIR types should be buildable empty and composable.'''
  rd = RelationDecl(rel_name="Edge", types=["int", "int"])
  prog = HirProgram(relation_decls=[rd])
  prog.strata.append(HirStratum(scc_members={"Edge"}))
  assert prog.strata[0].scc_members == {"Edge"}
  assert prog.relation_decls[0].rel_name == "Edge"


def test_hir_variant_links_back_to_dsl_rule():
  '''HirRuleVariant holds a reference to the user-facing Rule — confirms the
  hir_types → dsl edge is wired correctly.
  '''
  X, Y = Var("X"), Var("Y")
  edge = Relation("Edge", 2)
  path = Relation("Path", 2)
  rule = (path(X, Y) <= edge(X, Y)).named("TCBase")

  variant = HirRuleVariant(
    original_rule=rule,
    var_order=["X", "Y"],
    clause_order=[0],
  )
  variant.access_patterns.append(
    AccessPattern(
      rel_name="Edge",
      version=Version.FULL,
      access_order=["X", "Y"],
      index_cols=[0, 1],
      clause_idx=0,
    )
  )
  assert variant.original_rule.name == "TCBase"
  assert variant.access_patterns[0].index_cols == [0, 1]


if __name__ == "__main__":
  test_tc_dsl_construction()
  test_negation_via_invert()
  test_program_add()
  test_hir_types_default_construct()
  test_hir_variant_links_back_to_dsl_rule()
  print("OK")
