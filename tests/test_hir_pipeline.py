'''Pipeline-level tests for the HIR orchestrator.

Verifies that the Pipeline class registers passes correctly, runs them in
order, and rejects misregistered passes (wrong PassLevel).
'''
import sys
from pathlib import Path


from srdatalog.dsl import Var, Relation, Program, Rule
from srdatalog.hir.types import HirProgram, RelationDecl
from srdatalog.hir.pass_ import (
  Pipeline,
  PassInfo,
  PassLevel,
  Dialect,
  RuleRewritePass,
  HirTransformPass,
)
from srdatalog.hir import compile_to_hir


def _tc() -> Program:
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


class _RecordingRewrite:
  '''Rule-rewrite that appends a tag to the trace list for ordering assertions.'''
  def __init__(self, tag: str, order: int, trace: list[str]):
    self.info = PassInfo(
      name=f"Record({tag})",
      level=PassLevel.RULE_REWRITE,
      order=order,
      source_dialect=Dialect.HIR,
      target_dialect=Dialect.HIR,
    )
    self._tag = tag
    self._trace = trace

  def run(
    self, rules: list[Rule], decls: list[RelationDecl]
  ) -> tuple[list[Rule], list[RelationDecl]]:
    self._trace.append(self._tag)
    return rules, decls


class _RecordingHirTransform:
  def __init__(self, tag: str, order: int, trace: list[str]):
    self.info = PassInfo(
      name=f"Record({tag})",
      level=PassLevel.HIR_TRANSFORM,
      order=order,
      source_dialect=Dialect.HIR,
      target_dialect=Dialect.HIR,
    )
    self._tag = tag
    self._trace = trace

  def run(self, hir: HirProgram) -> HirProgram:
    self._trace.append(self._tag)
    return hir


def test_default_pipeline_produces_tc_3_strata():
  '''The default (empty-user-passes) pipeline == bare stratify on tc.'''
  hir = compile_to_hir(_tc())
  assert len(hir.strata) == 3


def test_rule_rewrites_run_before_stratify_in_order():
  trace: list[str] = []
  p = Pipeline()
  p.add_rule_rewrite(_RecordingRewrite("B", order=10, trace=trace))
  p.add_rule_rewrite(_RecordingRewrite("A", order=1, trace=trace))
  p.add_rule_rewrite(_RecordingRewrite("C", order=5, trace=trace))
  p.compile_to_hir(_tc())
  # Lower order first, regardless of add() order.
  assert trace == ["A", "C", "B"]


def test_hir_transforms_run_after_stratify_in_order():
  trace: list[str] = []
  p = Pipeline()
  p.add_hir_transform(_RecordingHirTransform("Y", order=2, trace=trace))
  p.add_hir_transform(_RecordingHirTransform("X", order=1, trace=trace))
  hir = p.compile_to_hir(_tc())
  assert trace == ["X", "Y"]
  # Stratification still happens (3 strata) — transforms ran after it.
  assert len(hir.strata) == 3


def test_wrong_level_registration_rejected():
  p = Pipeline()
  hir_pass = _RecordingHirTransform("bad", order=0, trace=[])
  try:
    p.add_rule_rewrite(hir_pass)  # type: ignore[arg-type]
  except ValueError as e:
    assert "level is" in str(e)
  else:
    raise AssertionError("expected ValueError for wrong-level registration")


def test_pipeline_preserves_rule_and_decl_identity_when_empty():
  '''Empty pipeline should be equivalent to stratify(rules, decls) directly.'''
  from srdatalog.hir.stratify import stratify
  from srdatalog.hir.pass_ import program_to_decls
  prog = _tc()
  bare = stratify(list(prog.rules), program_to_decls(prog))
  piped = compile_to_hir(prog)
  # Compare structural shape
  shape = lambda h: [
    (sorted(s.scc_members), s.is_recursive, [r.name for r in s.stratum_rules])
    for s in h.strata
  ]
  assert shape(bare) == shape(piped)


if __name__ == "__main__":
  tests = [
    test_default_pipeline_produces_tc_3_strata,
    test_rule_rewrites_run_before_stratify_in_order,
    test_hir_transforms_run_after_stratify_in_order,
    test_wrong_level_registration_rejected,
    test_pipeline_preserves_rule_and_decl_identity_when_empty,
  ]
  for t in tests:
    t()
  print(f"OK ({len(tests)} tests)")
