'''Tests for srdatalog.viz.bundle.get_visualization_bundle.'''

from __future__ import annotations

from srdatalog.dsl import Program, Relation, Var
from srdatalog.viz.bundle import get_visualization_bundle


def _tc_program() -> Program:
  X, Y, Z = Var("x"), Var("y"), Var("z")
  edge = Relation("Edge", 2)
  path = Relation("Path", 2)
  return Program(
    rules=[
      (path(X, Y) <= edge(X, Y)).named("TCBase"),
      (path(X, Z) <= path(X, Y) & edge(Y, Z)).named("TCRec").with_plan(var_order=["x", "y", "z"]),
    ]
  )


def test_bundle_contains_hir_mir_jit_and_rules():
  b = get_visualization_bundle(_tc_program(), project_name="TCViz")
  assert b["project_name"] == "TCViz"
  # HIR JSON: must have strata + relations keys the TS renderer expects.
  assert "strata" in b["hir"]
  assert "relations" in b["hir"]
  # MIR S-expr always starts with (program.
  assert b["mir"].startswith("(program")
  # Per-runner JIT code keyed by runner name. Recursive rules get a
  # `_D<n>` suffix per delta variant — matches how the Nim extension
  # sees them and how jit_batch files are named.
  assert "TCBase" in b["jit"]
  assert "TCRec_D0" in b["jit"]
  assert "JitRunner_TCBase" in b["jit"]["TCBase"]
  # Rules summary.
  names = [r["name"] for r in b["rules"]]
  assert names == ["TCBase", "TCRec"]
  # TCRec carries a plan entry echoing var_order.
  rec = next(r for r in b["rules"] if r["name"] == "TCRec")
  assert rec["plans"] == [{"delta": -1, "var_order": ["x", "y", "z"], "clause_order": []}]


def test_bundle_relations_match_rule_first_occurrence_order():
  b = get_visualization_bundle(_tc_program(), project_name="TCViz")
  # TC rule order: head Path, body Edge, ... second rule adds nothing new.
  # So bundle relations = [Path, Edge].
  assert b["relations"] == ["Path", "Edge"]
