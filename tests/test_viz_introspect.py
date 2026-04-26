'''Tests for srdatalog.viz.introspect.load_program.'''

from __future__ import annotations

import textwrap

import pytest

from srdatalog.viz.introspect import ProgramDiscoveryError, load_program


def test_auto_discovers_build_fn(tmp_path):
  (tmp_path / "triangle.py").write_text(
    textwrap.dedent(
      """\
      from srdatalog.dsl import Program, Relation, Var


      def build_triangle_program():
        X, Y = Var("x"), Var("y")
        edge = Relation("Edge", 2)
        path = Relation("Path", 2)
        return Program(rules=[(path(X, Y) <= edge(X, Y)).named("Base")])
      """
    )
  )
  prog = load_program(tmp_path / "triangle.py")
  assert len(prog.rules) == 1
  assert prog.rules[0].name == "Base"


def test_explicit_entry(tmp_path):
  (tmp_path / "two_builders.py").write_text(
    textwrap.dedent(
      """\
      from srdatalog.dsl import Program, Relation, Var


      def build_a_program():
        X = Var("x")
        a = Relation("A", 1)
        return Program(rules=[(a(X) <= a(X)).named("RuleA")])


      def build_b_program():
        X = Var("x")
        b = Relation("B", 1)
        return Program(rules=[(b(X) <= b(X)).named("RuleB")])
      """
    )
  )
  prog_a = load_program(tmp_path / "two_builders.py", entry="build_a_program")
  prog_b = load_program(tmp_path / "two_builders.py", entry="build_b_program")
  assert prog_a.rules[0].name == "RuleA"
  assert prog_b.rules[0].name == "RuleB"


def test_top_level_program_instance(tmp_path):
  (tmp_path / "inline.py").write_text(
    textwrap.dedent(
      """\
      from srdatalog.dsl import Program, Relation, Var

      X = Var("x")
      edge = Relation("Edge", 2)
      path = Relation("Path", 2)
      program = Program(rules=[(path(X, X) <= edge(X, X)).named("Self")])
      """
    )
  )
  prog = load_program(tmp_path / "inline.py")
  assert prog.rules[0].name == "Self"


def test_missing_entry_raises(tmp_path):
  (tmp_path / "empty.py").write_text("x = 1\n")
  with pytest.raises(ProgramDiscoveryError):
    load_program(tmp_path / "empty.py")


def test_meta_is_passed_to_builder_requiring_arg(tmp_path):
  (tmp_path / "with_meta.py").write_text(
    textwrap.dedent(
      """\
      from srdatalog.dsl import Program, Relation, Var, Const


      def build_cm_program(meta):
        X = Var("x")
        r = Relation("R", 1)
        C = Const(meta["k"])
        return Program(rules=[(r(X) <= r(C)).named("UseConst")])
      """
    )
  )
  prog = load_program(tmp_path / "with_meta.py", meta={"k": 42})
  assert prog.rules[0].name == "UseConst"


def test_missing_meta_when_required_raises_helpfully(tmp_path):
  (tmp_path / "needs_meta.py").write_text(
    textwrap.dedent(
      """\
      from srdatalog.dsl import Program, Relation, Var


      def build_needs_meta_program(meta):
        X = Var("x")
        r = Relation("R", 1)
        return Program(rules=[(r(X) <= r(X)).named("X")])
      """
    )
  )
  with pytest.raises(ProgramDiscoveryError, match="pass --meta"):
    load_program(tmp_path / "needs_meta.py")
