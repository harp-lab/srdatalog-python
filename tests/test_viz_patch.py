'''Tests for srdatalog.viz.patch — plan kwarg rewriting.'''

from __future__ import annotations

import textwrap

import pytest

from srdatalog.viz.patch import PlanPatchError, patch_rule_plan


def test_replace_existing_var_order():
  src = textwrap.dedent(
    """\
    rules = [
      (p(x, z) <= p(x, y) & e(y, z)).named("TCRec").with_plan(var_order=["a", "b"]),
    ]
    """
  )
  out = patch_rule_plan(src, "TCRec", var_order=["x", "y", "z"])
  assert 'var_order=["x", "y", "z"]' in out
  assert 'var_order=["a", "b"]' not in out
  # Should leave surrounding text untouched.
  assert out.startswith("rules = [")


def test_add_clause_order_to_existing_with_plan():
  src = textwrap.dedent(
    """\
    r = (p(x, z) <= p(x, y) & e(y, z)).named("TCRec").with_plan(var_order=["x", "y", "z"])
    """
  )
  out = patch_rule_plan(src, "TCRec", clause_order=[1, 0])
  assert 'var_order=["x", "y", "z"]' in out
  assert "clause_order=[1, 0]" in out


def test_append_with_plan_when_missing():
  src = textwrap.dedent(
    """\
    rule = (p(X, Y) <= e(X, Y)).named("Base")
    """
  )
  out = patch_rule_plan(src, "Base", var_order=["x", "y"])
  assert '.named("Base").with_plan(var_order=["x", "y"])' in out


def test_both_var_and_clause_order_together():
  src = textwrap.dedent(
    """\
    r = (p(x, z) <= p(x, y) & e(y, z)).named("TCRec")
    """
  )
  out = patch_rule_plan(src, "TCRec", var_order=["x", "y", "z"], clause_order=[1, 0])
  assert 'var_order=["x", "y", "z"]' in out
  assert "clause_order=[1, 0]" in out


def test_rule_not_found_raises():
  src = 'r = (p(X) <= q(X)).named("A")\n'
  with pytest.raises(PlanPatchError) as excinfo:
    patch_rule_plan(src, "B", var_order=["x"])
  assert "Rules seen: A" in str(excinfo.value)


def test_no_edits_raises():
  src = 'r = (p(X) <= q(X)).named("A")\n'
  with pytest.raises(PlanPatchError):
    patch_rule_plan(src, "A")


def test_patch_preserves_surrounding_kwargs():
  '''A delta= kwarg already on the call should survive var_order rewrite.'''
  src = textwrap.dedent(
    """\
    r = (p(x, z) <= a(x, y) & b(y, z)).named("TCRec").with_plan(delta=0, var_order=["a"])
    """
  )
  out = patch_rule_plan(src, "TCRec", var_order=["x", "y", "z"], delta=0)
  assert "delta=0" in out
  assert 'var_order=["x", "y", "z"]' in out
