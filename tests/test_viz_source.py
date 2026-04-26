'''Tests for srdatalog.viz.source — AST-based rule locator.'''

from __future__ import annotations

import textwrap

from srdatalog.viz.source import find_rule_locations


def test_named_rule_no_plan():
  src = textwrap.dedent(
    """\
    from srdatalog.dsl import Program, Relation, Var

    X, Y = Var("x"), Var("y")
    Edge = Relation("Edge", 2)
    Path = Relation("Path", 2)

    prog = Program(
      rules=[
        (Path(X, Y) <= Edge(X, Y)).named("TCBase"),
      ],
    )
    """
  )
  locs = find_rule_locations(src)
  assert [l.name for l in locs] == ["TCBase"]
  loc = locs[0]
  assert loc.plan_calls == []
  # The rule expression sits on line 9 (1-based); should be within that line.
  assert loc.start_line == 9
  assert src[loc.start : loc.end].startswith("(Path(X, Y) <= Edge(X, Y)).named(\"TCBase\")")


def test_with_plan_single_kwarg_exposes_value_span():
  src = textwrap.dedent(
    """\
    from srdatalog.dsl import Program

    # dummy
    r = None
    r2 = (
      (Path(X, Z) <= Path(X, Y) & Edge(Y, Z))
      .named("TCRec")
      .with_plan(var_order=["x", "y", "z"])
    )
    """
  )
  locs = find_rule_locations(src)
  assert [l.name for l in locs] == ["TCRec"]
  loc = locs[0]
  assert len(loc.plan_calls) == 1
  pc = loc.plan_calls[0]
  kwargs = {k.kwarg: src[k.start : k.end] for k in pc.kwargs}
  assert kwargs == {"var_order": '["x", "y", "z"]'}


def test_multiple_named_rules_each_located():
  src = textwrap.dedent(
    """\
    rules = [
      (p(x, y) <= e(x, y)).named("A"),
      (p(x, z) <= p(x, y) & e(y, z)).named("B").with_plan(var_order=["x", "y", "z"]),
      (q(x) <= p(x, x)).named("C"),
    ]
    """
  )
  locs = find_rule_locations(src)
  assert [l.name for l in locs] == ["A", "B", "C"]
  # B has a with_plan, A and C don't.
  assert locs[0].plan_calls == []
  assert len(locs[1].plan_calls) == 1
  assert locs[2].plan_calls == []


def test_anonymous_rules_are_skipped():
  '''A rule without .named("...") has no HIR identity and is skipped.'''
  src = "x = (p(X) <= q(X))\n"
  assert find_rule_locations(src) == []
