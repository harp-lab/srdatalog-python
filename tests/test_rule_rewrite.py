'''Rule-rewrite pass tests: body-constant rewriting (Pass 0) and head-
constant rewriting (Pass 1).

Both e2e-byte-diff against python/tests/fixtures/rewrite_consts.* and
unit-test the pass functions directly.
'''
import json
import sys
from pathlib import Path


from srdatalog.dsl import Var, Relation, Program, Atom, Filter, Let, ClauseArg, ArgKind
from srdatalog.hir.types import RelationDecl
from srdatalog.hir import compile_to_hir, compile_to_mir
from srdatalog.hir.emit import hir_to_obj
from srdatalog.mir.emit import print_mir_sexpr
from srdatalog.rule_rewrite import rewrite_constants, rewrite_head_constants


FIXTURES = Path(__file__).resolve().parent / "fixtures"


# -----------------------------------------------------------------------------
# Unit tests on the bare pass functions
# -----------------------------------------------------------------------------

def test_rewrite_constants_replaces_body_const_and_inserts_filter():
  X = Var("x")
  inp = Relation("In", 3)
  p = Relation("P", 1)
  # P(x) :- In(x, 42, 99)  -> two const args in body
  rule = (p(X) <= inp(X, 42, 99)).named("R")
  new_rules, new_decls = rewrite_constants([rule], [])
  assert len(new_rules) == 1
  r = new_rules[0]
  # Body should be [Atom(In), Filter(...)] (filter immediately after atom).
  assert len(r.body) == 2
  a, f = r.body
  assert isinstance(a, Atom) and a.rel == "In"
  assert [arg.var_name for arg in a.args if arg.kind is ArgKind.LVAR] == ["x", "_c0", "_c1"]
  assert isinstance(f, Filter)
  assert f.vars == ("_c0", "_c1")
  assert f.code == "return _c0 == 42 && _c1 == 99;"


def test_rewrite_constants_is_noop_when_no_body_consts():
  X, Y = Var("x"), Var("y")
  a = Relation("A", 2)
  b = Relation("B", 2)
  rule = (a(X, Y) <= b(X, Y)).named("R")
  new_rules, _ = rewrite_constants([rule], [])
  # Rule is unchanged (same body tuple structure).
  assert new_rules[0].body == rule.body


def test_rewrite_head_constants_replaces_head_const_and_appends_let():
  X = Var("x")
  in2 = Relation("In2", 1)
  q = Relation("Q", 2)
  # Q(x, 7) :- In2(x)  -> head has one const
  rule = (q(X, 7) <= in2(X)).named("R")
  new_rules, _ = rewrite_head_constants([rule], [])
  r = new_rules[0]
  head_var_names = [arg.var_name for arg in r.head.args if arg.kind is ArgKind.LVAR]
  assert head_var_names == ["x", "_hc0"]
  # Body should end with a Let binding _hc0 to "7".
  last = r.body[-1]
  assert isinstance(last, Let)
  assert last.var_name == "_hc0"
  assert last.code == "7"


def test_rewrite_head_constants_is_noop_when_no_head_consts():
  X = Var("x")
  a = Relation("A", 1)
  b = Relation("B", 1)
  rule = (a(X) <= b(X)).named("R")
  out_rules, _ = rewrite_head_constants([rule], [])
  # Rule object passes through unchanged when needs_rewrite is False.
  assert out_rules[0] is rule


def test_counter_resets_per_call():
  '''Each call starts counter at 0.'''
  X = Var("x")
  inp = Relation("In", 2)
  p = Relation("P", 1)
  rule = (p(X) <= inp(X, 1)).named("R")
  r1, _ = rewrite_constants([rule], [])
  r2, _ = rewrite_constants([rule], [])
  # Both should use _c0 for the rewritten constant.
  assert r1[0].body[0].args[1].var_name == "_c0"
  assert r2[0].body[0].args[1].var_name == "_c0"


# -----------------------------------------------------------------------------
# End-to-end byte-match against the Nim fixture
# -----------------------------------------------------------------------------

def build_rewrite_consts_program() -> Program:
  X = Var("x")
  inp = Relation("In", 3)
  in2 = Relation("In2", 1)
  p = Relation("P", 1)
  q = Relation("Q", 2)
  return Program(
    relations=[inp, in2, p, q],
    rules=[
      (p(X) <= inp(X, 42, 99)).named("BodyConst"),
      (q(X, 7) <= in2(X)).named("HeadConst"),
    ],
  )


def _canonical(obj: dict) -> str:
  return json.dumps(obj, indent=2, ensure_ascii=False)


def test_rewrite_consts_hir_byte_match():
  hir = compile_to_hir(build_rewrite_consts_program())
  actual = hir_to_obj(hir)
  golden = json.loads((FIXTURES / "rewrite_consts.hir.json").read_text())
  golden.pop("hirSExpr", None)
  if _canonical(actual) != _canonical(golden):
    import difflib
    diff = "\n".join(
      difflib.unified_diff(
        _canonical(golden).splitlines(),
        _canonical(actual).splitlines(),
        fromfile="nim-golden",
        tofile="python",
        lineterm="",
      )
    )
    raise AssertionError("HIR mismatch:\n" + diff)


def test_rewrite_consts_mir_byte_match():
  mir_prog = compile_to_mir(build_rewrite_consts_program())
  actual = print_mir_sexpr(mir_prog)
  golden = (FIXTURES / "rewrite_consts.mir.sexpr").read_text().rstrip("\n")
  if actual != golden:
    import difflib
    diff = "\n".join(
      difflib.unified_diff(
        golden.splitlines(),
        actual.splitlines(),
        fromfile="nim-golden",
        tofile="python",
        lineterm="",
      )
    )
    raise AssertionError("MIR mismatch:\n" + diff)


if __name__ == "__main__":
  tests = [
    test_rewrite_constants_replaces_body_const_and_inserts_filter,
    test_rewrite_constants_is_noop_when_no_body_consts,
    test_rewrite_head_constants_replaces_head_const_and_appends_let,
    test_rewrite_head_constants_is_noop_when_no_head_consts,
    test_counter_resets_per_call,
    test_rewrite_consts_hir_byte_match,
    test_rewrite_consts_mir_byte_match,
  ]
  for t in tests:
    t()
  print(f"OK ({len(tests)} tests)")
