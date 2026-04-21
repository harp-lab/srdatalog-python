'''Split-rule end-to-end tests.

Builds the split_test.nim program via the Python DSL and byte-matches
HIR + MIR against the Nim golden. Exercises detect_split, compute_temp_vars,
TempRelSynthesisPass, TempIndexRegistrationPass, lower_split_above,
lower_split_below, and the split-aware stratum wrapping.
'''

import json
from pathlib import Path

from srdatalog.dsl import SPLIT, Program, Relation, Var
from srdatalog.hir import compile_to_hir, compile_to_mir
from srdatalog.hir.emit import hir_to_obj
from srdatalog.hir.plan import compute_temp_vars, detect_split
from srdatalog.mir.emit import print_mir_sexpr

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def build_split_program() -> Program:
  '''Mirror of fixtures/split_test.nim.
  Q(x, z) :- A(x, y), ~B(y), split, C(x, z)
  '''
  X, Y, Z = Var("x"), Var("y"), Var("z")
  a = Relation("A", 2)
  b = Relation("B", 1)
  c = Relation("C", 2)
  q = Relation("Q", 2)
  rule = (q(X, Z) <= a(X, Y) & ~b(Y) & SPLIT & c(X, Z)).named("SplitTest")
  return Program(relations=[a, b, c, q], rules=[rule])


# -----------------------------------------------------------------------------
# Planner helpers
# -----------------------------------------------------------------------------


def test_detect_split_finds_marker_index():
  prog = build_split_program()
  rule = prog.rules[0]
  assert detect_split(rule) == 2


def test_detect_split_returns_minus_one_when_no_split():
  X, Y = Var("x"), Var("y")
  a = Relation("A", 2)
  q = Relation("Q", 2)
  rule = (q(X, Y) <= a(X, Y)).named("NoSplit")
  assert detect_split(rule) == -1


def test_compute_temp_vars_returns_shared_vars_only():
  '''y is only above split (in A, ~B); x is shared with C(x,z) below.'''
  prog = build_split_program()
  rule = prog.rules[0]
  assert compute_temp_vars(rule, split_at=2) == ["x"]


def test_variant_has_split_metadata_after_planning():
  hir = compile_to_hir(build_split_program())
  v = hir.strata[0].base_variants[0]
  assert v.split_at == 2
  assert v.temp_vars == ["x"]
  assert v.temp_rel_name == "_temp_SplitTest"


# -----------------------------------------------------------------------------
# Temp relation + index registration
# -----------------------------------------------------------------------------


def test_temp_rel_decl_synthesised():
  hir = compile_to_hir(build_split_program())
  temp_decls = [d for d in hir.relation_decls if d.rel_name == "_temp_SplitTest"]
  assert len(temp_decls) == 1
  td = temp_decls[0]
  assert td.types == ["int"]  # inferred from A's type at x's col
  assert td.is_generated is True
  assert td.is_temp is True


def test_temp_index_registered_in_stratum():
  hir = compile_to_hir(build_split_program())
  s = hir.strata[0]
  assert s.required_indices.get("_temp_SplitTest") == [[0]]
  assert s.canonical_index.get("_temp_SplitTest") == [0]


# -----------------------------------------------------------------------------
# End-to-end byte-match
# -----------------------------------------------------------------------------


def _canonical(obj: dict) -> str:
  return json.dumps(obj, indent=2, ensure_ascii=False)


def test_split_rule_hir_byte_match():
  hir = compile_to_hir(build_split_program())
  actual = hir_to_obj(hir)
  golden = json.loads((FIXTURES / "split_test.hir.json").read_text())
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


def test_split_rule_mir_byte_match():
  mir_prog = compile_to_mir(build_split_program())
  actual = print_mir_sexpr(mir_prog)
  golden = (FIXTURES / "split_test.mir.sexpr").read_text().rstrip("\n")
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
    test_detect_split_finds_marker_index,
    test_detect_split_returns_minus_one_when_no_split,
    test_compute_temp_vars_returns_shared_vars_only,
    test_variant_has_split_metadata_after_planning,
    test_temp_rel_decl_synthesised,
    test_temp_index_registered_in_stratum,
    test_split_rule_hir_byte_match,
    test_split_rule_mir_byte_match,
  ]
  for t in tests:
    t()
  print(f"OK ({len(tests)} tests)")
