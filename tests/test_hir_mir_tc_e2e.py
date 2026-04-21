'''End-to-end MIR byte-diff test on tc.

Builds tc as a DSL Program, runs the full Python pipeline (HIR + MIR
lowering), emits canonical MIR S-expr, and asserts byte-identical to
python/tests/fixtures/tc.mir.sexpr produced by srdatalog_plan.nim (with
its `emitMir` branch -- pre-MIR-pass output, matching what
lower_hir_to_mir produces in Python).

This closes the loop for tc: DSL -> HIR -> MIR, both byte-verified.
'''

from pathlib import Path

from srdatalog.dsl import Program, Relation, Var
from srdatalog.hir import compile_to_mir
from srdatalog.mir.emit import print_mir_sexpr

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def build_tc() -> Program:
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


def test_tc_mir_sexpr_matches_golden():
  mir_prog = compile_to_mir(build_tc())
  actual = print_mir_sexpr(mir_prog)

  golden = (FIXTURES / "tc.mir.sexpr").read_text().rstrip("\n")

  if actual != golden:
    import difflib

    diff = "\n".join(
      difflib.unified_diff(
        golden.splitlines(),
        actual.splitlines(),
        fromfile="nim-golden",
        tofile="python-mir",
        lineterm="",
      )
    )
    raise AssertionError("MIR S-expr mismatch:\n" + diff)


if __name__ == "__main__":
  test_tc_mir_sexpr_matches_golden()
  print("OK")
