'''End-to-end test: DSL program -> full HIR pipeline -> emit JSON ->
byte-identical to the Nim golden for tc.

This is the milestone that validates stratification + semi-naive variants
+ join planning all produce the same HIR shape as Nim. Previously the
emitter diff test hand-constructed the HIR; now we build only the DSL
Program and let the pipeline fill in everything.
'''
import json
import sys
from pathlib import Path


from srdatalog.dsl import Var, Relation, Program
from srdatalog.hir import compile_to_hir
from srdatalog.hir_emit import hir_to_obj


FIXTURES = Path(__file__).resolve().parent / "fixtures"


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


def _canonical(obj: dict) -> str:
  return json.dumps(obj, indent=2, ensure_ascii=False)


def test_tc_full_pipeline_matches_golden():
  prog = build_tc_program()
  hir = compile_to_hir(prog)
  actual = hir_to_obj(hir)

  golden = json.loads((FIXTURES / "tc.hir.json").read_text())
  golden.pop("hirSExpr", None)

  actual_s = _canonical(actual)
  golden_s = _canonical(golden)
  if actual_s != golden_s:
    import difflib
    diff = "\n".join(
      difflib.unified_diff(
        golden_s.splitlines(),
        actual_s.splitlines(),
        fromfile="nim-golden",
        tofile="python-pipeline",
        lineterm="",
      )
    )
    raise AssertionError("Full pipeline output != golden:\n" + diff)


if __name__ == "__main__":
  test_tc_full_pipeline_matches_golden()
  print("OK")
