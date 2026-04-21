'''End-to-end test on PathCompose — a recursive rule with TWO SCC-member
body clauses, producing two delta variants (delta_idx=0 and delta_idx=1).

Stresses the planner on:
  - Dual-delta recursive variant generation (each body Path clause
    independently becomes the delta seed).
  - Different delta positions yielding different clause_order and
    var_order: delta_idx=0 -> [0,1] / [y,x,z]; delta_idx=1 -> [1,0] / [y,z,x].

The golden fixture is python/tests/fixtures/path_compose.hir.json,
generated from python/tests/fixtures/path_compose.nim via srdatalog_plan.
'''
import json
import sys
from pathlib import Path


from srdatalog.dsl import Var, Relation, Program
from srdatalog.hir import compile_to_hir
from srdatalog.hir_emit import hir_to_obj


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def build_path_compose_program() -> Program:
  X, Y, Z = Var("x"), Var("y"), Var("z")
  seed = Relation("Seed", 2)
  path = Relation("Path", 2)
  return Program(
    relations=[seed, path],
    rules=[
      (path(X, Y) <= seed(X, Y)).named("PathSeed"),
      (path(X, Z) <= path(X, Y) & path(Y, Z)).named("PathCompose"),
    ],
  )


def _canonical(obj: dict) -> str:
  return json.dumps(obj, indent=2, ensure_ascii=False)


def test_path_compose_full_pipeline_matches_golden():
  hir = compile_to_hir(build_path_compose_program())
  actual = hir_to_obj(hir)

  golden = json.loads((FIXTURES / "path_compose.hir.json").read_text())
  golden.pop("hirSExpr", None)

  a, g = _canonical(actual), _canonical(golden)
  if a != g:
    import difflib
    raise AssertionError(
      "PathCompose pipeline output != golden:\n"
      + "\n".join(
        difflib.unified_diff(
          g.splitlines(), a.splitlines(),
          fromfile="nim-golden", tofile="python-pipeline", lineterm="",
        )
      )
    )


if __name__ == "__main__":
  test_path_compose_full_pipeline_matches_golden()
  print("OK")
