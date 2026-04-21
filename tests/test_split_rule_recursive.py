'''Recursive split-rule end-to-end test.

Builds split_rec.nim via DSL, runs compile_to_mir, byte-matches both HIR
and MIR against Nim golden. Exercises the recursive-stratum split path:
ClearRelation temp NEW per iteration + Pipeline A + CreateFlatView temp
NEW + Pipeline B with temp_version=NEW.
'''
import json
import sys
from pathlib import Path


from srdatalog.dsl import Var, Relation, Program, SPLIT
from srdatalog.hir import compile_to_hir, compile_to_mir
from srdatalog.hir_emit import hir_to_obj
from srdatalog.mir_emit import print_mir_sexpr


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def build_split_rec_program() -> Program:
  '''Mirror of fixtures/split_rec.nim.
    R(x, y) :- Seed(x, y)                              (RBase, non-recursive)
    R(x, z) :- R(x, y), ~Filter(y), split, Edge(y, z)  (RRec, recursive split)
  '''
  X, Y, Z = Var("x"), Var("y"), Var("z")
  seed = Relation("Seed", 2)
  filter_rel = Relation("Filter", 1)
  edge = Relation("Edge", 2)
  r = Relation("R", 2)
  return Program(
    relations=[seed, filter_rel, edge, r],
    rules=[
      (r(X, Y) <= seed(X, Y)).named("RBase"),
      (r(X, Z) <= r(X, Y) & ~filter_rel(Y) & SPLIT & edge(Y, Z)).named("RRec"),
    ],
  )


def test_recursive_split_variant_has_metadata():
  hir = compile_to_hir(build_split_rec_program())
  # Stratum 0: base {R}, Stratum 1: recursive {R}
  assert len(hir.strata) == 2
  rec_variant = hir.strata[1].recursive_variants[0]
  assert rec_variant.delta_idx == 0
  assert rec_variant.split_at == 2
  # tempVars = [y, x]: y is the join var with below body (Edge y z);
  # x is a head-only var carried through the temp.
  assert rec_variant.temp_vars == ["y", "x"]
  assert rec_variant.temp_rel_name == "_temp_RRec"


def test_recursive_split_temp_rel_synthesised():
  hir = compile_to_hir(build_split_rec_program())
  temp_decls = [d for d in hir.relation_decls if d.rel_name == "_temp_RRec"]
  assert len(temp_decls) == 1
  assert temp_decls[0].types == ["int", "int"]   # y:int (from R col 1), x:int (R col 0)
  assert temp_decls[0].is_temp is True


def _canonical(obj: dict) -> str:
  return json.dumps(obj, indent=2, ensure_ascii=False)


def test_recursive_split_hir_byte_match():
  hir = compile_to_hir(build_split_rec_program())
  actual = hir_to_obj(hir)
  golden = json.loads((FIXTURES / "split_rec.hir.json").read_text())
  golden.pop("hirSExpr", None)
  if _canonical(actual) != _canonical(golden):
    import difflib
    diff = "\n".join(
      difflib.unified_diff(
        _canonical(golden).splitlines(), _canonical(actual).splitlines(),
        fromfile="nim-golden", tofile="python", lineterm="",
      )
    )
    raise AssertionError("HIR mismatch:\n" + diff)


def test_recursive_split_mir_byte_match():
  '''Recursive split stratum MIR: ClearRelation(temp, NEW) +
  ExecutePipeline(splitA) + CreateFlatView(temp, NEW) +
  ExecutePipeline(splitB, temp Scan uses NEW) + loop maintenance.
  '''
  mir_prog = compile_to_mir(build_split_rec_program())
  actual = print_mir_sexpr(mir_prog)
  golden = (FIXTURES / "split_rec.mir.sexpr").read_text().rstrip("\n")
  if actual != golden:
    import difflib
    diff = "\n".join(
      difflib.unified_diff(
        golden.splitlines(), actual.splitlines(),
        fromfile="nim-golden", tofile="python", lineterm="",
      )
    )
    raise AssertionError("MIR mismatch:\n" + diff)


if __name__ == "__main__":
  tests = [
    test_recursive_split_variant_has_metadata,
    test_recursive_split_temp_rel_synthesised,
    test_recursive_split_hir_byte_match,
    test_recursive_split_mir_byte_match,
  ]
  for t in tests:
    t()
  print(f"OK ({len(tests)} tests)")
