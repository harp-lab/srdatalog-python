'''Differential test: hand-construct the HIR for tc in Python, emit it, and
assert the structure matches the golden fixture produced by the Nim emitter
(minus the hirSExpr debug field).

Passing this test is the milestone that says "the Python HIR emitter agrees
with Nim's on a real example." Once hit, subsequent HIR passes can be tested
the same way: build HIR from DSL via the pass, emit, diff against golden.
'''

import json
from pathlib import Path

from srdatalog.dsl import Relation, Var
from srdatalog.hir.emit import hir_to_obj
from srdatalog.hir.types import (
  AccessPattern,
  HirProgram,
  HirRuleVariant,
  HirStratum,
  RelationDecl,
  Version,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def build_tc_hir() -> HirProgram:
  '''Hand-construct the HIR for tc/tc.nim. Mirrors what Nim's compileToHir
  produces from tc.nim; used before the DSL->HIR pass exists in Python.
  '''
  X, Y, Z = Var("x"), Var("y"), Var("z")
  arc = Relation("ArcInput", 2)
  edge = Relation("Edge", 2)
  path = Relation("Path", 2)

  edge_load = (edge(X, Y) <= arc(X, Y)).named("EdgeLoad")
  tc_base = (path(X, Y) <= edge(X, Y)).named("TCBase")
  tc_rec = (path(X, Z) <= path(X, Y) & edge(Y, Z)).named("TCRec")

  # Stratum 0: {Edge}, non-recursive, base = EdgeLoad
  s0 = HirStratum(
    scc_members={"Edge"},
    is_recursive=False,
    base_variants=[
      HirRuleVariant(
        original_rule=edge_load,
        delta_idx=-1,
        clause_order=[0],
        var_order=["x", "y"],
        access_patterns=[
          AccessPattern(
            rel_name="ArcInput",
            version=Version.FULL,
            access_order=["x", "y"],
            index_cols=[0, 1],
            prefix_len=0,
            clause_idx=0,
          ),
        ],
      ),
    ],
  )

  # Stratum 1: {Path}, non-recursive, base = TCBase
  s1 = HirStratum(
    scc_members={"Path"},
    is_recursive=False,
    base_variants=[
      HirRuleVariant(
        original_rule=tc_base,
        delta_idx=-1,
        clause_order=[0],
        var_order=["x", "y"],
        access_patterns=[
          AccessPattern(
            rel_name="Edge",
            version=Version.FULL,
            access_order=["x", "y"],
            index_cols=[0, 1],
            prefix_len=0,
            clause_idx=0,
          ),
        ],
      ),
    ],
  )

  # Stratum 2: {Path}, recursive, recursive = TCRec (delta at body clause 0)
  s2 = HirStratum(
    scc_members={"Path"},
    is_recursive=True,
    recursive_variants=[
      HirRuleVariant(
        original_rule=tc_rec,
        delta_idx=0,
        clause_order=[0, 1],
        var_order=["y", "x", "z"],
        access_patterns=[
          AccessPattern(
            rel_name="Path",
            version=Version.DELTA,
            access_order=["y", "x"],
            index_cols=[1, 0],
            prefix_len=1,
            clause_idx=0,
          ),
          AccessPattern(
            rel_name="Edge",
            version=Version.FULL,
            access_order=["y", "z"],
            index_cols=[0, 1],
            prefix_len=1,
            clause_idx=1,
          ),
        ],
      ),
    ],
  )

  return HirProgram(
    strata=[s0, s1, s2],
    relation_decls=[
      RelationDecl(rel_name="ArcInput", types=["int", "int"], semiring="NoProvenance"),
      RelationDecl(rel_name="Edge", types=["int", "int"], semiring="NoProvenance"),
      RelationDecl(rel_name="Path", types=["int", "int"], semiring="NoProvenance"),
    ],
  )


def _canonicalize(obj: dict) -> str:
  '''Serialize with fixed formatting. sort_keys=False preserves insertion
  order, so ordering mismatches between Nim and Python emitters are caught.
  '''
  return json.dumps(obj, indent=2, ensure_ascii=False)


def test_tc_emitter_matches_golden():
  prog = build_tc_hir()
  actual = hir_to_obj(prog)

  golden = json.loads((FIXTURES / "tc.hir.json").read_text())
  golden.pop("hirSExpr", None)

  actual_s = _canonicalize(actual)
  golden_s = _canonicalize(golden)
  if actual_s != golden_s:
    # Dump side-by-side for easier diagnosis
    import difflib

    diff = "\n".join(
      difflib.unified_diff(
        golden_s.splitlines(),
        actual_s.splitlines(),
        fromfile="nim-golden",
        tofile="python-emit",
        lineterm="",
      )
    )
    raise AssertionError("HIR emit mismatch:\n" + diff)


if __name__ == "__main__":
  test_tc_emitter_matches_golden()
  print("OK")
