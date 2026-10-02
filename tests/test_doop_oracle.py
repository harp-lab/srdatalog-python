"""Semantic checks for the conservative canonical-Program CPU translation."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from srdatalog.dsl import SPLIT, Const, Filter, Program, Relation, Var

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import doop
from doop_suite.cpu import _build, translate_program


def test_translation_preserves_recursive_multihead_filter_negation_and_split(tmp_path):
  souffle = shutil.which("souffle")
  if souffle is None:
    pytest.skip("Souffle is required to execute the translated logical program")
  x, y, z = Var("x"), Var("y"), Var("z")
  seed = Relation("Seed", 2, input_file="Seed.csv")
  blocked = Relation("Blocked", 2, input_file="Blocked.csv")
  forward = Relation("Forward", 2)
  reverse = Relation("Reverse", 2)
  from_one = Relation("FromOne", 1)
  program = Program(
    rules=[
      (
        (forward(x, y) | reverse(y, x))
        <= seed(x, y)
        & SPLIT
        & ~blocked(x, Var("_"))
        & Filter(("x", "y"), "return x != -1 && y != 8;")
      ).with_plan(var_order=["y", "x"]),
      forward(x, z) <= forward(x, y) & forward(y, z),
      from_one(y) <= forward(x, y) & Filter(("x",), "return x == 1;"),
    ]
  )
  text, _ = translate_program(program)
  source = tmp_path / "semantic.dl"
  source.write_text(text, encoding="utf-8")
  (tmp_path / "Seed.csv").write_text("1\t2\n2\t3\n3\t4\n5\t6\n1\t2\n7\t8\n-1\t0\n")
  (tmp_path / "Blocked.csv").write_text("3\t99\n5\t1\n")
  outputs = tmp_path / "outputs"
  outputs.mkdir()
  subprocess.run(
    [souffle, "-F", str(tmp_path), "-D", str(outputs), str(source)],
    check=True,
    capture_output=True,
    text=True,
    timeout=30,
  )

  def tuples(name):
    rows = (outputs / f"{name}.tsv").read_text().splitlines()
    parsed = [tuple(map(int, row.split("\t"))) for row in rows]
    assert len(parsed) == len(set(parsed)), "The oracle must export set, not bag, results"
    return set(parsed)

  assert tuples("Forward") == {(1, 2), (2, 3), (1, 3)}
  assert tuples("Reverse") == {(2, 1), (3, 2)}
  assert tuples("FromOne") == {(2,), (3,)}


def test_compute_only_driver_retains_recursive_unused_and_empty_relations(tmp_path):
  if shutil.which(os.environ.get("SOUFFLE", "souffle")) is None:
    pytest.skip("Souffle and development headers are required for the embedded driver")
  x, y, z = Var("x"), Var("y"), Var("z")
  seed = Relation("Seed", 2, input_file="Seed.csv")
  reach = Relation("Reach", 2)
  unused = Relation("Unused", 2)
  empty = Relation("Empty", 2)
  program = Program(
    rules=[
      reach(x, y) <= seed(x, y),
      reach(x, z) <= reach(x, y) & seed(y, z),
      unused(y, x) <= seed(x, y),
      empty(x, y) <= seed(x, y) & Filter(("x",), "return x == 99;"),
    ]
  )
  text, _ = translate_program(program, export_tuples=False)
  source = tmp_path / "doop_reference.dl"
  source.write_text(text, encoding="utf-8")
  (tmp_path / "Seed.csv").write_text("1\t2\n2\t3\n3\t4\n")
  env = os.environ.copy()
  env["OMP_NUM_THREADS"] = "2"
  build = _build(source, tmp_path, 2, env, 120)
  tuples = tmp_path / "tuples"
  report = tmp_path / "timing.json"
  completed = subprocess.run(
    [build["binary"], build["factory_name"], str(tmp_path), str(tuples), "2", str(report), "none"],
    check=True, capture_output=True, text=True, timeout=30, env=env,
  )
  timing = json.loads(report.read_text())
  assert timing["relation_counts"] == {"Seed": 3, "Reach": 6, "Unused": 3, "Empty": 0}
  assert timing["exported_relations"] == []
  assert timing["export_seconds"] == 0
  assert not tuples.exists()
  assert completed.stdout == "", "Generated printsize I/O must stay outside the fixedpoint"


def test_canonical_object_array_store_matches_upstream_type_join(tmp_path):
  if shutil.which(os.environ.get("SOUFFLE", "souffle")) is None:
    pytest.skip("Souffle and development headers are required for the embedded driver")
  meta_path = Path(__file__).parent / "fixtures" / "integration" / "doop_meta.json"
  meta = json.loads(meta_path.read_text())
  canonical = doop.build_doopdb_program(meta)
  selected_names = {
    "IsObjectArrayHeap_rule",
    "EligibleObjectArrayHeap_rule",
    "ReachableSortedIndex_rule",
    "AIPT_Store_ObjectArray",
  }
  rules = [rule for rule in canonical.rules if rule.name in selected_names]
  heap, heaptype = Var("heap"), Var("heaptype")
  baseheap, baseheaptype = Var("baseheap"), Var("baseheaptype")
  frm, base, method = Var("frm"), Var("base"), Var("method")
  comptype = Var("comptype")
  seed_points = Relation("SeedPoints", 2, input_file="SeedPoints.csv")
  seed_supertype = Relation("SeedSupertype", 2, input_file="SeedSupertype.csv")
  seed_reachable = Relation("SeedReachable", 1, input_file="SeedReachable.csv")
  upstream = Relation("UpstreamArrayIndexPointsTo", 2)
  rules.extend([
    doop.VarPointsTo(heap, frm) <= seed_points(heap, frm),
    doop.SupertypeOf(comptype, heaptype) <= seed_supertype(comptype, heaptype),
    doop.Reachable(method) <= seed_reachable(method),
    # Original unfactored store formula: both allocation types and the
    # component/supertype join must exist, even when the base is Object[].
    upstream(baseheap, heap)
    <= doop.Reachable(method)
    & doop.StoreArrayIndex(frm, base, method)
    & doop.VarPointsTo(baseheap, base)
    & doop.VarPointsTo(heap, frm)
    & doop.HeapAllocation_Type(baseheap, baseheaptype)
    & doop.HeapAllocation_Type(heap, heaptype)
    & doop.ComponentType(baseheaptype, comptype)
    & doop.SupertypeOf(comptype, heaptype),
  ])
  text, _ = translate_program(Program(rules=rules), export_tuples=False)
  source = tmp_path / "doop_reference.dl"
  source.write_text(text, encoding="utf-8")
  env = os.environ.copy()
  env["OMP_NUM_THREADS"] = "2"
  build = _build(source, tmp_path, 2, env, 120)

  # 101 is compatible, 102 incompatible, 103 untyped, and 104 has two
  # compatible types plus an incompatible one. Two source variables create
  # distinct derivations of the same store, not merely duplicate input rows.
  object_array = meta["java_lang_Object_array"]
  facts = {
    "HeapAllocation_Type.csv": (
      f"10\t{object_array}\n101\t201\n102\t202\n"
      "104\t201\n104\t203\n104\t202\n"
    ),
    "SeedPoints.csv": (
      "10\t1\n101\t2\n102\t2\n103\t2\n104\t2\n104\t3\n"
    ),
    "SeedSupertype.csv": "200\t201\n200\t203\n",
    "SeedReachable.csv": "9\n",
    "StoreArrayIndex.csv": "2\t1\t9\n3\t1\t9\n",
  }
  for filename, contents in facts.items():
    (tmp_path / filename).write_text(contents, encoding="utf-8")

  for case, components, expected in [
    ("component-present", f"{object_array}\t200\n", {(10, 101), (10, 104)}),
    ("component-missing", "", set()),
  ]:
    (tmp_path / "ComponentType.csv").write_text(components, encoding="utf-8")
    outputs = tmp_path / case
    subprocess.run(
      [
        build["binary"], build["factory_name"], str(tmp_path), str(outputs),
        "2", str(tmp_path / f"{case}.json"), "tsv",
      ],
      check=True, capture_output=True, text=True, timeout=30, env=env,
    )

    def tuples(name, outputs=outputs, case=case):
      rows = (outputs / f"{name}.tsv").read_text().splitlines()
      parsed = [tuple(map(int, row.split("\t"))) for row in rows]
      assert len(parsed) == len(set(parsed)), f"{case}: {name} must have set semantics"
      return set(parsed)

    assert tuples("UpstreamArrayIndexPointsTo") == expected
    assert tuples("ArrayIndexPointsTo") == expected
    assert tuples("EligibleObjectArrayHeap") == {(heap,) for _, heap in expected}



@pytest.mark.parametrize(
  "code",
  [
    "return x != 1 || x != 2;",  # Disjunction must not be silently treated as conjunction.
    "return x == 010;",  # C++ octal is not a decimal identifier.
    "return x == 2147483648;",  # GPU's integer domain is signed int32.
  ],
)
def test_unsupported_filter_semantics_are_rejected(code):
  x = Var("x")
  seed = Relation("Seed", 1, input_file="Seed.csv")
  result = Relation("Result", 1)
  with pytest.raises(ValueError):
    translate_program(Program(rules=[result(x) <= seed(x) & Filter(("x",), code)]))


def test_constant_cpp_expression_cannot_override_metadata_literal():
  x = Var("x")
  seed = Relation("Seed", 1, input_file="Seed.csv")
  result = Relation("Result", 2)
  with pytest.raises(ValueError):
    translate_program(Program(rules=[result(x, Const(1, cpp_expr="2")) <= seed(x)]))
