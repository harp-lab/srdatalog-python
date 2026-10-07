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


def test_canonical_array_stores_match_upstream_type_join_and_propagate_new_points(tmp_path):
  if shutil.which(os.environ.get("SOUFFLE", "souffle")) is None:
    pytest.skip("Souffle and development headers are required for the embedded driver")
  meta_path = Path(__file__).parent / "fixtures" / "integration" / "doop_meta.json"
  meta = json.loads(meta_path.read_text())
  canonical = doop.build_doopdb_program(meta)
  selected_names = {
    "HeapAllocHelper",
    "IsObjectArrayHeap_rule",
    "EligibleObjectArrayHeap_rule",
    "IsStringArrayHeap_rule",
    "EligibleStringArrayHeap_rule",
    "ReachableSortedIndex_rule",
    "AIPT_Store",
    "AIPT_Store_ObjectArray",
    "AIPT_Store_StringArray",
    "ArrayTypeCompatPrecompute",
    "VPT_LoadArray",
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
    # Independent upstream formula, before either special-array split.
    # Both allocation types and the component/supertype join are required.
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

  # Source heaps: 101/105 fit String[], 102 fits only the ordinary array,
  # 103 is untyped, and 104 has both compatible and incompatible types.
  # Base heaps: 10/50/51 are Object[], 11/15 are String[], 12 is ordinary,
  # 13 is untyped, and 14 has no component relation.
  object_array = meta["java_lang_Object_array"]
  string_array = meta["java_lang_String_array"]
  ordinary_array = 204
  facts = {
    "HeapAllocation_Type.csv": (
      f"10\t{object_array}\n50\t{object_array}\n51\t{object_array}\n"
      f"11\t{string_array}\n15\t{string_array}\n12\t{ordinary_array}\n14\t205\n"
      "101\t201\n102\t202\n104\t201\n104\t203\n104\t202\n105\t203\n"
    ),
    "SeedPoints.csv": (
      "10\t1\n12\t5\n13\t6\n14\t7\n15\t8\n101\t2\n102\t2\n103\t2\n104\t2\n"
      "104\t3\n50\t30\n11\t31\n51\t32\n105\t33\n"
    ),
    "SeedReachable.csv": "9\n",
    "StoreArrayIndex.csv": (
      "2\t1\t9\n3\t1\t9\n2\t4\t9\n2\t5\t9\n2\t6\t9\n2\t7\t9\n2\t8\t9\n"
      "31\t30\t9\n33\t32\t9\n2\t8\t99\n"
    ),
    # These two loads discover a new destination array and a new source
    # heap only after stores have fired, exercising both recursive inputs.
    "LoadArrayIndex.csv": "30\t4\t9\n32\t2\t9\n",
    "Var_Type.csv": "4\t200\n2\t200\n",
  }
  for filename, contents in facts.items():
    (tmp_path / filename).write_text(contents, encoding="utf-8")

  for case, missing_component, string_compatible in [
    ("all-components", None, True),
    ("missing-string-component", string_array, True),
    ("missing-object-component", object_array, True),
    ("missing-ordinary-component", ordinary_array, True),
    ("missing-string-supertype", None, False),
  ]:
    components = [(object_array, 200), (string_array, 201), (ordinary_array, 202)]
    (tmp_path / "ComponentType.csv").write_text(
      "".join(f"{array}\t{component}\n" for array, component in components
              if array != missing_component),
      encoding="utf-8",
    )
    (tmp_path / "SeedSupertype.csv").write_text(
      f"200\t201\n200\t202\n200\t203\n200\t200\n200\t{string_array}\n202\t202\n"
      + ("201\t201\n201\t203\n" if string_compatible else ""),
      encoding="utf-8",
    )
    object_present = missing_component != object_array
    string_present = missing_component != string_array and string_compatible
    ordinary_present = missing_component != ordinary_array
    expected = set()
    if object_present:
      expected.update({(10, 101), (10, 102), (10, 104), (10, 105), (50, 11), (51, 105)})
    if string_present:
      expected.update({(15, 101), (15, 104)})
      if object_present:
        expected.update({(11, 101), (11, 104), (11, 105), (15, 105)})
    if ordinary_present:
      expected.update({(12, 102), (12, 104)})
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

    assert tuples("UpstreamArrayIndexPointsTo") == expected, case
    assert tuples("ArrayIndexPointsTo") == expected, case
    assert tuples("EligibleStringArrayHeap") == (
      {(101,), (104,), (105,)} if string_present else set()
    ), case
    assert tuples("EligibleObjectArrayHeap") == (
      {(11,), (15,), (101,), (102,), (104,), (105,)} if object_present else set()
    ), case
    assert tuples("NonStringHeapAllocSuperType") == (
      {(102, 12), (104, 12)} if ordinary_present else set()
    ), case
    points = tuples("VarPointsTo")
    assert ((11, 4) in points) == object_present, case
    assert ((105, 2) in points) == object_present, case


def test_canonical_array_metadata_requires_distinct_string_and_object_types():
  meta_path = Path(__file__).parent / "fixtures" / "integration" / "doop_meta.json"
  meta = json.loads(meta_path.read_text())
  meta["java_lang_String_array"] = meta["java_lang_Object_array"]
  with pytest.raises(ValueError):
    doop.build_doopdb_program(meta)



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
