"""Semantic regressions for symbolic DOOP normalization, without any engine."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from doop_suite.prepare import prepare

RAW_RELATIONS = """
DirectSuperclass DirectSuperinterface MainClass FormalParam ComponentType
AssignReturnValue ActualParam Method-Modifier Var-Type ClassType ArrayType
InterfaceType Var-DeclaringMethod ApplicationClass ThisVar Field Method
NormalHeap StringConstant AssignHeapAllocation AssignLocal AssignCast
LoadInstanceField StoreInstanceField LoadStaticField StoreStaticField
LoadArrayIndex StoreArrayIndex Return StaticMethodInvocation
SpecialMethodInvocation VirtualMethodInvocation
""".split()
MAIN = "<MainC: void main(java.lang.String[])>"
GET = "<MainC: java.lang.Object get()>"
EXCLUDED = "<java.util.prefs.Base64: void main(java.lang.String[])>"
OTHER = "<Other: void main(java.lang.String[])>"
FIELD = "<MainC: java.lang.Object field>"


def raw_facts(tmp_path, rows=None):
  source = tmp_path / "raw"
  source.mkdir()
  for name in RAW_RELATIONS:
    (source / f"{name}.facts").write_text("", encoding="utf-8")
  for name, values in (rows or {}).items():
    (source / f"{name}.facts").write_text(
      "".join("\t".join(map(str, row)) + "\n" for row in values), encoding="utf-8"
    )
  return source


def prepared_rows(output, name):
  return {
    tuple(map(int, line.split("\t"))) for line in (output / f"{name}.csv").read_text().splitlines()
  }


def test_complete_projection_preserves_shared_ids_roots_and_set_semantics(tmp_path):
  rows = {
    "MainClass": [("MainC",), ("MainC",), ("java.util.prefs.Base64",)],
    "ClassType": [("MainC",), ("java.lang.Object",)],
    "Method": [
      (MAIN, "main", "java.lang.String[]", "MainC", "void", "unused", 1),
      (MAIN, "main", "(java.lang.String[])", "MainC", "void", "unused", 1),
      (GET, "get", "", "MainC", "java.lang.Object", "unused", 0),
      (EXCLUDED, "main", "java.lang.String[]", "java.util.prefs.Base64", "void", "unused", 1),
      (OTHER, "main", "java.lang.String[]", "Other", "void", "unused", 1),
    ],
    "Method-Modifier": [
      (modifier, method) for method in (MAIN, EXCLUDED, OTHER) for modifier in ("public", "static")
    ],
    "FormalParam": [(0, MAIN, "args")],
    "ActualParam": [
      (0, "static-call", "args"),
      (-2147483648, "virtual-call", "value"),
      (2147483647, "virtual-call", "value"),
    ],
    "Var-Type": [("value", "java.lang.Object")],
    "Var-DeclaringMethod": [("value", MAIN)],
    "Field": [(FIELD, "MainC", "field", "java.lang.Object")],
    "NormalHeap": [("heap", "MainC"), ("heap", "MainC")],
    "StringConstant": [("string-heap",)],
    "AssignHeapAllocation": [
      ("alloc", 0, "heap", "value", MAIN, 12),
      ("alloc", 0, "heap", "value", MAIN, 13),
      ("alloc-again", 1, "heap", "value", MAIN, -1),
    ],
    "AssignLocal": [("local", 2, "value", "copy", MAIN)],
    "AssignCast": [("cast", 3, "copy", "cast-value", "MainC", MAIN)],
    "LoadInstanceField": [("load-field", 4, "loaded", "base", FIELD, MAIN)],
    "StoreInstanceField": [("store-field", 5, "value", "base", FIELD, MAIN)],
    "LoadStaticField": [("load-static", 6, "static-value", FIELD, MAIN)],
    "StoreStaticField": [("store-static", 7, "value", FIELD, MAIN)],
    "LoadArrayIndex": [("load-array", 8, "element", "array", MAIN)],
    # Multiarray instructions legitimately belong to several categories.
    "StoreArrayIndex": [("alloc", 0, "value", "array", MAIN)],
    "Return": [("return", 9, "value", MAIN)],
    "StaticMethodInvocation": [("static-call", 10, GET, MAIN)],
    "SpecialMethodInvocation": [("special-call", 11, GET, "base", MAIN)],
    "VirtualMethodInvocation": [("virtual-call", 12, GET, "base", MAIN)],
  }
  source = raw_facts(tmp_path, rows)
  output = tmp_path / "prepared"
  manifest = prepare(source, output, provenance={"release": "fixture"})
  symbols = json.loads((output / "str2num.json").read_text())
  expected = {
    "AssignHeapAllocation": [("heap", "value", MAIN)],
    "AssignLocal": [("value", "copy", MAIN)],
    "AssignCast": [("MainC", "copy", "cast-value", MAIN)],
    "LoadInstanceField": [("base", FIELD, "loaded", MAIN)],
    "StoreInstanceField": [("value", "base", FIELD, MAIN)],
    "LoadStaticField": [(FIELD, "static-value", MAIN)],
    "StoreStaticField": [("value", FIELD, MAIN)],
    "LoadArrayIndex": [("array", "element", MAIN)],
    "StoreArrayIndex": [("value", "array", MAIN)],
    "ReturnVar": [("value", MAIN)],
    "StaticMethodInvocation": [("static-call", GET, MAIN)],
    "SpecialMethodInvocation_Base": [("special-call", "base")],
    "VirtualMethodInvocation_Base": [("virtual-call", "base")],
    "VirtualMethodInvocation_SimpleName": [("virtual-call", "get")],
    "VirtualMethodInvocation_Descriptor": [("virtual-call", "java.lang.Object()")],
    "Field_DeclaringType": [(FIELD, "MainC")],
    "HeapAllocation_Type": [("heap", "MainC"), ("string-heap", "java.lang.String")],
    "FormalParam": [(0, MAIN, "args")],
    "ActualParam": [
      (0, "static-call", "args"),
      (-2147483648, "virtual-call", "value"),
      (2147483647, "virtual-call", "value"),
    ],
    "MainClass": [("MainC",), ("java.util.prefs.Base64",)],
    "Var_Type": [("value", "java.lang.Object")],
    "Var_DeclaringMethod": [("value", MAIN)],
  }
  for name, tuples in expected.items():
    assert prepared_rows(output, name) == {
      tuple(symbols[value] if isinstance(value, str) else value for value in row) for row in tuples
    }
  assert prepared_rows(output, "Instruction_Method") == {
    (symbols[row[0]], symbols[MAIN])
    for name, tuples in rows.items()
    if name
    in (
      "AssignHeapAllocation",
      "AssignLocal",
      "AssignCast",
      "LoadInstanceField",
      "StoreInstanceField",
      "LoadStaticField",
      "StoreStaticField",
      "LoadArrayIndex",
      "StoreArrayIndex",
      "Return",
      "StaticMethodInvocation",
      "SpecialMethodInvocation",
      "VirtualMethodInvocation",
    )
    for row in tuples
  }
  assert manifest["entrypoints"]["selected_main_methods"] == [{"id": symbols[MAIN], "symbol": MAIN}]
  assert manifest["entrypoints"]["excluded_candidates"] == [symbols[EXCLUDED]]
  assert {item["symbol"] for item in manifest["entrypoints"]["candidate_main_methods"]} == {
    MAIN,
    EXCLUDED,
  }
  meta = json.loads((output / "meta.json").read_text())
  assert meta["class_init_method"] == symbols[EXCLUDED]
  assert meta["main_descriptor"] == symbols["void(java.lang.String[])"]
  assert manifest["relations"]["AssignHeapAllocation"]["rows_before_set_dedup"] == 3
  assert manifest["relations"]["AssignHeapAllocation"]["rows"] == 1
  assert manifest["source_relations"]["AssignHeapAllocation"]["rows"] == 3
  assert manifest["status"] == "prepared_not_engine_validated"
  assert len(manifest["relations"]) == 39
  for name, info in manifest["relations"].items():
    content = (output / info["path"]).read_bytes()
    assert info["sha256"] == hashlib.sha256(content).hexdigest()
    assert info["rows"] == len(prepared_rows(output, name))
    assert info["bytes"] == len(content)
  second = tmp_path / "prepared-again"
  prepare(source, second)
  for name in (
    *[f"{relation}.csv" for relation in manifest["relations"]],
    "meta.json",
    "str2num.json",
  ):
    assert (output / name).read_bytes() == (second / name).read_bytes()


@pytest.mark.parametrize(
  "relation,values",
  [
    (
      "Method",
      [
        (GET, "get", "", "MainC", "java.lang.Object", "unused", 0),
        (GET, "get", "java.lang.String", "MainC", "void", "unused", 1),
      ],
    ),
    ("StoreArrayIndex", [("shared", 0, "value", "array", OTHER)]),
    ("AssignLocal", [("shared", 0, "value", "different-target", MAIN)]),
  ],
)
def test_nonfunctional_attributes_fail_without_losing_relational_tuples(tmp_path, relation, values):
  source = raw_facts(
    tmp_path,
    {
      "AssignHeapAllocation": [("shared", 0, "heap", "value", MAIN, 1)],
      relation: values,
    },
  )
  output = tmp_path / "prepared"
  with pytest.raises(ValueError, match="Nonfunctional"):
    prepare(source, output)
  assert not output.exists()


@pytest.mark.parametrize(
  "relation,values",
  [
    ("ActualParam", [(2147483648, "call", "value")]),
    # Numeric columns projected away must still satisfy the source oracle schema.
    ("Method", [(GET, "get", "", "MainC", "void", "unused", -2147483649)]),
    ("AssignHeapAllocation", [("alloc", "not-an-index", "heap", "value", MAIN, 1)]),
    ("Field", [(FIELD, "MainC", "field")]),
  ],
)
def test_invalid_source_domains_and_arity_are_not_published(tmp_path, relation, values):
  source = raw_facts(tmp_path, {relation: values})
  output = tmp_path / "prepared"
  with pytest.raises(ValueError):
    prepare(source, output)
  assert not output.exists()


def test_missing_relation_is_not_treated_as_an_empty_relation(tmp_path):
  source = raw_facts(tmp_path)
  (source / "MainClass.facts").unlink()
  output = tmp_path / "prepared"
  with pytest.raises(FileNotFoundError, match="MainClass"):
    prepare(source, output)
  assert not output.exists()


def test_external_sort_failure_does_not_publish_partial_preparation(tmp_path, monkeypatch):
  source = raw_facts(tmp_path)
  output = tmp_path / "prepared"

  def fail_sort(command, **kwargs):
    raise subprocess.CalledProcessError(1, command)

  monkeypatch.setattr(subprocess, "run", fail_sort)
  with pytest.raises(subprocess.CalledProcessError):
    prepare(source, output)
  assert not output.exists()


def test_refuses_existing_output_and_preserves_its_contents(tmp_path):
  source = raw_facts(tmp_path)
  output = tmp_path / "prepared"
  output.mkdir()
  marker = output / "existing"
  marker.write_text("keep me")
  with pytest.raises(FileExistsError):
    prepare(source, output)
  assert marker.read_text() == "keep me"
