"""Bound-key eligibility is a filter, not a generator or a destructive projection."""

import os
import subprocess
import sys

import pytest
from integration_helpers import check_mir_lifetimes

from srdatalog.dsl import Program, Relation, Var
from srdatalog.ir.hir import compile_to_hir, compile_to_mir
from srdatalog.ir.hir.types import Version
from srdatalog.ir.mir import types as mir


def _program():
  key, value, tag, next_value = map(Var, ("key", "value", "tag", "next_value"))
  items = Relation("Items", 2, input_file="Items.csv")
  matches = Relation("Matches", 2, input_file="Matches.csv")
  allowed_input = Relation("AllowedInput", 1, input_file="AllowedInput.csv")
  allowed = Relation("Allowed", 1)
  empty = Relation("Empty", 1, input_file="Empty.csv")
  seed = Relation("Seed", 1, input_file="Seed.csv")
  step = Relation("Step", 2, input_file="Step.csv")
  filtered = Relation("Filtered", 2)
  selected = Relation("Selected", 3)
  ordered = Relation("Ordered", 3)
  guard_first = Relation("GuardFirst", 3)
  rejected = Relation("Rejected", 2)
  original = Relation("Original", 2)
  reach = Relation("Reach", 2)
  eligible = Relation("Eligible", 1)
  live = Relation("Live", 2)
  a, b, c, d = map(Var, ("a", "b", "c", "d"))
  triple_seed = Relation("TripleSeed", 2, input_file="TripleSeed.csv")
  triple_guard = Relation("TripleGuard", 1, input_file="TripleGuard.csv")
  triple_path = Relation("TriplePath", 2, index_type="SRDatalog::GPU::Device2LevelIndex")
  shared_seed = Relation("SharedSeed", 2, input_file="SharedSeed.csv")
  shared_path = Relation("SharedPath", 2, index_type="SRDatalog::GPU::Device2LevelIndex")
  body = items(key, value) & matches(key, tag) & allowed(value)
  return Program(rules=[
    (allowed(value) <= allowed_input(value)).named("LoadAllowed"),
    (filtered(key, value) <= items(key, value) & allowed(value)).named("FilterItems"),
    (selected(key, value, tag) <= body).named("SelectItems"),
    (ordered(key, value, tag) <= body).named("ExplicitOrder").with_plan(
      var_order=["value", "key", "tag"], clause_order=[0, 2, 1],
    ),
    (guard_first(key, value, tag) <= body).named("GuardFirstOrder").with_plan(
      var_order=["value", "key", "tag"], clause_order=[2, 0, 1],
    ),
    (rejected(key, value) <= items(key, value) & empty(value)).named("RejectAll"),
    (original(key, value) <= items(key, value)).named("PreserveItems"),
    (reach(key, value) <= filtered(key, value)).named("ReachSeed"),
    (reach(key, next_value) <= reach(key, value) & step(value, next_value)
     & allowed(key) & allowed(next_value)).named("ReachStep"),
    (eligible(value) <= seed(value)).named("EligibilitySeed"),
    (live(key, value) <= items(key, value) & eligible(value)).named("NewEligibility"),
    (eligible(next_value) <= live(key, value) & step(value, next_value)).named("AdvanceEligibility"),
    (triple_path(a, b) <= triple_seed(a, b)).named("TripleSeedRule"),
    (triple_path(a, d) <= triple_path(a, b) & triple_path(b, c)
     & triple_path(c, d) & triple_guard(b)).named("TripleStep"),
    (shared_path(a, b) <= shared_seed(a, b)).named("SharedSeedRule"),
    (shared_path(b, c) <= shared_path(a, b) & shared_path(a, c)
     & shared_path(a, d) & triple_guard(d)).named("SharedStep"),
  ])


def _variants(hir, name):
  return [
    variant
    for stratum in hir.strata
    for variant in stratum.base_variants + stratum.recursive_variants
    if variant.original_rule.name == name
  ]


def _pipelines(program):
  for node, _ in compile_to_mir(program).steps:
    if isinstance(node, mir.FixpointPlan):
      for instruction in node.instructions:
        for pipeline in instruction.ops if isinstance(instruction, mir.ParallelGroup) else [instruction]:
          if isinstance(pipeline, mir.ExecutePipeline):
            yield pipeline


def test_unary_filter_does_not_compete_with_a_real_join_or_add_columns():
  program = _program()
  hir = compile_to_hir(program)
  variant, = _variants(hir, "SelectItems")
  assert variant.var_order[0] == "key"
  assert {pattern.rel_name for pattern in variant.access_patterns} == {"Items", "Matches"}
  probe, = variant.semijoin_patterns
  assert (probe.rel_name, probe.version, probe.access_order) == ("Allowed", Version.FULL, ["value"])
  pipeline, = [p for p in _pipelines(program) if p.rule_name == "SelectItems"]
  bound = set()
  probes = []
  for op in pipeline.pipeline:
    if isinstance(op, mir.ColumnJoin):
      bound.add(op.var_name)
    elif isinstance(op, (mir.Scan, mir.CartesianJoin)):
      bound.update(op.vars)
    elif isinstance(op, mir.SemiJoin):
      assert set(op.prefix_vars) <= bound
      probes.append(op)
    elif isinstance(op, mir.InsertInto):
      assert op.vars == ["key", "value", "tag"]
      assert set(op.vars) <= bound
  assert [(op.rel_name, op.prefix_vars) for op in probes] == [("Allowed", ["value"])]
  assert bound == {"key", "value", "tag"}


@pytest.mark.parametrize("name,clause_order", [
  ("ExplicitOrder", [0, 2, 1]),
  ("GuardFirstOrder", [2, 0, 1]),
])
def test_explicit_orders_are_preserved_with_eligibility(name, clause_order):
  variant, = _variants(compile_to_hir(_program()), name)
  assert variant.var_order == ["value", "key", "tag"]
  assert variant.clause_order == clause_order
  if name == "GuardFirstOrder":
    assert not variant.semijoin_patterns
    assert any(pattern.rel_name == "Allowed" for pattern in variant.access_patterns)


def test_lone_unary_relation_remains_a_generator():
  value = Var("value")
  source, output = Relation("Source", 1), Relation("Output", 1)
  hir = compile_to_hir(Program([(output(value) <= source(value)).named("Copy")]))
  variant, = _variants(hir, "Copy")
  assert not variant.semijoin_patterns
  assert [(p.rel_name, p.access_order) for p in variant.access_patterns] == [("Source", ["value"])]


@pytest.mark.parametrize("options", [
  {"semiring": "BooleanSR"},
  {"index_type": "SRDatalog::GPU::Device2LevelIndex"},
  {"index_type": "SRDatalog::GPU::DeviceTvjoinIndex"},
])
def test_provenance_and_multisegment_eligibility_keep_the_original_join(options):
  key, value = Var("key"), Var("value")
  items, output = Relation("Items", 2), Relation("Output", 2)
  allowed = Relation("Allowed", 1, **options)
  rule = (output(key, value) <= items(key, value) & allowed(value)).named("Filter")
  variant, = _variants(compile_to_hir(Program([rule])), "Filter")
  assert not variant.semijoin_patterns
  assert {p.rel_name for p in variant.access_patterns} == {"Items", "Allowed"}


def test_recursive_generators_and_probe_index_lifetimes():
  program = _program()
  hir = compile_to_hir(program)
  check_mir_lifetimes(program)
  reach, = _variants(hir, "ReachStep")
  assert any(p.rel_name == "Reach" and p.version is Version.DELTA for p in reach.access_patterns)
  assert [(p.rel_name, p.version) for p in reach.semijoin_patterns] == [
    ("Allowed", Version.FULL), ("Allowed", Version.FULL),
  ]
  assert reach.var_order[0] == "key"
  newly_eligible, = _variants(hir, "NewEligibility")
  assert newly_eligible.delta_idx == 1
  assert newly_eligible.clause_versions == [Version.FULL, Version.DELTA]
  assert not newly_eligible.semijoin_patterns
  assert any(p.rel_name == "Eligible" and p.version is Version.DELTA for p in newly_eligible.access_patterns)


def test_native_generated_semijoin_preserves_exact_relations(tmp_path):
  if os.environ.get("SRDATALOG_JIT_RUN_COMPILE_TESTS") != "1":
    pytest.skip("Opt in to native CUDA compilation with SRDATALOG_JIT_RUN_COMPILE_TESTS=1")

  from srdatalog import CompilerConfig, build_project, compile_jit_project
  from srdatalog.runtime import (
    cuda_compile_flags,
    cuda_include_paths,
    cuda_libs,
    cuda_link_flags,
    runtime_defines,
    runtime_include_paths,
  )

  program = _program()
  # Ensure this executes the new compiled operation, not just equivalent ordinary joins.
  pipelines = list(_pipelines(program))
  for name in ("FilterItems", "SelectItems", "ExplicitOrder", "RejectAll", "ReachStep_D0"):
    pipeline, = [p for p in pipelines if p.rule_name == name]
    assert any(isinstance(op, mir.SemiJoin) for op in pipeline.pipeline)
  for name in ("TripleStep", "SharedStep"):
    for delta in range(3):
      pipeline, = [p for p in pipelines if p.rule_name == f"{name}_D{delta}"]
      assert any(isinstance(op, mir.SemiJoin) for op in pipeline.pipeline)
  project = build_project(program, "SemijoinRegression", cache_base=str(tmp_path / "cache"))
  compiled = compile_jit_project(project, CompilerConfig(
    include_paths=runtime_include_paths() + cuda_include_paths(),
    defines=runtime_defines(),
    cxx_flags=cuda_compile_flags() + ["-fPIC"],
    link_flags=cuda_link_flags(),
    libs=cuda_libs() + ["boost_container"],
    shared=True,
    jobs=2,
  ))
  assert compiled.ok(), compiled
  items = {(key, (key + offset) % 5) for key in range(257) for offset in range(3)}
  # More than 2048 sparse keys exercise the cooperative large-range lookup;
  # each eligible value has adjacent rejected values and a shared join key.
  items.update(
    (node % 257, 100 + 3 * node + offset)
    for node in range(3000) for offset in (-1, 0, 1)
  )
  matches = {(key, 1000 + 3 * key + offset) for key in range(257) for offset in range(3)}
  allowed = {0, 2, 9} | {100 + 3 * node for node in range(3000)}
  steps = {(0, 2), (2, 4), (4, 1)}
  # Disconnected seed rows keep recursive additions below the 10% compaction
  # threshold: subsequent iterations must read both FULL and nonempty HEAD.
  triple_seed = (
    {(node, node + 1) for node in range(23)}
    | {(node, node + 2) for node in range(0, 21, 3)}
    | {(1000 + 2 * node, 1001 + 2 * node) for node in range(4096)}
  )
  triple_guard = {node for node in range(24) if node % 4 != 2}
  triple_expected = set(triple_seed)
  while True:
    successors = {}
    for start, end in triple_expected:
      successors.setdefault(start, set()).add(end)
    additions = {
      (a, d)
      for a, b in triple_expected if b in triple_guard
      for c in successors.get(b, ())
      for d in successors.get(c, ())
    } - triple_expected
    if not additions:
      break
    triple_expected.update(additions)
  assert (0, 5) in triple_expected
  assert len(triple_expected - triple_seed) <= len(triple_seed) // 10
  # These occurrences share relation, index AND prefix, but their segment
  # selections are independent. In particular, FULL offsets from one
  # occurrence must not be used with the HEAD view selected by another.
  shared_seed = {
    (0, 1), (0, 2), (0, 3), (1, 3), (1, 4), (2, 4),
    (2, 5), (3, 5), (3, 6), (4, 6), (4, 7), (5, 7),
  } | {(1000 + 2 * node, 1001 + 2 * node) for node in range(4096)}
  shared_expected = set(shared_seed)
  while True:
    successors = {}
    for start, end in shared_expected:
      successors.setdefault(start, set()).add(end)
    additions = {
      (b, c)
      for targets in successors.values() if targets & triple_guard
      for b in targets for c in targets
    } - shared_expected
    if not additions:
      break
    shared_expected.update(additions)
  assert (7, 1) in shared_expected
  assert len(shared_expected - shared_seed) <= len(shared_seed) // 10
  facts = tmp_path / "facts"
  facts.mkdir()
  rows = {
    "Items": sorted(items), "Matches": sorted(matches),
    "AllowedInput": [(value,) for value in sorted(allowed)], "Empty": [],
    "Seed": [(0,)], "Step": sorted(steps),
    "TripleSeed": sorted(triple_seed),
    "TripleGuard": [(value,) for value in sorted(triple_guard)],
    "SharedSeed": sorted(shared_seed),
  }
  for relation, tuples in rows.items():
    (facts / f"{relation}.csv").write_text("".join("\t".join(map(str, row)) + "\n" for row in tuples))
  filtered = {(key, value) for key, value in items if value in allowed}
  selected = {
    (key, value, tag) for key, value in filtered
    for match_key, tag in matches if match_key == key
  }
  reach = filtered | {(key, 2) for key, value in filtered if value == 0 and key in allowed}
  expected = {
    "Items": items, "Original": items, "AllowedInput": {(value,) for value in allowed},
    "SharedSeed": shared_seed, "SharedPath": shared_expected,
    "Allowed": {(value,) for value in allowed}, "Empty": set(),
    "Filtered": filtered, "Selected": selected, "Ordered": selected,
    "GuardFirst": selected, "Rejected": set(), "Reach": reach,
    "Eligible": {(0,), (1,), (2,), (4,)},
    "Live": {(key, value) for key, value in items if value in {0, 1, 2, 4}},
    "TripleSeed": triple_seed, "TriplePath": triple_expected,
  }
  worker = r"""
import ctypes as c
import os
import sys
from pathlib import Path

lib = c.CDLL(sys.argv[1], mode=c.RTLD_GLOBAL)
for name, arguments in {
    "init": [], "load_all": [c.c_char_p], "prepare": [],
    "run": [c.c_ulonglong], "shutdown": [],
    "export_tsv": [c.c_char_p, c.c_char_p],
    "get_size": [c.c_char_p, c.POINTER(c.c_ulonglong)],
}.items():
    function = getattr(lib, "srdatalog_" + name)
    function.argtypes = arguments
    function.restype = c.c_int
assert lib.srdatalog_init() == 0
try:
    assert lib.srdatalog_load_all(os.fsencode(sys.argv[2])) == 0
    assert lib.srdatalog_prepare() == 0
    assert lib.srdatalog_run(0) == 0
    for name in sys.argv[4:]:
        path = Path(sys.argv[3]) / (name + ".tsv")
        assert lib.srdatalog_export_tsv(name.encode(), os.fsencode(path)) == 0
        size = c.c_ulonglong()
        assert lib.srdatalog_get_size(name.encode(), c.byref(size)) == 0
        assert size.value == len(path.read_text().splitlines()), name
finally:
    assert lib.srdatalog_shutdown() == 0
"""
  completed = subprocess.run(
    [sys.executable, "-c", worker, str(compiled.artifact), str(facts), str(tmp_path), *expected],
    cwd=tmp_path, capture_output=True, text=True, timeout=120,
  )
  assert completed.returncode == 0, completed.stdout + completed.stderr
  for relation, tuples in expected.items():
    observed = [
      tuple(map(int, line.split("\t")))
      for line in (tmp_path / f"{relation}.tsv").read_text().splitlines()
    ]
    assert set(observed) == tuples, relation
    assert len(observed) == len(tuples), relation
