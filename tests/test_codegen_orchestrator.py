'''Orchestrator-level C++ codegen tests.

The non-recursive fixture is taken directly from mhk's
`TestNonrecursive.test_block` (Andersen's Base rule); built with our
`mir_types` nodes, normalized-whitespace-compared against mhk's expected
C++ output. Proves the shared-IR port is byte-equivalent for the easy
shape.

The recursive fixture runs our real `compile_to_mir(build_andersen())`
through the orchestrator and asserts the emitted function bodies contain
the key phase markers (setup/count/scan_all/materialize plus the tail
maintenance). Our MIR structures `PostStratumReconstructInternCols` as
its own step, so the recursive step and the reconstruct step emit
separately — this test covers both.
'''

import sys

import srdatalog.mir.types as m
from srdatalog.codegen.orchestrator import (
  SRDatalogProgram,
  generate_dest_stream_map,
  generate_step,
  generate_step_body,
  get_canonical_specs,
)
from srdatalog.codegen.schema import FactDefinition, Pragma, SchemaDefinition
from srdatalog.hir.types import Version


def _nows(s: str) -> str:
  return s.replace(" ", "").replace("\n", "")


# =============================================================================
# Helpers: get_canonical_specs, generate_dest_stream_map
# =============================================================================


def test_get_canonical_specs_empty():
  assert get_canonical_specs([]) == []


def test_get_canonical_specs_compute_delta_index():
  instrs = [m.ComputeDeltaIndex(rel_name="PointsTo", canonical_index=[0, 1])]
  assert get_canonical_specs(instrs) == [("PointsTo", [0, 1])]


def test_get_canonical_specs_deduplicates():
  instrs = [
    m.ComputeDeltaIndex(rel_name="PointsTo", canonical_index=[0, 1]),
    m.MergeIndex(rel_name="PointsTo", index=[1, 0]),
  ]
  # First occurrence wins
  assert get_canonical_specs(instrs) == [("PointsTo", [0, 1])]


def test_get_canonical_specs_multi_rel():
  instrs = [
    m.ComputeDeltaIndex(rel_name="PointsTo", canonical_index=[0, 1]),
    m.MergeIndex(rel_name="Store", index=[0, 1]),
  ]
  assert get_canonical_specs(instrs) == [("PointsTo", [0, 1]), ("Store", [0, 1])]


def test_get_canonical_specs_ignores_irrelevant():
  instrs = [
    m.RebuildIndex(rel_name="PointsTo", version=Version.NEW, index=[0, 1]),
    m.ClearRelation(rel_name="PointsTo", version=Version.NEW),
    m.CheckSize(rel_name="PointsTo", version=Version.NEW),
  ]
  assert get_canonical_specs(instrs) == []


def test_generate_dest_stream_map_one_dest_per_pipeline():
  pipelines = [
    m.ExecutePipeline(
      pipeline=[],
      source_specs=[],
      dest_specs=[
        m.InsertInto(rel_name="PointsTo", version=Version.NEW, vars=["y", "x"], index=[0, 1])
      ],
      rule_name="Assign_D0",
    ),
    m.ExecutePipeline(
      pipeline=[],
      source_specs=[],
      dest_specs=[
        m.InsertInto(rel_name="PointsTo", version=Version.NEW, vars=["y", "w"], index=[0, 1])
      ],
      rule_name="Load_D0",
    ),
  ]
  assert generate_dest_stream_map(pipelines) == {"PointsTo": [0, 1]}


# =============================================================================
# Non-recursive fixture — mhk's TestNonrecursive, rebuilt on our types
# =============================================================================


def _nonrecursive_andersen_base_step() -> m.FixpointPlan:
  '''Matches `TestNonrecursive.setUp` fixture in test_mir_commands_notemplate.py:
  Base pipeline + the full maintenance tail.'''
  pipeline = m.ExecutePipeline(
    pipeline=[
      m.InsertInto(rel_name="PointsTo", version=Version.NEW, vars=["y", "x"], index=[1, 0]),
      m.Scan(vars=["y", "x"], rel_name="AddressOf", version=Version.FULL, index=[0, 1]),
    ],
    source_specs=[
      m.Scan(vars=["y", "x"], rel_name="AddressOf", version=Version.FULL, index=[0, 1]),
    ],
    dest_specs=[
      m.InsertInto(rel_name="PointsTo", version=Version.NEW, vars=["y", "x"], index=[1, 0]),
    ],
    rule_name="Base",
  )
  return m.FixpointPlan(
    instructions=[
      pipeline,
      m.RebuildIndex(rel_name="PointsTo", version=Version.NEW, index=[1, 0]),
      m.CheckSize(rel_name="PointsTo", version=Version.NEW),
      m.ComputeDeltaIndex(rel_name="PointsTo", canonical_index=[1, 0]),
      m.ClearRelation(rel_name="PointsTo", version=Version.NEW),
      m.MergeIndex(rel_name="PointsTo", index=[1, 0]),
      m.RebuildIndexFromIndex(
        rel_name="PointsTo", source_index=[1, 0], target_index=[0, 1], version=Version.DELTA
      ),
      m.MergeIndex(rel_name="PointsTo", index=[0, 1]),
    ]
  )


def test_nonrecursive_step_matches_mhk_expected():
  '''Byte-match (whitespace-normalized) against mhk's expected step_0.'''
  plan = _nonrecursive_andersen_base_step()
  actual = generate_step(0, plan, is_recursive=False)

  expected = '''
  template <typename DB>
  static void step_0(DB& db, std::size_t max_iterations) {
    using PointsTo_canonical_spec_t = SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>;
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<AddressOf, std::integer_sequence<int, 0, 1>, FULL_VER>>(db, 0);
    mir_helpers::create_index_fn<SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 1, 0>, NEW_VER>>(db, 0);
    JitRunner_Base::execute(db, 0);
    SRDatalog::GPU::mir_helpers::rebuild_index_fn<SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 1, 0>, NEW_VER>>(db);
    SRDatalog::GPU::mir_helpers::check_size_fn<PointsTo, NEW_VER, PointsTo_canonical_spec_t>(db);
    SRDatalog::GPU::mir_helpers::compute_delta_index_fn<SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 1, 0>, NEW_VER>, SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>, SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>>(db);
    SRDatalog::GPU::mir_helpers::clear_relation_fn<PointsTo, NEW_VER>(db);
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 1, 0>, FULL_VER>>(db);
    SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 1, 0>, DELTA_VER>, SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, DELTA_VER>>(db);
    SRDatalog::GPU::mir_helpers::merge_index_fn<SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>>(db);
  }
    '''
  assert _nows(actual) == _nows(expected), f"mismatch:\nexpected:\n{expected}\nactual:\n{actual}"


def test_nonrecursive_step_body_directly():
  '''Same fixture, checked without the outer function wrapper.'''
  plan = _nonrecursive_andersen_base_step()
  body = generate_step_body(plan, is_recursive=False)
  assert "JitRunner_Base::execute" in body
  assert "PointsTo_canonical_spec_t" in body
  assert "AddressOf" in body
  assert "reconstruct_fn" not in body  # reconstruct is a separate step in our MIR


# =============================================================================
# Recursive fixture — drive real compile_to_mir(build_andersen())
# =============================================================================


def test_andersen_recursive_step_emits_four_phases():
  from test_integration_andersen import build_andersen

  from srdatalog.hir import compile_to_mir

  prog = compile_to_mir(build_andersen())

  # Find the recursive FixpointPlan step
  rec_step = next(
    (node, i)
    for i, (node, is_rec) in enumerate(prog.steps)
    if is_rec and isinstance(node, m.FixpointPlan)
  )
  rec_node, rec_idx = rec_step
  out = generate_step(rec_idx, rec_node, is_recursive=True)

  for marker in [
    "for (std::size_t iter = 0; iter < max_iterations; ++iter)",
    "static SRDatalog::GPU::StreamPool _stream_pool",
    "// Phase 1: Setup all rules",
    "// Phase 2b: Launch count kernels",
    "// Phase 3a: Scan shared buffers",
    "// Phase 3b: Single sync",
    "// Phase 3c: Resize once per unique dest",
    "// Phase 4: Launch all materialize kernels",
    "_stream_pool.wait_event(",  # RebuildIndex-level wait after the parallel group
    "rebuild_index_fn<",
    "compute_delta_index_fn<",
    "merge_index_fn<",
    "GPU_DEVICE_SYNCHRONIZE();",
  ]:
    assert marker in out, f"missing marker in recursive step:\n{marker}\n---\n{out}"


def test_andersen_reconstruct_step():
  '''The PostStratumReconstructInternCols step follows the fixpoint step
  in our MIR and emits its own step_N function.'''
  from test_integration_andersen import build_andersen

  from srdatalog.hir import compile_to_mir

  prog = compile_to_mir(build_andersen())

  recon_idx, (recon_node, _) = next(
    (i, (n, r))
    for i, (n, r) in enumerate(prog.steps)
    if isinstance(n, m.PostStratumReconstructInternCols)
  )
  out = generate_step(recon_idx, recon_node, is_recursive=False)
  assert "reconstruct_fn<" in out
  assert "GPU_DEVICE_SYNCHRONIZE()" in out


# =============================================================================
# SRDatalogProgram driver
# =============================================================================


def test_srdatalog_program_full_orchestrator():
  from test_integration_andersen import build_andersen

  from srdatalog.hir import compile_to_mir

  mir = compile_to_mir(build_andersen())

  schema = SchemaDefinition(
    facts=[
      FactDefinition("AddressOf", [int, int], pragmas={Pragma.INPUT: "ao.csv"}),
      FactDefinition("Assign", [int, int]),
      FactDefinition("Load", [int, int]),
      FactDefinition("Store", [int, int]),
      FactDefinition("PointsTo", [int, int]),
    ]
  )
  prog = SRDatalogProgram(name="Andersen", database=schema, program=mir)
  out = prog.generate_orchestrator(include_ffi=True)

  # Prelude wiring
  assert '#include "srdatalog.h"' in out
  assert "using namespace SRDatalog::mir::dsl" in out
  # One step function per MIR step
  n_steps = len(mir.steps)
  for i in range(n_steps):
    assert f"static void step_{i}(DB& db, std::size_t max_iterations)" in out
  # FFI header and load_data
  assert "typedef void *DBHandle;" in out
  assert "static void load_data(DB& db, std::string root_dir)" in out
  assert 'load_from_file<AddressOf>(db, root_dir + "/ao.csv")' in out


if __name__ == "__main__":
  import inspect

  this = sys.modules[__name__]
  passed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if name.startswith("test_"):
      fn()
      passed += 1
  print(f"{passed} passed")
