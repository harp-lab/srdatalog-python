'''Batchfile codegen tests.

The `generate_prelude` test byte-matches mhk's expected C++ verbatim
(whitespace-normalized). `generate_runner` tests verify the struct
scaffold (type aliases, LaunchParams fields, phase method decls, and
the histogram kernel when BalancedScan is the root). A top-level
integration test runs `compile_to_mir(build_andersen())` through
`generate_batchfile` and asserts one JitRunner per pipeline.
'''
import sys
from pathlib import Path


import srdatalog.mir.types as m
from srdatalog.hir.types import Version
from srdatalog.codegen.schema import FactDefinition, Pragma, SchemaDefinition
from srdatalog.codegen.batchfile import (
  generate_prelude,
  generate_runner,
  generate_pipeline,
  generate_batchfile,
)
from srdatalog.codegen.helpers import CodeGenContext


def _nows(s: str) -> str:
  return s.replace(" ", "").replace("\n", "")


# -----------------------------------------------------------------------------
# Prelude
# -----------------------------------------------------------------------------

def test_generate_prelude_andersen():
  schema = SchemaDefinition(facts=[
    FactDefinition("AddressOf", [int, int], pragmas={"semiring": "NoProvenance"}),
    FactDefinition("Assign", [int, int], pragmas={"semiring": "NoProvenance"}),
    FactDefinition("Load", [int, int], pragmas={"semiring": "NoProvenance"}),
    FactDefinition("Store", [int, int], pragmas={"semiring": "NoProvenance"}),
    FactDefinition("PointsTo", [int, int], pragmas={"semiring": "NoProvenance"}),
  ])
  actual = generate_prelude(schema, "Andersen")
  expected = '''
// JIT-Generated Rule Kernel Batch
// This file is auto-generated - do not edit
#define SRDATALOG_JIT_BATCH  // Guard: exclude host-side helpers from JIT compilation
// Main project header - includes all necessary boost/hana, etc.
#include "srdatalog.h"
#include <cstdint>
#include <cooperative_groups.h>
// JIT-specific headers (relative to generalized_datalog/)
#include "gpu/device_sorted_array_index.h"
#include "gpu/runtime/output_context.h"
#include "gpu/runtime/jit/intersect_handles.h"
#include "gpu/runtime/jit/jit_executor.h"
#include "gpu/runtime/jit/materialized_join.h"
#include "gpu/runtime/jit/ws_infrastructure.h"  // WCOJTask, WCOJTaskQueue, ChunkedOutputContext
#include "gpu/runtime/query.h"  // For DeviceRelationType
namespace cg = cooperative_groups;
// Make JIT helpers visible without full namespace qualification
using SRDatalog::GPU::JIT::intersect_handles;

// Project-specific schema definitions (inlined)
using namespace SRDatalog::AST::Literals;  // For _s string literal
using AddressOf = SRDatalog::AST::RelationSchema<decltype("AddressOf"_s), NoProvenance, std::tuple<int, int>>;
using Assign = SRDatalog::AST::RelationSchema<decltype("Assign"_s), NoProvenance, std::tuple<int, int>>;
using Load = SRDatalog::AST::RelationSchema<decltype("Load"_s), NoProvenance, std::tuple<int, int>>;
using Store = SRDatalog::AST::RelationSchema<decltype("Store"_s), NoProvenance, std::tuple<int, int>>;
using PointsTo = SRDatalog::AST::RelationSchema<decltype("PointsTo"_s), NoProvenance, std::tuple<int, int>>;
using AndersenFixpoint_DB_Blueprint = SRDatalog::AST::Database<AddressOf, Assign, Load, Store, PointsTo>;
using AndersenFixpoint_DB_DeviceDB = SRDatalog::AST::SemiNaiveDatabase<AndersenFixpoint_DB_Blueprint, SRDatalog::GPU::DeviceRelationType>;
'''
  assert _nows(actual) == _nows(expected), (
    f"prelude mismatch:\n--expected--\n{expected}\n--actual--\n{actual}"
  )


# -----------------------------------------------------------------------------
# generate_pipeline — state mutation on first-op dispatch
# -----------------------------------------------------------------------------

def _cs(rel, ver, idx, prefix=()):
  return m.ColumnSource(rel_name=rel, version=ver, index=idx,
                        prefix_vars=list(prefix))


def test_pipeline_cartesian_root_sets_bound_vars():
  ep = m.ExecutePipeline(
    pipeline=[
      m.CartesianJoin(
        vars=["x", "y"],
        sources=[
          _cs("PointsTo", Version.DELTA, [0, 1], prefix=("z",)),
          _cs("Assign", Version.FULL, [1, 0], prefix=("z",)),
        ],
        var_from_source=[["x"], ["y"]],
      ),
      m.InsertInto(rel_name="PointsTo", version=Version.NEW,
                   vars=["y", "x"], index=[0, 1]),
    ],
    source_specs=[
      _cs("PointsTo", Version.DELTA, [0, 1]),
      _cs("Assign", Version.FULL, [1, 0]),
    ],
    dest_specs=[m.InsertInto(rel_name="PointsTo", version=Version.NEW,
                              vars=["y", "x"], index=[0, 1])],
    rule_name="Assign_D0_cart",
  )
  ctx = CodeGenContext(output_name="output_ctx", is_counting=True, is_jit_mode=True)
  out = generate_pipeline(ep, ctx)
  assert ctx.inside_cartesian_join is True
  assert ctx.cartesian_bound_vars == ["x", "y"]
  assert "Pipeline calls for step Assign_D0_cart" in out


def test_pipeline_balanced_scan_root_sets_bound_vars():
  ep = m.ExecutePipeline(
    pipeline=[
      m.BalancedScan(
        group_var="z",
        source1=_cs("VarPointsTo", Version.DELTA, [0, 1]),
        source2=_cs("Assign", Version.FULL, [1, 0]),
        vars1=["x"],
        vars2=["y"],
      ),
    ],
    source_specs=[
      _cs("VarPointsTo", Version.DELTA, [0, 1]),
      _cs("Assign", Version.FULL, [1, 0]),
    ],
    dest_specs=[],
    rule_name="Bal",
  )
  ctx = CodeGenContext(output_name="output_ctx", is_counting=True, is_jit_mode=True)
  generate_pipeline(ep, ctx)
  assert ctx.inside_cartesian_join is True
  assert ctx.cartesian_bound_vars == ["z", "x", "y"]


# -----------------------------------------------------------------------------
# generate_runner — struct scaffold
# -----------------------------------------------------------------------------

def _andersen_base_pipeline() -> m.ExecutePipeline:
  '''Simple Scan-rooted base pipeline: PointsTo(y,x) <- AddressOf(y,x).'''
  scan = m.Scan(vars=["y", "x"], rel_name="AddressOf",
                version=Version.FULL, index=[0, 1])
  insert = m.InsertInto(rel_name="PointsTo", version=Version.NEW,
                         vars=["y", "x"], index=[0, 1])
  return m.ExecutePipeline(
    pipeline=[scan, insert],
    source_specs=[scan],
    dest_specs=[insert],
    rule_name="Base",
  )


def test_generate_runner_struct_scaffold():
  ep = _andersen_base_pipeline()
  full, skeleton = generate_runner(ep, "Andersen")
  assert skeleton == ""
  assert "struct JitRunner_Base {" in full
  # Type aliases
  assert "using DB = AndersenFixpoint_DB_DeviceDB;" in full
  assert "using FirstSchema = AddressOf;" in full
  assert "using DestSchema = PointsTo;" in full
  assert "using SR = NoProvenance;" in full
  assert "using ViewType = typename IndexType::NodeView;" in full
  # Arity constants
  assert "static constexpr std::size_t OutputArity_0 = 2;" in full
  assert "static constexpr std::size_t OutputArity = OutputArity_0" in full
  assert "static constexpr std::size_t NumSources = 1;" in full
  # Kernels
  assert "static __global__ void __launch_bounds__(kBlockSize) kernel_count" in full
  assert "static __global__ void __launch_bounds__(kBlockSize) kernel_materialize" in full
  # LaunchParams fields
  assert "struct LaunchParams {" in full
  assert "std::vector<ViewType> views_vec;" in full
  assert "uint32_t old_size_0 = 0;" in full
  # Phase-method forward decls (no duplication)
  assert full.count("static LaunchParams setup(DB& db, uint32_t iteration") == 1
  assert "static void launch_count(LaunchParams& p, GPU_STREAM_T stream = 0);" in full
  assert "static uint32_t scan_and_resize(DB& db, LaunchParams& p, GPU_STREAM_T stream = 0);" in full
  assert "static void scan_only(LaunchParams& p, GPU_STREAM_T stream = 0);" in full
  assert "static uint32_t read_total(LaunchParams& p);" in full
  assert "static void launch_materialize(DB& db, LaunchParams& p, uint32_t total_count" in full
  assert "static void execute(DB& db, uint32_t iteration);" in full


def test_generate_runner_no_dest_means_void_dest_schema():
  scan = m.Scan(vars=["x"], rel_name="R", version=Version.FULL, index=[0])
  ep = m.ExecutePipeline(
    pipeline=[scan],
    source_specs=[scan],
    dest_specs=[],
    rule_name="NoDest",
  )
  full, _ = generate_runner(ep, "P")
  assert "using DestSchema = void;" in full
  # With no dests, no old_size_ field should appear
  assert "old_size_0" not in full


def test_generate_runner_balanced_scan_emits_histogram_kernel():
  bs = m.BalancedScan(
    group_var="z",
    source1=_cs("VarPointsTo", Version.DELTA, [0, 1]),
    source2=_cs("Assign", Version.FULL, [1, 0]),
    vars1=["x"],
    vars2=["y"],
  )
  insert = m.InsertInto(rel_name="PointsTo", version=Version.NEW,
                         vars=["y", "x"], index=[0, 1])
  ep = m.ExecutePipeline(
    pipeline=[bs, insert],
    source_specs=[bs.source1, bs.source2],
    dest_specs=[insert],
    rule_name="BalRun",
  )
  full, _ = generate_runner(ep, "Andersen")
  assert "kernel_histogram" in full
  # kernel_count signature takes the balanced params
  assert "prefix_fanouts" in full
  assert "total_balanced_work" in full
  # LaunchParams gets the balanced state
  assert "fanouts{0};" in full
  assert "prefix_fanouts{0};" in full


def test_generate_runner_work_stealing_marker():
  ep = _andersen_base_pipeline()
  ep.work_stealing = True
  full, _ = generate_runner(ep, "Andersen")
  assert "TODO: Implement work-stealing logic" in full


# -----------------------------------------------------------------------------
# Top-level generate_batchfile — drive compile_to_mir output
# -----------------------------------------------------------------------------

def test_generate_batchfile_andersen_end_to_end():
  from test_integration_andersen import build_andersen
  from srdatalog.hir import compile_to_mir
  mir = compile_to_mir(build_andersen())
  schema = SchemaDefinition(facts=[
    FactDefinition("AddressOf", [int, int]),
    FactDefinition("Assign", [int, int]),
    FactDefinition("Load", [int, int]),
    FactDefinition("Store", [int, int]),
    FactDefinition("PointsTo", [int, int]),
  ])
  out = generate_batchfile(mir, schema, "Andersen")

  # Prelude wired
  assert "#define SRDATALOG_JIT_BATCH" in out
  assert "using AndersenFixpoint_DB_Blueprint =" in out

  # One JitRunner per ExecutePipeline in program.steps
  from srdatalog.codegen.batchfile import _collect_pipelines
  pipelines = _collect_pipelines(mir)
  assert len(pipelines) >= 1
  for ep in pipelines:
    assert f"struct JitRunner_{ep.rule_name} {{" in out


if __name__ == "__main__":
  import inspect
  this = sys.modules[__name__]
  passed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if name.startswith("test_"):
      fn()
      passed += 1
  print(f"{passed} passed")
