'''JIT batch-file generation.

Ported from `generate_batch_files_no_template.py` on python-api-notemplate.
Drives a `mir_types.Program` through the runner-struct skeleton: per
ExecutePipeline, emit a `JitRunner_<rule_name>` struct wrapping the
kernel_count / kernel_materialize / phase-decomposed setup functions.

Much of the original is a scaffold — `_generate_body` returns the literal
string `"unimplemented\n"`, and `generate_pipeline` threads codegen state
but dispatches to commented-out handlers for the first op. This port
preserves that posture verbatim; filling in the actual C++ body for
ColumnJoin / CartesianJoin / Scan / Negation / Aggregate / Filter /
ConstantBind / BalancedScan roots is follow-up work to match what
`jit_kernel.nim` emits today.

Also fixes one bug in the original: mhk's `generate_runner` duplicated
the `setup()` forward declaration twice in the phase-method block; the
port drops the duplicate.
'''
from __future__ import annotations
from typing import Tuple

import srdatalog.mir.types as m
from srdatalog.hir_types import Version
from srdatalog.codegen.schema import SchemaDefinition
from srdatalog.codegen.helpers import (
  CodeGenContext,
  assign_handles,
  spec_key,
  find_source_idx,
  emit_view_declarations,
)


# -----------------------------------------------------------------------------
# Prelude
# -----------------------------------------------------------------------------

PRELUDE = """
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
"""


def generate_prelude(schema: SchemaDefinition, name: str) -> str:
  '''Batch-file prelude: static C++ includes plus the schema-driven
  `using` aliases and device-DB typedefs.'''
  return PRELUDE + schema.get_batch_prelude(name)


# -----------------------------------------------------------------------------
# Pipeline body emission (skeleton — see module docstring)
# -----------------------------------------------------------------------------

def _generate_body(rest_ops: list[m.MirNode], ctx: CodeGenContext) -> str:
  '''Placeholder — per-op emitters for ColumnJoin / Filter / Negation / etc.
  go here once the root dispatch in `generate_pipeline` is wired up.
  '''
  return "unimplemented\n"


def generate_pipeline(pipeline: m.ExecutePipeline, ctx: CodeGenContext) -> str:
  '''Emit the kernel body for one pipeline. Dispatches on the first op's
  shape to set up the outer iteration, then delegates the rest to
  `_generate_body`.

  Matches mhk's original: this is mostly state mutation + a few
  comments; the actual per-op handlers are still commented-out TODOs.
  '''
  assert isinstance(pipeline, m.ExecutePipeline)
  if not pipeline.pipeline:
    return ""

  res = f"// Pipeline calls for step {pipeline.rule_name}\n"
  res += emit_view_declarations(pipeline, ctx)

  first_op = pipeline.pipeline[0]
  rest_ops = pipeline.pipeline[1:]

  # ColumnJoin root: record per-source handle vars in ctx.
  if isinstance(first_op, m.ColumnJoin) and len(first_op.sources) > 1:
    root_var_name = "TODO"  # mhk's placeholder — cjVarName equivalent
    for src in first_op.sources:
      if not isinstance(src, (m.ColumnSource, m.Scan, m.Negation, m.Aggregate)):
        continue
      state_key = spec_key(src.rel_name, src.version, list(src.index)) + "_" + root_var_name
      handle_var = f"h_{src.rel_name}_{getattr(src, 'handle_start', 0)}_root"
      ctx.view_vars[state_key] = handle_var

  # CartesianJoin root: mark codegen as inside-cartesian and record
  # bound vars emitted by the join.
  elif isinstance(first_op, m.CartesianJoin):
    ctx.inside_cartesian_join = True
    for vfc in first_op.var_from_source:
      for var_name in vfc:
        ctx.cartesian_bound_vars.append(var_name)

  # BalancedScan root: bind the group var plus the per-source output vars.
  elif isinstance(first_op, m.BalancedScan):
    ctx.inside_cartesian_join = True
    ctx.cartesian_bound_vars.append(first_op.group_var)
    for v in first_op.vars1:
      ctx.cartesian_bound_vars.append(v)
    for v in first_op.vars2:
      ctx.cartesian_bound_vars.append(v)

  res += _generate_body(rest_ops, ctx)
  res += "//First operations unimplemented \n"
  # Intentional parity with mhk: the per-op first-root handlers
  # (jitRootColumnJoin / jitRootCartesianJoin / jitRootScan /
  # jitNegation / jitAggregate / jitFilter / jitConstantBind /
  # jitRootBalancedScan / jitInsertInto) remain to be ported from
  # jit_kernel.nim. Leaving the dispatch comment in source so the
  # work is easy to slot in.
  return res


# -----------------------------------------------------------------------------
# JitRunner struct emission
# -----------------------------------------------------------------------------

def _insert_source_specs_count(num_sources: int) -> str:
  return f"static constexpr std::size_t NumSources = {num_sources};\n\n"


def _dest_arity(dest: m.MirNode) -> int:
  '''Columns written by an InsertInto (or arity of a leaf dest).'''
  if isinstance(dest, m.InsertInto):
    return len(dest.index)
  if hasattr(dest, "index"):
    return len(dest.index)
  return 0


def _first_dest_rel(pipeline: m.ExecutePipeline) -> str:
  if not pipeline.dest_specs:
    return "void"
  d = pipeline.dest_specs[0]
  return getattr(d, "rel_name", "void") or "void"


def _first_source_rel_and_version(pipeline: m.ExecutePipeline) -> Tuple[str, Version]:
  src = pipeline.source_specs[0]
  return src.rel_name, src.version


def _find_balanced_scan(pipeline: m.ExecutePipeline) -> m.BalancedScan | None:
  '''If the pipeline root is a BalancedScan, return it — otherwise None.'''
  if pipeline.pipeline and isinstance(pipeline.pipeline[0], m.BalancedScan):
    return pipeline.pipeline[0]
  return None


def generate_runner(
  pipeline: m.ExecutePipeline, program_name: str,
) -> tuple[str, str]:
  '''Emit `struct JitRunner_<rule_name> { ... }` for one pipeline.

  Returns `(full, skeleton)`. `full` is the complete struct; `skeleton`
  is reserved for a future header-only forward-decl variant (empty for
  now, matching mhk's scaffolding).
  '''
  assert isinstance(pipeline, m.ExecutePipeline)

  provenance = "NoProvenance"
  first_rel, first_ver = _first_source_rel_and_version(pipeline)
  dest_arities = [_dest_arity(d) for d in pipeline.dest_specs]
  first_dest = _first_dest_rel(pipeline)

  # -- Type aliases --
  full = f'''
struct JitRunner_{pipeline.rule_name} {{
  // Type aliases - all concrete, resolved at Nim JIT time
  using DB = {program_name}Fixpoint_DB_DeviceDB;
  using FirstSchema = {first_rel};
  using DestSchema = {first_dest};
  using SR = {provenance};
  using ValueType = typename FirstSchema::intern_value_type;
  using RelType = std::decay_t<decltype(get_relation_by_schema<FirstSchema, {first_ver.name}>(std::declval<DB&>()))>;
  using IndexType = typename RelType::IndexTypeInst;
  using ViewType = typename IndexType::NodeView;
  static constexpr auto Layout = SRDatalog::GPU::StorageLayout::SoA;
  static constexpr int kBlockSize = 256;
  static constexpr int kGroupSize = 32;
  '''
  for i, arity in enumerate(dest_arities):
    full += f"static constexpr std::size_t OutputArity_{i} = {arity};\n"
    if i == 0:
      full += "static constexpr std::size_t OutputArity = OutputArity_0; // Legacy alias\n"
  full += _insert_source_specs_count(len(pipeline.source_specs))

  # -- kernel_count --
  bs = _find_balanced_scan(pipeline)
  balanced_insertion = ""
  if bs is not None:
    balanced_insertion = (
      "\n    const uint64_t* __restrict__ prefix_fanouts,"
      "\n    const uint32_t* __restrict__ deg2_arr,"
      "\n    uint64_t total_balanced_work,"
      "\n    uint64_t work_per_warp,"
    )

  full += f'''
  static __global__ void __launch_bounds__(kBlockSize) kernel_count(
      const ViewType* __restrict__ views,
      const ValueType* __restrict__ root_unique_values,
      uint32_t num_unique_root_keys,
      uint32_t num_root_keys,{balanced_insertion}
      uint32_t* __restrict__ thread_counts) {{
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<kGroupSize>(block);
    auto single_thread = cg::tiled_partition<1>(block);  // For per-thread search inside Cartesian
    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t warp_id = thread_id / kGroupSize;
    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;
    uint32_t num_threads = num_warps;  // Alias for scalar mode (kGroupSize=1)
'''

  out_arity = "OutputArity_0" if dest_arities else "0"
  full += f'''
    using OutputCtx = SRDatalog::GPU::OutputContext<ValueType, SR, true, Layout, {out_arity}>;
    OutputCtx output_ctx{{nullptr, nullptr, 0, 0}};'''

  mutable_pipeline = assign_handles(pipeline)
  ctx = CodeGenContext(output_name="output_ctx", is_counting=True, is_jit_mode=True)
  ctx.set_output_vars(pipeline.dest_specs)
  full += generate_pipeline(mutable_pipeline, ctx)
  full += "thread_counts[thread_id] = output_ctx.count();\n}\n\n"

  # -- kernel_histogram (BalancedScan only) --
  if bs is not None:
    spec_key_1 = bs.source1.rel_name + "_" + "".join(str(a) for a in bs.source1.index)
    spec_key_2 = bs.source2.rel_name + "_" + "".join(str(a) for a in bs.source2.index)
    src_1_idx = find_source_idx(
      pipeline.source_specs, bs.source1.rel_name, list(bs.source1.index), None,
    )
    src_2_idx = find_source_idx(
      pipeline.source_specs, bs.source2.rel_name, list(bs.source2.index), None,
    )
    if src_1_idx < 0 or src_2_idx < 0:
      print(f"[WARNING] Balanced source not found in BalancedScan.body: src1={spec_key_1} src2={spec_key_2}")
    full += f'''
      // Histogram kernel for balanced work partitioning
      // Computes fanout[key] = deg1(key) * deg2(key) for prefix-sum
      static __global__ void __launch_bounds__(kBlockSize) kernel_histogram(
        const ViewType* __restrict__ views,
        const ValueType* __restrict__ root_unique_values,
        uint32_t num_unique_root_keys,
        uint64_t* __restrict__ fanouts,
        uint32_t* __restrict__ deg1_arr,
        uint32_t* __restrict__ deg2_arr)
      {{
        auto block = cg::this_thread_block();
        auto tile = cg::tiled_partition<kGroupSize>(block);
        auto single_thread = cg::tiled_partition<1>(block);  // For per-thread search inside Cartesian
        uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
        uint32_t warp_id = thread_id / kGroupSize;
        uint32_t lane_id = thread_id % kGroupSize;
        uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;
        uint32_t num_threads = num_warps;  // Alias for scalar mode (kGroupSize=1)
        using HandleType = ViewType::NodeHandle;
        // Get view for both balanced sources
        auto& view1 = views[{src_1_idx}];  // {spec_key_1}
        auto& view2 = views[{src_2_idx}];  // {spec_key_2}
        for (uint32_t key_idx = warp_id; key_idx < num_unique_root_keys; key_idx += num_warps) {{
          auto key = root_unique_values[key_idx];
          auto h1 = HandleType(0, view1.num_rows_, 0).prefix(key, tile, view1);
          auto h2 = HandleType(0, view2.num_rows_, 0).prefix(key, tile, view2);
          uint32_t d1 = h1.degree();
          uint32_t d2 = h2.degree();
          if (lane_id == 0) {{
            fanouts[key_idx] = static_cast<uint64_t>(d1) * static_cast<uint64_t>(d2);
            deg1_arr[key_idx] = d1;
            deg2_arr[key_idx] = d2;
          }}
        }}
      }}
'''

  # -- kernel_materialize --
  full += '''
  // Non-template kernel_materialize (concrete ViewType)
  static __global__ void __launch_bounds__(kBlockSize) kernel_materialize(
    const ViewType* __restrict__ views,
    const ValueType* __restrict__ root_unique_values,
    uint32_t num_unique_root_keys,
    uint32_t num_root_keys,'''
  if bs is not None:
    full += '''
      const uint64_t* __restrict__ prefix_fanouts,
      const uint32_t* __restrict__ deg2_arr,
      uint64_t total_balanced_work,
      uint64_t work_per_warp,'''
  full += "\n    const uint32_t* __restrict__ thread_offsets,\n"
  for i, _ in enumerate(pipeline.dest_specs):
    full += (
      f"      ValueType* __restrict__ output_data_{i},\n"
      f"      semiring_value_t<SR>* __restrict__ output_prov_{i},\n"
      f"      std::size_t output_stride_{i},\n"
      f"      uint32_t old_size_{i}\n"
    )
    if i == len(pipeline.dest_specs) - 1:
      full += ")\n{\n"
    else:
      full += ",\n"

  full += '''
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<kGroupSize>(block);
    auto single_thread = cg::tiled_partition<1>(block);  // For per-thread search inside Cartesian
    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t warp_id = thread_id / kGroupSize;
    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;
    uint32_t num_threads = num_warps;  // Alias for scalar mode (kGroupSize=1)
    uint32_t thread_offset = thread_offsets[thread_id];
    '''

  ctx = CodeGenContext(output_name="output_ctx", is_counting=False, is_jit_mode=True)
  for i, dest_spec in enumerate(pipeline.dest_specs):
    output_var = f"output_ctx_{i}"
    arity = dest_arities[i]
    full += f'''
    using OutputCtx_{i} = SRDatalog::GPU::OutputContext<ValueType, SR, false, Layout, OutputArity_{i}>;
    OutputCtx_{i} {output_var}{{output_data_{i}, output_prov_{i}, output_stride_{i}, old_size_{i} + thread_offset}};
    '''
    rel = getattr(dest_spec, "rel_name", "")
    if rel:
      ctx.output_vars[rel] = output_var
    if i == 0:
      ctx.output_name = output_var
  full += generate_pipeline(mutable_pipeline, ctx)

  if pipeline.work_stealing:
    full += "    // TODO: Implement work-stealing logic (jit_kernel.nim 1084-1589)\n"

  # -- LaunchParams struct --
  full += '''
  // State carried between decomposed phases
  struct LaunchParams {
    std::vector<ViewType> views_vec;
    SRDatalog::GPU::DeviceArray<ViewType> d_views{0};
    SRDatalog::GPU::DeviceArray<uint32_t> thread_counts{0};
    uint32_t* thread_counts_ptr = nullptr;
    uint32_t num_unique_root_keys = 0;
    uint32_t num_root_keys = 0;
    const ValueType* root_unique_values_ptr = nullptr;
    uint32_t num_blocks = 0;
    uint32_t num_threads = 0;'''
  if bs is not None:
    full += '''
      SRDatalog::GPU::DeviceArray<uint64_t> fanouts{0};
      SRDatalog::GPU::DeviceArray<uint32_t> deg1_arr{0};
      SRDatalog::GPU::DeviceArray<uint32_t> deg2_arr{0};
      SRDatalog::GPU::DeviceArray<uint64_t> prefix_fanouts{0};
      uint64_t total_balanced_work = 0;
      uint64_t work_per_warp = 0;'''
  for i, _ in enumerate(pipeline.dest_specs):
    full += f"\n    uint32_t old_size_{i} = 0;"
  if pipeline.work_stealing:
    full += ""  # jitWSLaunchParamsFields() TODO
  full += "\n  };\n\n"

  # -- Phase method declarations --
  full += '''  // Phase-decomposed methods for stream-parallel execution
  static LaunchParams setup(DB& db, uint32_t iteration, GPU_STREAM_T stream = 0);
  static void launch_count(LaunchParams& p, GPU_STREAM_T stream = 0);
  static uint32_t scan_and_resize(DB& db, LaunchParams& p, GPU_STREAM_T stream = 0);
  static void scan_only(LaunchParams& p, GPU_STREAM_T stream = 0);
  static uint32_t read_total(LaunchParams& p);
  static void launch_materialize(DB& db, LaunchParams& p, uint32_t total_count, GPU_STREAM_T stream = 0);

  static void execute(DB& db, uint32_t iteration);
  };
'''

  skeleton = ""
  return full, skeleton


# -----------------------------------------------------------------------------
# Top-level batchfile generator
# -----------------------------------------------------------------------------

def _collect_pipelines(program: m.Program) -> list[m.ExecutePipeline]:
  '''Walk a Program's FixpointPlan / ParallelGroup steps and flatten out
  every ExecutePipeline that gets a JitRunner. Reconstruct and
  stand-alone maintenance steps contribute no runners.
  '''
  pipelines: list[m.ExecutePipeline] = []
  for node, _is_rec in program.steps:
    if isinstance(node, m.FixpointPlan):
      for instr in node.instructions:
        if isinstance(instr, m.ExecutePipeline):
          pipelines.append(instr)
        elif isinstance(instr, m.ParallelGroup):
          for op in instr.ops:
            if isinstance(op, m.ExecutePipeline):
              pipelines.append(op)
  return pipelines


def generate_batchfile(
  program: m.Program, schema: SchemaDefinition, name: str,
) -> str:
  '''Top-level batchfile generator. Consumes our `mir_types.Program`
  (produced by `compile_to_mir`) plus the schema + program name.
  Emits the prelude followed by one `JitRunner_...` per pipeline.
  '''
  result = generate_prelude(schema, name)
  for ep in _collect_pipelines(program):
    full, _skel = generate_runner(ep, name)
    result += full
  return result
