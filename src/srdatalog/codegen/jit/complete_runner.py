'''Complete runner struct emission.

Port of src/srdatalog/codegen/target_jit/jit_complete_runner.nim.

Emits `struct JitRunner_<rule>` — the per-rule launcher scaffold that
wraps count, materialize, and (optionally) fused __global__ kernels
with host-side setup / launch phase methods and `execute()` entry.

Entry point:
  gen_complete_runner(node, db_type_name, rel_index_types={}) -> (decl, full)

`full` is the complete struct + out-of-line phase method definitions
(goes into the JIT batch file). `decl` is a minimal forward-declaration
struct (goes into the main compile unit).

Scope: baseline non-recursive emit supporting:
  - ColumnJoin / ColumnSource sources with FULL_VER / DELTA_VER / NEW_VER
  - kernel_count + kernel_materialize + kernel_fused
  - setup / launch_count / scan_and_resize / scan_only / read_total
    / launch_materialize / launch_fused / read_fused_result
  - execute + execute_fused
  - Tiled Cartesian materialize body (when eligible)
  - count-only pipelines (ep.count): kernel_materialize discarded

Unsupported (raise NotImplementedError): work_stealing, block_group,
dedup_hash, balanced scan, fan-out, materialized pipelines.
'''

from __future__ import annotations

# Side-effect import: registers the Device2LevelIndex plugin. Without this,
# pluginViewCount falls back to the default plugin (view_count=1 for every
# version) for any relation declared with `index_type="...Device2LevelIndex"`,
# which silently undercounts NumSources and diverges from the Nim build.
# Nim gets this registration for free via its import-graph; we need to
# pin it here because `gen_complete_runner` is the consumer that cares.
import srdatalog.codegen.jit.indexes.two_level  # noqa: F401
import srdatalog.mir.types as m
from srdatalog.codegen.jit.emit_helpers import (
  assign_handle_positions,
  has_balanced_scan,
  has_tiled_cartesian_eligible,
)
from srdatalog.codegen.jit.materialized import is_materialized_pipeline
from srdatalog.codegen.jit.plugin import plugin_gen_host_view_setup, plugin_view_count
from srdatalog.codegen.jit.view_management import (
  compute_total_view_count,
  source_spec_key,
)

# Pure-template phase emitters now live in the dialect's runner module.
# Local aliases preserve the legacy call sites until the rest of this
# file migrates over.
from srdatalog.dialects.target.cuda.runner import emit_execute as _gen_execute
from srdatalog.dialects.target.cuda.runner import emit_execute_fused as _gen_execute_fused
from srdatalog.dialects.target.cuda.runner import emit_grid_config_code as _gen_grid_config_code
from srdatalog.dialects.target.cuda.runner import emit_launch_count as _gen_launch_count
from srdatalog.dialects.target.cuda.runner import emit_launch_fused as _gen_launch_fused
from srdatalog.dialects.target.cuda.runner import emit_launch_materialize as _gen_launch_materialize
from srdatalog.dialects.target.cuda.runner import (
  emit_launch_params_struct as _gen_launch_params_struct,
)
from srdatalog.dialects.target.cuda.runner import (
  emit_method_forward_decls as _gen_method_forward_decls,
)
from srdatalog.dialects.target.cuda.runner import emit_read_fused_result as _gen_read_fused_result
from srdatalog.dialects.target.cuda.runner import emit_read_total as _gen_read_total
from srdatalog.dialects.target.cuda.runner import emit_scan_and_resize as _gen_scan_and_resize
from srdatalog.dialects.target.cuda.runner import emit_scan_only as _gen_scan_only
from srdatalog.dialects.target.cuda.runner import (
  emit_struct_type_aliases as _gen_struct_type_aliases,
)

# -----------------------------------------------------------------------------
# Source spec extraction helpers (mirror Nim's inline lambdas)
# -----------------------------------------------------------------------------


def _version_to_cpp(ver: str) -> str:
  v = str(ver)
  if v in ("DELTA", "DeltaVer", "DELTA_VER"):
    return "DELTA_VER"
  if v in ("FULL", "FullVer", "FULL_VER"):
    return "FULL_VER"
  if v in ("NEW", "NewVer", "NEW_VER"):
    return "NEW_VER"
  return "FULL_VER"


def _index_to_list(idx: list[int]) -> str:
  return "{{" + ", ".join(str(i) for i in idx) + "}}"


def _src_schema(src: m.MirNode) -> str:
  if isinstance(src, m.ColumnSource):
    return src.rel_name
  if isinstance(src, m.Scan):
    return src.rel_name
  if isinstance(src, m.Negation):
    return src.rel_name
  raise AssertionError(f"unsupported source spec kind: {type(src).__name__}")


def _src_version_cpp(src: m.MirNode) -> str:
  if isinstance(src, m.ColumnSource):
    return _version_to_cpp(src.version.code)
  if isinstance(src, m.Scan):
    return _version_to_cpp(src.version.code)
  if isinstance(src, m.Negation):
    return _version_to_cpp(src.version.code)
  raise AssertionError(f"unsupported source spec kind: {type(src).__name__}")


def _src_mir_version(src: m.MirNode) -> str:
  if isinstance(src, m.ColumnSource):
    return src.version.code
  if isinstance(src, m.Scan):
    return src.version.code
  if isinstance(src, m.Negation):
    return src.version.code
  if isinstance(src, m.Aggregate):
    return src.version.code
  raise AssertionError(f"unsupported source spec kind: {type(src).__name__}")


def _src_index(src: m.MirNode) -> list[int]:
  if isinstance(src, m.ColumnSource):
    return list(src.index)
  if isinstance(src, m.Scan):
    return list(src.index)
  if isinstance(src, m.Negation):
    return list(src.index)
  raise AssertionError(f"unsupported source spec kind: {type(src).__name__}")


def _root_is_scan(pipeline: list[m.MirNode]) -> bool:
  return len(pipeline) > 0 and isinstance(pipeline[0], m.Scan)


# -----------------------------------------------------------------------------
# Kernel emitters
# -----------------------------------------------------------------------------


def _gen_kernel_count(
  node: m.ExecutePipeline,
  pipeline: list[m.MirNode],
  rel_index_types: dict[str, str],
) -> str:
  code = "  // Non-template kernel_count (concrete ViewType)\n"
  code += "  static __global__ void __launch_bounds__(kBlockSize) kernel_count(\n"
  code += "      const ViewType* __restrict__ views,\n"
  code += "      const ValueType* __restrict__ root_unique_values,\n"
  code += "      uint32_t num_unique_root_keys,\n"
  code += "      uint32_t num_root_keys,\n"
  code += "      uint32_t* __restrict__ thread_counts) {\n"
  code += "    auto block = cg::this_thread_block();\n"
  code += "    auto tile = cg::tiled_partition<kGroupSize>(block);\n"
  code += (
    "    auto single_thread = cg::tiled_partition<1>(block);"
    "  // For per-thread search inside Cartesian\n"
  )
  code += "    __shared__ char s_views_buf[NumSources * sizeof(ViewType)];\n"
  code += "    auto* s_views = reinterpret_cast<ViewType*>(s_views_buf);\n"
  code += "    if (threadIdx.x < NumSources) { s_views[threadIdx.x] = views[threadIdx.x]; }\n"
  code += "    __syncthreads();\n"
  code += "    views = s_views;  // redirect to shared memory copy\n"
  code += "    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;\n"
  code += "    uint32_t warp_id = thread_id / kGroupSize;\n"
  code += "    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;\n"
  code += "    uint32_t num_threads = num_warps;  // Alias for scalar mode (kGroupSize=1)\n\n"

  dest_arities = [len(d.index) for d in node.dest_specs]
  if dest_arities:
    code += (
      "    using OutputCtx = SRDatalog::GPU::OutputContext<"
      "ValueType, SR, true, Layout, OutputArity_0>;\n"
    )
  else:
    code += "    using OutputCtx = SRDatalog::GPU::OutputContext<ValueType, SR, true, Layout, 0>;\n"
  code += "    OutputCtx output_ctx{nullptr, nullptr, 0, 0};\n\n"

  output_vars: dict[str, str] = {}
  if node.dest_specs:
    output_vars[node.dest_specs[0].rel_name] = "output_ctx"
    for i in range(1, len(node.dest_specs)):
      output_vars[node.dest_specs[i].rel_name] = "__skip_counting__"

  from srdatalog.compile import compile_kernel_body

  code += compile_kernel_body(
    node,
    is_counting=True,
    output_var_name="output_ctx",
    output_vars=output_vars,
    rel_index_types=rel_index_types,
  )

  code += "    thread_counts[thread_id] = output_ctx.count();\n"
  code += "  }\n\n"
  return code


def _gen_kernel_materialize(
  node: m.ExecutePipeline,
  pipeline: list[m.MirNode],
  rel_index_types: dict[str, str],
  tiled_cartesian_eligible: bool,
) -> tuple[str, str]:
  '''Emit kernel_materialize. Returns (code, pipeline_body_output) —
  the pipeline body is returned separately so kernel_fused can reuse it
  when tiled Cartesian is NOT eligible (matches Nim behavior).
  '''
  dest_specs = node.dest_specs
  code = "  // Non-template kernel_materialize (concrete ViewType)\n"
  code += "  static __global__ void __launch_bounds__(kBlockSize) kernel_materialize(\n"
  code += "      const ViewType* __restrict__ views,\n"
  code += "      const ValueType* __restrict__ root_unique_values,\n"
  code += "      uint32_t num_unique_root_keys,\n"
  code += "      uint32_t num_root_keys,\n"
  code += "      const uint32_t* __restrict__ thread_offsets,\n"
  for i, dest in enumerate(dest_specs):
    code += f"      ValueType* __restrict__ output_data_{i},\n"
    code += f"      semiring_value_t<SR>* __restrict__ output_prov_{i},\n"
    code += f"      std::size_t output_stride_{i},\n"
    code += f"      uint32_t old_size_{i}"
    code += ") {\n" if i == len(dest_specs) - 1 else ",\n"
  code += "    auto block = cg::this_thread_block();\n"
  code += "    auto tile = cg::tiled_partition<kGroupSize>(block);\n"
  code += (
    "    auto single_thread = cg::tiled_partition<1>(block);"
    "  // For per-thread search inside Cartesian\n"
  )
  code += "    __shared__ char s_views_buf[NumSources * sizeof(ViewType)];\n"
  code += "    auto* s_views = reinterpret_cast<ViewType*>(s_views_buf);\n"
  code += "    if (threadIdx.x < NumSources) { s_views[threadIdx.x] = views[threadIdx.x]; }\n"
  code += "    __syncthreads();\n"
  code += "    views = s_views;\n"
  code += "    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;\n"
  code += "    uint32_t warp_id = thread_id / kGroupSize;\n"
  code += "    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;\n"
  code += "    uint32_t num_threads = num_warps;  // Alias for scalar mode (kGroupSize=1)\n"
  code += "    uint32_t thread_offset = thread_offsets[thread_id];\n\n"

  if tiled_cartesian_eligible:
    code += "    // Tiled Cartesian: per-warp smem tiles + coalesced write state\n"
    code += "    constexpr int kWarpsPerBlock = kBlockSize / kGroupSize;\n"
    code += "    constexpr int kCartTileSize = 256;\n"
    code += "    __shared__ ValueType s_cart[kWarpsPerBlock][2][kCartTileSize];\n"
    code += "    uint32_t warp_in_block = threadIdx.x / kGroupSize;\n"
    code += (
      "    uint32_t warp_write_base = tile.shfl(thread_offset, 0);  // broadcast lane 0 offset\n"
    )
    code += "    uint32_t warp_local_count = 0;\n\n"

  output_vars: dict[str, str] = {}
  output_var_name = 'output'
  for i, dest in enumerate(dest_specs):
    output_var = f"output_ctx_{i}"
    arity_const = f"OutputArity_{i}"
    code += (
      f"    using OutputCtx_{i} = SRDatalog::GPU::OutputContext<"
      f"ValueType, SR, false, Layout, {arity_const}>;\n"
    )
    code += (
      f"    OutputCtx_{i} {output_var}"
      f"{{output_data_{i}, output_prov_{i}, output_stride_{i}, "
      f"old_size_{i} + thread_offset}};\n"
    )
    output_vars[dest.rel_name] = output_var
    if i == 0:
      output_var_name = output_var
  code += "\n"

  from srdatalog.compile import compile_kernel_body

  body = compile_kernel_body(
    node,
    is_counting=False,
    output_var_name=output_var_name,
    output_vars=output_vars,
    rel_index_types=rel_index_types,
    tiled_cartesian=tiled_cartesian_eligible,
  )

  code += body
  code += "  }\n\n"
  return code, body


def _gen_kernel_fused(
  node: m.ExecutePipeline,
  pipeline: list[m.MirNode],
  rel_index_types: dict[str, str],
  materialize_pipeline_body: str,
  tiled_cartesian_eligible: bool,
) -> str:
  dest_specs = node.dest_specs
  code = "  // Fused kernel: single-pass join with atomic output (tail mode)\n"
  code += "  static __global__ void __launch_bounds__(kBlockSize) kernel_fused(\n"
  code += "      const ViewType* __restrict__ views,\n"
  code += "      const ValueType* __restrict__ root_unique_values,\n"
  code += "      uint32_t num_unique_root_keys,\n"
  code += "      uint32_t num_root_keys,\n"
  for j, dest in enumerate(dest_specs):
    code += f"      ValueType* __restrict__ output_data_{j},\n"
    code += f"      std::size_t output_stride_{j},\n"
    code += f"      uint32_t old_size_{j},\n"
    code += f"      uint32_t* __restrict__ atomic_write_pos_{j},\n"
  code += "      uint32_t capacity,\n"
  code += "      uint32_t* __restrict__ overflow_flag) {\n"
  code += "    auto block = cg::this_thread_block();\n"
  code += "    auto tile = cg::tiled_partition<kGroupSize>(block);\n"
  code += "    auto single_thread = cg::tiled_partition<1>(block);\n"
  code += "    __shared__ char s_views_buf[NumSources * sizeof(ViewType)];\n"
  code += "    auto* s_views = reinterpret_cast<ViewType*>(s_views_buf);\n"
  code += "    if (threadIdx.x < NumSources) { s_views[threadIdx.x] = views[threadIdx.x]; }\n"
  code += "    __syncthreads();\n"
  code += "    views = s_views;\n"
  code += "    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;\n"
  code += "    uint32_t warp_id = thread_id / kGroupSize;\n"
  code += "    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;\n"
  code += "    uint32_t num_threads = num_warps;\n\n"
  for j, dest in enumerate(dest_specs):
    code += (
      f"    using SpecCtx_{j} = SRDatalog::GPU::JIT::WS::"
      f"SpeculativeOutputContext<ValueType, OutputArity_{j}, 16>;\n"
    )
    code += (
      f"    SpecCtx_{j} output_ctx_{j}{{output_data_{j}, atomic_write_pos_{j}, overflow_flag,\n"
    )
    code += (
      f"                         static_cast<uint32_t>(output_stride_{j}), "
      f"old_size_{j}, capacity}};\n\n"
    )

  # When tiled Cartesian is eligible, re-render the pipeline WITHOUT tiled
  # mode (the ballot-path bypasses SpeculativeOutputContext). Otherwise
  # reuse the materialize pipeline body verbatim (matches Nim exactly).
  if tiled_cartesian_eligible:
    output_vars = {dest.rel_name: f"output_ctx_{i}" for i, dest in enumerate(dest_specs)}
    from srdatalog.compile import compile_kernel_body

    code += compile_kernel_body(
      node,
      is_counting=False,
      output_var_name="output_ctx_0",
      output_vars=output_vars,
      rel_index_types=rel_index_types,
    )
  else:
    code += materialize_pipeline_body

  for j in range(len(dest_specs)):
    code += f"    output_ctx_{j}.flush();\n"
  code += "  }\n\n"
  return code


# -----------------------------------------------------------------------------
# Block-group kernel variants
# -----------------------------------------------------------------------------


def _gen_kernel_bg_histogram(
  node: m.ExecutePipeline,
  rel_index_types: dict[str, str],
) -> str:
  '''Delegate to `par.data.block_group.emit_bg_histogram_kernel`.

  The histogram kernel is the first BG component lifted into the dialect
  (N4.0). Future N4.x milestones replace `_gen_kernel_bg_count` /
  `_bg_materialize` / `_bg_fused` with native dialect emission too.
  '''
  from srdatalog.dialects.parallel.data.block_group import emit_bg_histogram_kernel

  return emit_bg_histogram_kernel(node, rel_index_types)


def _gen_kernel_bg_count(
  node: m.ExecutePipeline,
  pipeline: list[m.MirNode],
  rel_index_types: dict[str, str],
) -> str:
  '''Emit kernel_bg_count — same body as kernel_count but with the BG
  kernel signature (extra head_root_unique_values / num_full_unique_root_keys
  / bg_cumulative_work / bg_total_work params) and bg_enabled=True ctx
  so jit_root_column_join dispatches to the BG variant.'''
  code = "  // Block-group partitioned count kernel\n"
  code += "  static __global__ void __launch_bounds__(kBlockSize) kernel_bg_count(\n"
  code += "      const ViewType* __restrict__ views,\n"
  code += "      const ValueType* __restrict__ root_unique_values,\n"
  code += "      const ValueType* __restrict__ head_root_unique_values,\n"
  code += "      uint32_t num_unique_root_keys,\n"
  code += "      uint32_t num_full_unique_root_keys,\n"
  code += "      uint32_t num_root_keys,\n"
  code += "      const uint64_t* __restrict__ bg_cumulative_work,\n"
  code += "      uint64_t bg_total_work,\n"
  code += "      uint32_t* __restrict__ thread_counts) {\n"
  code += "    auto block = cg::this_thread_block();\n"
  code += "    auto tile = cg::tiled_partition<kGroupSize>(block);\n"
  code += "    auto single_thread = cg::tiled_partition<1>(block);\n"
  code += "    __shared__ char s_views_buf[NumSources * sizeof(ViewType)];\n"
  code += "    auto* s_views = reinterpret_cast<ViewType*>(s_views_buf);\n"
  code += "    if (threadIdx.x < NumSources) { s_views[threadIdx.x] = views[threadIdx.x]; }\n"
  code += "    __syncthreads();\n"
  code += "    views = s_views;\n"
  code += "    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;\n"
  code += "    uint32_t warp_id = thread_id / kGroupSize;\n"
  code += "    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;\n"
  code += "    uint32_t num_threads = num_warps;\n\n"
  dest_arities = [len(d.index) for d in node.dest_specs]
  if dest_arities:
    code += (
      "    using OutputCtx = SRDatalog::GPU::OutputContext<"
      "ValueType, SR, true, Layout, OutputArity_0>;\n"
    )
  else:
    code += "    using OutputCtx = SRDatalog::GPU::OutputContext<ValueType, SR, true, Layout, 0>;\n"
  code += "    OutputCtx output_ctx{nullptr, nullptr, 0, 0};\n\n"

  output_vars: dict[str, str] = {}
  if node.dest_specs:
    output_vars[node.dest_specs[0].rel_name] = "output_ctx"
    for i in range(1, len(node.dest_specs)):
      output_vars[node.dest_specs[i].rel_name] = "__skip_counting__"
  from srdatalog.compile import compile_kernel_body
  code += compile_kernel_body(
    node,
    is_counting=True,
    output_var_name="output_ctx",
    output_vars=output_vars,
    rel_index_types=rel_index_types,
    bg_enabled=True,
  )
  code += "    thread_counts[thread_id] = output_ctx.count();\n"
  code += "  }\n\n"
  return code


def _gen_kernel_bg_materialize(
  node: m.ExecutePipeline,
  pipeline: list[m.MirNode],
  rel_index_types: dict[str, str],
) -> str:
  dest_specs = node.dest_specs
  code = "  // Block-group partitioned materialize kernel\n"
  code += "  static __global__ void __launch_bounds__(kBlockSize) kernel_bg_materialize(\n"
  code += "      const ViewType* __restrict__ views,\n"
  code += "      const ValueType* __restrict__ root_unique_values,\n"
  code += "      const ValueType* __restrict__ head_root_unique_values,\n"
  code += "      uint32_t num_unique_root_keys,\n"
  code += "      uint32_t num_full_unique_root_keys,\n"
  code += "      uint32_t num_root_keys,\n"
  code += "      const uint64_t* __restrict__ bg_cumulative_work,\n"
  code += "      uint64_t bg_total_work,\n"
  code += "      const uint32_t* __restrict__ thread_offsets,\n"
  for i, dest in enumerate(dest_specs):
    code += f"      ValueType* __restrict__ output_data_{i},\n"
    code += f"      semiring_value_t<SR>* __restrict__ output_prov_{i},\n"
    code += f"      std::size_t output_stride_{i},\n"
    code += f"      uint32_t old_size_{i}"
    code += ") {\n" if i == len(dest_specs) - 1 else ",\n"
  code += "    auto block = cg::this_thread_block();\n"
  code += "    auto tile = cg::tiled_partition<kGroupSize>(block);\n"
  code += "    auto single_thread = cg::tiled_partition<1>(block);\n"
  code += "    __shared__ char s_views_buf[NumSources * sizeof(ViewType)];\n"
  code += "    auto* s_views = reinterpret_cast<ViewType*>(s_views_buf);\n"
  code += "    if (threadIdx.x < NumSources) { s_views[threadIdx.x] = views[threadIdx.x]; }\n"
  code += "    __syncthreads();\n"
  code += "    views = s_views;\n"
  code += "    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;\n"
  code += "    uint32_t warp_id = thread_id / kGroupSize;\n"
  code += "    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;\n"
  code += "    uint32_t num_threads = num_warps;\n"
  code += "    uint32_t thread_offset = thread_offsets[thread_id];\n\n"

  output_vars: dict[str, str] = {}
  output_var_name = 'output'
  for i, dest in enumerate(dest_specs):
    output_var = f"output_ctx_{i}"
    arity_const = f"OutputArity_{i}"
    code += (
      f"    using OutputCtx_{i} = SRDatalog::GPU::OutputContext<"
      f"ValueType, SR, false, Layout, {arity_const}>;\n"
    )
    code += (
      f"    OutputCtx_{i} {output_var}"
      f"{{output_data_{i}, output_prov_{i}, output_stride_{i}, "
      f"old_size_{i} + thread_offset}};\n"
    )
    output_vars[dest.rel_name] = output_var
    if i == 0:
      output_var_name = output_var
  code += "\n"
  from srdatalog.compile import compile_kernel_body
  code += compile_kernel_body(
    node,
    is_counting=False,
    output_var_name=output_var_name,
    output_vars=output_vars,
    rel_index_types=rel_index_types,
    bg_enabled=True,
  )
  code += "  }\n\n"
  return code


def _gen_kernel_bg_fused(
  node: m.ExecutePipeline,
  pipeline: list[m.MirNode],
  rel_index_types: dict[str, str],
) -> str:
  dest_specs = node.dest_specs
  code = "  // Block-group partitioned fused kernel (speculative atomic output)\n"
  code += "  static __global__ void __launch_bounds__(kBlockSize) kernel_bg_fused(\n"
  code += "      const ViewType* __restrict__ views,\n"
  code += "      const ValueType* __restrict__ root_unique_values,\n"
  code += "      const ValueType* __restrict__ head_root_unique_values,\n"
  code += "      uint32_t num_unique_root_keys,\n"
  code += "      uint32_t num_full_unique_root_keys,\n"
  code += "      uint32_t num_root_keys,\n"
  code += "      const uint64_t* __restrict__ bg_cumulative_work,\n"
  code += "      uint64_t bg_total_work,\n"
  for j, dest in enumerate(dest_specs):
    code += f"      ValueType* __restrict__ output_data_{j},\n"
    code += f"      std::size_t output_stride_{j},\n"
    code += f"      uint32_t old_size_{j},\n"
    code += f"      uint32_t* __restrict__ atomic_write_pos_{j},\n"
  code += "      uint32_t capacity,\n"
  code += "      uint32_t* __restrict__ overflow_flag) {\n"
  code += "    auto block = cg::this_thread_block();\n"
  code += "    auto tile = cg::tiled_partition<kGroupSize>(block);\n"
  code += "    auto single_thread = cg::tiled_partition<1>(block);\n"
  code += "    __shared__ char s_views_buf[NumSources * sizeof(ViewType)];\n"
  code += "    auto* s_views = reinterpret_cast<ViewType*>(s_views_buf);\n"
  code += "    if (threadIdx.x < NumSources) { s_views[threadIdx.x] = views[threadIdx.x]; }\n"
  code += "    __syncthreads();\n"
  code += "    views = s_views;\n"
  code += "    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;\n"
  code += "    uint32_t warp_id = thread_id / kGroupSize;\n"
  code += "    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;\n"
  code += "    uint32_t num_threads = num_warps;\n\n"
  for j, dest in enumerate(dest_specs):
    code += (
      f"    using SpecCtx_{j} = SRDatalog::GPU::JIT::WS::"
      f"SpeculativeOutputContext<ValueType, OutputArity_{j}, 16>;\n"
    )
    code += (
      f"    SpecCtx_{j} output_ctx_{j}{{output_data_{j}, atomic_write_pos_{j}, overflow_flag,\n"
    )
    code += (
      f"                         static_cast<uint32_t>(output_stride_{j}), "
      f"old_size_{j}, capacity}};\n\n"
    )

  output_vars = {dest.rel_name: f"output_ctx_{i}" for i, dest in enumerate(dest_specs)}
  from srdatalog.compile import compile_kernel_body
  code += compile_kernel_body(
    node,
    is_counting=False,
    output_var_name="output_ctx_0",
    output_vars=output_vars,
    rel_index_types=rel_index_types,
    bg_enabled=True,
  )
  for j in range(len(dest_specs)):
    code += f"    output_ctx_{j}.flush();\n"
  code += "  }\n\n"
  return code


# -----------------------------------------------------------------------------
# LaunchParams + forward declarations
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Phase method definitions (out-of-line)
# -----------------------------------------------------------------------------


def _gen_view_setup_for_source(
  i: int,
  src: m.MirNode,
  views_var: str,
  rel_index_types: dict[str, str],
) -> str:
  schema = _src_schema(src)
  ver = _src_version_cpp(src)
  mir_ver = _src_mir_version(src)
  idx_str = _index_to_list(_src_index(src))
  index_type = rel_index_types.get(schema, "")
  code = f"  // Source {i}: {schema} version {ver}\n"
  code += "  {\n"
  if ver == "DELTA_VER":
    code += (
      f"    auto& rel_{i} = (iteration == 0) ? "
      f"get_relation_by_schema<{schema}, FULL_VER>(db) : "
      f"get_relation_by_schema<{schema}, DELTA_VER>(db);\n"
    )
  else:
    code += f"    auto& rel_{i} = get_relation_by_schema<{schema}, {ver}>(db);\n"
  force_rebuild = "true" if isinstance(src, m.Negation) else "false"
  code += (
    f"    auto& idx_{i} = rel_{i}.ensure_index(SRDatalog::IndexSpec{idx_str}, {force_rebuild});\n"
  )
  # Push view(s) via the plugin hook — a multi-view plugin (e.g.
  # Device2LevelIndex on FULL_VER) returns both head and full view
  # expressions; the single-view default returns just `.view()`.
  # Mirrors Nim's jit_complete_runner.nim call to pluginGenHostViewSetup.
  for expr in plugin_gen_host_view_setup(f"idx_{i}", mir_ver, index_type):
    code += f"    {views_var}.push_back({expr});\n"
  code += "  }\n\n"
  return code


def _gen_root_keys_code(
  prefix: str,
  first_schema: str,
  first_version: str,
  first_index: list[int],
  rel_index_types: dict[str, str],
) -> str:
  code = "  // First source for root keys\n"
  if first_version == "DELTA_VER":
    code += (
      f"  auto& first_rel = (iteration == 0) ? "
      f"get_relation_by_schema<{first_schema}, FULL_VER>(db) : "
      f"get_relation_by_schema<{first_schema}, DELTA_VER>(db);\n"
    )
  else:
    code += f"  auto& first_rel = get_relation_by_schema<{first_schema}, {first_version}>(db);\n"
  code += (
    f"  auto& first_idx = first_rel.get_index(SRDatalog::IndexSpec{_index_to_list(first_index)});\n"
  )
  code += f"  {prefix}num_root_keys = first_idx.root().degree();\n"
  code += (
    f"  {prefix}num_unique_root_keys = static_cast<uint32_t>(first_idx.num_unique_root_values());\n"
  )
  code += (
    f"  {prefix}root_unique_values_ptr = ({prefix}num_unique_root_keys > 0) "
    "? first_idx.root_unique_values().data() : nullptr;\n"
  )
  # 2-level first source: expose HEAD segment for BG dual-pointer iteration.
  first_index_type = rel_index_types.get(first_schema, "")
  first_view_count = plugin_view_count(first_version, first_index_type)
  if first_view_count > 1:
    code += f"  {prefix}num_full_unique_root_keys = {prefix}num_unique_root_keys;\n"
    code += (
      f"  {prefix}num_head_unique_root_keys = "
      "static_cast<uint32_t>(first_idx.head_num_unique_root_values());\n"
    )
    code += (
      f"  {prefix}head_root_unique_values_ptr = "
      f"({prefix}num_head_unique_root_keys > 0) ? "
      "first_idx.head_root_unique_values().data() : nullptr;\n"
    )
    code += f"  {prefix}num_unique_root_keys += {prefix}num_head_unique_root_keys;\n"
    code += f"  {prefix}num_root_keys += first_idx.head().root().degree();\n"
  else:
    code += f"  {prefix}num_full_unique_root_keys = {prefix}num_unique_root_keys;\n"
  code += "\n"
  return code


def _gen_setup(
  node: m.ExecutePipeline,
  runner_prefix: str,
  first_schema: str,
  first_version: str,
  first_index: list[int],
  rel_index_types: dict[str, str],
) -> str:
  code = "// Phase 1: Setup views and compute grid config\n"
  code += (
    f"{runner_prefix}::LaunchParams {runner_prefix}"
    "::setup(DB& db, uint32_t iteration, GPU_STREAM_T stream) {\n"
  )
  code += "  LaunchParams p;\n"
  code += "  p.views_vec.reserve(NumSources);\n\n"

  seen: list[str] = []
  for i, src in enumerate(node.source_specs):
    key = source_spec_key(src)
    if key not in seen:
      seen.append(key)
      code += _gen_view_setup_for_source(i, src, "p.views_vec", rel_index_types)

  code += _gen_root_keys_code(
    "p.",
    first_schema,
    first_version,
    first_index,
    rel_index_types,
  )
  code += "  // Copy views to device using provided stream (NOT stream 0)\n"
  code += "  p.d_views = SRDatalog::GPU::DeviceArray<ViewType>(p.views_vec.size());\n"
  code += (
    "  GPU_MEMCPY_ASYNC(p.d_views.data(), p.views_vec.data(), "
    "p.views_vec.size() * sizeof(ViewType), GPU_HOST_TO_DEVICE, stream);\n\n"
  )
  root_is_scan = _root_is_scan(node.pipeline)
  code += _gen_grid_config_code("p.", root_is_scan)

  if node.block_group:
    code += "  // Block-group: pre-allocate and compute work histogram in setup\n"
    code += "  // Both thresholds must pass: enough total rows AND enough unique keys\n"
    code += "  if (p.num_root_keys >= 256 && p.num_unique_root_keys >= 32) {\n"
    code += "    // BG buffers: static rmm::device_uvector, resize only when needed\n"
    code += "    static rmm::device_uvector<uint64_t> s_bg_wk(0, rmm::cuda_stream_default);\n"
    code += "    static rmm::device_uvector<uint64_t> s_bg_cw(0, rmm::cuda_stream_default);\n"
    code += "    if (s_bg_wk.size() < p.num_unique_root_keys) {\n"
    code += "      s_bg_wk.resize(p.num_unique_root_keys, rmm::cuda_stream_view{stream});\n"
    code += "      s_bg_cw.resize(p.num_unique_root_keys, rmm::cuda_stream_view{stream});\n"
    code += "    }\n"
    code += "    p.bg_cumulative_work_ptr = s_bg_cw.data();\n"
    code += "    uint64_t* bg_wk_ptr = s_bg_wk.data();\n"
    code += "    int bg_num_sms = 0;\n"
    code += "    GPU_DEVICE_GET_ATTRIBUTE(&bg_num_sms, GPU_DEV_ATTR_MULTIPROCESSOR_COUNT, 0);\n"
    code += (
      "    uint32_t hist_blocks = std::max((uint32_t)"
      "((p.num_unique_root_keys + (kBlockSize/kGroupSize) - 1) / "
      "(kBlockSize/kGroupSize)), (uint32_t)(bg_num_sms * 4));\n"
    )
    code += (
      "    kernel_bg_histogram<<<hist_blocks, kBlockSize, 0, stream>>>"
      "(p.d_views.data(), p.root_unique_values_ptr, "
      "p.head_root_unique_values_ptr, p.num_unique_root_keys, "
      "p.num_full_unique_root_keys, p.num_root_keys, bg_wk_ptr);\n"
    )
    code += (
      "    thrust::inclusive_scan(rmm::exec_policy(stream), bg_wk_ptr, "
      "bg_wk_ptr + p.num_unique_root_keys, p.bg_cumulative_work_ptr);\n"
    )
    code += (
      "    GPU_MEMCPY_ASYNC(&p.bg_total_work, "
      "p.bg_cumulative_work_ptr + p.num_unique_root_keys - 1, "
      "sizeof(uint64_t), GPU_DEVICE_TO_HOST, stream);\n"
    )
    code += "    // Decide: is BG worth it? Compare max per-key work vs fair share.\n"
    code += (
      "    auto bg_max_it = thrust::max_element(rmm::exec_policy(stream), "
      "bg_wk_ptr, bg_wk_ptr + p.num_unique_root_keys);\n"
    )
    code += "    uint64_t bg_max_work = 0;\n"
    code += (
      "    GPU_MEMCPY_ASYNC(&bg_max_work, "
      "thrust::raw_pointer_cast(bg_max_it), sizeof(uint64_t), "
      "GPU_DEVICE_TO_HOST, stream);\n"
    )
    code += "    GPU_STREAM_SYNCHRONIZE(stream);\n"
    code += "    uint32_t bg_num_warps = (p.num_blocks * kBlockSize) / kGroupSize;\n"
    code += "    // BG helps when the hottest key's work exceeds what a warp processes\n"
    code += "    // in one grid-stride pass: total_work / num_warps.\n"
    code += "    // If max_key fits within a warp's stride batch, baseline is fine.\n"
    code += "    uint64_t bg_warp_stride_work = p.bg_total_work / bg_num_warps;\n"
    code += "    // BG helps when the hottest key dominates the total runtime.\n"
    code += "    // If max_key < 10% of total work, baseline's tail is short enough.\n"
    code += "    // Adaptive fallback disabled — always use BG when threshold met\n"
    code += "    // if (bg_max_work * 10 <= p.bg_total_work) { p.bg_total_work = 0; }\n"
    code += "    p.bg_num_blocks = bg_num_sms * 8;\n"
    code += "    p.bg_num_threads = p.bg_num_blocks * kBlockSize;\n"
    code += "  }\n\n"

  code += "  p.thread_counts = SRDatalog::GPU::DeviceArray<uint32_t>(p.num_threads + 1);\n"
  code += (
    "  cudaMemsetAsync(p.thread_counts.data(), 0, "
    "(p.num_threads + 1) * sizeof(uint32_t), stream);\n"
  )
  code += "  p.thread_counts_ptr = p.thread_counts.data();\n"
  code += "  return p;\n"
  code += "}\n\n"
  return code


# -----------------------------------------------------------------------------
# Top-level entry point
# -----------------------------------------------------------------------------


def gen_complete_runner(
  node: m.ExecutePipeline,
  db_type_name: str,
  rel_index_types: dict[str, str] | None = None,
) -> tuple[str, str]:
  '''Generate `(decl, full)` for one ExecutePipeline.

  `full` is the complete struct with kernel definitions + out-of-line
  phase method bodies (goes into the JIT batch file). `decl` is a
  minimal forward-declaration struct with type aliases + LaunchParams +
  method declarations only (goes into the main compile unit so the
  orchestrator can call `JitRunner_X::execute()`).
  '''
  assert isinstance(node, m.ExecutePipeline)
  if rel_index_types is None:
    rel_index_types = {}

  # Materialized join: out of scope for this port (jit_materialized.py
  # already emits its own runner).
  if is_materialized_pipeline(node.pipeline):
    raise NotImplementedError(
      "gen_complete_runner: materialized pipelines handled by "
      "jit_materialized.gen_materialized_runner — caller should dispatch."
    )

  # Feature-flag guards: not covered in Phase 2 baseline port.
  if node.work_stealing:
    raise NotImplementedError("gen_complete_runner: work_stealing not yet ported")
  if node.dedup_hash:
    raise NotImplementedError("gen_complete_runner: dedup_hash not yet ported")
  if has_balanced_scan(node.pipeline):
    raise NotImplementedError("gen_complete_runner: balanced-scan runner not yet ported")

  rule_name = node.rule_name
  runner_prefix = f"JitRunner_{rule_name}"
  is_count = node.count
  is_fused_eligible = not is_count

  # Mutate a copy so handle positions don't leak back.
  mutable_pipe = list(node.pipeline)
  assign_handle_positions(mutable_pipe)

  first_src = node.source_specs[0]
  first_schema = _src_schema(first_src)
  first_version = _src_version_cpp(first_src)
  first_index = _src_index(first_src)
  dest_arities = [len(d.index) for d in node.dest_specs]
  total_view_count = compute_total_view_count(node.source_specs, rel_index_types)

  # Mirror Nim's tiledCartesianEligible condition (jit_complete_runner.nim:158-160):
  # disable tiled on WS / BG / dedup_hash / count / concurrent-write rules.
  # `concurrent_write` is set by orchestrator_jit when multiple rules in a
  # ParallelGroup share a destination relation — tiled's coalesced writes
  # can't interleave safely with concurrent kernels into the same region.
  tiled_cartesian_eligible = (
    has_tiled_cartesian_eligible(mutable_pipe)
    and not node.work_stealing
    and not node.block_group
    and not node.dedup_hash
    and not is_count
    and not node.concurrent_write
  )

  # -------------------------------------------------------------------------
  # Build `full` (struct + kernel defs + out-of-line phase methods)
  # -------------------------------------------------------------------------
  full = ""
  full += "// =============================================================\n"
  full += f"// JIT-Generated Complete Runner: {rule_name}\n"
  full += "// Fully concrete - NO C++ templates/metaprogramming\n"
  full += "// =============================================================\n\n"
  full += f"struct {runner_prefix} {{\n"
  full += _gen_struct_type_aliases(
    rule_name,
    db_type_name,
    first_schema,
    first_version,
    node.dest_specs,
    dest_arities,
    total_view_count,
  )

  full += _gen_kernel_count(node, mutable_pipe, rel_index_types)

  materialize_body = ""
  if is_count:
    # Emit materialize to a discard buffer (matches Nim — it still emits
    # so jit_pipeline side effects on handle registration etc. fire, but
    # drops the resulting code).
    _mat_code, materialize_body = _gen_kernel_materialize(
      node,
      mutable_pipe,
      rel_index_types,
      tiled_cartesian_eligible,
    )
  else:
    mat_code, materialize_body = _gen_kernel_materialize(
      node,
      mutable_pipe,
      rel_index_types,
      tiled_cartesian_eligible,
    )
    full += mat_code

  # Baseline `kernel_fused` is skipped when BG is enabled — Nim emits
  # only `kernel_bg_fused` for BG rules (see jit_complete_runner.nim:516:
  # `if isFusedEligible and not isBlockGroup`).
  if is_fused_eligible and not node.block_group:
    full += _gen_kernel_fused(
      node,
      mutable_pipe,
      rel_index_types,
      materialize_body,
      tiled_cartesian_eligible,
    )

  # Block-group kernels: histogram + count + materialize + bg_fused.
  # Emitted alongside baseline count/materialize; the runner dispatches
  # between paths at execute() time based on `bg_total_work > 0`.
  if node.block_group:
    full += _gen_kernel_bg_histogram(node, rel_index_types)
    full += _gen_kernel_bg_count(node, mutable_pipe, rel_index_types)
    if not is_count:
      full += _gen_kernel_bg_materialize(node, mutable_pipe, rel_index_types)
    if is_fused_eligible:
      full += _gen_kernel_bg_fused(node, mutable_pipe, rel_index_types)

  full += _gen_launch_params_struct(
    len(node.dest_specs),
    is_fused_eligible,
    is_block_group=node.block_group,
  )
  full += _gen_method_forward_decls(is_count, is_fused_eligible)
  full += "};\n\n"

  full += _gen_setup(
    node,
    runner_prefix,
    first_schema,
    first_version,
    first_index,
    rel_index_types,
  )
  full += _gen_launch_count(runner_prefix, is_block_group=node.block_group)
  full += _gen_scan_and_resize(node, runner_prefix)
  full += _gen_scan_only(runner_prefix)
  full += _gen_read_total(runner_prefix)
  if not is_count:
    full += _gen_launch_materialize(node, runner_prefix)
  if is_fused_eligible:
    full += _gen_launch_fused(node, runner_prefix)
    full += _gen_read_fused_result(node, runner_prefix)
  full += _gen_execute(
    rule_name,
    runner_prefix,
    is_count,
    is_block_group=node.block_group,
    dest_specs=node.dest_specs,
  )
  if is_fused_eligible:
    full += _gen_execute_fused(node, runner_prefix)

  # -------------------------------------------------------------------------
  # Build `decl` (forward declaration struct — same type aliases +
  # LaunchParams + method decls, no kernel bodies)
  # -------------------------------------------------------------------------
  decl = f"struct {runner_prefix} {{\n"
  # Note: decl variant uses "OutputArity = OutputArity_0;" without
  # the "// Legacy alias" comment — matches the Nim structDecl branch.
  decl_aliases = _gen_struct_type_aliases(
    rule_name,
    db_type_name,
    first_schema,
    first_version,
    node.dest_specs,
    dest_arities,
    total_view_count,
  )
  decl_aliases = decl_aliases.replace(
    "static constexpr std::size_t OutputArity = OutputArity_0;  // Legacy alias\n",
    "static constexpr std::size_t OutputArity = OutputArity_0;\n",
  )
  # Decl also drops the "// Type aliases - all concrete..." comment.
  decl_aliases = decl_aliases.replace(
    "  // Type aliases - all concrete, resolved at Nim JIT time\n",
    "",
  )
  decl += decl_aliases
  decl += _gen_launch_params_struct(
    len(node.dest_specs),
    is_fused_eligible,
    is_block_group=node.block_group,
    for_decl=True,
  )
  # The decl variant drops the "// State carried..." comment (matches Nim).
  decl = decl.replace("  // State carried between decomposed phases\n", "")
  decl += _gen_method_forward_decls(is_count, is_fused_eligible)
  # Drop the "// Phase-decomposed methods..." comment in decl (matches Nim).
  decl = decl.replace(
    "  // Phase-decomposed methods for stream-parallel execution\n",
    "",
  )
  # Drop the "// Non-template execute..." comment in decl (matches Nim).
  decl = decl.replace("  // Non-template execute - calls kernels directly\n", "")
  decl += "};\n\n"

  return decl, full
