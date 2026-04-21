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

import srdatalog.mir_types as m
from srdatalog.codegen.jit.context import CodeGenContext, new_code_gen_context
from srdatalog.codegen.jit.emit_helpers import (
  assign_handle_positions,
  has_balanced_scan,
  has_tiled_cartesian_eligible,
)
from srdatalog.codegen.jit.view_management import (
  build_root_slot_map,
  compute_total_view_count,
  compute_view_slot_offsets,
  register_pipeline_handles,
  source_spec_key,
)
from srdatalog.codegen.jit.pipeline import jit_pipeline
from srdatalog.codegen.jit.materialized import is_materialized_pipeline
from srdatalog.codegen.jit.plugin import plugin_view_count


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
# Pipeline context builder — fresh ctx per kernel
# -----------------------------------------------------------------------------

def _make_kernel_ctx(
  source_specs: list[m.MirNode],
  pipeline: list[m.MirNode],
  rel_index_types: dict[str, str],
  *,
  is_counting: bool,
  tiled_cartesian: bool = False,
  dedup_hash: bool = False,
  bg_enabled: bool = False,
  output_var_name: str = "output_ctx",
) -> CodeGenContext:
  ctx = new_code_gen_context()
  ctx.indent = 4
  ctx.is_counting = is_counting
  ctx.is_jit_mode = True
  ctx.dedup_hash_enabled = dedup_hash
  ctx.tiled_cartesian_enabled = tiled_cartesian
  ctx.bg_enabled = bg_enabled
  ctx.rel_index_types = dict(rel_index_types)
  ctx.view_slot_offsets = compute_view_slot_offsets(source_specs, rel_index_types)
  ctx.output_var_name = output_var_name
  root_slots = build_root_slot_map(source_specs, rel_index_types)
  register_pipeline_handles(
    ctx.view_slot_offsets, pipeline, rel_index_types, root_slots,
  )
  return ctx


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
  code += (
    "    if (threadIdx.x < NumSources) "
    "{ s_views[threadIdx.x] = views[threadIdx.x]; }\n"
  )
  code += "    __syncthreads();\n"
  code += "    views = s_views;  // redirect to shared memory copy\n"
  code += "    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;\n"
  code += "    uint32_t warp_id = thread_id / kGroupSize;\n"
  code += "    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;\n"
  code += (
    "    uint32_t num_threads = num_warps;"
    "  // Alias for scalar mode (kGroupSize=1)\n\n"
  )

  dest_arities = [len(d.index) for d in node.dest_specs]
  if dest_arities:
    code += (
      "    using OutputCtx = SRDatalog::GPU::OutputContext<"
      "ValueType, SR, true, Layout, OutputArity_0>;\n"
    )
  else:
    code += (
      "    using OutputCtx = SRDatalog::GPU::OutputContext<"
      "ValueType, SR, true, Layout, 0>;\n"
    )
  code += "    OutputCtx output_ctx{nullptr, nullptr, 0, 0};\n\n"

  ctx = _make_kernel_ctx(
    node.source_specs, pipeline, rel_index_types,
    is_counting=True, output_var_name="output_ctx",
  )
  # Primary output registered to output_ctx; secondary dests skipped in count phase.
  if node.dest_specs:
    ctx.output_vars[node.dest_specs[0].rel_name] = "output_ctx"
    for i in range(1, len(node.dest_specs)):
      ctx.output_vars[node.dest_specs[i].rel_name] = "__skip_counting__"

  code += jit_pipeline(pipeline, node.source_specs, ctx)
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
  code += (
    "  static __global__ void __launch_bounds__(kBlockSize) "
    "kernel_materialize(\n"
  )
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
  code += (
    "    if (threadIdx.x < NumSources) "
    "{ s_views[threadIdx.x] = views[threadIdx.x]; }\n"
  )
  code += "    __syncthreads();\n"
  code += "    views = s_views;\n"
  code += "    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;\n"
  code += "    uint32_t warp_id = thread_id / kGroupSize;\n"
  code += "    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;\n"
  code += (
    "    uint32_t num_threads = num_warps;"
    "  // Alias for scalar mode (kGroupSize=1)\n"
  )
  code += "    uint32_t thread_offset = thread_offsets[thread_id];\n\n"

  if tiled_cartesian_eligible:
    code += (
      "    // Tiled Cartesian: per-warp smem tiles + coalesced write state\n"
    )
    code += "    constexpr int kWarpsPerBlock = kBlockSize / kGroupSize;\n"
    code += "    constexpr int kCartTileSize = 256;\n"
    code += "    __shared__ ValueType s_cart[kWarpsPerBlock][2][kCartTileSize];\n"
    code += "    uint32_t warp_in_block = threadIdx.x / kGroupSize;\n"
    code += (
      "    uint32_t warp_write_base = tile.shfl(thread_offset, 0);"
      "  // broadcast lane 0 offset\n"
    )
    code += "    uint32_t warp_local_count = 0;\n\n"

  ctx = _make_kernel_ctx(
    node.source_specs, pipeline, rel_index_types,
    is_counting=False, tiled_cartesian=tiled_cartesian_eligible,
  )
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
    ctx.output_vars[dest.rel_name] = output_var
    if i == 0:
      ctx.output_var_name = output_var
  code += "\n"

  body = jit_pipeline(pipeline, node.source_specs, ctx)
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
  code += (
    "    if (threadIdx.x < NumSources) "
    "{ s_views[threadIdx.x] = views[threadIdx.x]; }\n"
  )
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
      f"    SpecCtx_{j} output_ctx_{j}{{output_data_{j}, atomic_write_pos_{j}, "
      "overflow_flag,\n"
    )
    code += (
      f"                         static_cast<uint32_t>(output_stride_{j}), "
      f"old_size_{j}, capacity}};\n\n"
    )

  # When tiled Cartesian is eligible, re-render the pipeline WITHOUT tiled
  # mode (the ballot-path bypasses SpeculativeOutputContext). Otherwise
  # reuse the materialize pipeline body verbatim (matches Nim exactly).
  if tiled_cartesian_eligible:
    ctx = _make_kernel_ctx(
      node.source_specs, pipeline, rel_index_types,
      is_counting=False, tiled_cartesian=False,
    )
    for i, dest in enumerate(dest_specs):
      ctx.output_vars[dest.rel_name] = f"output_ctx_{i}"
    ctx.output_var_name = "output_ctx_0"
    code += jit_pipeline(pipeline, node.source_specs, ctx)
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
  node: m.ExecutePipeline, rel_index_types: dict[str, str],
) -> str:
  '''Emit the BG histogram kernel — a grid-stride loop over unique keys
  that writes the per-key work estimate (product of root-source degrees)
  into `bg_work_per_key[]`. Body is a hand-crafted prefix+degree sweep,
  not a jit_pipeline render.
  '''
  source_specs = node.source_specs
  # Get root source handles from the first ColumnJoin in the pipeline.
  # The histogram only cares about the root sources (those with narrow
  # by `root_val`). We pull them from the MIR's first CJ, which also
  # tells us which handle_start values map to root slots.
  first_op = node.pipeline[0]
  if not isinstance(first_op, m.ColumnJoin):
    raise NotImplementedError(
      "_gen_kernel_bg_histogram: only ColumnJoin-root pipelines supported"
    )
  root_sources = [s for s in first_op.sources if isinstance(s, m.ColumnSource)]

  # Deduplicate source specs for view declarations (same as regular
  # kernels).
  from srdatalog.codegen.jit.view_management import (
    collect_unique_view_specs, spec_key as vs_spec_key,
  )
  mutable_pipe = list(node.pipeline)
  view_specs = collect_unique_view_specs(mutable_pipe)

  code = "  // Block-group histogram: per-key work via grid-stride + warp reduction\n"
  code += "  static __global__ void __launch_bounds__(kBlockSize) kernel_bg_histogram(\n"
  code += "      const ViewType* __restrict__ views,\n"
  code += "      const ValueType* __restrict__ root_unique_values,\n"
  code += "      const ValueType* __restrict__ head_root_unique_values,\n"
  code += "      uint32_t num_unique_root_keys,\n"
  code += "      uint32_t num_full_unique_root_keys,\n"
  code += "      uint32_t num_root_keys,\n"
  code += "      uint64_t* __restrict__ bg_work_per_key) {\n"
  code += "    auto block = cg::this_thread_block();\n"
  code += "    auto tile = cg::tiled_partition<kGroupSize>(block);\n"
  code += "    auto single_thread = cg::tiled_partition<1>(block);\n"
  code += "    __shared__ char s_views_buf[NumSources * sizeof(ViewType)];\n"
  code += "    auto* s_views = reinterpret_cast<ViewType*>(s_views_buf);\n"
  code += (
    "    if (threadIdx.x < NumSources) "
    "{ s_views[threadIdx.x] = views[threadIdx.x]; }\n"
  )
  code += "    __syncthreads();\n"
  code += "    views = s_views;\n"
  code += "    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;\n"
  code += "    uint32_t warp_id = thread_id / kGroupSize;\n"
  code += "    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;\n"
  code += "    uint32_t num_threads = num_warps;\n\n"

  code += "    using HandleType = ViewType::NodeHandle;\n"
  code += "    using ViewType = std::remove_cvref_t<decltype(views[0])>;\n"
  code += "    using HandleType = ViewType::NodeHandle;\n\n"

  # View declarations (one per unique spec, using view slot lookup).
  code += (
    f"    // View declarations (deduplicated by spec, "
    f"{len(view_specs)} unique views)\n"
  )
  # Compute view slots.
  view_slot_offsets = compute_view_slot_offsets(source_specs, rel_index_types)
  for sp in view_specs:
    idx_str = "_".join(str(v) for v in sp.index)
    view_var = f"view_{sp.rel_name}_{idx_str}_{sp.version}"
    slot = view_slot_offsets.get(sp.handle_idx, sp.handle_idx)
    code += f"    auto {view_var} = views[{slot}];\n"
  code += "\n"

  # Declared but unused in this baseline case.
  dest_arities = [len(d.index) for d in node.dest_specs]
  if dest_arities:
    code += (
      "    using OutputCtx = SRDatalog::GPU::OutputContext<"
      "ValueType, SR, true, Layout, OutputArity_0>;\n"
    )
  code += "    // Grid-stride histogram: per-key warp-reduced work counts\n"
  code += (
    "    for (uint32_t bg_hist_key = warp_id; "
    "bg_hist_key < num_unique_root_keys; bg_hist_key += num_warps) {\n"
  )
  if dest_arities:
    code += "      OutputCtx hist_output_ctx{nullptr, nullptr, 0, 0};\n"
  code += "      auto bg_hist_root_val = root_unique_values[bg_hist_key];\n"

  # Emit per-root-source narrowing + degree. First source uses
  # hint_lo/hi; later sources use gen_root_handle + prefix.
  for idx_, src in enumerate(root_sources):
    src_idx = src.handle_start
    rel_name = src.rel_name
    src_index_type = rel_index_types.get(rel_name, "")
    handle_var = f"h_{rel_name}_{src_idx}_root"
    # Find view_var for this source.
    idx_str = "_".join(str(v) for v in src.index)
    view_var = f"view_{rel_name}_{idx_str}_{src.version.code}"
    if idx_ == 0:
      code += "      uint32_t bg_hlo = bg_hist_key;\n"
      code += (
        "      uint32_t bg_hhi = "
        f"{view_var}.num_rows_ - (num_unique_root_keys - bg_hist_key - 1);\n"
      )
      code += (
        f"      bg_hhi = (bg_hhi <= {view_var}.num_rows_) ? "
        f"bg_hhi : {view_var}.num_rows_;\n"
      )
      code += (
        f"      bg_hhi = (bg_hhi > bg_hlo) ? bg_hhi : {view_var}.num_rows_;\n"
      )
      code += (
        f"      auto {handle_var} = HandleType(bg_hlo, bg_hhi, 0)"
        f".prefix(bg_hist_root_val, tile, {view_var});\n"
      )
    else:
      from srdatalog.codegen.jit.context import gen_root_handle as _gen_rh
      code += (
        f"      auto {handle_var} = {_gen_rh(view_var, src_index_type)}"
        f".prefix(bg_hist_root_val, tile, {view_var});\n"
      )
    code += (
      f"      if (!{handle_var}.valid()) {{ "
      "bg_work_per_key[bg_hist_key] = 0; continue; }\n"
    )

  code += "      uint64_t bg_deg = 1;\n"
  for src in root_sources:
    src_idx = src.handle_start
    rel_name = src.rel_name
    handle_var = f"h_{rel_name}_{src_idx}_root"
    code += f"      bg_deg *= {handle_var}.degree();\n"
  code += (
    "      if (tile.thread_rank() == 0) bg_work_per_key[bg_hist_key] = bg_deg;\n"
  )
  code += "    }\n"
  code += "  }\n\n"
  return code


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
  code += (
    "    if (threadIdx.x < NumSources) "
    "{ s_views[threadIdx.x] = views[threadIdx.x]; }\n"
  )
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
    code += (
      "    using OutputCtx = SRDatalog::GPU::OutputContext<"
      "ValueType, SR, true, Layout, 0>;\n"
    )
  code += "    OutputCtx output_ctx{nullptr, nullptr, 0, 0};\n\n"

  ctx = _make_kernel_ctx(
    node.source_specs, pipeline, rel_index_types,
    is_counting=True, bg_enabled=True, output_var_name="output_ctx",
  )
  if node.dest_specs:
    ctx.output_vars[node.dest_specs[0].rel_name] = "output_ctx"
    for i in range(1, len(node.dest_specs)):
      ctx.output_vars[node.dest_specs[i].rel_name] = "__skip_counting__"
  code += jit_pipeline(pipeline, node.source_specs, ctx)
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
  code += (
    "  static __global__ void __launch_bounds__(kBlockSize) kernel_bg_materialize(\n"
  )
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
  code += (
    "    if (threadIdx.x < NumSources) "
    "{ s_views[threadIdx.x] = views[threadIdx.x]; }\n"
  )
  code += "    __syncthreads();\n"
  code += "    views = s_views;\n"
  code += "    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;\n"
  code += "    uint32_t warp_id = thread_id / kGroupSize;\n"
  code += "    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;\n"
  code += "    uint32_t num_threads = num_warps;\n"
  code += "    uint32_t thread_offset = thread_offsets[thread_id];\n\n"

  ctx = _make_kernel_ctx(
    node.source_specs, pipeline, rel_index_types,
    is_counting=False, tiled_cartesian=False, bg_enabled=True,
  )
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
    ctx.output_vars[dest.rel_name] = output_var
    if i == 0:
      ctx.output_var_name = output_var
  code += "\n"
  code += jit_pipeline(pipeline, node.source_specs, ctx)
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
  code += (
    "    if (threadIdx.x < NumSources) "
    "{ s_views[threadIdx.x] = views[threadIdx.x]; }\n"
  )
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
      f"    SpecCtx_{j} output_ctx_{j}{{output_data_{j}, atomic_write_pos_{j}, "
      "overflow_flag,\n"
    )
    code += (
      f"                         static_cast<uint32_t>(output_stride_{j}), "
      f"old_size_{j}, capacity}};\n\n"
    )

  ctx = _make_kernel_ctx(
    node.source_specs, pipeline, rel_index_types,
    is_counting=False, tiled_cartesian=False, bg_enabled=True,
  )
  for i, dest in enumerate(dest_specs):
    ctx.output_vars[dest.rel_name] = f"output_ctx_{i}"
  ctx.output_var_name = "output_ctx_0"
  code += jit_pipeline(pipeline, node.source_specs, ctx)
  for j in range(len(dest_specs)):
    code += f"    output_ctx_{j}.flush();\n"
  code += "  }\n\n"
  return code


# -----------------------------------------------------------------------------
# LaunchParams + forward declarations
# -----------------------------------------------------------------------------

def _gen_launch_params_struct(
  num_dests: int, is_fused_eligible: bool, is_block_group: bool = False,
  for_decl: bool = False,
) -> str:
  '''LaunchParams block — shared between `full` and `decl` emission.
  When `for_decl` is True the BG-block comment uses the decl variant
  ("must match JIT batch definition exactly!") to mirror Nim exactly.
  '''
  code = "  // State carried between decomposed phases\n"
  code += "  struct LaunchParams {\n"
  code += "    std::vector<ViewType> views_vec;\n"
  code += "    SRDatalog::GPU::DeviceArray<ViewType> d_views{0};\n"
  code += "    SRDatalog::GPU::DeviceArray<uint32_t> thread_counts{0};\n"
  code += "    uint32_t* thread_counts_ptr = nullptr;\n"
  code += "    uint32_t num_unique_root_keys = 0;\n"
  code += "    uint32_t num_root_keys = 0;\n"
  code += "    const ValueType* root_unique_values_ptr = nullptr;\n"
  code += "    const ValueType* head_root_unique_values_ptr = nullptr;\n"
  code += "    uint32_t num_full_unique_root_keys = 0;\n"
  code += "    uint32_t num_head_unique_root_keys = 0;\n"
  code += "    uint32_t num_blocks = 0;\n"
  code += "    uint32_t num_threads = 0;\n"
  if is_block_group:
    if for_decl:
      code += (
        "    // Block-group state (must match JIT batch definition exactly!)\n"
      )
    else:
      code += (
        "    // Block-group state "
        "(raw ptr to avoid DeviceArray default ctor using wrong stream)\n"
      )
    code += "    uint64_t* bg_cumulative_work_ptr = nullptr;\n"
    code += "    uint64_t bg_total_work = 0;\n"
    code += "    uint32_t bg_num_blocks = 0;\n"
    code += "    uint32_t bg_num_threads = 0;\n"
  for i in range(num_dests):
    code += f"    uint32_t old_size_{i} = 0;\n"
  if is_fused_eligible:
    code += "    // Fused execution state (set by orchestrator)\n"
    for j in range(num_dests):
      code += f"    uint32_t* fused_wp_ptr_{j} = nullptr;\n"
    code += "    uint32_t* fused_of_ptr = nullptr;\n"
    code += "    uint32_t fused_capacity = 0;\n"
    for j in range(num_dests):
      code += f"    uint32_t fused_h_wp_{j} = 0;\n"
    code += "    bool fused_overflow = false;\n"
  code += "  };\n\n"
  return code


def _gen_method_forward_decls(
  is_count: bool, is_fused_eligible: bool,
) -> str:
  code = "  // Phase-decomposed methods for stream-parallel execution\n"
  code += (
    "  static LaunchParams setup(DB& db, uint32_t iteration, "
    "GPU_STREAM_T stream = 0);\n"
  )
  code += (
    "  static void launch_count(LaunchParams& p, GPU_STREAM_T stream = 0);\n"
  )
  code += (
    "  static uint32_t scan_and_resize(DB& db, LaunchParams& p, "
    "GPU_STREAM_T stream = 0);\n"
  )
  code += "  static void scan_only(LaunchParams& p, GPU_STREAM_T stream = 0);\n"
  code += "  static uint32_t read_total(LaunchParams& p);\n"
  if not is_count:
    code += (
      "  static void launch_materialize(DB& db, LaunchParams& p, "
      "uint32_t total_count, GPU_STREAM_T stream = 0);\n\n"
    )

  code += "  // Non-template execute - calls kernels directly\n"
  code += "  static void execute(DB& db, uint32_t iteration);\n"
  if is_fused_eligible:
    code += "  static void execute_fused(DB& db, uint32_t iteration);\n"
    code += (
      "  static void launch_fused(DB& db, LaunchParams& p, "
      "GPU_STREAM_T stream = 0);\n"
    )
    code += "  static void read_fused_result(LaunchParams& p);\n"
    code += "  static inline uint32_t prev_fused_total_ = 4096;\n"
  return code


# -----------------------------------------------------------------------------
# Phase method definitions (out-of-line)
# -----------------------------------------------------------------------------

def _gen_view_setup_for_source(
  i: int, src: m.MirNode, views_var: str, rel_index_types: dict[str, str],
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
    code += (
      f"    auto& rel_{i} = get_relation_by_schema<{schema}, {ver}>(db);\n"
    )
  force_rebuild = "true" if isinstance(src, m.Negation) else "false"
  code += (
    f"    auto& idx_{i} = rel_{i}.ensure_index(SRDatalog::IndexSpec"
    f"{idx_str}, {force_rebuild});\n"
  )
  view_count = plugin_view_count(mir_ver, index_type)
  if view_count == 1:
    code += f"    {views_var}.push_back(idx_{i}.view());\n"
  else:
    # Multi-view plugin (e.g. Device2LevelIndex FULL → 2 views)
    code += f"    {views_var}.push_back(idx_{i}.head().view());\n"
    code += f"    {views_var}.push_back(idx_{i}.view());\n"
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
    code += (
      f"  auto& first_rel = "
      f"get_relation_by_schema<{first_schema}, {first_version}>(db);\n"
    )
  code += (
    f"  auto& first_idx = first_rel.get_index(SRDatalog::IndexSpec"
    f"{_index_to_list(first_index)});\n"
  )
  code += f"  {prefix}num_root_keys = first_idx.root().degree();\n"
  code += (
    f"  {prefix}num_unique_root_keys = "
    "static_cast<uint32_t>(first_idx.num_unique_root_values());\n"
  )
  code += (
    f"  {prefix}root_unique_values_ptr = ({prefix}num_unique_root_keys > 0) "
    "? first_idx.root_unique_values().data() : nullptr;\n"
  )
  # 2-level first source: expose HEAD segment for BG dual-pointer iteration.
  first_index_type = rel_index_types.get(first_schema, "")
  first_view_count = plugin_view_count(first_version, first_index_type)
  if first_view_count > 1:
    code += (
      f"  {prefix}num_full_unique_root_keys = {prefix}num_unique_root_keys;\n"
    )
    code += (
      f"  {prefix}num_head_unique_root_keys = "
      "static_cast<uint32_t>(first_idx.head_num_unique_root_values());\n"
    )
    code += (
      f"  {prefix}head_root_unique_values_ptr = "
      f"({prefix}num_head_unique_root_keys > 0) ? "
      "first_idx.head_root_unique_values().data() : nullptr;\n"
    )
    code += (
      f"  {prefix}num_unique_root_keys += {prefix}num_head_unique_root_keys;\n"
    )
    code += f"  {prefix}num_root_keys += first_idx.head().root().degree();\n"
  else:
    code += (
      f"  {prefix}num_full_unique_root_keys = {prefix}num_unique_root_keys;\n"
    )
  code += "\n"
  return code


def _gen_grid_config_code(prefix: str, root_is_scan: bool) -> str:
  code = "  int num_sms = 0;\n"
  code += (
    "  GPU_DEVICE_GET_ATTRIBUTE(&num_sms, "
    "GPU_DEV_ATTR_MULTIPROCESSOR_COUNT, 0);\n"
  )
  if root_is_scan:
    code += "  // Binary join: row-based iteration over ALL rows\n"
    code += (
      f"  {prefix}num_threads = (({prefix}num_root_keys + kGroupSize - 1) "
      "/ kGroupSize) * kGroupSize;\n"
    )
  else:
    code += "  // WCOJ: unique-key-based iteration\n"
    code += (
      f"  {prefix}num_threads = (({prefix}num_unique_root_keys + "
      "kGroupSize - 1) / kGroupSize) * kGroupSize;\n"
    )
  code += (
    f"  {prefix}num_threads = std::max("
    f"{prefix}num_threads, static_cast<uint32_t>(kBlockSize));\n"
  )
  code += (
    f"  {prefix}num_blocks = ({prefix}num_threads + kBlockSize - 1) "
    "/ kBlockSize;\n"
  )
  code += (
    f"  {prefix}num_blocks = std::max("
    f"{prefix}num_blocks, static_cast<uint32_t>(num_sms) * 8);\n"
  )
  code += f"  {prefix}num_threads = {prefix}num_blocks * kBlockSize;\n\n"
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
    "p.", first_schema, first_version, first_index, rel_index_types,
  )
  code += "  // Copy views to device using provided stream (NOT stream 0)\n"
  code += (
    "  p.d_views = SRDatalog::GPU::DeviceArray<ViewType>(p.views_vec.size());\n"
  )
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
    code += (
      "    static rmm::device_uvector<uint64_t> s_bg_wk(0, "
      "rmm::cuda_stream_default);\n"
    )
    code += (
      "    static rmm::device_uvector<uint64_t> s_bg_cw(0, "
      "rmm::cuda_stream_default);\n"
    )
    code += "    if (s_bg_wk.size() < p.num_unique_root_keys) {\n"
    code += (
      "      s_bg_wk.resize(p.num_unique_root_keys, "
      "rmm::cuda_stream_view{stream});\n"
    )
    code += (
      "      s_bg_cw.resize(p.num_unique_root_keys, "
      "rmm::cuda_stream_view{stream});\n"
    )
    code += "    }\n"
    code += "    p.bg_cumulative_work_ptr = s_bg_cw.data();\n"
    code += "    uint64_t* bg_wk_ptr = s_bg_wk.data();\n"
    code += "    int bg_num_sms = 0;\n"
    code += (
      "    GPU_DEVICE_GET_ATTRIBUTE(&bg_num_sms, "
      "GPU_DEV_ATTR_MULTIPROCESSOR_COUNT, 0);\n"
    )
    code += (
      "    uint32_t hist_blocks = std::max((uint32_t)"
      "((p.num_unique_root_keys + (kBlockSize / kGroupSize) - 1) / "
      "(kBlockSize / kGroupSize)), (uint32_t)(bg_num_sms * 4));\n"
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
    code += (
      "    // BG helps when the hottest key's work exceeds what a warp processes\n"
    )
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

  code += (
    "  p.thread_counts = SRDatalog::GPU::DeviceArray<uint32_t>"
    "(p.num_threads + 1);\n"
  )
  code += (
    "  cudaMemsetAsync(p.thread_counts.data(), 0, "
    "(p.num_threads + 1) * sizeof(uint32_t), stream);\n"
  )
  code += "  p.thread_counts_ptr = p.thread_counts.data();\n"
  code += "  return p;\n"
  code += "}\n\n"
  return code


def _gen_launch_count(runner_prefix: str, is_block_group: bool = False) -> str:
  code = (
    f"void {runner_prefix}::launch_count"
    "(LaunchParams& p, GPU_STREAM_T stream) {\n"
  )
  code += "  if (p.num_threads == 0) return;\n"
  code += "  if (p.num_unique_root_keys == 0) {\n"
  code += (
    "    cudaMemsetAsync(p.thread_counts_ptr, 0, "
    "p.num_threads * sizeof(uint32_t), stream);\n"
  )
  code += "    return;\n"
  code += "  }\n"
  if is_block_group:
    code += "  if (p.bg_total_work > 0) {\n"
    code += (
      "    kernel_bg_count<<<p.num_blocks, kBlockSize, 0, stream>>>"
      "(p.d_views.data(), p.root_unique_values_ptr, "
      "p.head_root_unique_values_ptr, p.num_unique_root_keys, "
      "p.num_full_unique_root_keys, p.num_root_keys, "
      "p.bg_cumulative_work_ptr, p.bg_total_work, p.thread_counts_ptr);\n"
    )
    code += "  } else {\n"
    code += (
      "    kernel_count<<<p.num_blocks, kBlockSize, 0, stream>>>"
      "(p.d_views.data(), p.root_unique_values_ptr, "
      "p.num_unique_root_keys, p.num_root_keys, p.thread_counts_ptr);\n"
    )
    code += "  }\n"
  else:
    code += (
      "  kernel_count<<<p.num_blocks, kBlockSize, 0, stream>>>"
      "(p.d_views.data(), p.root_unique_values_ptr, "
      "p.num_unique_root_keys, p.num_root_keys, p.thread_counts_ptr);\n"
    )
  code += "}\n\n"
  return code


def _gen_scan_and_resize(
  node: m.ExecutePipeline, runner_prefix: str,
) -> str:
  code = "// Phase 3: Prefix scan + readback total + resize destinations\n"
  code += (
    f"uint32_t {runner_prefix}::scan_and_resize"
    "(DB& db, LaunchParams& p, GPU_STREAM_T stream) {\n"
  )
  code += (
    "  thrust::exclusive_scan(rmm::exec_policy(stream), "
    "p.thread_counts_ptr, p.thread_counts_ptr + p.num_threads + 1, "
    "p.thread_counts_ptr, 0, thrust::plus<uint32_t>());\n"
  )
  code += "  uint32_t total_count = 0;\n"
  code += (
    "  GPU_MEMCPY_ASYNC(&total_count, p.thread_counts_ptr + p.num_threads, "
    "sizeof(uint32_t), GPU_DEVICE_TO_HOST, stream);\n"
  )
  code += "  GPU_STREAM_SYNCHRONIZE(stream);\n"
  code += "  if (total_count == 0) return 0;\n\n"
  for i, dest in enumerate(node.dest_specs):
    code += (
      f"  auto& dest_rel_{i} = "
      f"get_relation_by_schema<{dest.rel_name}, NEW_VER>(db);\n"
    )
    code += (
      f"  p.old_size_{i} = static_cast<uint32_t>(dest_rel_{i}.size());\n"
    )
    code += (
      f"  dest_rel_{i}.resize_interned_columns"
      f"(p.old_size_{i} + total_count, stream);\n"
    )
  code += "  return total_count;\n"
  code += "}\n\n"
  return code


def _gen_scan_only(runner_prefix: str) -> str:
  code = "// Phase 3a: Prefix scan only (async, no sync)\n"
  code += (
    f"void {runner_prefix}::scan_only"
    "(LaunchParams& p, GPU_STREAM_T stream) {\n"
  )
  code += "  if (p.num_threads == 0) return;\n"
  code += (
    "  thrust::exclusive_scan(rmm::exec_policy(stream), "
    "p.thread_counts_ptr, p.thread_counts_ptr + p.num_threads + 1, "
    "p.thread_counts_ptr, 0, thrust::plus<uint32_t>());\n"
  )
  code += "}\n\n"
  return code


def _gen_read_total(runner_prefix: str) -> str:
  code = "// Phase 3b: Read total count (call after device sync)\n"
  code += f"uint32_t {runner_prefix}::read_total(LaunchParams& p) {{\n"
  code += "  if (p.num_threads == 0) return 0;\n"
  code += "  uint32_t total_count = 0;\n"
  code += (
    "  GPU_MEMCPY(&total_count, p.thread_counts_ptr + p.num_threads, "
    "sizeof(uint32_t), GPU_DEVICE_TO_HOST);\n"
  )
  code += "  return total_count;\n"
  code += "}\n\n"
  return code


def _gen_launch_materialize(
  node: m.ExecutePipeline, runner_prefix: str,
) -> str:
  code = "// Phase 4: Launch materialize kernel on given stream (no sync)\n"
  code += (
    f"void {runner_prefix}::launch_materialize"
    "(DB& db, LaunchParams& p, uint32_t total_count, GPU_STREAM_T stream) {\n"
  )
  code += "  using ProvPtrType = semiring_value_t<SR>*;\n"
  code += "  ProvPtrType prov_ptr = nullptr;\n\n"
  for i, dest in enumerate(node.dest_specs):
    code += (
      f"  auto& dest_rel_{i} = "
      f"get_relation_by_schema<{dest.rel_name}, NEW_VER>(db);\n"
    )
    code += f"  uint32_t old_size_{i} = p.old_size_{i};\n"

  def _baseline_launch() -> str:
    out = "  kernel_materialize<<<p.num_blocks, kBlockSize, 0, stream>>>(\n"
    out += (
      "      p.d_views.data(), p.root_unique_values_ptr, "
      "p.num_unique_root_keys, p.num_root_keys,\n"
    )
    out += "      p.thread_counts_ptr"
    for i in range(len(node.dest_specs)):
      out += (
        f", dest_rel_{i}.template interned_column<0>(), "
        f"prov_ptr, dest_rel_{i}.interned_stride(), old_size_{i}"
      )
    out += ");\n"
    return out

  if node.block_group:
    code += "\n  if (p.bg_total_work > 0) {\n"
    code += (
      "    kernel_bg_materialize<<<p.bg_num_blocks, kBlockSize, 0, stream>>>(\n"
    )
    code += (
      "        p.d_views.data(), p.root_unique_values_ptr, "
      "p.head_root_unique_values_ptr, p.num_unique_root_keys, "
      "p.num_full_unique_root_keys, p.num_root_keys, "
      "p.bg_cumulative_work_ptr, p.bg_total_work, p.thread_counts_ptr"
    )
    for i in range(len(node.dest_specs)):
      code += (
        f", dest_rel_{i}.template interned_column<0>(), prov_ptr, "
        f"dest_rel_{i}.interned_stride(), old_size_{i}"
      )
    code += ");\n"
    code += "  } else {\n  "
    code += _baseline_launch().replace("\n", "\n  ").rstrip("  ")
    code += "}\n"
  else:
    code += _baseline_launch()
  code += "}\n\n"
  return code


def _gen_launch_fused(
  node: m.ExecutePipeline, runner_prefix: str,
) -> str:
  code = "// launch_fused: launch fused kernel on given stream (no sync)\n"
  code += (
    f"void {runner_prefix}::launch_fused"
    "(DB& db, LaunchParams& p, GPU_STREAM_T stream) {\n"
  )
  code += "  if (p.num_unique_root_keys == 0) return;\n\n"
  for i, dest in enumerate(node.dest_specs):
    code += (
      f"  auto& dest_rel_{i} = "
      f"get_relation_by_schema<{dest.rel_name}, NEW_VER>(db);\n"
    )

  if node.block_group:
    # BG fused: compute histogram each call, launch kernel_bg_fused.
    code += "\n  // BG fused: compute histogram for block assignment (stream-ordered)\n"
    code += "  int num_sms = 0;\n"
    code += (
      "  GPU_DEVICE_GET_ATTRIBUTE(&num_sms, "
      "GPU_DEV_ATTR_MULTIPROCESSOR_COUNT, 0);\n"
    )
    code += (
      "  static SRDatalog::GPU::DeviceArray<uint64_t> bg_work_per_key;\n"
    )
    code += "  bg_work_per_key.resize(p.num_unique_root_keys);\n"
    code += "  {\n"
    code += (
      "    uint32_t hist_blocks = std::max((uint32_t)"
      "((p.num_unique_root_keys + (kBlockSize / kGroupSize) - 1) / "
      "(kBlockSize / kGroupSize)), (uint32_t)(num_sms * 4));\n"
    )
    code += (
      "    kernel_bg_histogram<<<hist_blocks, kBlockSize, 0, stream>>>"
      "(p.d_views.data(), p.root_unique_values_ptr, "
      "p.head_root_unique_values_ptr, p.num_unique_root_keys, "
      "p.num_full_unique_root_keys, p.num_root_keys, "
      "bg_work_per_key.data());\n"
    )
    code += "  }\n"
    code += (
      "  static SRDatalog::GPU::DeviceArray<uint64_t> bg_cumulative_work;\n"
    )
    code += "  bg_cumulative_work.resize(p.num_unique_root_keys);\n"
    code += (
      "  thrust::inclusive_scan(rmm::exec_policy(stream), "
      "bg_work_per_key.data(), bg_work_per_key.data() + "
      "p.num_unique_root_keys, bg_cumulative_work.data());\n"
    )
    code += "  uint64_t bg_total_work = 0;\n"
    code += "  cudaStreamSynchronize(stream);\n"
    code += (
      "  GPU_MEMCPY(&bg_total_work, bg_cumulative_work.data() + "
      "p.num_unique_root_keys - 1, sizeof(uint64_t), GPU_DEVICE_TO_HOST);\n"
    )
    code += "  if (bg_total_work == 0) return;\n"
    code += "  uint32_t bg_num_blocks = num_sms * 8;\n\n"

    code += (
      "  kernel_bg_fused<<<bg_num_blocks, kBlockSize, 0, stream>>>(\n"
    )
    code += (
      "      p.d_views.data(), p.root_unique_values_ptr, "
      "p.head_root_unique_values_ptr, p.num_unique_root_keys, "
      "p.num_full_unique_root_keys, p.num_root_keys, "
      "bg_cumulative_work.data(), bg_total_work"
    )
    for i in range(len(node.dest_specs)):
      code += (
        f", dest_rel_{i}.template interned_column<0>(), "
        f"dest_rel_{i}.interned_stride(), p.old_size_{i}, "
        f"p.fused_wp_ptr_{i}"
      )
    code += ", p.fused_capacity, p.fused_of_ptr);\n"
  else:
    code += "  kernel_fused<<<p.num_blocks, kBlockSize, 0, stream>>>(\n"
    code += (
      "      p.d_views.data(), p.root_unique_values_ptr, "
      "p.num_unique_root_keys, p.num_root_keys,\n"
    )
    parts = []
    for i in range(len(node.dest_specs)):
      parts.append(
        f"dest_rel_{i}.template interned_column<0>(), "
        f"dest_rel_{i}.interned_stride(), p.old_size_{i}, p.fused_wp_ptr_{i}"
      )
    code += "      " + ", ".join(parts) + ",\n"
    code += "      p.fused_capacity, p.fused_of_ptr);\n"
  code += "}\n\n"
  return code


def _gen_read_fused_result(
  node: m.ExecutePipeline, runner_prefix: str,
) -> str:
  code = "// read_fused_result: readback fused write counts (call after device sync)\n"
  code += f"void {runner_prefix}::read_fused_result(LaunchParams& p) {{\n"
  for i in range(len(node.dest_specs)):
    code += (
      f"  GPU_MEMCPY(&p.fused_h_wp_{i}, p.fused_wp_ptr_{i}, "
      "sizeof(uint32_t), GPU_DEVICE_TO_HOST);\n"
    )
  code += "  uint32_t h_of = 0;\n"
  code += (
    "  GPU_MEMCPY(&h_of, p.fused_of_ptr, sizeof(uint32_t), "
    "GPU_DEVICE_TO_HOST);\n"
  )
  code += "  p.fused_overflow = (h_of != 0);\n"
  code += "}\n\n"
  return code


def _gen_execute(
  rule_name: str, runner_prefix: str, is_count: bool,
  is_block_group: bool = False, dest_specs: list | None = None,
) -> str:
  code = "// Execute definition - calls decomposed phases sequentially\n"
  code += f"void {runner_prefix}::execute(DB& db, uint32_t iteration) {{\n"
  code += f'  nvtxRangePushA("{rule_name}");\n\n'

  if is_block_group and not is_count:
    # BG execute: adaptive fallback for small deltas, then 5-step BG.
    code += "  auto p = setup(db, iteration);\n"
    code += (
      "  if (p.num_unique_root_keys == 0) { nvtxRangePop(); return; }\n\n"
    )
    code += (
      "  // Adaptive: fall back to baseline for small deltas "
      "(histogram overhead > benefit)\n"
    )
    code += "  constexpr uint32_t kBGAdaptiveThreshold = 256;\n"
    code += (
      "  if (p.num_root_keys < kBGAdaptiveThreshold || "
      "p.num_unique_root_keys < 32) {\n"
    )
    code += "    launch_count(p, 0);\n"
    code += "    uint32_t total_count = scan_and_resize(db, p, 0);\n"
    code += (
      "    if (total_count == 0) { nvtxRangePop(); return; }\n"
    )
    code += "    launch_materialize(db, p, total_count, 0);\n"
    code += "    nvtxRangePop();\n"
    code += "    return;\n"
    code += "  }\n\n"
    code += "  int num_sms = 0;\n"
    code += (
      "  GPU_DEVICE_GET_ATTRIBUTE(&num_sms, "
      "GPU_DEV_ATTR_MULTIPROCESSOR_COUNT, 0);\n\n"
    )
    # Step 1: histogram
    code += (
      "  // BG Step 1: Per-key work histogram (grid-stride, warp-reduced)\n"
    )
    code += (
      "  SRDatalog::GPU::DeviceArray<uint64_t> "
      "bg_work_per_key(p.num_unique_root_keys);\n"
    )
    code += "  {\n"
    code += (
      "    uint32_t hist_blocks = std::max((uint32_t)"
      "((p.num_unique_root_keys + (kBlockSize / kGroupSize) - 1) / "
      "(kBlockSize / kGroupSize)), (uint32_t)(num_sms * 4));\n"
    )
    code += (
      "    kernel_bg_histogram<<<hist_blocks, kBlockSize>>>"
      "(p.d_views.data(), p.root_unique_values_ptr, "
      "p.head_root_unique_values_ptr, p.num_unique_root_keys, "
      "p.num_full_unique_root_keys, p.num_root_keys, "
      "bg_work_per_key.data());\n"
    )
    code += "    cudaDeviceSynchronize();\n"
    code += "  }\n\n"
    # Step 2: prefix sum
    code += "  // BG Step 2: Inclusive prefix sum on per-key work\n"
    code += (
      "  SRDatalog::GPU::DeviceArray<uint64_t> "
      "bg_cumulative_work(p.num_unique_root_keys);\n"
    )
    code += (
      "  thrust::inclusive_scan(thrust::device, bg_work_per_key.data(), "
      "bg_work_per_key.data() + p.num_unique_root_keys, "
      "bg_cumulative_work.data());\n"
    )
    code += "  cudaDeviceSynchronize();\n"
    code += (
      "  GPU_MEMCPY(&p.bg_total_work, bg_cumulative_work.data() + "
      "p.num_unique_root_keys - 1, sizeof(uint64_t), GPU_DEVICE_TO_HOST);\n"
    )
    code += (
      "  if (p.bg_total_work == 0) { nvtxRangePop(); return; }\n\n"
    )
    # Step 3: BG count kernel
    code += "  // BG Step 3: Block-group count kernel\n"
    code += "  p.bg_num_blocks = num_sms * 8;\n"
    code += "  p.bg_num_threads = p.bg_num_blocks * kBlockSize;\n"
    code += (
      "  p.thread_counts = SRDatalog::GPU::DeviceArray<uint32_t>"
      "(p.bg_num_threads + 1);\n"
    )
    code += (
      "  cudaMemset(p.thread_counts.data(), 0, "
      "(p.bg_num_threads + 1) * sizeof(uint32_t));\n"
    )
    code += "  p.thread_counts_ptr = p.thread_counts.data();\n"
    code += (
      "  kernel_bg_count<<<p.bg_num_blocks, kBlockSize>>>"
      "(p.d_views.data(), p.root_unique_values_ptr, "
      "p.head_root_unique_values_ptr, p.num_unique_root_keys, "
      "p.num_full_unique_root_keys, p.num_root_keys, "
      "bg_cumulative_work.data(), p.bg_total_work, "
      "p.thread_counts_ptr);\n"
    )
    code += "  cudaDeviceSynchronize();\n\n"
    # Step 4: prefix sum + resize
    code += "  // BG Step 4: Prefix sum + resize\n"
    code += (
      "  thrust::exclusive_scan(thrust::device, p.thread_counts_ptr, "
      "p.thread_counts_ptr + p.bg_num_threads + 1, p.thread_counts_ptr, "
      "0, thrust::plus<uint32_t>());\n"
    )
    code += "  uint32_t total_count = 0;\n"
    code += (
      "  GPU_MEMCPY(&total_count, p.thread_counts_ptr + p.bg_num_threads, "
      "sizeof(uint32_t), GPU_DEVICE_TO_HOST);\n"
    )
    code += (
      "  if (total_count == 0) { nvtxRangePop(); return; }\n\n"
    )
    # Resize dest + launch bg_materialize
    assert dest_specs is not None
    for i, dest in enumerate(dest_specs):
      code += (
        f"  auto& bg_dest_{i} = "
        f"get_relation_by_schema<{dest.rel_name}, NEW_VER>(db);\n"
      )
      code += (
        f"  uint32_t bg_old_size_{i} = "
        f"static_cast<uint32_t>(bg_dest_{i}.size());\n"
      )
      code += (
        f"  bg_dest_{i}.resize_interned_columns(bg_old_size_{i} + total_count);\n"
      )
    code += "\n  // BG Step 5: Block-group materialize kernel\n"
    code += (
      "  kernel_bg_materialize<<<p.bg_num_blocks, kBlockSize>>>"
      "(p.d_views.data(), p.root_unique_values_ptr, "
      "p.head_root_unique_values_ptr, p.num_unique_root_keys, "
      "p.num_full_unique_root_keys, p.num_root_keys, "
      "bg_cumulative_work.data(), p.bg_total_work, "
      "p.thread_counts_ptr"
    )
    for i in range(len(dest_specs)):
      code += (
        f", bg_dest_{i}.template interned_column<0>(), "
        f"static_cast<semiring_value_t<SR>*>(nullptr), "
        f"bg_dest_{i}.interned_stride(), bg_old_size_{i}"
      )
    code += ");\n"
    code += "  cudaDeviceSynchronize();\n"
    code += "  nvtxRangePop();\n"
    code += "}\n\n"
    return code

  # Baseline (non-BG) execute.
  code += "  auto p = setup(db, iteration);\n"
  code += "  launch_count(p, 0);\n"
  if is_count:
    code += "  scan_only(p, 0);\n"
    code += "  GPU_STREAM_SYNCHRONIZE(0);\n"
    code += "  uint32_t total_count = read_total(p);\n"
    code += "  if (total_count == 0) { nvtxRangePop(); return; }\n\n"
  else:
    code += "  uint32_t total_count = scan_and_resize(db, p, 0);\n"
    code += "  if (total_count == 0) { nvtxRangePop(); return; }\n\n"
    code += "  launch_materialize(db, p, total_count, 0);\n"
  code += "  nvtxRangePop();\n"
  code += "}\n\n"
  return code


def _gen_execute_fused(
  node: m.ExecutePipeline, runner_prefix: str,
) -> str:
  code = "// Tail-mode fused execution: single kernel, no count/scan phase\n"
  code += f"void {runner_prefix}::execute_fused(DB& db, uint32_t iteration) {{\n"
  code += "  auto p = setup(db, iteration);\n"
  code += "  if (p.num_unique_root_keys == 0) return;\n\n"
  for i, dest in enumerate(node.dest_specs):
    code += (
      f"  auto& dest_rel_{i} = "
      f"get_relation_by_schema<{dest.rel_name}, NEW_VER>(db);\n"
    )
    code += (
      f"  uint32_t old_size_{i} = static_cast<uint32_t>(dest_rel_{i}.size());\n"
    )
  code += "  uint32_t capacity = prev_fused_total_;\n"
  for i in range(len(node.dest_specs)):
    code += (
      f"  dest_rel_{i}.resize_interned_columns(old_size_{i} + capacity);\n"
    )
  code += "\n"
  for i in range(len(node.dest_specs)):
    code += (
      f"  static SRDatalog::GPU::DeviceArray<uint32_t> s_wp_{i}(1);\n"
    )
    code += (
      f"  cudaMemsetAsync(s_wp_{i}.data(), 0, sizeof(uint32_t), 0);\n"
    )
  code += "  static SRDatalog::GPU::DeviceArray<uint32_t> s_of(1);\n"
  code += "  cudaMemsetAsync(s_of.data(), 0, sizeof(uint32_t), 0);\n"

  if node.block_group:
    code += "\n  // BG fused: compute histogram for block assignment\n"
    code += "  int num_sms = 0;\n"
    code += (
      "  GPU_DEVICE_GET_ATTRIBUTE(&num_sms, "
      "GPU_DEV_ATTR_MULTIPROCESSOR_COUNT, 0);\n"
    )
    code += (
      "  SRDatalog::GPU::DeviceArray<uint64_t> "
      "bg_work_per_key(p.num_unique_root_keys);\n"
    )
    code += "  {\n"
    code += (
      "    uint32_t hist_blocks = std::max((uint32_t)"
      "((p.num_unique_root_keys + (kBlockSize / kGroupSize) - 1) / "
      "(kBlockSize / kGroupSize)), (uint32_t)(num_sms * 4));\n"
    )
    code += (
      "    kernel_bg_histogram<<<hist_blocks, kBlockSize>>>"
      "(p.d_views.data(), p.root_unique_values_ptr, "
      "p.head_root_unique_values_ptr, p.num_unique_root_keys, "
      "p.num_full_unique_root_keys, p.num_root_keys, "
      "bg_work_per_key.data());\n"
    )
    code += "    cudaDeviceSynchronize();\n"
    code += "  }\n"
    code += (
      "  SRDatalog::GPU::DeviceArray<uint64_t> "
      "bg_cumulative_work(p.num_unique_root_keys);\n"
    )
    code += (
      "  thrust::inclusive_scan(thrust::device, bg_work_per_key.data(), "
      "bg_work_per_key.data() + p.num_unique_root_keys, "
      "bg_cumulative_work.data());\n"
    )
    code += "  uint64_t bg_total_work = 0;\n"
    code += (
      "  GPU_MEMCPY(&bg_total_work, bg_cumulative_work.data() + "
      "p.num_unique_root_keys - 1, sizeof(uint64_t), GPU_DEVICE_TO_HOST);\n"
    )
    code += "  if (bg_total_work == 0) {\n"
    for i in range(len(node.dest_specs)):
      code += f"    dest_rel_{i}.resize_interned_columns(old_size_{i});\n"
    code += "    return;\n"
    code += "  }\n"
    code += "  uint32_t bg_num_blocks = num_sms * 8;\n\n"

    code += "  kernel_bg_fused<<<bg_num_blocks, kBlockSize>>>(\n"
    code += (
      "      p.d_views.data(), p.root_unique_values_ptr, "
      "p.head_root_unique_values_ptr, p.num_unique_root_keys, "
      "p.num_full_unique_root_keys, p.num_root_keys, "
      "bg_cumulative_work.data(), bg_total_work"
    )
    for i in range(len(node.dest_specs)):
      code += (
        f", dest_rel_{i}.template interned_column<0>(), "
        f"dest_rel_{i}.interned_stride(), old_size_{i}, s_wp_{i}.data()"
      )
    code += ", capacity, s_of.data());\n"
  else:
    code += "  kernel_fused<<<p.num_blocks, kBlockSize>>>(\n"
    code += (
      "      p.d_views.data(), p.root_unique_values_ptr, "
      "p.num_unique_root_keys, p.num_root_keys,\n"
    )
    parts = []
    for i in range(len(node.dest_specs)):
      parts.append(
        f"dest_rel_{i}.template interned_column<0>(), "
        f"dest_rel_{i}.interned_stride(), old_size_{i}, s_wp_{i}.data()"
      )
    code += "      " + ", ".join(parts) + ",\n"
    code += "      capacity, s_of.data());\n"

  code += "  GPU_DEVICE_SYNCHRONIZE();\n"
  code += "  uint32_t h_of = 0;\n"
  for i in range(len(node.dest_specs)):
    code += f"  uint32_t h_wp_{i} = 0;\n"
    code += (
      f"  GPU_MEMCPY(&h_wp_{i}, s_wp_{i}.data(), sizeof(uint32_t), "
      "GPU_DEVICE_TO_HOST);\n"
    )
  code += (
    "  GPU_MEMCPY(&h_of, s_of.data(), sizeof(uint32_t), GPU_DEVICE_TO_HOST);\n"
  )
  code += "  if (!h_of) {\n"
  for i in range(len(node.dest_specs)):
    code += (
      f"    dest_rel_{i}.resize_interned_columns(old_size_{i} + h_wp_{i});\n"
    )
  code += "    uint32_t max_wp = 0;\n"
  for i in range(len(node.dest_specs)):
    code += f"    max_wp = std::max(max_wp, h_wp_{i});\n"
  code += "    prev_fused_total_ = std::max(max_wp, 1u) * 4;\n"
  code += "  } else {\n"
  for i in range(len(node.dest_specs)):
    code += f"    dest_rel_{i}.resize_interned_columns(old_size_{i});\n"
  code += "    uint32_t max_wp = 0;\n"
  for i in range(len(node.dest_specs)):
    code += f"    max_wp = std::max(max_wp, h_wp_{i});\n"
  code += "    prev_fused_total_ = max_wp * 4;\n"
  code += "    execute(db, iteration);\n"
  code += "  }\n"
  code += "}\n"
  return code


# -----------------------------------------------------------------------------
# Struct-wide emission (typedefs header + declaration variant)
# -----------------------------------------------------------------------------

def _gen_struct_type_aliases(
  rule_name: str,
  db_type_name: str,
  first_schema: str,
  first_version: str,
  dest_specs: list[m.MirNode],
  dest_arities: list[int],
  total_view_count: int,
) -> str:
  '''Type alias block shared between `full` and `decl`. Does NOT include
  `struct JitRunner_X {` or the closing brace.
  '''
  code = "  // Type aliases - all concrete, resolved at Nim JIT time\n"
  code += f"  using DB = {db_type_name};\n"
  code += f"  using FirstSchema = {first_schema};\n"
  if dest_specs:
    code += f"  using DestSchema = {dest_specs[0].rel_name};\n"
  else:
    code += "  using DestSchema = void;\n"
  code += "  using SR = NoProvenance;\n"
  code += "  using ValueType = typename FirstSchema::intern_value_type;\n"
  code += (
    "  using RelType = std::decay_t<decltype(get_relation_by_schema"
    f"<FirstSchema, {first_version}>(std::declval<DB&>()))>;\n"
  )
  code += "  using IndexType = typename RelType::IndexTypeInst;\n"
  code += "  using ViewType = typename IndexType::NodeView;\n"
  code += (
    "  static constexpr auto Layout = SRDatalog::GPU::StorageLayout::SoA;\n"
  )
  code += "  static constexpr int kBlockSize = 256;\n"
  code += "  static constexpr int kGroupSize = 32;\n"
  for i, arity in enumerate(dest_arities):
    code += f"  static constexpr std::size_t OutputArity_{i} = {arity};\n"
    if i == 0:
      code += (
        "  static constexpr std::size_t OutputArity = OutputArity_0;"
        "  // Legacy alias\n"
      )
  code += (
    f"  static constexpr std::size_t NumSources = {total_view_count};\n\n"
  )
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
    raise NotImplementedError(
      "gen_complete_runner: balanced-scan runner not yet ported"
    )

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
    rule_name, db_type_name, first_schema, first_version,
    node.dest_specs, dest_arities, total_view_count,
  )

  full += _gen_kernel_count(node, mutable_pipe, rel_index_types)

  materialize_body = ""
  if is_count:
    # Emit materialize to a discard buffer (matches Nim — it still emits
    # so jit_pipeline side effects on handle registration etc. fire, but
    # drops the resulting code).
    _mat_code, materialize_body = _gen_kernel_materialize(
      node, mutable_pipe, rel_index_types, tiled_cartesian_eligible,
    )
  else:
    mat_code, materialize_body = _gen_kernel_materialize(
      node, mutable_pipe, rel_index_types, tiled_cartesian_eligible,
    )
    full += mat_code

  # Baseline `kernel_fused` is skipped when BG is enabled — Nim emits
  # only `kernel_bg_fused` for BG rules (see jit_complete_runner.nim:516:
  # `if isFusedEligible and not isBlockGroup`).
  if is_fused_eligible and not node.block_group:
    full += _gen_kernel_fused(
      node, mutable_pipe, rel_index_types,
      materialize_body, tiled_cartesian_eligible,
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
    len(node.dest_specs), is_fused_eligible,
    is_block_group=node.block_group,
  )
  full += _gen_method_forward_decls(is_count, is_fused_eligible)
  full += "};\n\n"

  full += _gen_setup(
    node, runner_prefix, first_schema, first_version, first_index,
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
    rule_name, runner_prefix, is_count,
    is_block_group=node.block_group, dest_specs=node.dest_specs,
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
    rule_name, db_type_name, first_schema, first_version,
    node.dest_specs, dest_arities, total_view_count,
  )
  decl_aliases = decl_aliases.replace(
    "static constexpr std::size_t OutputArity = OutputArity_0;"
    "  // Legacy alias\n",
    "static constexpr std::size_t OutputArity = OutputArity_0;\n",
  )
  # Decl also drops the "// Type aliases - all concrete..." comment.
  decl_aliases = decl_aliases.replace(
    "  // Type aliases - all concrete, resolved at Nim JIT time\n", "",
  )
  decl += decl_aliases
  decl += _gen_launch_params_struct(
    len(node.dest_specs), is_fused_eligible,
    is_block_group=node.block_group, for_decl=True,
  )
  # The decl variant drops the "// State carried..." comment (matches Nim).
  decl = decl.replace("  // State carried between decomposed phases\n", "")
  decl += _gen_method_forward_decls(is_count, is_fused_eligible)
  # Drop the "// Phase-decomposed methods..." comment in decl (matches Nim).
  decl = decl.replace(
    "  // Phase-decomposed methods for stream-parallel execution\n", "",
  )
  # Drop the "// Non-template execute..." comment in decl (matches Nim).
  decl = decl.replace("  // Non-template execute - calls kernels directly\n", "")
  decl += "};\n\n"

  return decl, full
