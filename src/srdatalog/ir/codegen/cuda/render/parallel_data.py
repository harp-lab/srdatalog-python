'''CUDA renderer for the parallel.data dialect.

Per docs/stage3a_execution_plan.md §7 tasks S3A.3 + S3A.9b.

Two pieces of CUDA emission live here:

  - `_render_bg_root_cj_multi` — the BgRootCjMulti op renderer
    (split from the legacy codegen/cuda/emit.py 41-case match).
  - `emit_bg_histogram_kernel` — the standalone histogram kernel
    template (a per-rule kernel that's not part of the BG body
    rendering; called by complete_runner.py during runner emit).
    Relocated S3A.9b from dialects/parallel/data/block_group.py
    so the dialect file contains only ops + their helper data
    (BgSourceSpec); CUDA emission lives in the codegen, not
    inside the dialect.
'''

from __future__ import annotations

import srdatalog.ir.mir.types as m
from srdatalog.ir.codegen.cuda.render import EmitCtx, emit, register_render
from srdatalog.ir.dialects.parallel.data.block_group import BgRootCjMulti


@register_render(BgRootCjMulti, mode='stmt')
def _render_bg_root_cj_multi(op: BgRootCjMulti, ctx: EmitCtx) -> str:
  '''Emit the block-group root multi-source ColumnJoin scaffold,
  lifted byte-for-byte from legacy `jit_root_column_join_block_group`.

  ctx.indent_level controls the outer indent. The body emits at
  `ctx.indent_level + 1 + segs` (where segs = number of multi-view
  non-first sources contributing a `_bg_seg_<idx>` loop). All other
  string-level indents are derived from `ctx.ind()`.
  '''
  i = ctx.ind()
  ii = i + '  '
  first = op.sources[0]
  parts: list[str] = []

  # Block-level work assignment preamble.
  parts.append(f'{i}static constexpr int kWarpsPerBlock = kBlockSize / kGroupSize;\n')
  parts.append(f'{i}uint64_t bg_work_per_block = (bg_total_work + gridDim.x - 1) / gridDim.x;\n')
  parts.append(f'{i}uint64_t bg_block_begin = (uint64_t)blockIdx.x * bg_work_per_block;\n')
  parts.append(f'{i}uint64_t bg_block_end = bg_block_begin + bg_work_per_block;\n')
  parts.append(f'{i}if (bg_block_end > bg_total_work) bg_block_end = bg_total_work;\n')
  parts.append(f'{i}if (bg_block_begin >= bg_total_work) {{\n')
  if op.is_counting:
    parts.append(f'{i}  thread_counts[thread_id] = 0;\n')
  parts.append(f'{i}  return;\n')
  parts.append(f'{i}}}\n\n')

  # Binary search cumulative_work for starting key.
  parts.append(f'{i}uint32_t bg_key_lo = 0, bg_key_hi = num_unique_root_keys;\n')
  parts.append(f'{i}while (bg_key_lo < bg_key_hi) {{\n')
  parts.append(f'{i}  uint32_t bg_mid = bg_key_lo + (bg_key_hi - bg_key_lo) / 2;\n')
  parts.append(
    f'{i}  if (bg_cumulative_work[bg_mid] <= (uint64_t)bg_block_begin) bg_key_lo = bg_mid + 1;\n'
  )
  parts.append(f'{i}  else bg_key_hi = bg_mid;\n')
  parts.append(f'{i}}}\n\n')

  parts.append(f'{i}uint64_t bg_remaining_begin = bg_block_begin;\n')
  parts.append(f'{i}uint64_t bg_remaining_end = bg_block_end;\n\n')

  # Key loop opens.
  parts.append(
    f'{i}for (uint32_t {op.key_idx_var} = bg_key_lo; '
    f'{op.key_idx_var} < num_unique_root_keys && '
    f'bg_remaining_begin < bg_remaining_end; '
    f'{op.key_idx_var}++) {{\n'
  )

  parts.append(f'{ii}auto {op.root_val_var} = root_unique_values[{op.key_idx_var}];\n')

  # Per-key work range.
  parts.append(
    f'{ii}uint64_t bg_key_work_start = ({op.key_idx_var} > 0) ? '
    f'bg_cumulative_work[{op.key_idx_var} - 1] : 0;\n'
  )
  parts.append(f'{ii}uint64_t bg_key_work_end = bg_cumulative_work[{op.key_idx_var}];\n')
  parts.append(f'{ii}if (bg_key_work_end <= bg_remaining_begin) continue;\n')
  parts.append(f'{ii}if (bg_key_work_start >= bg_remaining_end) break;\n\n')
  parts.append(
    f'{ii}uint64_t bg_my_begin_in_key = '
    f'(bg_remaining_begin > bg_key_work_start) ? '
    f'(bg_remaining_begin - bg_key_work_start) : 0;\n'
  )
  parts.append(
    f'{ii}uint64_t bg_my_end_in_key = '
    f'(bg_remaining_end < bg_key_work_end) ? '
    f'(bg_remaining_end - bg_key_work_start) : '
    f'(bg_key_work_end - bg_key_work_start);\n\n'
  )

  # Per-source handle prefix. First source uses key_idx hint; multi-view
  # non-first sources defer their handle bind to a segment loop. The
  # view variable is already declared at kernel start (via
  # `emit_view_declarations`) so we don't re-declare it here.
  bg_seg_specs: list[tuple] = []  # (idx, spec, seg_var)
  for idx_, src in enumerate(op.sources):
    is_first = idx_ == 0
    is_deferred = (not is_first) and src.view_count > 1
    if is_first:
      parts.append(f'{ii}uint32_t {op.hint_lo} = {op.key_idx_var};\n')
      parts.append(
        f'{ii}uint32_t {op.hint_hi} = {src.view_var}.num_rows_ - '
        f'(num_unique_root_keys - {op.key_idx_var} - 1);\n'
      )
      parts.append(
        f'{ii}{op.hint_hi} = ({op.hint_hi} <= {src.view_var}.num_rows_) ? '
        f'{op.hint_hi} : {src.view_var}.num_rows_;\n'
      )
      parts.append(
        f'{ii}{op.hint_hi} = ({op.hint_hi} > {op.hint_lo}) ? '
        f'{op.hint_hi} : {src.view_var}.num_rows_;\n'
      )
      parts.append(
        f'{ii}auto {src.handle_var} = HandleType({op.hint_lo}, {op.hint_hi}, 0)'
        f'.prefix({op.root_val_var}, {ctx.tile_var}, {src.view_var});\n'
      )
    elif is_deferred:
      seg_var = f'_bg_seg_{idx_}'
      bg_seg_specs.append((idx_, src, seg_var))
      parts.append(
        f'{ii}auto {src.handle_var} = HandleType(0, '
        f'{src.view_var}.num_rows_, 0)'
        f'.prefix({op.root_val_var}, {ctx.tile_var}, {src.view_var});\n'
      )
    else:
      parts.append(
        f'{ii}auto {src.handle_var} = HandleType(0, '
        f'{src.view_var}.num_rows_, 0)'
        f'.prefix({op.root_val_var}, {ctx.tile_var}, {src.view_var});\n'
      )
    if not is_deferred:
      parts.append(
        f'{ii}if (!{src.handle_var}.valid()) {{ '
        f'bg_remaining_begin = bg_key_work_end; continue; }}\n'
      )

  # Warp redistribution within block (row-proportional on first source).
  first_handle = first.handle_var
  parts.append('\n')
  parts.append(f'{ii}// Distribute within-key work across warps in block (row-proportional)\n')
  parts.append(f'{ii}uint32_t bg_warp_in_block = threadIdx.x / kGroupSize;\n')
  parts.append(f'{ii}uint64_t bg_key_total_work = bg_key_work_end - bg_key_work_start;\n')
  parts.append(
    f'{ii}uint32_t bg_deg_first = (uint32_t)({first_handle}.end() - {first_handle}.begin());\n'
  )
  parts.append(
    f'{ii}uint32_t bg_block_row_begin = (uint32_t)'
    f'((bg_my_begin_in_key * (uint64_t)bg_deg_first) / bg_key_total_work);\n'
  )
  parts.append(
    f'{ii}uint32_t bg_block_row_end = (uint32_t)'
    f'((bg_my_end_in_key * (uint64_t)bg_deg_first) / bg_key_total_work);\n'
  )
  parts.append(f'{ii}if (bg_my_end_in_key >= bg_key_total_work) bg_block_row_end = bg_deg_first;\n')
  parts.append(
    f'{ii}if (bg_block_row_begin >= bg_block_row_end) {{ '
    f'bg_remaining_begin = bg_key_work_end; continue; }}\n\n'
  )
  parts.append(f'{ii}uint32_t bg_rows_in_block = bg_block_row_end - bg_block_row_begin;\n')
  parts.append(
    f'{ii}uint32_t bg_warp_row_size = (bg_rows_in_block + kWarpsPerBlock - 1) / kWarpsPerBlock;\n'
  )
  parts.append(
    f'{ii}uint32_t bg_warp_row_begin = bg_block_row_begin + bg_warp_in_block * bg_warp_row_size;\n'
  )
  parts.append(f'{ii}uint32_t bg_warp_row_end = bg_warp_row_begin + bg_warp_row_size;\n')
  parts.append(f'{ii}if (bg_warp_row_end > bg_block_row_end) bg_warp_row_end = bg_block_row_end;\n')
  parts.append(
    f'{ii}if (bg_warp_row_begin >= bg_warp_row_end) {{ '
    f'bg_remaining_begin = bg_key_work_end; continue; }}\n\n'
  )
  parts.append(f'{ii}// Narrow first source handle to warp\'s row range\n')
  parts.append(f'{ii}{{\n')
  parts.append(f'{ii}  auto bg_narrow_begin = {first_handle}.begin() + bg_warp_row_begin;\n')
  parts.append(f'{ii}  auto bg_narrow_end = {first_handle}.begin() + bg_warp_row_end;\n')
  parts.append(
    f'{ii}  {first_handle} = HandleType(bg_narrow_begin, bg_narrow_end, {first_handle}.depth());\n'
  )
  parts.append(f'{ii}}}\n\n')

  # Segment loops for multi-view non-first sources.
  seg_indent = ii
  for _, src, seg_var in bg_seg_specs:
    parts.append(
      f'{seg_indent}for (int {seg_var} = 0; {seg_var} < {src.view_count}; {seg_var}++) {{\n'
    )
    seg_indent += '  '
    parts.append(f'{seg_indent}auto {src.view_var} = views[{src.base_slot} + {seg_var}];\n')
    parts.append(
      f'{seg_indent}auto {src.handle_var} = HandleType(0, '
      f'{src.view_var}.num_rows_, 0)'
      f'.prefix({op.root_val_var}, {ctx.tile_var}, {src.view_var});\n'
    )
    parts.append(f'{seg_indent}if (!{src.handle_var}.valid()) continue;\n')

  # Bind root var at deepest segment indent.
  parts.append(f'{seg_indent}auto {op.var_name} = {op.root_val_var};\n')

  # Body emits at ctx.indent_level + 1 + segs (matches legacy).
  saved = ctx.indent_level
  ctx.indent_level = saved + 1 + len(bg_seg_specs)
  try:
    parts.append(emit(op.body, ctx))
  finally:
    ctx.indent_level = saved

  # Close segment loops innermost-first.
  for k in range(len(bg_seg_specs) - 1, -1, -1):
    close_indent = ii + ('  ' * k)
    parts.append(f'{close_indent}}}\n')

  # Per-key trailer + close key loop.
  parts.append(f'{ii}bg_remaining_begin = bg_key_work_end;\n')
  parts.append(f'{i}}}\n')

  return ''.join(parts)


# -----------------------------------------------------------------------------
# Standalone histogram kernel template (relocated S3A.9b)
#
# Not a `@register_render` handler — this is a per-rule kernel
# template called by complete_runner.py during runner emit. It takes
# a MIR ExecutePipeline (legitimate: target needs MIR for kernel
# signature derivation) and renders the bg_histogram kernel body.
# -----------------------------------------------------------------------------


def emit_bg_histogram_kernel(
  ep: m.ExecutePipeline,
  rel_index_types: dict[str, str],
) -> str:
  '''Emit `kernel_bg_histogram` — a grid-stride loop over unique root
  keys that writes the per-key work estimate (product of root-source
  degrees) into `bg_work_per_key[]`.

  Body is a hand-crafted prefix+degree sweep, not a `jit_pipeline`
  render. Pulls plugin/view-management helpers from the codegen
  internals (gen_root_handle, plugin_view_count, view_slot helpers).
  '''
  from srdatalog.ir.codegen.cuda.context import gen_root_handle as _gen_rh
  from srdatalog.ir.codegen.cuda.plugin import plugin_view_count
  from srdatalog.ir.codegen.cuda.view_slots import (
    collect_unique_view_specs,
    compute_view_slot_offsets,
  )

  source_specs = ep.source_specs
  first_op = ep.pipeline[0]
  if not isinstance(first_op, m.ColumnJoin):
    raise NotImplementedError('emit_bg_histogram_kernel: only ColumnJoin-root pipelines supported')
  root_sources = [s for s in first_op.sources if isinstance(s, m.ColumnSource)]

  mutable_pipe = list(ep.pipeline)
  view_specs = collect_unique_view_specs(mutable_pipe)

  code = '  // Block-group histogram: per-key work via grid-stride + warp reduction\n'
  code += '  static __global__ void __launch_bounds__(kBlockSize) kernel_bg_histogram(\n'
  code += '      const ViewType* __restrict__ views,\n'
  code += '      const ValueType* __restrict__ root_unique_values,\n'
  code += '      const ValueType* __restrict__ head_root_unique_values,\n'
  code += '      uint32_t num_unique_root_keys,\n'
  code += '      uint32_t num_full_unique_root_keys,\n'
  code += '      uint32_t num_root_keys,\n'
  code += '      uint64_t* __restrict__ bg_work_per_key) {\n'
  code += '    auto block = cg::this_thread_block();\n'
  code += '    auto tile = cg::tiled_partition<kGroupSize>(block);\n'
  code += '    auto single_thread = cg::tiled_partition<1>(block);\n'
  code += '    __shared__ char s_views_buf[NumSources * sizeof(ViewType)];\n'
  code += '    auto* s_views = reinterpret_cast<ViewType*>(s_views_buf);\n'
  code += '    if (threadIdx.x < NumSources) { s_views[threadIdx.x] = views[threadIdx.x]; }\n'
  code += '    __syncthreads();\n'
  code += '    views = s_views;\n'
  code += '    uint32_t thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;\n'
  code += '    uint32_t warp_id = thread_id / kGroupSize;\n'
  code += '    uint32_t num_warps = (gridDim.x * blockDim.x) / kGroupSize;\n'
  code += '    uint32_t num_threads = num_warps;\n\n'

  code += '    using HandleType = ViewType::NodeHandle;\n'
  code += '    using ViewType = std::remove_cvref_t<decltype(views[0])>;\n'
  code += '    using HandleType = ViewType::NodeHandle;\n\n'

  code += f'    // View declarations (deduplicated by spec, {len(view_specs)} unique views)\n'
  view_slot_offsets = compute_view_slot_offsets(source_specs, rel_index_types)
  for sp in view_specs:
    idx_str = '_'.join(str(v) for v in sp.index)
    view_var = f'view_{sp.rel_name}_{idx_str}_{sp.version}'
    slot = view_slot_offsets.get(sp.handle_idx, sp.handle_idx)
    code += f'    auto {view_var} = views[{slot}];\n'
  code += '\n'

  dest_arities = [len(d.index) for d in ep.dest_specs]
  if dest_arities:
    code += (
      '    using OutputCtx = SRDatalog::GPU::OutputContext<'
      'ValueType, SR, true, Layout, OutputArity_0>;\n'
    )
  code += '    // Grid-stride histogram: per-key warp-reduced work counts\n'
  code += (
    '    for (uint32_t bg_hist_key = warp_id; '
    'bg_hist_key < num_unique_root_keys; bg_hist_key += num_warps) {\n'
  )
  if dest_arities:
    code += '      OutputCtx hist_output_ctx{nullptr, nullptr, 0, 0};\n'
  code += '      auto bg_hist_root_val = root_unique_values[bg_hist_key];\n'

  src_view_counts: list[int] = []
  for idx_, src in enumerate(root_sources):
    src_idx = src.handle_start
    rel_name = src.rel_name
    src_index_type = rel_index_types.get(rel_name, '')
    src_view_count = plugin_view_count(src.version.code, src_index_type)
    src_view_counts.append(src_view_count)
    handle_var = f'h_{rel_name}_{src_idx}_root'
    idx_str = '_'.join(str(v) for v in src.index)
    view_var = f'view_{rel_name}_{idx_str}_{src.version.code}'
    if idx_ == 0:
      code += '      uint32_t bg_hlo = bg_hist_key;\n'
      code += (
        '      uint32_t bg_hhi = '
        f'{view_var}.num_rows_ - (num_unique_root_keys - bg_hist_key - 1);\n'
      )
      code += f'      bg_hhi = (bg_hhi <= {view_var}.num_rows_) ? bg_hhi : {view_var}.num_rows_;\n'
      code += f'      bg_hhi = (bg_hhi > bg_hlo) ? bg_hhi : {view_var}.num_rows_;\n'
      code += (
        f'      auto {handle_var} = HandleType(bg_hlo, bg_hhi, 0)'
        f'.prefix(bg_hist_root_val, tile, {view_var});\n'
      )
      code += (
        f'      if (!{handle_var}.valid()) {{ bg_work_per_key[bg_hist_key] = 0; continue; }}\n'
      )
    else:
      code += (
        f'      auto {handle_var} = {_gen_rh(view_var, src_index_type)}'
        f'.prefix(bg_hist_root_val, tile, {view_var});\n'
      )
      if src_view_count <= 1:
        code += (
          f'      if (!{handle_var}.valid()) {{ bg_work_per_key[bg_hist_key] = 0; continue; }}\n'
        )

  code += '      uint64_t bg_deg = 1;\n'
  for idx_, src in enumerate(root_sources):
    src_idx = src.handle_start
    rel_name = src.rel_name
    src_index_type = rel_index_types.get(rel_name, '')
    src_view_count = src_view_counts[idx_]
    handle_var = f'h_{rel_name}_{src_idx}_root'
    if src_view_count > 1 and idx_ > 0:
      seg_deg_var = f'bg_seg_deg_{idx_}'
      src_base_slot = view_slot_offsets.get(src_idx, 0)
      code += f'      uint64_t {seg_deg_var} = {handle_var}.valid() ? {handle_var}.degree() : 0;\n'
      for seg in range(1, src_view_count):
        seg_view_var = f'bg_seg_view_{idx_}_{seg}'
        seg_handle_var = f'bg_seg_h_{idx_}_{seg}'
        code += '      {\n'
        code += f'        auto {seg_view_var} = views[{src_base_slot + seg}];\n'
        code += (
          f'        auto {seg_handle_var} = {_gen_rh(seg_view_var, src_index_type)}'
          f'.prefix(bg_hist_root_val, tile, {seg_view_var});\n'
        )
        code += (
          f'        if ({seg_handle_var}.valid()) {seg_deg_var} += {seg_handle_var}.degree();\n'
        )
        code += '      }\n'
      code += f'      if ({seg_deg_var} == 0) {{ bg_work_per_key[bg_hist_key] = 0; continue; }}\n'
      code += f'      bg_deg *= {seg_deg_var};\n'
    else:
      code += f'      bg_deg *= {handle_var}.degree();\n'
  code += '      if (tile.thread_rank() == 0) bg_work_per_key[bg_hist_key] = bg_deg;\n'
  code += '    }\n'
  code += '  }\n\n'
  return code


__all__ = ['emit_bg_histogram_kernel']
