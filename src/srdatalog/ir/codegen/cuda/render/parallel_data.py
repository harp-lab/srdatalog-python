'''CUDA renderer for the parallel.data dialect.

Per docs/stage3a_execution_plan.md §7 task S3A.3 — handler split out
from the legacy codegen/cuda/emit.py. Pure refactor.

Note: `emit_bg_histogram_kernel` (which emits the histogram kernel
template — different shape from BgRootCjMulti's body emission) still
lives in `dialects/parallel/data/block_group.py`. S3A.9b will relocate
it to this file alongside `_render_bg_root_cj_multi`.
'''

from __future__ import annotations

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
