'''par.data.block_group — block-group work-balanced parallelism strategy.

The block-group strategy assigns each CUDA block a contiguous slice of
the total flat work-space `[0, bg_total_work)`. Per-block:

  1. Binary-search `bg_cumulative_work[]` for the starting key.
  2. Iterate keys from there until the block's work budget is consumed.
  3. Inside each key, redistribute work across warps proportional to
     the first source's degree.

Used for skewed root-key workloads where uniform warp-strided dispatch
would leave warps idle. Three runtime arrays produced by the host:

  - `bg_work_per_key[]`   filled by `kernel_bg_histogram` (per-key
                          work estimate = product of root-source degrees)
  - `bg_cumulative_work[]` exclusive prefix-sum over the histogram
  - `bg_total_work`        sum of all per-key work

This module owns:

  - `emit_bg_histogram_kernel(ep, rel_index_types)` — the histogram
    kernel template (lifted from
    `ir.dialects.target.cuda.complete_runner._gen_kernel_bg_histogram`).
  - `BgRootCjMulti` op — the BG dispatch shape for root multi-source
    ColumnJoin (count/materialize/fused kernel bodies). Lifted N4.1
    from legacy `jit_root_column_join_block_group`. Bundles the
    work-assignment preamble, binary-search key loop, per-source
    handle narrowing, warp-row redistribution, optional D2L
    segment-loops, and the wrapped body into one IR op so the
    dialect's `compile_kernel_body` covers the BG kernel emission.
  - `BgSourceSpec` — per-source descriptor consumed by
    `BgRootCjMulti`.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import srdatalog.ir.mir.types as m
from srdatalog.ir.core import Op


@final
@dataclass(frozen=True, slots=True)
class BgSourceSpec:
  '''Per-source descriptor for `BgRootCjMulti`.

  Carries the legacy state that `jit_root_column_join_block_group`
  threads through its emit: rel_name + view/handle var names,
  multi-view view_count for D2L segment loops, base view-slot for
  per-segment view rebinding, and the index_type passed to legacy
  helpers (`gen_root_handle`, `gen_valid`).
  '''

  rel_name: str
  view_var: str
  handle_var: str
  view_count: int  # 1 = DSAI / single-view; 2 = D2L FULL_VER
  base_slot: int
  index_type: str


@final
@dataclass(frozen=True, slots=True)
class BgRootCjMulti(Op):
  '''Block-group root multi-source ColumnJoin (count/materialize body).

  Lifts legacy `jit_root_column_join_block_group` into a single
  dialect op. Emits the full BG scaffolding around `body`:
    - block-level work assignment preamble (work_per_block,
      block_begin/end, return-if-out-of-range — with
      `thread_counts[thread_id] = 0;` in count mode);
    - binary search for the starting key index;
    - per-key loop with key-range checks;
    - per-source handle narrow with the first source using the
      key_idx hint; multi-view (D2L FULL_VER) non-first sources
      defer their handle bind to a `_bg_seg_<idx>` segment loop;
    - warp-row redistribution narrowing the first source handle
      proportionally on its degree;
    - segment loops (when present) wrapping the body;
    - body emit;
    - segment-loop close braces, `bg_remaining_begin = ...;`,
      key-loop close brace.

  `var_name` is the sanitized inner-bind name (`auto <var> =
  root_val_<n>;`). `is_counting` toggles the count-phase
  `thread_counts[...] = 0;` early-exit branch. `key_idx_var`,
  `root_val_var`, `hint_lo`, `hint_hi` are the counter-allocated
  names.

  Sources is in pipeline order (sources[0] = first/hint source;
  sources[i>0] = either single-view direct narrow or multi-view
  segment-loop deferred).
  '''

  var_name: str
  is_counting: bool
  key_idx_var: str
  root_val_var: str
  hint_lo: str
  hint_hi: str
  sources: tuple[BgSourceSpec, ...]
  body: Op


def emit_bg_histogram_kernel(
  ep: m.ExecutePipeline,
  rel_index_types: dict[str, str],
) -> str:
  '''Emit `kernel_bg_histogram` — a grid-stride loop over unique root
  keys that writes the per-key work estimate (product of root-source
  degrees) into `bg_work_per_key[]`.

  Body is a hand-crafted prefix+degree sweep, not a `jit_pipeline`
  render. Imports plugin/view-management helpers from `codegen.jit`
  for now; those move with N5 (relation.d2l dialect).
  '''
  from srdatalog.ir.dialects.target.cuda.context import gen_root_handle as _gen_rh
  from srdatalog.ir.dialects.target.cuda.plugin import plugin_view_count
  from srdatalog.ir.dialects.target.cuda.view_slots import (
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


__all__ = ['BgRootCjMulti', 'BgSourceSpec', 'emit_bg_histogram_kernel']
