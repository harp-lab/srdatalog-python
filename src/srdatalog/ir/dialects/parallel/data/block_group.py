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

This module owns the block-group **dialect ops**:

  - `BgRootCjMulti` — the BG dispatch shape for root multi-source
    ColumnJoin (count/materialize/fused kernel bodies). Lifted N4.1
    from legacy `jit_root_column_join_block_group`. Bundles the
    work-assignment preamble, binary-search key loop, per-source
    handle narrowing, warp-row redistribution, optional D2L
    segment-loops, and the wrapped body into one IR op.
  - `BgSourceSpec` — per-source descriptor consumed by `BgRootCjMulti`.

CUDA emission lives in [codegen/cuda/render/parallel_data.py][1] —
both `_render_bg_root_cj_multi` (BgRootCjMulti's renderer) and
`emit_bg_histogram_kernel` (the standalone histogram template called
by the runner). Per docs/stage3a_execution_plan.md §7 task S3A.9b,
the dialect file holds only data; rendering is the codegen's job.

[1]: ../../codegen/cuda/render/parallel_data.py
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import final

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


__all__ = ['BgRootCjMulti', 'BgSourceSpec']
