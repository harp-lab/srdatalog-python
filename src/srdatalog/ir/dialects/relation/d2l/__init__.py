'''relation.d2l — Device2LevelIndex dialect.

Today this module owns:
  - `view_count(spec)` and `view_counts_for_specs` — view-slot
    accounting used by `compile.compile_kernel_body` to populate
    `view_counts` for the dialect's view declarations.
  - `D2lSegmentLoop` op (in `ops.py`) — the IIR segment-loop wrapper
    that lets nested CJ pipelines visit both HEAD and FULL segments
    of a D2L FULL_VER source.

Legacy-comment header (kept for the design narrative):

A 2-level index: each relation has a HEAD segment (smaller, dense) and
a FULL segment (the full sorted array). FULL_VER reads expose both
segments as separate views in `views[]`; iteration over the source
visits both segments in turn (HEAD first, FULL second).

Versions:
  - DELTA_VER, NEW_VER:  1 view (FULL only — incremental data)
  - FULL_VER:             2 views (HEAD + FULL)

Compared to `relation.sorted_array` (single-view-per-source DSAI):

  - View-slot accounting: a FULL_VER D2L source occupies 2 contiguous
    slots in `views[]`. `view_count(version) = 2 if version == FULL_VER
    else 1`. The dialect's `emit_view_declarations` honors per-spec
    `view_counts` to compute correct positional slots.

  - Nested CJ segment loop: a nested ColumnJoin whose RHS source is a
    D2L FULL_VER source must wrap its body in
    `for (_nseg = 0; _nseg < 2; ++_nseg) { view_X = views[base+_nseg];
    <cj body> }` so the join visits both segments. This op (planned:
    `D2lSegmentLoop`) is the missing piece keeping D2L pipelines on
    the legacy fallback today.

  - BG histogram segment-aware degree: BG histogram non-first sources
    accumulate degree across both segments before validity checks.
    Lives in `codegen.cuda.render.parallel_data.emit_bg_histogram_kernel`
    (which already handles `view_count > 1`).

Planned ops (N5.x):
  D2lRoot(view_full, view_head)         dual-segment root handle
  D2lPrefix(parent, key)                 cooperative narrow + segment
  D2lSegmentLoop(seg_var, view, body)    iterate over HEAD/FULL
  D2lDualSegmentDegree(handle, view_count)  segment-aware degree sum

The legacy `plugin_view_count` is the underlying source for
`view_count` until we expand the dialect to host its own ops + emit.
'''

from __future__ import annotations

from srdatalog.ir.codegen.cuda.envelope import ViewSpec

# Side-effect import: registers D2L's CUDA plugin with
# `codegen.cuda.plugin` so `plugin_view_count` etc. dispatch correctly
# for relations declared `index_type="...Device2LevelIndex"`.
from srdatalog.ir.dialects.relation.d2l import cuda as _cuda
from srdatalog.ir.dialects.relation.d2l.ops import D2lSegmentLoop


def view_count(version: str, index_type: str) -> int:
  '''Number of physical view slots a (version, index_type) pair
  consumes. D2L FULL_VER = 2; DSAI / D2L DELTA / NEW = 1.

  Delegates to the legacy `plugin_view_count` for the actual
  resolution. Once the relation-dialect registry hosts D2L ops,
  the dispatch shifts here and the legacy plugin module retires.
  '''
  from srdatalog.ir.codegen.cuda.plugin import plugin_view_count

  return plugin_view_count(version, index_type)


def view_counts_for_specs(
  view_specs: list[ViewSpec],
  rel_index_types: dict[str, str],
) -> list[int]:
  '''Per-spec view_count for a pipeline's view specs. Pass the result
  to `emit_view_declarations(..., view_counts=...)` so positional
  slots advance by the right amount per spec.
  '''
  return [view_count(sp.version, rel_index_types.get(sp.rel_name, '')) for sp in view_specs]


__all__ = ['D2lSegmentLoop', 'view_count', 'view_counts_for_specs']
