'''relation.d2l ops — Device2LevelIndex segment-aware nodes.

D2L FULL_VER reads expose two view slots (HEAD + FULL); operators that
consume them have to visit both segments. The legacy emitter does this
with a `for (_seg = 0; _seg < view_count; ...)` loop wrapping the join
body, reassigning the view variable per iteration. `D2lSegmentLoop`
captures that pattern as a first-class IIR op.

See also `relation.d2l.__init__.view_count` for the per-source slot
arithmetic.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from srdatalog.ir.core import Op


@final
@dataclass(frozen=True, slots=True)
class D2lSegmentLoop(Op):
  '''Iterate over a D2L source's segments (HEAD then FULL).

  Two emit shapes:

  Single-line (multi-source nested CJ — legacy
  `_nested_column_join_multi`): when `local_view_var == ''`, lowers to

      for (int <seg_var> = 0; <seg_var> < <view_count>; <seg_var>++) {
        [auto ]<view_var> = views[<base_slot> + <seg_var>];
        <body>
      }

    `declare=False` rebinds the kernel-start `view_var`; `declare=True`
    declares a fresh per-segment view (single-source nested CJ case,
    N5.3).

  Two-line (root CJ multi non-first source — legacy `_root_cj_multi`
  phase 2): when `local_view_var` is non-empty, lowers to

      for (int <seg_var> = 0; <seg_var> < <view_count>; <seg_var>++) {
        auto <local_view_var> = views[<base_slot> + <seg_var>];
        <view_var> = <local_view_var>;
        <body>
      }

    The local var is the SHORT name (`view_<rel>_<src_idx>`) the
    immediately-following prefix narrowing references; the fixed
    canonical kernel-start view is rebound so downstream code that
    uses the LONG name also sees the current segment's array. The
    assignment line is omitted when `local_view_var == view_var`
    (matches legacy `if fixed_view_var != view_var` guard).

  Indent semantics (both shapes): the wrapped `body` emits at one
  level deeper than the segment loop's `for` preamble, but
  `EmitCtx.segment_depth` is bumped so any inner `IntersectIter`
  anchors its body lines / body op back to the *outer* indent —
  matching the legacy `seg_indent`-vs-`ind(ctx)` indent quirk.
  '''

  seg_var: str
  view_var: str
  base_slot: int
  view_count: int
  declare: bool
  local_view_var: str
  body: Op


__all__ = ['D2lSegmentLoop']
