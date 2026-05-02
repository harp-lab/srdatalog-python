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

from srdatalog.ir_core import Op


@final
@dataclass(frozen=True, slots=True)
class D2lSegmentLoop(Op):
  '''Iterate over a D2L source's segments (HEAD then FULL).

  Lowers (target.cuda) to:

      for (int <seg_var> = 0; <seg_var> < <view_count>; <seg_var>++) {
        [auto ]<view_var> = views[<base_slot> + <seg_var>];
        <body>
      }

  - When `declare` is False, the assignment line drops the `auto` —
    the view variable was already declared at kernel start, and we
    rebind it per segment so all references downstream see the
    current segment's array.
  - When `declare` is True, the `auto` is emitted; this matches the
    single-source nested CJ case where the segment loop introduces a
    fresh per-segment view variable.

  Critically, the wrapped `body` emits at the SAME indent as the
  segment loop's opening `for (...)` line — i.e. the body's
  ctx.indent is NOT bumped. This mirrors the legacy quirk where
  `seg_indent` is purely a string-level offset; `ind(ctx)` (the
  legacy's structural indent) doesn't change. The view assignment
  line is the one piece emitted at +1 indent.

  See `dialects.relation.d2l.__init__` for the dialect overview and
  the planned op set (D2lRoot, D2lPrefix, D2lDualSegmentDegree).
  '''

  seg_var: str
  view_var: str
  base_slot: int
  view_count: int
  declare: bool
  body: Op


__all__ = ['D2lSegmentLoop']
