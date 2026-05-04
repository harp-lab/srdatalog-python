'''CUDA renderer for the relation.d2l dialect.

Per docs/stage3a_execution_plan.md §7 task S3A.3 — handler split out
from the legacy codegen/cuda/emit.py. Pure refactor.
'''

from __future__ import annotations

from srdatalog.ir.codegen.cuda.render import EmitCtx, emit, register_render
from srdatalog.ir.dialects.relation.d2l.ops import D2lSegmentLoop


@register_render(D2lSegmentLoop, mode='stmt')
def _render_d2l_segment_loop(op: D2lSegmentLoop, ctx: EmitCtx) -> str:
  # for-loop at ctx.ind(); view assignment(s) at +1; body at +1 with
  # segment_depth bumped so a wrapped IntersectIter anchors its body
  # lines back to the *outer* indent (this segment loop doesn't move
  # them deeper).
  head = (
    f'{ctx.ind()}for (int {op.seg_var} = 0; {op.seg_var} < {op.view_count}; {op.seg_var}++) {{\n'
  )
  inner_ind = ctx.ind() + '  '
  if op.local_view_var:
    # Two-line root-CJ shape: fresh local + reassign canonical.
    assign = f'{inner_ind}auto {op.local_view_var} = views[{op.base_slot} + {op.seg_var}];\n'
    if op.local_view_var != op.view_var:
      assign += f'{inner_ind}{op.view_var} = {op.local_view_var};\n'
  else:
    decl_kw = 'auto ' if op.declare else ''
    assign = f'{inner_ind}{decl_kw}{op.view_var} = views[{op.base_slot} + {op.seg_var}];\n'
  ctx.indent_level += 1
  ctx.segment_depth += 1
  try:
    body_str = emit(op.body, ctx)
  finally:
    ctx.indent_level -= 1
    ctx.segment_depth -= 1
  return head + assign + body_str + f'{ctx.ind()}}}\n'
