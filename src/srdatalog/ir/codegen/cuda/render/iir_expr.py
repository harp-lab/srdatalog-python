'''CUDA renderers for the iir.expr dialect.

Per docs/stage4_iir_vocabulary.md S4.2.
'''

from __future__ import annotations

from srdatalog.ir.codegen.cuda.render import EmitCtx, emit_expr, register_render
from srdatalog.ir.dialects.iir.expr.ops import BinOp


@register_render(BinOp, mode='expr')
def _render_bin_op(op: BinOp, ctx: EmitCtx) -> str:
  '''Render `<lhs> <op_str> <rhs>` with no surrounding parens. Matches
  the existing RawString-with-bare-text behavior these handlers replace
  — preserves byte-equivalence for the lowerings being migrated.

  Caller is responsible for any precedence-explicit grouping; future
  Parens / explicit-grouping support can be added without changing this
  handler's contract.
  '''
  return f'{emit_expr(op.lhs, ctx)} {op.op_str} {emit_expr(op.rhs, ctx)}'
