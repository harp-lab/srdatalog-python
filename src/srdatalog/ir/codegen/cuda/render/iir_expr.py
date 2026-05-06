'''CUDA renderers for the iir.expr dialect.

Per docs/stage4_iir_vocabulary.md S4.2.
'''

from __future__ import annotations

from srdatalog.ir.codegen.cuda.render import EmitCtx, emit_expr, register_render
from srdatalog.ir.dialects.iir.expr.ops import BinOp, IndexExpr, MemberAccess, MemberCall


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


@register_render(IndexExpr, mode='expr')
def _render_index_expr(op: IndexExpr, ctx: EmitCtx) -> str:
  return f'{emit_expr(op.arr, ctx)}[{emit_expr(op.idx, ctx)}]'


@register_render(MemberAccess, mode='expr')
def _render_member_access(op: MemberAccess, ctx: EmitCtx) -> str:
  return f'{emit_expr(op.obj, ctx)}.{op.member}'


@register_render(MemberCall, mode='expr')
def _render_member_call(op: MemberCall, ctx: EmitCtx) -> str:
  args = ', '.join(emit_expr(a, ctx) for a in op.args)
  return f'{emit_expr(op.obj, ctx)}.{op.method}({args})'
