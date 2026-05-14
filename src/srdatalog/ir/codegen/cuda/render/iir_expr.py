'''CUDA renderers for the iir.expr dialect.

Per docs/stage4_iir_vocabulary.md S4.2.
'''

from __future__ import annotations

from srdatalog.ir.codegen.cuda.render import EmitCtx, emit_expr, register_render
from srdatalog.ir.dialects.iir.expr.ops import (
  BinOp,
  CCast,
  FuncCall,
  IndexExpr,
  IntLit,
  MemberAccess,
  MemberCall,
  Parens,
  PostfixIncrement,
  StaticCast,
  Ternary,
  UnaryOp,
)


@register_render(CCast, mode='expr')
def _render_c_cast(op: CCast, ctx: EmitCtx) -> str:
  return f'({op.type_str}){emit_expr(op.expr, ctx)}'


@register_render(StaticCast, mode='expr')
def _render_static_cast(op: StaticCast, ctx: EmitCtx) -> str:
  return f'static_cast<{op.type_str}>({emit_expr(op.expr, ctx)})'


@register_render(Parens, mode='expr')
def _render_parens(op: Parens, ctx: EmitCtx) -> str:
  return f'({emit_expr(op.expr, ctx)})'


@register_render(Ternary, mode='expr')
def _render_ternary(op: Ternary, ctx: EmitCtx) -> str:
  '''Renders `<cond> ? <then> : <else>` with no surrounding parens.
  Caller wraps in `Parens(...)` if precedence with the surrounding
  context requires it.'''
  return f'{emit_expr(op.cond, ctx)} ? {emit_expr(op.then_, ctx)} : {emit_expr(op.else_, ctx)}'


@register_render(IntLit, mode='expr')
def _render_int_lit(op: IntLit, ctx: EmitCtx) -> str:
  return f'{op.value}{op.suffix}'


@register_render(UnaryOp, mode='expr')
def _render_unary_op(op: UnaryOp, ctx: EmitCtx) -> str:
  '''Renders `<op_str><expr>` with no surrounding parens. Matches the
  existing raw-text behavior for unary `!h.valid()` etc.'''
  return f'{op.op_str}{emit_expr(op.expr, ctx)}'


@register_render(PostfixIncrement, mode='expr')
def _render_postfix_increment(op: PostfixIncrement, ctx: EmitCtx) -> str:
  return f'{emit_expr(op.expr, ctx)}++'


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


@register_render(FuncCall, mode='expr')
def _render_func_call(op: FuncCall, ctx: EmitCtx) -> str:
  args = ', '.join(emit_expr(a, ctx) for a in op.args)
  return f'{op.func_name}({args})'
