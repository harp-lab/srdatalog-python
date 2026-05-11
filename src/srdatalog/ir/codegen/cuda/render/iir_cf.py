'''CUDA renderers for the iir.cf (control-flow) dialect.

Per docs/stage3a_execution_plan.md §7 task S3A.3 — handlers split out
from the legacy codegen/cuda/emit.py 41-case match. Every handler is
a verbatim port of one match arm (pure refactor; no semantic change).
Byte-equivalence preserved.

22 statement handlers + 2 expression handlers (RawString and VarRef
appear in both modes; iir.cf owns both forms here).
'''

from __future__ import annotations

from srdatalog.ir.codegen.cuda.render import EmitCtx, emit, emit_expr, register_render
from srdatalog.ir.dialects.iir.cf.ops import (
  AddCount,
  Assign,
  Bind,
  BlankLine,
  Block,
  Cartesian2DDecompose,
  CartesianFlatLoop,
  CartesianNDecompose,
  Comment,
  GridStrideLoop,
  If,
  IfContinue,
  IfContinueIfNot,
  IfReturn,
  IfReturnIfNot,
  IndentBlock,
  IntersectIter,
  LaneZeroGuard,
  OuterAnchor,
  ParallelFor,
  Phase,
  RawString,
  TiledBallotBlock,
  VarRef,
  WriteOutput,
)

# -----------------------------------------------------------------------------
# Statement-mode handlers
# -----------------------------------------------------------------------------


@register_render(Block, mode='stmt')
def _render_block(op: Block, ctx: EmitCtx) -> str:
  return ''.join(emit(s, ctx) for s in op.stmts)


@register_render(IndentBlock, mode='stmt')
def _render_indent_block(op: IndentBlock, ctx: EmitCtx) -> str:
  ctx.indent_level += op.extra
  try:
    return ''.join(emit(s, ctx) for s in op.stmts)
  finally:
    ctx.indent_level -= op.extra


@register_render(BlankLine, mode='stmt')
def _render_blank_line(op: BlankLine, ctx: EmitCtx) -> str:
  return '\n'


@register_render(Bind, mode='stmt')
def _render_bind(op: Bind, ctx: EmitCtx) -> str:
  return f'{ctx.ind()}{op.type_decl} {op.name} = {emit_expr(op.expr, ctx)};\n'


@register_render(Assign, mode='stmt')
def _render_assign(op: Assign, ctx: EmitCtx) -> str:
  return f'{ctx.ind()}{op.target} = {emit_expr(op.value, ctx)};\n'


@register_render(IfReturnIfNot, mode='stmt')
def _render_if_return_if_not(op: IfReturnIfNot, ctx: EmitCtx) -> str:
  return f'{ctx.ind()}if (!{emit_expr(op.cond, ctx)}) return;\n'


@register_render(IfContinueIfNot, mode='stmt')
def _render_if_continue_if_not(op: IfContinueIfNot, ctx: EmitCtx) -> str:
  return f'{ctx.ind()}if (!{emit_expr(op.cond, ctx)}) continue;\n'


@register_render(IfReturn, mode='stmt')
def _render_if_return(op: IfReturn, ctx: EmitCtx) -> str:
  return f'{ctx.ind()}if ({emit_expr(op.cond, ctx)}) return;\n'


@register_render(IfContinue, mode='stmt')
def _render_if_continue(op: IfContinue, ctx: EmitCtx) -> str:
  return f'{ctx.ind()}if ({emit_expr(op.cond, ctx)}) continue;\n'


@register_render(IntersectIter, mode='stmt')
def _render_intersect_iter(op: IntersectIter, ctx: EmitCtx) -> str:
  iter_args = ', '.join(emit_expr(e, ctx) for e in op.iterator_exprs)
  preamble = (
    f'{ctx.ind()}auto {op.intersect_var} = '
    f'intersect_handles({ctx.tile_var}, {iter_args});\n'
    f'{ctx.ind()}for (auto {op.iter_var} = {op.intersect_var}.begin(); '
    f'{op.iter_var}.valid(); {op.iter_var}.next()) {{\n'
  )
  # body_lines (auto value, positions) and the body itself anchor against
  # the *outer* indent — i.e. ctx.indent_level minus the number of
  # D2lSegmentLoops we're inside. Mirrors legacy `_nested_column_join_multi`
  # ind(ctx) trick.
  outer_il = ctx.indent_level - ctx.segment_depth
  outer_ind = '  ' * outer_il
  body_lines = (
    f'{outer_ind}  auto {op.value_var} = {op.iter_var}.value();\n'
    f'{outer_ind}  auto positions = {op.iter_var}.positions();\n'
  )
  saved_il = ctx.indent_level
  ctx.indent_level = outer_il
  try:
    body_str = emit(op.body, ctx)
  finally:
    ctx.indent_level = saved_il
  return preamble + body_lines + body_str + f'{ctx.ind()}}}\n'


@register_render(OuterAnchor, mode='stmt')
def _render_outer_anchor(op: OuterAnchor, ctx: EmitCtx) -> str:
  saved_il = ctx.indent_level
  saved_sd = ctx.segment_depth
  ctx.indent_level -= ctx.segment_depth
  ctx.segment_depth = 0
  try:
    return emit(op.body, ctx)
  finally:
    ctx.indent_level = saved_il
    ctx.segment_depth = saved_sd


@register_render(If, mode='stmt')
def _render_if(op: If, ctx: EmitCtx) -> str:
  # Body emitted at SAME indent as the wrapping if (legacy quirk —
  # body was rendered before the wrap was applied).
  return f'{ctx.ind()}if ({emit_expr(op.cond, ctx)}) {{\n' + emit(op.body, ctx) + f'{ctx.ind()}}}\n'


@register_render(CartesianFlatLoop, mode='stmt')
def _render_cartesian_flat_loop(op: CartesianFlatLoop, ctx: EmitCtx) -> str:
  return (
    f'{ctx.ind()}for (uint32_t {op.idx_var} = {op.lane_var}; '
    f'{op.idx_var} < {op.bound_var}; {op.idx_var} += {op.group_size_var}) {{\n'
    + emit(op.body, ctx)
    + f'{ctx.ind()}}}\n'
  )


@register_render(Cartesian2DDecompose, mode='stmt')
def _render_cartesian_2d_decompose(op: Cartesian2DDecompose, ctx: EmitCtx) -> str:
  i = ctx.ind()
  return (
    f'{i}const bool {op.major_var} = ({op.deg1_var} >= {op.deg0_var});\n'
    f'{i}uint32_t {op.idx0_var}, {op.idx1_var};\n'
    f'{i}if ({op.major_var}) {{\n'
    f'{i}  {op.idx0_var} = {op.flat_idx_var} / {op.deg1_var};\n'
    f'{i}  {op.idx1_var} = {op.flat_idx_var} % {op.deg1_var};\n'
    f'{i}}} else {{\n'
    f'{i}  {op.idx1_var} = {op.flat_idx_var} / {op.deg0_var};\n'
    f'{i}  {op.idx0_var} = {op.flat_idx_var} % {op.deg0_var};\n'
    f'{i}}}\n'
  )


@register_render(CartesianNDecompose, mode='stmt')
def _render_cartesian_n_decompose(op: CartesianNDecompose, ctx: EmitCtx) -> str:
  n = len(op.idx_vars)
  i = ctx.ind()
  lines = [f'{i}uint32_t remaining = {op.flat_idx_var};\n']
  for k in range(n - 1, -1, -1):
    lines.append(f'{i}uint32_t {op.idx_vars[k]} = remaining % {op.deg_vars[k]};\n')
    if k > 0:
      lines.append(f'{i}remaining /= {op.deg_vars[k]};\n')
  return ''.join(lines)


@register_render(GridStrideLoop, mode='stmt')
def _render_grid_stride_loop(op: GridStrideLoop, ctx: EmitCtx) -> str:
  # Body rendered at the SAME indent as the for-loop preamble.
  # Sub-parts that need increased indent should wrap themselves in
  # IndentBlock — matches the legacy emitter's mixed-indent quirk.
  return (
    f'{ctx.ind()}// WARP MODE: 32 threads cooperatively handle one row\n'
    f'{ctx.ind()}for (uint32_t {op.idx_name} = warp_id; '
    f'{op.idx_name} < {emit_expr(op.bound, ctx)}; '
    f'{op.idx_name} += num_warps) {{\n' + emit(op.body, ctx) + f'{ctx.ind()}}}\n'
  )


@register_render(ParallelFor, mode='stmt')
def _render_parallel_for(op: ParallelFor, ctx: EmitCtx) -> str:
  if op.strategy != 'warp_strided':
    raise NotImplementedError(
      f'codegen.cuda M1 supports strategy="warp_strided" only; got {op.strategy!r}'
    )
  # The strategy itself doesn't add code at the body's indent — the
  # warp_id/num_warps come from the kernel signature, and the
  # GridStrideLoop inside emits the actual loop.
  return emit(op.body, ctx)


@register_render(Phase, mode='stmt')
def _render_phase(op: Phase, ctx: EmitCtx) -> str:
  # M1: phase is a marker; the legacy emitter doesn't bracket phase
  # scopes in code (the OutputContext template handles it at runtime).
  return emit(op.body, ctx)


@register_render(LaneZeroGuard, mode='stmt')
def _render_lane_zero_guard(op: LaneZeroGuard, ctx: EmitCtx) -> str:
  # One-liner if body is a single RawString (matches the common legacy
  # form); otherwise braced block.
  if isinstance(op.body, RawString):
    return f'{ctx.ind()}if ({ctx.tile_var}.thread_rank() == 0) {op.body.text}\n'
  lz_head = f'{ctx.ind()}if ({ctx.tile_var}.thread_rank() == 0) {{\n'
  ctx.indent_level += 1
  try:
    body_str = emit(op.body, ctx)
  finally:
    ctx.indent_level -= 1
  return lz_head + body_str + f'{ctx.ind()}}}\n'


@register_render(Comment, mode='stmt')
def _render_comment(op: Comment, ctx: EmitCtx) -> str:
  return f'{ctx.ind()}// {op.text}\n'


@register_render(RawString, mode='stmt')
def _render_raw_string_stmt(op: RawString, ctx: EmitCtx) -> str:
  return f'{ctx.ind()}{op.text}\n'


@register_render(WriteOutput, mode='stmt')
def _render_write_output(op: WriteOutput, ctx: EmitCtx) -> str:
  args = ', '.join(emit_expr(v, ctx) for v in op.values)
  return f'{ctx.ind()}{op.output_var}.emit_direct({args});\n'


@register_render(TiledBallotBlock, mode='stmt')
def _render_tiled_ballot_block(op: TiledBallotBlock, ctx: EmitCtx) -> str:
  '''Lifted from legacy emit_helpers.jit_insert_into ballot path.
  Single ballot setup + per-output `if (valid) { write }` block,
  closing with `warp_local_count += _tc_active`.'''
  i = ctx.ind()
  tile = ctx.tile_var
  parts: list[str] = []
  for idx_, (dest_idx, values, debug) in enumerate(op.outputs):
    if debug:
      parts.append(f'{i}// {debug}\n')
    if idx_ == 0:
      parts.append(f'{i}{{\n')
      parts.append(f'{i}  uint32_t _tc_ballot = {tile}.ballot({op.valid_var});\n')
      parts.append(f'{i}  uint32_t _tc_active = __popc(_tc_ballot);\n')
      parts.append(f'{i}  if (_tc_active > 0) {{\n')
      parts.append(f'{i}    uint32_t _tc_mask = (1u << {tile}.thread_rank()) - 1u;\n')
      parts.append(f'{i}    uint32_t _tc_off = __popc(_tc_ballot & _tc_mask);\n')
    parts.append(f'{i}    if ({op.valid_var}) {{\n')
    parts.append(
      f'{i}      uint32_t _tc_pos_{dest_idx} = old_size_{dest_idx} '
      f'+ warp_write_base + warp_local_count + _tc_off;\n'
    )
    for col, name in enumerate(values):
      parts.append(
        f'{i}      output_data_{dest_idx}[{col} * '
        f'static_cast<uint32_t>(output_stride_{dest_idx}) + '
        f'_tc_pos_{dest_idx}] = {name};\n'
      )
    parts.append(f'{i}    }}\n')
  parts.append(f'{i}    warp_local_count += _tc_active;\n')
  parts.append(f'{i}  }}\n')
  parts.append(f'{i}}}\n')
  return ''.join(parts)


@register_render(AddCount, mode='stmt')
def _render_add_count(op: AddCount, ctx: EmitCtx) -> str:
  return f'{ctx.ind()}{op.output_var}.add_count({emit_expr(op.delta, ctx)});\n'


# -----------------------------------------------------------------------------
# Expression-mode handlers
# -----------------------------------------------------------------------------


@register_render(VarRef, mode='expr')
def _render_var_ref(op: VarRef, ctx: EmitCtx) -> str:
  return op.name


@register_render(RawString, mode='expr')
def _render_raw_string_expr(op: RawString, ctx: EmitCtx) -> str:
  return op.text
