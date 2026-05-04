'''CUDA renderers for the relation.sorted_array dialect.

Per docs/stage3a_execution_plan.md §7 task S3A.3 — handlers split out
from the legacy codegen/cuda/emit.py 41-case match. Pure refactor;
no semantic change. Byte-equivalence preserved.

11 expression handlers (the index-access ops) + 1 statement handler
(SaTiledCartesian2D — the tiled-Cartesian smem block).
'''

from __future__ import annotations

from srdatalog.ir.codegen.cuda.render import EmitCtx, emit, emit_expr, register_render
from srdatalog.ir.dialects.relation.sorted_array.ops import (
  SaChildRange,
  SaDegree,
  SaGetVal,
  SaGetValAt,
  SaGetValAtPos,
  SaHint,
  SaIterators,
  SaPrefCoop,
  SaPrefSeq,
  SaRoot,
  SaTiledCartesian2D,
  SaValid,
)

# -----------------------------------------------------------------------------
# Expression-mode handlers (11)
# -----------------------------------------------------------------------------


@register_render(SaRoot, mode='expr')
def _render_sa_root(op: SaRoot, ctx: EmitCtx) -> str:
  return f'HandleType(0, {op.view_name}.num_rows_, 0)'


@register_render(SaValid, mode='expr')
def _render_sa_valid(op: SaValid, ctx: EmitCtx) -> str:
  return f'{op.handle_name}.valid()'


@register_render(SaDegree, mode='expr')
def _render_sa_degree(op: SaDegree, ctx: EmitCtx) -> str:
  return f'{op.handle_name}.degree()'


@register_render(SaGetVal, mode='expr')
def _render_sa_get_val(op: SaGetVal, ctx: EmitCtx) -> str:
  return f'{op.view_name}.get_value({op.col}, {op.idx_var_name})'


@register_render(SaGetValAt, mode='expr')
def _render_sa_get_val_at(op: SaGetValAt, ctx: EmitCtx) -> str:
  return f'{op.handle_name}.get_value_at({op.view_name}, {op.idx_var_name})'


@register_render(SaHint, mode='expr')
def _render_sa_hint(op: SaHint, ctx: EmitCtx) -> str:
  return f'HandleType({op.lo_var}, {op.hi_var}, {op.depth})'


@register_render(SaPrefCoop, mode='expr')
def _render_sa_pref_coop(op: SaPrefCoop, ctx: EmitCtx) -> str:
  return f'{emit_expr(op.parent, ctx)}.prefix({op.key_var}, {ctx.tile_var}, {op.view_name})'


@register_render(SaPrefSeq, mode='expr')
def _render_sa_pref_seq(op: SaPrefSeq, ctx: EmitCtx) -> str:
  return f'{emit_expr(op.parent, ctx)}.prefix_seq({op.key_var}, {op.view_name})'


@register_render(SaIterators, mode='expr')
def _render_sa_iterators(op: SaIterators, ctx: EmitCtx) -> str:
  return f'{op.handle_name}.iterators({op.view_name})'


@register_render(SaChildRange, mode='expr')
def _render_sa_child_range(op: SaChildRange, ctx: EmitCtx) -> str:
  return (
    f'{op.handle_name}.child_range({op.pos_expr}, {op.key_var}, {ctx.tile_var}, {op.view_name})'
  )


@register_render(SaGetValAtPos, mode='expr')
def _render_sa_get_val_at_pos(op: SaGetValAtPos, ctx: EmitCtx) -> str:
  return f'{op.view_name}.get_value({op.col}, {op.handle_name}.begin() + {op.idx_var_name})'


# -----------------------------------------------------------------------------
# Statement-mode handler (1) — SaTiledCartesian2D
# -----------------------------------------------------------------------------


@register_render(SaTiledCartesian2D, mode='stmt')
def _render_sa_tiled_cartesian_2d(op: SaTiledCartesian2D, ctx: EmitCtx) -> str:
  '''Lifted from legacy `_emit_tiled_cartesian` (deleted instructions.py).
  The whole structure is string-level — body emits at the surrounding
  scope's indent (legacy quirk: bodies pre-rendered before the tiled
  wrap textually surrounds them).'''
  i = ctx.ind()
  tile = ctx.tile_var
  body_str = emit(op.body, ctx)
  parts: list[str] = [
    f'{i}if ({op.total_var} > 32) {{\n',
    f'{i}  // Tiled Cartesian: smem pre-load reads, standard emit_direct writes\n',
    f'{i}  for (uint32_t {op.t0_base} = 0; {op.t0_base} < {op.degree_var0}; '
    f'{op.t0_base} += kCartTileSize) {{\n',
    f'{i}    uint32_t {op.t0_len} = min({op.t0_base} + (uint32_t)kCartTileSize, '
    f'{op.degree_var0}) - {op.t0_base};\n',
    f'{i}    for (uint32_t _ti = {op.lane_var}; _ti < {op.t0_len}; _ti += {op.group_size_var})\n',
    f'{i}      s_cart[warp_in_block][0][_ti] = {op.view_var0}.get_value('
    f'{op.col0}, {op.handle_var0}.begin() + {op.t0_base} + _ti);\n',
    f'{i}    for (uint32_t {op.t1_base} = 0; {op.t1_base} < {op.degree_var1}; '
    f'{op.t1_base} += kCartTileSize) {{\n',
    f'{i}      uint32_t {op.t1_len} = min({op.t1_base} + (uint32_t)kCartTileSize, '
    f'{op.degree_var1}) - {op.t1_base};\n',
    f'{i}      for (uint32_t _ti = {op.lane_var}; _ti < {op.t1_len}; _ti += {op.group_size_var})\n',
    f'{i}        s_cart[warp_in_block][1][_ti] = {op.view_var1}.get_value('
    f'{op.col1}, {op.handle_var1}.begin() + {op.t1_base} + _ti);\n',
    f'{i}      {tile}.sync();\n',
    f'{i}      uint32_t {op.tile_total} = {op.t0_len} * {op.t1_len};\n',
    f'{i}      for (uint32_t {op.batch_var} = 0; {op.batch_var} < {op.tile_total}; '
    f'{op.batch_var} += {op.group_size_var}) {{\n',
    f'{i}        uint32_t {op.flat_idx_var} = {op.batch_var} + {op.lane_var};\n',
    f'{i}        bool {op.valid_var} = {op.flat_idx_var} < {op.tile_total};\n',
    f'{i}        auto {op.var_name0} = {op.valid_var} ? '
    f's_cart[warp_in_block][0][{op.flat_idx_var} / {op.t1_len}] : ValueType{{0}};\n',
    f'{i}        auto {op.var_name1} = {op.valid_var} ? '
    f's_cart[warp_in_block][1][{op.flat_idx_var} % {op.t1_len}] : ValueType{{0}};\n',
    body_str,
    f'{i}      }}\n',
    f'{i}      {tile}.sync();\n',
    f'{i}    }}\n',
    f'{i}  }}\n',
    f'{i}}} else {{\n',
    f'{i}  for (uint32_t {op.fb_batch_var} = 0; {op.fb_batch_var} < {op.total_var}; '
    f'{op.fb_batch_var} += {op.group_size_var}) {{\n',
    f'{i}    uint32_t {op.flat_idx_var} = {op.fb_batch_var} + {op.lane_var};\n',
    f'{i}    bool {op.valid_var} = {op.flat_idx_var} < {op.total_var};\n',
    f'{i}    const bool {op.major_var} = ({op.degree_var1} >= {op.degree_var0});\n',
    f'{i}    uint32_t {op.idx0_var}, {op.idx1_var};\n',
    f'{i}    if ({op.major_var}) {{ {op.idx0_var} = {op.flat_idx_var} / {op.degree_var1}; '
    f'{op.idx1_var} = {op.flat_idx_var} % {op.degree_var1}; }}\n',
    f'{i}    else {{ {op.idx1_var} = {op.flat_idx_var} / {op.degree_var0}; '
    f'{op.idx0_var} = {op.flat_idx_var} % {op.degree_var0}; }}\n',
    f'{i}    auto {op.var_name0} = {op.view_var0}.get_value('
    f'{op.col0}, {op.handle_var0}.begin() + {op.idx0_var});\n',
    f'{i}    auto {op.var_name1} = {op.view_var1}.get_value('
    f'{op.col1}, {op.handle_var1}.begin() + {op.idx1_var});\n',
    body_str,
    f'{i}  }}\n',
    f'{i}}}\n',
  ]
  return ''.join(parts)
