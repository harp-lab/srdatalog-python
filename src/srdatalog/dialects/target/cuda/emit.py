'''target.cuda — IIR -> C++ source emission.

`emit(op, ctx) -> str` walks an IIR tree and produces CUDA C++ source.
The output is structured to match the legacy `jit_pipeline()` emitter
on M1's supported shapes (Scan + InsertInto, no feature flags).

Two emission modes:
  - **statement mode**: produces lines ending in `\n`. Used for ops
    that emit full statements (Bind, GridStrideLoop, Block, ...).
  - **expression mode** (`emit_expr`): produces a string with no
    trailing newline, no leading indent. Used inside `<expr>` slots
    of statements.

The split is intentional: every IIR op is unambiguously one or the
other. Mixing them produces malformed C++.
'''

from __future__ import annotations

from dataclasses import dataclass

from srdatalog.dialects.iir.cf import (
  AddCount,
  Bind,
  BlankLine,
  Block,
  Cartesian2DDecompose,
  CartesianFlatLoop,
  CartesianNDecompose,
  Comment,
  GridStrideLoop,
  If,
  IfContinueIfNot,
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
from srdatalog.dialects.relation.d2l.ops import D2lSegmentLoop
from srdatalog.dialects.relation.sorted_array.ops import (
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
from srdatalog.ir_core import Op, assert_never


@dataclass
class EmitCtx:
  '''Mutable emission state during the C++ walk.

  - `indent_level`: counts of 2-space units. Legacy emitter starts
    at indent=2 (operator() body); M1 callers pass that in to match.
  - `tile_var`: name of the tile/thread-group variable. "tile" by default.
  - `segment_depth`: how many `D2lSegmentLoop`s wrap the current
    point. Lets `IntersectIter` reproduce the legacy
    `_nested_column_join_multi` indent quirk where segment loops
    bump the structural indent (alias_binds, intersect at +segs)
    but the for-iter body lines (auto value, positions, child_binds,
    body_op) anchor against the *outer* indent (+1, +1, +1, +0
    respectively).
  '''

  indent_level: int = 2
  tile_var: str = 'tile'
  segment_depth: int = 0

  def ind(self) -> str:
    return '  ' * self.indent_level


def emit(op: Op, ctx: EmitCtx) -> str:
  '''Emit a statement-shaped IIR op. Always returns a string ending
  in `\n` (or empty for trivial cases).'''
  match op:
    case Block(stmts=stmts):
      return ''.join(emit(s, ctx) for s in stmts)

    case IndentBlock(extra=extra, stmts=stmts):
      ctx.indent_level += extra
      try:
        return ''.join(emit(s, ctx) for s in stmts)
      finally:
        ctx.indent_level -= extra

    case BlankLine():
      return '\n'

    case Bind(name=name, expr=expr, type_decl=tdecl):
      return f'{ctx.ind()}{tdecl} {name} = {emit_expr(expr, ctx)};\n'

    case IfReturnIfNot(cond=cond):
      return f'{ctx.ind()}if (!{emit_expr(cond, ctx)}) return;\n'

    case IfContinueIfNot(cond=cond):
      return f'{ctx.ind()}if (!{emit_expr(cond, ctx)}) continue;\n'

    case IntersectIter(
      intersect_var=ivar, iter_var=itvar,
      iterator_exprs=iters, value_var=vvar, body=body,
    ):
      iter_args = ', '.join(emit_expr(e, ctx) for e in iters)
      preamble = (
        f'{ctx.ind()}auto {ivar} = intersect_handles({ctx.tile_var}, {iter_args});\n'
        f'{ctx.ind()}for (auto {itvar} = {ivar}.begin(); '
        f'{itvar}.valid(); {itvar}.next()) {{\n'
      )
      # body_lines (auto value, positions) and the body itself anchor
      # against the *outer* indent — i.e. ctx.indent_level minus the
      # number of D2lSegmentLoops we're inside. Segment loops bump
      # ctx.indent_level (so alias_binds + intersect emit deeper) but
      # NOT the body-line / body-op indent. Mirrors the legacy
      # `_nested_column_join_multi` ind(ctx) trick.
      outer_il = ctx.indent_level - ctx.segment_depth
      outer_ind = '  ' * outer_il
      body_lines = (
        f'{outer_ind}  auto {vvar} = {itvar}.value();\n'
        f'{outer_ind}  auto positions = {itvar}.positions();\n'
      )
      saved_il = ctx.indent_level
      ctx.indent_level = outer_il
      try:
        body_str = emit(body, ctx)
      finally:
        ctx.indent_level = saved_il
      return preamble + body_lines + body_str + f'{ctx.ind()}}}\n'

    case D2lSegmentLoop(
      seg_var=sv, view_var=vv, base_slot=bs,
      view_count=vc, declare=declare,
      local_view_var=local_vv, body=body,
    ):
      # for-loop at ctx.ind(); view assignment(s) at +1; body at +1
      # with segment_depth bumped so a wrapped IntersectIter anchors
      # its body lines back to the *outer* indent (this segment loop
      # doesn't move them deeper).
      head = f'{ctx.ind()}for (int {sv} = 0; {sv} < {vc}; {sv}++) {{\n'
      inner_ind = ctx.ind() + '  '
      if local_vv:
        # Two-line root-CJ shape: fresh local + reassign canonical.
        assign = f'{inner_ind}auto {local_vv} = views[{bs} + {sv}];\n'
        if local_vv != vv:
          assign += f'{inner_ind}{vv} = {local_vv};\n'
      else:
        decl_kw = 'auto ' if declare else ''
        assign = f'{inner_ind}{decl_kw}{vv} = views[{bs} + {sv}];\n'
      ctx.indent_level += 1
      ctx.segment_depth += 1
      try:
        body_str = emit(body, ctx)
      finally:
        ctx.indent_level -= 1
        ctx.segment_depth -= 1
      return head + assign + body_str + f'{ctx.ind()}}}\n'

    case OuterAnchor(body=body):
      # Drop indent_level by segment_depth (so emit lands at the
      # surrounding scope's indent) and reset segment_depth to 0
      # inside body so any inner D2lSegmentLoop / IntersectIter
      # anchors against this fresh outer base.
      saved_il = ctx.indent_level
      saved_sd = ctx.segment_depth
      ctx.indent_level -= ctx.segment_depth
      ctx.segment_depth = 0
      try:
        return emit(body, ctx)
      finally:
        ctx.indent_level = saved_il
        ctx.segment_depth = saved_sd

    case If(cond=cond, body=body):
      # Body emitted at SAME indent as the wrapping if (legacy
      # quirk — body was rendered before the wrap was applied).
      return (
        f'{ctx.ind()}if ({emit_expr(cond, ctx)}) {{\n'
        + emit(body, ctx)
        + f'{ctx.ind()}}}\n'
      )

    case CartesianFlatLoop(
      idx_var=idx, bound_var=bound, lane_var=lane,
      group_size_var=gs, body=body,
    ):
      return (
        f'{ctx.ind()}for (uint32_t {idx} = {lane}; '
        f'{idx} < {bound}; {idx} += {gs}) {{\n'
        + emit(body, ctx)
        + f'{ctx.ind()}}}\n'
      )

    case Cartesian2DDecompose(
      major_var=mv, idx0_var=i0, idx1_var=i1,
      flat_idx_var=fi, deg0_var=d0, deg1_var=d1,
    ):
      return (
        f'{ctx.ind()}const bool {mv} = ({d1} >= {d0});\n'
        f'{ctx.ind()}uint32_t {i0}, {i1};\n'
        f'{ctx.ind()}if ({mv}) {{\n'
        f'{ctx.ind()}  {i0} = {fi} / {d1};\n'
        f'{ctx.ind()}  {i1} = {fi} % {d1};\n'
        f'{ctx.ind()}}} else {{\n'
        f'{ctx.ind()}  {i1} = {fi} / {d0};\n'
        f'{ctx.ind()}  {i0} = {fi} % {d0};\n'
        f'{ctx.ind()}}}\n'
      )

    case CartesianNDecompose(
      flat_idx_var=fi, idx_vars=idxs, deg_vars=degs,
    ):
      n = len(idxs)
      lines = [f'{ctx.ind()}uint32_t remaining = {fi};\n']
      for k in range(n - 1, -1, -1):
        lines.append(f'{ctx.ind()}uint32_t {idxs[k]} = remaining % {degs[k]};\n')
        if k > 0:
          lines.append(f'{ctx.ind()}remaining /= {degs[k]};\n')
      return ''.join(lines)

    case GridStrideLoop(idx_name=idx, bound=bound, body=body):
      # Body is rendered at the SAME indent as the for-loop preamble.
      # Sub-parts that need increased indent should wrap themselves
      # in IndentBlock — this matches the legacy emitter's convention
      # where some children of a scope are at +1 indent and others
      # at the surrounding indent.
      return (
        f'{ctx.ind()}// WARP MODE: 32 threads cooperatively handle one row\n'
        f'{ctx.ind()}for (uint32_t {idx} = warp_id; {idx} < {emit_expr(bound, ctx)}; '
        f'{idx} += num_warps) {{\n'
        + emit(body, ctx)
        + f'{ctx.ind()}}}\n'
      )

    case ParallelFor(strategy=strategy, body=body):
      if strategy != 'warp_strided':
        raise NotImplementedError(
          f'target.cuda M1 supports strategy="warp_strided" only; got {strategy!r}'
        )
      # The strategy itself doesn't add code at the body's indent — the
      # warp_id/num_warps come from the kernel signature, and the
      # GridStrideLoop inside emits the actual loop.
      return emit(body, ctx)

    case Phase(mode=_mode, body=body):
      # M1: phase is a marker; the legacy emitter doesn't bracket
      # phase scopes in code (the OutputContext template handles it
      # at runtime). Just emit the body.
      return emit(body, ctx)

    case LaneZeroGuard(body=body):
      # Renders as `if (tile.thread_rank() == 0) <body>` on a single line
      # if the body is a single RawString; otherwise as a braced block.
      # The legacy form for the common case (single emit_direct call) is
      # the one-liner — match that.
      if isinstance(body, RawString):
        return f'{ctx.ind()}if ({ctx.tile_var}.thread_rank() == 0) {body.text}\n'
      # Multi-statement guard — emit a brace block. (Not exercised by M1
      # fixtures; included for completeness.)
      lz_head = f'{ctx.ind()}if ({ctx.tile_var}.thread_rank() == 0) {{\n'
      ctx.indent_level += 1
      try:
        body_str = emit(body, ctx)
      finally:
        ctx.indent_level -= 1
      return lz_head + body_str + f'{ctx.ind()}}}\n'

    case Comment(text=text):
      return f'{ctx.ind()}// {text}\n'

    case RawString(text=text):
      return f'{ctx.ind()}{text}\n'

    case WriteOutput(output_var=out, values=values):
      args = ', '.join(emit_expr(v, ctx) for v in values)
      return f'{ctx.ind()}{out}.emit_direct({args});\n'

    case SaTiledCartesian2D(
      view_var0=vv0, view_var1=vv1, handle_var0=hv0, handle_var1=hv1,
      col0=col0, col1=col1, var_name0=vn0, var_name1=vn1,
      lane_var=lv, group_size_var=gv, total_var=tv,
      degree_var0=dv0, degree_var1=dv1, flat_idx_var=fiv,
      t0_base=t0b, t1_base=t1b, t0_len=t0l, t1_len=t1l,
      tile_total=tt, batch_var=bv, valid_var=vvar,
      fb_batch_var=fbv, major_var=mv, idx0_var=i0v, idx1_var=i1v,
      body=body,
    ):
      return _emit_tiled_cartesian_2d(
        ctx, vv0, vv1, hv0, hv1, col0, col1, vn0, vn1,
        lv, gv, tv, dv0, dv1, fiv, t0b, t1b, t0l, t1l,
        tt, bv, vvar, fbv, mv, i0v, i1v, body,
      )

    case TiledBallotBlock(valid_var=vvar, outputs=outputs):
      return _emit_tiled_ballot_block(ctx, vvar, outputs)

    case AddCount(output_var=out, delta=delta):
      return f'{ctx.ind()}{out}.add_count({emit_expr(delta, ctx)});\n'

    case _:
      assert_never(op)


def emit_expr(op: Op, ctx: EmitCtx) -> str:
  '''Emit an expression-shaped IIR op. Returns a string with no
  trailing newline and no leading indent — suitable to plug into a
  statement template.'''
  match op:
    case VarRef(name=name):
      return name

    case SaRoot(view_name=view):
      return f'HandleType(0, {view}.num_rows_, 0)'

    case SaValid(handle_name=h):
      return f'{h}.valid()'

    case SaDegree(handle_name=h):
      return f'{h}.degree()'

    case SaGetVal(view_name=view, col=col, idx_var_name=idx):
      return f'{view}.get_value({col}, {idx})'

    case SaGetValAt(handle_name=h, view_name=view, idx_var_name=idx):
      return f'{h}.get_value_at({view}, {idx})'

    case SaHint(lo_var=lo, hi_var=hi, depth=d):
      return f'HandleType({lo}, {hi}, {d})'

    case SaPrefCoop(parent=parent, key_var=k, view_name=view):
      return f'{emit_expr(parent, ctx)}.prefix({k}, {ctx.tile_var}, {view})'

    case SaPrefSeq(parent=parent, key_var=k, view_name=view):
      return f'{emit_expr(parent, ctx)}.prefix_seq({k}, {view})'

    case SaIterators(handle_name=h, view_name=view):
      return f'{h}.iterators({view})'

    case SaChildRange(handle_name=h, pos_expr=pos, key_var=k, view_name=view):
      return f'{h}.child_range({pos}, {k}, {ctx.tile_var}, {view})'

    case SaGetValAtPos(view_name=view, col=col, handle_name=h, idx_var_name=idx):
      return f'{view}.get_value({col}, {h}.begin() + {idx})'

    case RawString(text=text):
      return text

    case _:
      raise NotImplementedError(
        f'target.cuda M1: emit_expr does not yet handle {type(op).__name__}; '
        f'add a case as the dialect grows'
      )


def _emit_tiled_cartesian_2d(
  ctx: EmitCtx,
  vv0: str, vv1: str, hv0: str, hv1: str,
  col0: int, col1: int,
  vn0: str, vn1: str,
  lv: str, gv: str, tv: str,
  dv0: str, dv1: str, fiv: str,
  t0b: str, t1b: str, t0l: str, t1l: str,
  tt: str, bv: str, vvar: str,
  fbv: str, mv: str, i0v: str, i1v: str,
  body: Op,
) -> str:
  '''Lifted from legacy `_emit_tiled_cartesian` (codegen/jit/
  instructions.py). The whole structure is string-level — the body
  IR emits at the surrounding scope's indent (legacy quirk where
  bodies are pre-rendered before tiled wrap textually surrounds
  them).'''
  i = ctx.ind()
  tile = ctx.tile_var
  body_str = emit(body, ctx)
  parts: list[str] = [
    f'{i}if ({tv} > 32) {{\n',
    f'{i}  // Tiled Cartesian: smem pre-load reads, '
    f'standard emit_direct writes\n',
    f'{i}  for (uint32_t {t0b} = 0; {t0b} < {dv0}; '
    f'{t0b} += kCartTileSize) {{\n',
    f'{i}    uint32_t {t0l} = min({t0b} + (uint32_t)kCartTileSize, '
    f'{dv0}) - {t0b};\n',
    f'{i}    for (uint32_t _ti = {lv}; _ti < {t0l}; _ti += {gv})\n',
    f'{i}      s_cart[warp_in_block][0][_ti] = {vv0}.get_value('
    f'{col0}, {hv0}.begin() + {t0b} + _ti);\n',
    f'{i}    for (uint32_t {t1b} = 0; {t1b} < {dv1}; '
    f'{t1b} += kCartTileSize) {{\n',
    f'{i}      uint32_t {t1l} = min({t1b} + (uint32_t)kCartTileSize, '
    f'{dv1}) - {t1b};\n',
    f'{i}      for (uint32_t _ti = {lv}; _ti < {t1l}; _ti += {gv})\n',
    f'{i}        s_cart[warp_in_block][1][_ti] = {vv1}.get_value('
    f'{col1}, {hv1}.begin() + {t1b} + _ti);\n',
    f'{i}      {tile}.sync();\n',
    f'{i}      uint32_t {tt} = {t0l} * {t1l};\n',
    f'{i}      for (uint32_t {bv} = 0; {bv} < {tt}; '
    f'{bv} += {gv}) {{\n',
    f'{i}        uint32_t {fiv} = {bv} + {lv};\n',
    f'{i}        bool {vvar} = {fiv} < {tt};\n',
    f'{i}        auto {vn0} = {vvar} ? '
    f's_cart[warp_in_block][0][{fiv} / {t1l}] : ValueType{{0}};\n',
    f'{i}        auto {vn1} = {vvar} ? '
    f's_cart[warp_in_block][1][{fiv} % {t1l}] : ValueType{{0}};\n',
    body_str,
    f'{i}      }}\n',
    f'{i}      {tile}.sync();\n',
    f'{i}    }}\n',
    f'{i}  }}\n',
    f'{i}}} else {{\n',
    f'{i}  for (uint32_t {fbv} = 0; {fbv} < {tv}; '
    f'{fbv} += {gv}) {{\n',
    f'{i}    uint32_t {fiv} = {fbv} + {lv};\n',
    f'{i}    bool {vvar} = {fiv} < {tv};\n',
    f'{i}    const bool {mv} = ({dv1} >= {dv0});\n',
    f'{i}    uint32_t {i0v}, {i1v};\n',
    f'{i}    if ({mv}) {{ {i0v} = {fiv} / {dv1}; '
    f'{i1v} = {fiv} % {dv1}; }}\n',
    f'{i}    else {{ {i1v} = {fiv} / {dv0}; '
    f'{i0v} = {fiv} % {dv0}; }}\n',
    f'{i}    auto {vn0} = {vv0}.get_value({col0}, '
    f'{hv0}.begin() + {i0v});\n',
    f'{i}    auto {vn1} = {vv1}.get_value({col1}, '
    f'{hv1}.begin() + {i1v});\n',
    body_str,
    f'{i}  }}\n',
    f'{i}}}\n',
  ]
  return ''.join(parts)


def _emit_tiled_ballot_block(
  ctx: EmitCtx,
  vvar: str,
  outputs: tuple[tuple[int, tuple[str, ...], str], ...],
) -> str:
  '''Lifted from legacy `emit_helpers.jit_insert_into` ballot path.
  Single ballot setup + per-output `if (valid) { write }` block,
  closing with `warp_local_count += _tc_active`.'''
  i = ctx.ind()
  tile = ctx.tile_var
  parts: list[str] = []
  for idx_, (dest_idx, values, debug) in enumerate(outputs):
    if debug:
      parts.append(f'{i}// {debug}\n')
    if idx_ == 0:
      parts.append(f'{i}{{\n')
      parts.append(f'{i}  uint32_t _tc_ballot = {tile}.ballot({vvar});\n')
      parts.append(f'{i}  uint32_t _tc_active = __popc(_tc_ballot);\n')
      parts.append(f'{i}  if (_tc_active > 0) {{\n')
      parts.append(
        f'{i}    uint32_t _tc_mask = (1u << {tile}.thread_rank()) - 1u;\n'
      )
      parts.append(
        f'{i}    uint32_t _tc_off = __popc(_tc_ballot & _tc_mask);\n'
      )
    parts.append(f'{i}    if ({vvar}) {{\n')
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
