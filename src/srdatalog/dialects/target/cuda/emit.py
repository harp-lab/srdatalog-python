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
  Comment,
  GridStrideLoop,
  IfReturnIfNot,
  IndentBlock,
  LaneZeroGuard,
  ParallelFor,
  Phase,
  RawString,
  VarRef,
  WriteOutput,
)
from srdatalog.dialects.relation.sorted_array.ops import (
  SaDegree,
  SaGetVal,
  SaGetValAt,
  SaRoot,
  SaValid,
)
from srdatalog.ir_core import Op, assert_never


@dataclass
class EmitCtx:
  '''Mutable emission state during the C++ walk.

  - `indent_level`: counts of 2-space units. Legacy emitter starts
    at indent=2 (operator() body); M1 callers pass that in to match.
  - `tile_var`: name of the tile/thread-group variable. "tile" by default.
  '''

  indent_level: int = 2
  tile_var: str = 'tile'

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
      lines = f'{ctx.ind()}if ({ctx.tile_var}.thread_rank() == 0) {{\n'
      ctx.indent_level += 1
      try:
        body_str = emit(body, ctx)
      finally:
        ctx.indent_level -= 1
      return lines + body_str + f'{ctx.ind()}}}\n'

    case Comment(text=text):
      return f'{ctx.ind()}// {text}\n'

    case RawString(text=text):
      return f'{ctx.ind()}{text}\n'

    case WriteOutput(output_var=out, values=values):
      args = ', '.join(emit_expr(v, ctx) for v in values)
      return f'{ctx.ind()}{out}.emit_direct({args});\n'

    case AddCount(delta=delta):
      return f'{ctx.ind()}output.add_count({emit_expr(delta, ctx)});\n'

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
      return f'{view}.get_value_at({h}.begin(), {idx})'

    case RawString(text=text):
      return text

    case _:
      raise NotImplementedError(
        f'target.cuda M1: emit_expr does not yet handle {type(op).__name__}; '
        f'add a case as the dialect grows'
      )
