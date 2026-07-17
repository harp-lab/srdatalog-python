'''Print_i for the iir.cf (control-flow) dialect.

Per docs/stage3a_execution_plan.md §1: Print is one of three distinct
operations on IR data (Lowering / Render / Print). This module is
iir.cf's contribution to Print_i. Style mirrors mir/print.py — Racket-
canonical `#:keyword` form.

Children of an op recurse through `srdatalog.ir.print_iir.print_iir`
so a sub-tree mixing dialects renders cleanly.
'''

from __future__ import annotations

from srdatalog.ir.core import Op
from srdatalog.ir.dialects.iir.cf.ops import (
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
from srdatalog.ir.print_iir import _bool, _ind, _quoted, _str_tuple, print_iir

OPS: tuple[type, ...] = (
  AddCount,
  BlankLine,
  Bind,
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


def print_op(op: Op, indent: int = 0) -> str:
  p = _ind(indent)

  # --- Sequencing primitives ---

  if isinstance(op, Block):
    if not op.stmts:
      return p + '(block)'
    body = '\n'.join(print_iir(s, indent + 1) for s in op.stmts)
    return p + '(block\n' + body + ')'

  if isinstance(op, IndentBlock):
    if not op.stmts:
      return p + f'(indent-block #:extra {op.extra})'
    body = '\n'.join(print_iir(s, indent + 1) for s in op.stmts)
    return p + f'(indent-block #:extra {op.extra}\n' + body + ')'

  if isinstance(op, BlankLine):
    return p + '(blank-line)'

  # --- Bindings & references ---

  if isinstance(op, Bind):
    expr = print_iir(op.expr, indent + 1)
    return (
      p + f'(bind #:name {op.name} #:type-decl {op.type_decl}\n' + p + '  #:expr\n' + expr + ')'
    )

  if isinstance(op, VarRef):
    return p + f'(var-ref #:name {op.name})'

  # --- Conditionals ---

  if isinstance(op, IfReturnIfNot):
    cond = print_iir(op.cond, indent + 1)
    return p + '(if-return-if-not\n' + cond + ')'

  if isinstance(op, IfContinueIfNot):
    cond = print_iir(op.cond, indent + 1)
    return p + '(if-continue-if-not\n' + cond + ')'

  if isinstance(op, If):
    cond = print_iir(op.cond, indent + 1)
    body = print_iir(op.body, indent + 1)
    return p + '(if\n' + p + '  #:cond\n' + cond + '\n' + p + '  #:body\n' + body + ')'

  # --- Cartesian decomposition ---

  if isinstance(op, CartesianFlatLoop):
    body = print_iir(op.body, indent + 1)
    return (
      p
      + f'(cartesian-flat-loop #:idx-var {op.idx_var} #:bound-var {op.bound_var} '
      + f'#:lane-var {op.lane_var} #:group-size-var {op.group_size_var}\n'
      + body
      + ')'
    )

  if isinstance(op, Cartesian2DDecompose):
    return (
      p
      + f'(cartesian-2d-decompose #:major-var {op.major_var} '
      + f'#:idx0-var {op.idx0_var} #:idx1-var {op.idx1_var} '
      + f'#:flat-idx-var {op.flat_idx_var} '
      + f'#:deg0-var {op.deg0_var} #:deg1-var {op.deg1_var})'
    )

  if isinstance(op, CartesianNDecompose):
    return (
      p
      + f'(cartesian-n-decompose #:flat-idx-var {op.flat_idx_var} '
      + f'#:idx-vars {_str_tuple(op.idx_vars)} '
      + f'#:deg-vars {_str_tuple(op.deg_vars)})'
    )

  # --- Iteration ---

  if isinstance(op, IntersectIter):
    iters = '\n'.join(print_iir(e, indent + 2) for e in op.iterator_exprs)
    body = print_iir(op.body, indent + 1)
    return (
      p
      + f'(intersect-iter #:intersect-var {op.intersect_var} '
      + f'#:iter-var {op.iter_var} #:value-var {op.value_var}\n'
      + p
      + '  #:iterator-exprs (\n'
      + iters
      + ')\n'
      + p
      + '  #:body\n'
      + body
      + ')'
    )

  if isinstance(op, GridStrideLoop):
    bound = print_iir(op.bound, indent + 1)
    body = print_iir(op.body, indent + 1)
    return (
      p
      + f'(grid-stride-loop #:idx-name {op.idx_name}\n'
      + p
      + '  #:bound\n'
      + bound
      + '\n'
      + p
      + '  #:body\n'
      + body
      + ')'
    )

  if isinstance(op, ParallelFor):
    body = print_iir(op.body, indent + 1)
    return p + f'(parallel-for #:strategy {op.strategy}\n' + body + ')'

  # --- Phase / guards ---

  if isinstance(op, Phase):
    body = print_iir(op.body, indent + 1)
    return p + f'(phase #:mode {op.mode}\n' + body + ')'

  if isinstance(op, LaneZeroGuard):
    body = print_iir(op.body, indent + 1)
    return p + '(lane-zero-guard\n' + body + ')'

  # --- Outputs ---

  if isinstance(op, WriteOutput):
    if not op.values:
      return p + f'(write-output #:output-var {op.output_var})'
    values = '\n'.join(print_iir(v, indent + 2) for v in op.values)
    return (
      p + f'(write-output #:output-var {op.output_var}\n' + p + '  #:values (\n' + values + '))'
    )

  if isinstance(op, AddCount):
    delta = print_iir(op.delta, indent + 1)
    return p + f'(add-count #:output-var {op.output_var}\n' + delta + ')'

  # --- Pass-through escape hatches ---

  if isinstance(op, Comment):
    return p + f'(comment #:text {_quoted(op.text)})'

  if isinstance(op, RawString):
    return p + f'(raw-string #:text {_quoted(op.text)})'

  # --- Tiled-Cartesian ballot block ---

  if isinstance(op, TiledBallotBlock):
    if not op.outputs:
      return p + f'(tiled-ballot-block #:valid-var {op.valid_var})'
    out_lines = []
    for dest_idx, vals, debug in op.outputs:
      out_lines.append(
        f'{p}    (output #:dest-idx {dest_idx} '
        f'#:values {_str_tuple(vals)} '
        f'#:debug {_quoted(debug)})'
      )
    outs = '\n'.join(out_lines)
    return (
      p + f'(tiled-ballot-block #:valid-var {op.valid_var}\n' + p + '  #:outputs (\n' + outs + '))'
    )

  # --- Indent anchor ---

  if isinstance(op, OuterAnchor):
    body = print_iir(op.body, indent + 1)
    return p + '(outer-anchor\n' + body + ')'

  # mypy-strict exhaustiveness assertion
  raise TypeError(f'iir.cf print_op: unknown op {type(op).__name__}')


# Silence ruff unused-imports for helpers re-exported as building blocks.
_ = (_bool,)

__all__ = ['OPS', 'print_op']
