'''Print_i for the relation.sorted_array dialect.

See docs/stage3a_execution_plan.md §1 + ../../iir/cf/print.py for the
contract. Twelve ops, each maps to a single s-expression form.
'''

from __future__ import annotations

from srdatalog.ir.core import Op
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
from srdatalog.ir.print_iir import _ind, print_iir

OPS: tuple[type, ...] = (
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


def print_op(op: Op, indent: int = 0) -> str:
  p = _ind(indent)

  if isinstance(op, SaRoot):
    return p + f'(sa-root #:view-name {op.view_name})'

  if isinstance(op, SaValid):
    return p + f'(sa-valid #:handle-name {op.handle_name})'

  if isinstance(op, SaDegree):
    return p + f'(sa-degree #:handle-name {op.handle_name})'

  if isinstance(op, SaGetVal):
    return (
      p
      + f'(sa-get-val #:view-name {op.view_name} '
      + f'#:col {op.col} #:idx-var-name {op.idx_var_name})'
    )

  if isinstance(op, SaGetValAt):
    return (
      p
      + f'(sa-get-val-at #:handle-name {op.handle_name} '
      + f'#:view-name {op.view_name} '
      + f'#:idx-var-name {op.idx_var_name})'
    )

  if isinstance(op, SaHint):
    return p + (f'(sa-hint #:lo-var {op.lo_var} #:hi-var {op.hi_var} #:depth {op.depth})')

  if isinstance(op, SaPrefCoop):
    parent = print_iir(op.parent, indent + 1)
    return (
      p
      + f'(sa-pref-coop #:key-var {op.key_var} #:view-name {op.view_name}\n'
      + p
      + '  #:parent\n'
      + parent
      + ')'
    )

  if isinstance(op, SaPrefSeq):
    parent = print_iir(op.parent, indent + 1)
    return (
      p
      + f'(sa-pref-seq #:key-var {op.key_var} #:view-name {op.view_name}\n'
      + p
      + '  #:parent\n'
      + parent
      + ')'
    )

  if isinstance(op, SaIterators):
    return p + f'(sa-iterators #:handle-name {op.handle_name} #:view-name {op.view_name})'

  if isinstance(op, SaChildRange):
    return (
      p
      + f'(sa-child-range #:handle-name {op.handle_name} '
      + f'#:pos-expr {op.pos_expr} '
      + f'#:key-var {op.key_var} #:view-name {op.view_name})'
    )

  if isinstance(op, SaGetValAtPos):
    return (
      p
      + f'(sa-get-val-at-pos #:view-name {op.view_name} '
      + f'#:col {op.col} #:handle-name {op.handle_name} '
      + f'#:idx-var-name {op.idx_var_name})'
    )

  if isinstance(op, SaTiledCartesian2D):
    body = print_iir(op.body, indent + 1)
    return (
      p
      + '(sa-tiled-cartesian-2d '
      + f'#:view-var0 {op.view_var0} #:view-var1 {op.view_var1} '
      + f'#:handle-var0 {op.handle_var0} #:handle-var1 {op.handle_var1} '
      + f'#:col0 {op.col0} #:col1 {op.col1} '
      + f'#:var-name0 {op.var_name0} #:var-name1 {op.var_name1} '
      + f'#:lane-var {op.lane_var} #:group-size-var {op.group_size_var} '
      + f'#:total-var {op.total_var} '
      + f'#:degree-var0 {op.degree_var0} #:degree-var1 {op.degree_var1} '
      + f'#:flat-idx-var {op.flat_idx_var} '
      + f'#:t0-base {op.t0_base} #:t1-base {op.t1_base} '
      + f'#:t0-len {op.t0_len} #:t1-len {op.t1_len} '
      + f'#:tile-total {op.tile_total} #:batch-var {op.batch_var} '
      + f'#:valid-var {op.valid_var} #:fb-batch-var {op.fb_batch_var} '
      + f'#:major-var {op.major_var} '
      + f'#:idx0-var {op.idx0_var} #:idx1-var {op.idx1_var}\n'
      + p
      + '  #:body\n'
      + body
      + ')'
    )

  raise TypeError(f'sorted_array print_op: unknown op {type(op).__name__}')


__all__ = ['OPS', 'print_op']
