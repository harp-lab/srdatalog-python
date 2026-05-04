'''Print_i for the relation.d2l dialect — D2lSegmentLoop only.'''

from __future__ import annotations

from srdatalog.ir.dialects.relation.d2l.ops import D2lSegmentLoop
from srdatalog.ir.print_iir import _bool, _ind, print_iir

OPS: tuple[type, ...] = (D2lSegmentLoop,)


def print_op(op, indent: int = 0) -> str:
  p = _ind(indent)

  if isinstance(op, D2lSegmentLoop):
    body = print_iir(op.body, indent + 1)
    return (
      p
      + f'(d2l-segment-loop #:seg-var {op.seg_var} '
      + f'#:view-var {op.view_var} '
      + f'#:base-slot {op.base_slot} '
      + f'#:view-count {op.view_count} '
      + f'#:declare {_bool(op.declare)} '
      + f'#:local-view-var {op.local_view_var}\n'
      + p
      + '  #:body\n'
      + body
      + ')'
    )

  raise TypeError(f'd2l print_op: unknown op {type(op).__name__}')


__all__ = ['OPS', 'print_op']
