'''Print_i for the iir.expr dialect.'''

from __future__ import annotations

from srdatalog.ir.dialects.iir.expr.ops import BinOp
from srdatalog.ir.print_iir import _ind, print_iir

OPS: tuple[type, ...] = (BinOp,)


def print_op(op, indent: int = 0) -> str:
  p = _ind(indent)

  if isinstance(op, BinOp):
    lhs = print_iir(op.lhs, indent + 1)
    rhs = print_iir(op.rhs, indent + 1)
    return (
      p
      + f'(bin-op #:op-str "{op.op_str}"\n'
      + p
      + '  #:lhs\n'
      + lhs
      + '\n'
      + p
      + '  #:rhs\n'
      + rhs
      + ')'
    )

  raise TypeError(f'iir.expr print_op: unknown op {type(op).__name__}')


__all__ = ['OPS', 'print_op']
