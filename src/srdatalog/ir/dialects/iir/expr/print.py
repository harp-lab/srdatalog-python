'''Print_i for the iir.expr dialect.'''

from __future__ import annotations

from srdatalog.ir.dialects.iir.expr.ops import (
  BinOp,
  CCast,
  IndexExpr,
  IntLit,
  MemberAccess,
  MemberCall,
  Parens,
  StaticCast,
  Ternary,
  UnaryOp,
)
from srdatalog.ir.print_iir import _ind, print_iir

OPS: tuple[type, ...] = (
  BinOp,
  CCast,
  IndexExpr,
  IntLit,
  MemberAccess,
  MemberCall,
  Parens,
  StaticCast,
  Ternary,
  UnaryOp,
)


def print_op(op, indent: int = 0) -> str:
  p = _ind(indent)

  if isinstance(op, IntLit):
    return p + f'(int-lit #:value {op.value})'

  if isinstance(op, CCast):
    expr = print_iir(op.expr, indent + 1)
    return p + f'(c-cast #:type-str "{op.type_str}"\n' + p + '  #:expr\n' + expr + ')'

  if isinstance(op, StaticCast):
    expr = print_iir(op.expr, indent + 1)
    return p + f'(static-cast #:type-str "{op.type_str}"\n' + p + '  #:expr\n' + expr + ')'

  if isinstance(op, UnaryOp):
    expr = print_iir(op.expr, indent + 1)
    return p + f'(unary-op #:op-str "{op.op_str}"\n' + p + '  #:expr\n' + expr + ')'

  if isinstance(op, Parens):
    expr = print_iir(op.expr, indent + 1)
    return p + '(parens\n' + expr + ')'

  if isinstance(op, Ternary):
    cond = print_iir(op.cond, indent + 1)
    then_ = print_iir(op.then_, indent + 1)
    else_ = print_iir(op.else_, indent + 1)
    return (
      p
      + '(ternary\n'
      + p
      + '  #:cond\n'
      + cond
      + '\n'
      + p
      + '  #:then\n'
      + then_
      + '\n'
      + p
      + '  #:else\n'
      + else_
      + ')'
    )

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

  if isinstance(op, IndexExpr):
    arr = print_iir(op.arr, indent + 1)
    idx = print_iir(op.idx, indent + 1)
    return p + '(index-expr\n' + p + '  #:arr\n' + arr + '\n' + p + '  #:idx\n' + idx + ')'

  if isinstance(op, MemberAccess):
    obj = print_iir(op.obj, indent + 1)
    return p + f'(member-access #:member "{op.member}"\n' + p + '  #:obj\n' + obj + ')'

  if isinstance(op, MemberCall):
    obj = print_iir(op.obj, indent + 1)
    if not op.args:
      return p + f'(member-call #:method "{op.method}"\n' + p + '  #:obj\n' + obj + ')'
    args = '\n'.join(print_iir(a, indent + 2) for a in op.args)
    return (
      p
      + f'(member-call #:method "{op.method}"\n'
      + p
      + '  #:obj\n'
      + obj
      + '\n'
      + p
      + '  #:args (\n'
      + args
      + '))'
    )

  raise TypeError(f'iir.expr print_op: unknown op {type(op).__name__}')


__all__ = ['OPS', 'print_op']
