'''iir.expr op definitions — expression-shaped IIR.

Per the project memory note `feedback_decorator_registries.md`,
all ops are pure data (D1), frozen+slots dataclasses (D2/D3),
@final to lock the closed sum (D11).

S4.2 ships only `BinOp`; subsequent S4 tasks add the rest of the
expression vocabulary (literals, index expressions, member access,
ternaries, boolean negation).
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from srdatalog.ir.core import Op


@final
@dataclass(frozen=True, slots=True)
class BinOp(Op):
  '''Binary operator expression.

  `op_str` is the C++ operator literal (e.g. `'+'`, `'*'`, `'/'`,
  `'%'`, `'=='`, `'<'`, `'&&'`, `'<<'`, etc.). Per the open design
  decision in docs/stage4_iir_vocabulary.md §3.1, we use a generic
  `op_str` rather than per-operator subclasses (`Mul`, `Div`, ...)
  to keep the op surface small.

  Renders to `<lhs> <op_str> <rhs>` with no surrounding parens.
  Callers needing precedence-explicit grouping should construct
  the IR with explicit grouping (or wait for a future Parens op).

  N-ary chains (e.g. `a * b * c`) are left-folded at construction
  time: `BinOp("*", BinOp("*", a, b), c)`. A `bin_op_chain(op_str,
  exprs)` helper makes this ergonomic.
  '''

  op_str: str
  lhs: Op
  rhs: Op


def bin_op_chain(op_str: str, exprs: list[Op]) -> Op:
  '''Left-fold `exprs` with the given binary operator. Requires
  `len(exprs) >= 1`. For `len == 1`, returns the lone expression
  unchanged (no wrapping BinOp emitted).

  Used by lowerings that produce variadic-arity expressions like
  `degree0 * degree1 * degree2`.
  '''
  if not exprs:
    raise ValueError(f'bin_op_chain({op_str!r}, ...) requires at least one expression')
  result = exprs[0]
  for e in exprs[1:]:
    result = BinOp(op_str=op_str, lhs=result, rhs=e)
  return result


__all__ = ['BinOp', 'bin_op_chain']
