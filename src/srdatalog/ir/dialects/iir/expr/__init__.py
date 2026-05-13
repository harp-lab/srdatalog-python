'''iir.expr — expression-shaped IIR ops.

Per docs/stage4_iir_vocabulary.md §2 (S4.2): structured replacements
for arithmetic / index / member-access / ternary RawString sites in
sorted_array/lowerings.py.

Currently registered ops:
  - BinOp(op_str, lhs, rhs) — binary expression. Generic op_str
    (per the open design decision §3.1: chose generic over per-operator).

Planned (later S4 tasks):
  - IntLit / BoolLit       — typed literals
  - IndexExpr(arr, idx)    — subscript expression
  - MemberAccess(obj, m)   — `obj.member`
  - MemberCall(obj, m, *)  — `obj.method(args)`
  - Ternary(cond, t, e)    — `cond ? t : e`
  - BoolNot(expr)          — `!expr`

The split between iir.cf (statements) and iir.expr (expressions)
mirrors the existing emit/emit_expr distinction in the renderer
registry — it is not a purely cosmetic split.
'''

from __future__ import annotations

from srdatalog.ir.core import Dialect
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

DIALECT = Dialect(
  name='iir.expr',
  ops=[
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
  ],
)


# Verifier scaffolding — expression-level invariants (operator string
# in the allowed set, lhs/rhs are expression-shaped, etc.) land
# incrementally as we encode them.
def _register_passes() -> None:
  from srdatalog.ir.core.passes import verifier

  @verifier(DIALECT)
  def _verify(_prog):
    return []


_register_passes()


__all__ = [
  'DIALECT',
  'BinOp',
  'CCast',
  'IndexExpr',
  'IntLit',
  'MemberAccess',
  'MemberCall',
  'Parens',
  'StaticCast',
  'Ternary',
  'UnaryOp',
]
