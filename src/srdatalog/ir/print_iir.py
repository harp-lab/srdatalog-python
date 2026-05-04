'''Print_i — canonical s-expression form for IIR ops.

Per [`docs/stage3a_execution_plan.md` §1](../../../docs/stage3a_execution_plan.md),
"Print" is one of the three distinct operations on IR data:

    Lowering  D1.Op → D2.Op*  (data → data, framework-dispatched)
    Render    D.Op  → str     (data → target text, codegen-dispatched)
    Print     D.Op  → s-expr  (data → canonical text, dialect-owned)

This module is the top-level dispatcher for IIR Print. Each
IIR-contributing dialect (iir.cf, relation.sorted_array, relation.d2l,
parallel.data) ships a `print.py` with:

    OPS:      tuple[type, ...]                 # op classes the module handles
    print_op(op, indent: int = 0) -> str       # per-dialect dispatcher

`print_iir(op, indent)` routes by `isinstance(op, dialect.OPS)` to the
owning dialect's `print_op`. Children inside an op recurse through
`print_iir` so a sub-tree mixing dialects renders cleanly.

Mirrors `mir/emit.py` (the MIR s-expr printer) in style: Racket-canonical
`#:keyword` form, two-space indent per level, parens-balanced.

Test surface: byte-equal s-expr at the IIR layer. See
`tests/test_iir_print.py`.
'''

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from srdatalog.ir.core import Op


# -----------------------------------------------------------------------------
# Helpers (used by every dialect's print.py)
# -----------------------------------------------------------------------------


def _ind(level: int) -> str:
  '''Two-space indent per level, matching mir/emit.py.'''
  return '  ' * level


def _str_tuple(strs: tuple[str, ...]) -> str:
  '''"(a b c)" — space-separated, no commas. Matches mir/emit.py _var_tuple.'''
  return '(' + ' '.join(strs) + ')'


def _quoted(text: str) -> str:
  '''Free-text fields (Comment.text, RawString.text) — escape and quote.'''
  return '"' + text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') + '"'


def _bool(b: bool) -> str:
  '''Lowercase, matching mir/emit.py's `(step #:recursive true)` form.'''
  return 'true' if b else 'false'


# -----------------------------------------------------------------------------
# Top-level dispatcher
# -----------------------------------------------------------------------------


def print_iir(op: Op, indent: int = 0) -> str:
  '''Render an IIR op (from any IIR-contributing dialect) as s-expression text.

  Routes by isinstance against each dialect's `OPS` tuple. Children
  recurse through this function, so an IIR tree mixing dialects (e.g.
  iir.cf.Block containing sorted_array.SaPrefCoop) renders correctly.

  Raises TypeError on an unknown op kind — useful as the "did the
  dispatcher get out of sync with a new dialect" signal.
  '''
  # Lazy imports to avoid module-load-time cycle (each dialect's
  # print.py imports helpers from this module).
  from srdatalog.ir.dialects.iir.cf.print import OPS as cf_OPS
  from srdatalog.ir.dialects.iir.cf.print import print_op as cf_print
  from srdatalog.ir.dialects.parallel.data.print import OPS as pd_OPS
  from srdatalog.ir.dialects.parallel.data.print import print_op as pd_print
  from srdatalog.ir.dialects.relation.d2l.print import OPS as d2l_OPS
  from srdatalog.ir.dialects.relation.d2l.print import print_op as d2l_print
  from srdatalog.ir.dialects.relation.sorted_array.print import OPS as sa_OPS
  from srdatalog.ir.dialects.relation.sorted_array.print import print_op as sa_print

  if isinstance(op, cf_OPS):
    return cf_print(op, indent)
  if isinstance(op, sa_OPS):
    return sa_print(op, indent)
  if isinstance(op, d2l_OPS):
    return d2l_print(op, indent)
  if isinstance(op, pd_OPS):
    return pd_print(op, indent)
  raise TypeError(
    f'print_iir: unknown IIR op {type(op).__name__!r} (module={type(op).__module__!r}). '
    'Dispatcher missing the dialect, or the dialect missing its OPS entry?'
  )


__all__ = ['print_iir']
