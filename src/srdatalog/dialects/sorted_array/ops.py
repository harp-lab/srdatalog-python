'''Sorted-array dialect ops — M1 subset.

Each op references its operands by *name* (string) for M1 pragmatism.
The legacy emitter passes view/handle variable names through string
keys; matching that lets the byte-equivalence gate compare directly.
A later milestone replaces this with lexical binding (D8) once the
gate has been validated end-to-end.

Operand naming convention:
  view_name   — name of a previously-declared view variable
                (`auto view_<rel>_<cols>_<ver> = views[<slot>];`).
  handle_name — name of a previously-bound handle variable (declared
                via iir.cf.Bind).
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from srdatalog.ir_core import Op


@final
@dataclass(frozen=True, slots=True)
class SaRoot(Op):
  '''Root handle into the sorted-array's full row-range.

  Lowers (target.cuda) to: `HandleType(0, <view_name>.num_rows_, 0)`.
  '''

  view_name: str


@final
@dataclass(frozen=True, slots=True)
class SaValid(Op):
  '''Whether a handle is non-empty / non-degenerate.

  Lowers (target.cuda) to: `<handle_name>.valid()`.
  '''

  handle_name: str


@final
@dataclass(frozen=True, slots=True)
class SaDegree(Op):
  '''Branching factor / row count at a handle position.

  Lowers (target.cuda) to: `<handle_name>.degree()`.
  '''

  handle_name: str


@final
@dataclass(frozen=True, slots=True)
class SaGetVal(Op):
  '''Get the value at column `col`, row `idx_var_name`, in the view.

  Used inside a root scan: each var binding fetches its column.

  Lowers (target.cuda) to: `<view_name>.get_value(<col>, <idx_var_name>)`.
  '''

  view_name: str
  col: int
  idx_var_name: str


@final
@dataclass(frozen=True, slots=True)
class SaGetValAt(Op):
  '''Get the value at column position `col` of a narrowed handle's
  child at slot `idx_var_name`. Used in nested ColumnJoin paths;
  M1 declares the op for completeness but doesn't yet lower it.

  Lowers (target.cuda) to:
      `<view_name>.get_value_at(<handle_name>.begin(), <idx_var_name>)`.
  '''

  handle_name: str
  view_name: str
  idx_var_name: str
