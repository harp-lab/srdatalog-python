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


@final
@dataclass(frozen=True, slots=True)
class SaHint(Op):
  '''Range-narrowed root handle constructor (expression-shaped).

  Lowers (target.cuda) to:
      `HandleType(<lo_var>, <hi_var>, <depth>)`.

  Typically composed with SaPrefCoop:
      `HandleType(lo, hi, 0).prefix(root_val, tile, view)`.
  '''

  lo_var: str
  hi_var: str
  depth: int = 0


@final
@dataclass(frozen=True, slots=True)
class SaPrefCoop(Op):
  '''Cooperative prefix-narrowing on a parent handle expression.

  Lowers (target.cuda) to:
      `<parent>.prefix(<key>, tile, <view>)`.

  Used in multi-source root CJ where 32 threads cooperatively
  binary-search the parent handle for `key`.
  '''

  parent: Op
  key_var: str
  view_name: str


@final
@dataclass(frozen=True, slots=True)
class SaIterators(Op):
  '''Iterator pair for a handle, suitable to hand to
  `intersect_handles`.

  Lowers (target.cuda) to: `<handle>.iterators(<view>)`.
  '''

  handle_name: str
  view_name: str


@final
@dataclass(frozen=True, slots=True)
class SaChildRange(Op):
  '''Narrowed child range from a handle.

  Lowers (target.cuda) to:
      `<handle>.child_range(<pos_expr>, <key_var>, tile, <view>)`.

  Used inside an IntersectIter body to produce per-source child
  handles for the next nesting level.
  '''

  handle_name: str
  pos_expr: str
  key_var: str
  view_name: str
