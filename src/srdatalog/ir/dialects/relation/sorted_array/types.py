'''Sorted-array dialect types.

These are markers for the type system; they don't carry runtime
value, just IR-level type identity. The target.cuda lowering knows
how to map these to concrete C++ types (`HandleType`, etc.).
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from srdatalog.ir.core import Type


@final
@dataclass(frozen=True, slots=True)
class SaView(Type):
  '''A sorted-array view of a relation. M1 carries this as a marker;
  the actual view variable is referenced by name in IR ops (M1
  pragmatic choice — a future refactor adopts proper lexical view
  binding).'''


@final
@dataclass(frozen=True, slots=True)
class SaHandle(Type):
  '''A NodeHandle into a sorted-array index. Marker type; runtime
  shape is `(view, lo, hi, depth)` per spec §9, exposed in the C++
  runtime as `HandleType`.'''
