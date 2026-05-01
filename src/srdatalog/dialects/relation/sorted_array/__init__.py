'''relation.sorted_array dialect.

Index-aware ops for sorted-array relation storage. Currently covers
the M1-M3 subset:

  M1: SaRoot, SaValid, SaDegree, SaGetVal, SaGetValAt.
  M3: SaHint, SaPrefCoop, SaIterators, SaChildRange.

Planned (M4+): SaPref (for nested CJ), SaExists (for negation),
SaValues, SaPrefLb (lower-bound prefix).

See docs/ir_lowering_semantics.md §10 for the lowering rules and
docs/stage2_emitter_audit.md §6 for the plugin-dispatched expression
shapes the target lowering produces.
'''

from __future__ import annotations

from srdatalog.dialects.relation.sorted_array.ops import (
  SaChildRange,
  SaDegree,
  SaGetVal,
  SaGetValAt,
  SaHint,
  SaIterators,
  SaPrefCoop,
  SaRoot,
  SaValid,
)
from srdatalog.dialects.relation.sorted_array.types import (
  SaHandle,
  SaView,
)
from srdatalog.ir_core import Dialect

DIALECT = Dialect(
  name='relation.sorted_array',
  types=[SaHandle, SaView],
  ops=[
    SaChildRange,
    SaDegree,
    SaGetVal,
    SaGetValAt,
    SaHint,
    SaIterators,
    SaPrefCoop,
    SaRoot,
    SaValid,
  ],
)

__all__ = [
  'DIALECT',
  'SaChildRange',
  'SaDegree',
  'SaGetVal',
  'SaGetValAt',
  'SaHandle',
  'SaHint',
  'SaIterators',
  'SaPrefCoop',
  'SaRoot',
  'SaValid',
  'SaView',
]
