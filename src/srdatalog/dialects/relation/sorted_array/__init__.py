'''relation.sorted_array dialect.

Index-aware ops for sorted-array relation storage. M1 covers the
subset needed for `Scan + InsertInto` lowering: SaRoot, SaValid,
SaDegree, SaGetValAt. Later milestones add SaPref, SaChild, SaHint,
SaExists, SaValues, SaPrefLb.

See docs/ir_lowering_semantics.md §10 for the lowering rules and
docs/stage2_emitter_audit.md §6 for the plugin-dispatched expression
shapes the target lowering produces.
'''

from __future__ import annotations

from srdatalog.dialects.relation.sorted_array.ops import (
  SaDegree,
  SaGetVal,
  SaGetValAt,
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
  ops=[SaRoot, SaValid, SaDegree, SaGetVal, SaGetValAt],
)

__all__ = [
  'DIALECT',
  'SaDegree',
  'SaGetVal',
  'SaGetValAt',
  'SaHandle',
  'SaRoot',
  'SaValid',
  'SaView',
]
