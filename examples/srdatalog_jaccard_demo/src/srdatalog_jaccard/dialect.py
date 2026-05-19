'''`relation.jaccard` dialect — sparse-similarity index.

The dialect's vocabulary today is exactly one Op type: `JaccardIndex`,
a wrap op around an `mir.InsertInto` that signals "this emission
should pass through a Jaccard-similarity gate". Materialized by the
`Jaccard` typed pragma (see `pragmas/jaccard.py`) during
`MirPragmaPass`.

Why a wrap op (not a new MIR-level field)? The op carries the inner
`InsertInto` so the registered `@lowering(target=DIALECT, source=
JaccardIndex)` rule has everything it needs to emit IIR (vars,
rel_name, index). This mirrors the built-in `DedupGate` / `WSScope`
pattern (`src/srdatalog/ir/mir/types.py:DedupGate`,
`src/srdatalog/ir/dialects/parallel/atomic_ws/__init__.py`); the
difference is that `JaccardIndex` is defined HERE in an external
package — proving the framework's claim that new wrap ops do not
require core-side enum updates.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from srdatalog.ir.core import Dialect, Op
from srdatalog.ir.mir.types import InsertInto


@final
@dataclass(frozen=True, slots=True)
class JaccardIndex(Op):
  '''Wrap op: route an emission through a Jaccard-similarity gate.

  Inserted by `srdatalog_jaccard.pragmas.jaccard.materialize_jaccard`
  during `MirPragmaPass` whenever an `ExecutePipeline` carries a
  `Jaccard` pragma. Lowered by the
  `@lowering(target=DIALECT, source=JaccardIndex)` rule registered
  in `srdatalog_jaccard.lowerings` (which delegates back into the
  sorted_array dialect's `_lower_insert_into` helper to emit IIR).

  Fields:

    inner     — the `InsertInto` op being gated. Carrying it (rather
                than a relation name + var list) lets the lowering
                reuse the existing sorted_array machinery verbatim.
    threshold — the Jaccard similarity threshold (0.0–1.0). Recorded
                here so the lowering can lift it into a generated
                kernel-side constant. The threshold is a structural
                field of the wrap op (not a free-floating context
                attribute) per discipline D1 (Op subclasses are pure
                data).
  '''

  inner: InsertInto
  threshold: float


DIALECT = Dialect(
  name='relation.jaccard',
  ops=[JaccardIndex],
)


__all__ = ['DIALECT', 'JaccardIndex']
