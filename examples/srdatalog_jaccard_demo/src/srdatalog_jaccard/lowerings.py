'''Lowering: `JaccardIndex` -> IIR.

Registers an `@lowering(target=DIALECT, source=JaccardIndex)` rule
that lowers our wrap op to an IIR `Block` containing:

  1. A `// jaccard threshold=<t>` comment marker so the threshold
     surfaces in the rendered C++ (useful as a discriminator in
     golden snapshots).
  2. The same IIR shape that the sorted_array dialect's
     `_lower_insert_into` produces for the dedup-hash branch — we
     reuse that helper verbatim so the demo's emission is a
     well-tested existing shape, not novel codegen we'd have to
     port. The PURPOSE of this demo is the REGISTRATION pathway,
     not new lowering semantics; piggybacking on an existing IIR
     shape lets the test verify "the plugin's lowering rule fires
     and produces well-formed IIR" without us inventing new ops.

The registration runs as a module-import side effect — the parent
`srdatalog_jaccard` package imports this module exactly for that
side effect.

Importantly, this file's `from srdatalog.ir.dialects.relation.
sorted_array.lowerings import _lower_insert_into` import is the
ONLY cross-package dependency on srdatalog's lowering internals.
A more polished plugin would re-implement the IIR shape itself; we
delegate to demonstrate that cross-dialect reuse is supported.
'''

from __future__ import annotations

from typing import Any

from srdatalog.ir.core import Op
from srdatalog.ir.core.passes import lowering
from srdatalog.ir.dialects.iir.cf import Block, Comment
from srdatalog_jaccard.dialect import DIALECT, JaccardIndex


@lowering(
  DIALECT,
  JaccardIndex,
  consumes=('mir', 'relation.jaccard'),
  produces=('iir.cf', 'relation.sorted_array'),
)
def lower_jaccard_index(op: JaccardIndex, ctx: Any) -> Op:
  '''Emit the IIR for `JaccardIndex(inner=InsertInto, threshold=t)`.

  Returns a `Block` containing:

    1. `Comment(text="jaccard threshold=<t>")` — a marker for golden-
       snapshot tests; survives all the way to the rendered C++.
    2. The IIR statements produced by the sorted_array dialect's
       `_lower_insert_into` under `ctx.dedup_hash=True`. The dedup-
       hash branch produces a `dedup_table.try_insert(...) + if (_p)
       {...}` gate — semantically a close-enough analogue for a
       Jaccard threshold check (both decide whether to materialize
       a tuple based on a per-emission predicate). Byte-equivalence
       between dedup_hash and Jaccard is NOT a goal; reusing the
       branch is a demo-simplification, not a contract.

  The save/restore around `ctx.dedup_hash` is defensive: if a
  future caller invokes this lowering from a partial-ctx scope
  where `dedup_hash` is False, the flag still flips on for the
  duration of the gate (the lowering helper is the byte-equivalence
  anchor for the dedup_hash branch; we reuse it as-is).

  Args:
    op  — the `JaccardIndex` wrap op to lower. `op.inner` is the
          `mir.InsertInto` being gated.
    ctx — the lowering context (`LoweringCtx` from sorted_array's
          lowerings module). Non-frozen dataclass; we mutate
          `dedup_hash` directly per discipline D10's parallel rule
          for `LowerCtx`.

  Returns:
    A `Block` IIR op the runner's emit pipeline renders to a
    single `{ ... }` C++ block.
  '''
  # Deferred import: the sorted_array monolith module imports
  # nothing from this package. Importing at function-call time
  # keeps the dialect import graph linear AND ensures sorted_array
  # is loaded only when we actually need to lower (i.e., when the
  # user has used the Jaccard pragma at least once).
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _lower_insert_into,
  )

  prev = getattr(ctx, 'dedup_hash', False)
  try:
    ctx.dedup_hash = True
    stmts = list(_lower_insert_into(op.inner, ctx))
  finally:
    ctx.dedup_hash = prev

  marker = Comment(text=f'jaccard threshold={op.threshold}')
  return Block(stmts=(marker, *stmts))


__all__ = ['lower_jaccard_index']
