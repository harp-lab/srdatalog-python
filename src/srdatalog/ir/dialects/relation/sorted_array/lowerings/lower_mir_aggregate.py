'''Lowering: `mir.Aggregate` -> iir.cf — Wave 2A per-op migration (order 8).

Per `docs/phase_b_lowering_dispatcher.md` §4 (per-MIR-op work-unit
table, row B-Aggregate, difficulty `hard`) and §4.1 (per-PR
template): each MIR op type gets one `@lowering(target=iir.cf,
source=mir.X)` rule in its own file under
`dialects/relation/sorted_array/lowerings/`. The registry
registration lives alongside the dialect's other lowerings in
`__init__.py:_register_passes`.

Byte-equivalence contract (§4.2): the migrated op produces the same
IIR tree as the legacy `if isinstance(head, mir.Aggregate):` branch
inside `_lower_inner_chain` on every fixture that exercises it.

Aggregate-specific note (status as of Wave 2A, mirrors the long-
standing Nim-also-broken state documented in `tests/test_aggregate.py`):

  The Nim HIR pipeline parses `AggClause` into HIR JSON
  (`kind="aggregation"`) but NEVER constructs a `moAggregate` MIR
  node from it in its lowering pipeline. Python mirrors that exactly
  — DSL `agg(...)` / `count(...)` round-trips through HIR but the
  `Agg` clause disappears during MIR lowering, so `mir.Aggregate`
  is never produced by `compile_to_mir`. The `mir.Aggregate`
  dataclass exists for parity with the runtime C++
  `mir::Aggregate<...>` template (see
  `runtime/generalized_datalog/mir_def.h`) and for the structural
  helpers (`_var_used_in_op`, `view_slots.py`, the codegen
  `Scan|Negation|Aggregate` triples) that treat Aggregate as a
  spec-list source.

  Consequently the legacy `_lower_inner_chain` has NO
  `if isinstance(head, mir.Aggregate):` branch — Aggregate falls
  through to the terminal `raise ValueError(f'unsupported inner
  op: {type(head).__name__}')`. Likewise `_supported_pipeline`
  REJECTS any pipeline containing a `mir.Aggregate`, so an
  Aggregate can only reach `_lower_inner_chain` via a direct
  call (tests / future paths) — never via `compile_pipeline`.

  This migration therefore registers `mir.Aggregate` for dialect
  ownership + completeness (so the per-op `@lowering` table covers
  every MIR op family per the Phase B sign-off bullet "every concrete
  MIR op has an `@lowering`") and pins the chain-aware variant to
  the same "unsupported" raise. Byte-equivalence holds vacuously
  on real pipelines (Aggregate never appears) and structurally on
  direct calls (both legacy and new paths raise an identical
  `ValueError` for the unsupported op).

  When a future PR teaches the dialect to lower `mir.Aggregate` for
  real (analogous to `_lower_negation` for `mir.Negation`), it will
  replace the `_unsupported` body here with the real emission —
  the registration scaffold + dispatch wiring will already be in
  place.

Chain-aware split (mirrors B-Filter / B-ConstantBind / B-InsertInto):
two entry points whose names match the established Wave 2A
convention so the table-of-contents in
`lowerings/__init__.py:_register_passes` stays uniform:

  - `lower_mir_aggregate_in_chain(op, tail, ctx)`: the real entry
    called from `_lower_inner_chain` (legacy) when
    `type(head) in USE_DECLARATIVE`. Today this raises a
    `ValueError` whose text matches the legacy fall-through
    exception ("unsupported inner op: Aggregate") so the byte-
    equivalent test can pin both paths to the same error string.
  - `lower_mir_aggregate(op, ctx)`: the `@lowering`-registered
    stub. Asserts on direct invocation — the framework dispatch
    path is reserved for a future MIR-IIR walker that no longer
    routes through `_lower_inner_chain` and can plumb `tail`
    through.

The split lets the registry pin dialect ownership (`Aggregate`
belongs to `relation.sorted_array`) without forcing the chain
dispatcher through the registry today.
'''

from __future__ import annotations

from typing import Any

import srdatalog.ir.mir.types as mir
from srdatalog.ir.core import Op


def lower_mir_aggregate_in_chain(
  op: mir.Aggregate,
  tail: list[Any],
  ctx: Any,
) -> Op:
  '''Emit the IIR for a `mir.Aggregate` chain head with trailing
  `tail`.

  Mirrors the (vacuous) legacy fall-through in `_lower_inner_chain`
  byte-for-byte: there is no `if isinstance(head, mir.Aggregate):`
  branch upstream, so the legacy path raises `ValueError` with the
  text `'unsupported inner op: Aggregate'` when an Aggregate ever
  reaches the chain dispatcher. We raise the same `ValueError`
  here so the dispatch under `USE_DECLARATIVE` matches the legacy
  fall-through exactly — the byte-equivalent test asserts both
  paths raise the same error.

  See module docstring for why Aggregate is "vacuous today"
  (Nim-also-broken: AggClause -> HIR JSON only, never -> MIR
  Aggregate) and the upgrade path when a real lowering lands.

  The `tail` and `ctx` parameters are accepted to match the
  Wave 2A `_in_chain` signature; both are ignored until a real
  lowering replaces this body.
  '''
  # Match the legacy `raise ValueError(f'unsupported inner op:
  # {type(head).__name__}')` text exactly so the byte-equivalence
  # test can pin both paths to the same error string. Tail and ctx
  # are unused (no real emission yet) — del to silence linters and
  # signal intent to future readers.
  del tail, ctx
  raise ValueError(f'unsupported inner op: {type(op).__name__}')


def lower_mir_aggregate(op: mir.Aggregate, ctx: Any) -> Op:
  '''Framework-registry stub for `@lowering(target=iir.cf,
  source=mir.Aggregate)`. The actual dispatch lives in
  `lowerings._lower_inner_chain` (via the `USE_DECLARATIVE`
  ratchet), which calls `lower_mir_aggregate_in_chain` with the
  trailing chain in scope.

  This stub exists so the dialect's `lowerings` list pins the
  (consumes, produces) contract for `mir.Aggregate` — the
  discipline test consults the dialect to verify ownership, and
  the Phase B sign-off bullet "every concrete MIR op has an
  `@lowering`" requires this registration regardless of whether
  the chain dispatcher reaches a real Aggregate today.

  Calling this stub directly raises a structural assertion — the
  framework path is reserved for future readers who can plumb in
  the missing `tail` (e.g. a refactored MIR-IIR walker that no
  longer routes through `_lower_inner_chain`).
  '''
  raise AssertionError(
    'lower_mir_aggregate: dispatch goes through '
    '`lowerings._lower_inner_chain` -> `lower_mir_aggregate_in_chain` '
    'so the trailing chain is in scope. Direct invocation '
    'indicates a refactor that bypassed the chain dispatch — '
    'plumb the `tail` through and call '
    '`lower_mir_aggregate_in_chain` instead.'
  )


__all__ = [
  'lower_mir_aggregate',
  'lower_mir_aggregate_in_chain',
]
