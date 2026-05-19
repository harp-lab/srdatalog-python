'''Byte-equivalence test for Wave 2A B-Aggregate migration.

Per `docs/phase_b_lowering_dispatcher.md` §4.2 (per-PR acceptance
gate): each per-MIR-op migration ships a
`test_lower_mir_<op>_byte_equivalent` test that runs the migrated
path on every relevant fixture and asserts byte equality with the
legacy `if isinstance(head, mir.X):` branch.

Aggregate-specific notes (mirrors the long-standing Nim-also-broken
state documented in `tests/test_aggregate.py` and in
`lowerings/lower_mir_aggregate.py`'s module docstring):

  The Nim HIR pipeline parses `AggClause` into HIR JSON
  (`kind="aggregation"`) but NEVER constructs a `moAggregate`
  MIR node from it during HIR -> MIR lowering. Python mirrors
  that exactly — DSL `agg(...)` / `count(...)` round-trips
  through HIR but the `Agg` clause disappears, so `mir.Aggregate`
  is never produced by `compile_to_mir` today. The
  `mir.Aggregate` dataclass exists for parity with the runtime
  C++ `mir::Aggregate<...>` template + structural helpers
  (`_var_used_in_op`, `view_slots.py`, the codegen
  `Scan|Negation|Aggregate` triples).

  Consequently the legacy `_lower_inner_chain` has NO
  `if isinstance(head, mir.Aggregate):` branch — Aggregate falls
  through to the terminal `raise ValueError(f'unsupported inner
  op: {type(head).__name__}')`. The B-Aggregate migration
  preserves this contract byte-for-byte (both paths raise the
  same `ValueError` text) so the registration scaffold can land
  ahead of a future real lowering replacing the raise.

  Likewise `_supported_pipeline` REJECTS any pipeline containing
  a `mir.Aggregate`, so `mir.Aggregate` can only reach the
  chain dispatcher via direct `_lower_inner_chain([agg, ins], ctx)`
  calls — never via `compile_pipeline`. This file therefore
  exercises the dispatch surface directly; there is no full-
  `compile_pipeline` smoke test (it would be DOA at
  `_supported_pipeline`).

The test compares two compilation paths:

  - LEGACY: `USE_DECLARATIVE` patched to NOT contain
    `mir.Aggregate`, so `_lower_inner_chain` falls into the
    terminal `raise ValueError(...)` for unsupported inner ops.
  - NEW: `USE_DECLARATIVE` left alone (Aggregate IS in the set),
    so `_lower_inner_chain` routes through
    `lower_mir_aggregate_in_chain` (which raises the same
    `ValueError`).

Byte-equivalence is asserted on the exception text (the only
observable output today). When a future PR replaces the raise
with a real lowering, this file gains rendered-IIR assertions
mirroring the other Wave 2A tests.
'''

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

import srdatalog.ir.mir.types as mir
from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
  LoweringCtx,
  _lower_inner_chain,
)
from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_aggregate import (
  lower_mir_aggregate,
  lower_mir_aggregate_in_chain,
)
from srdatalog.ir.hir.types import Version

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


@contextmanager
def _force_legacy_branch():
  '''Temporarily strip `mir.Aggregate` from `USE_DECLARATIVE` so
  `_lower_inner_chain` falls into the legacy fall-through (which
  raises `ValueError('unsupported inner op: Aggregate')`).

  Save / restore as a context manager — discipline test
  `test_use_declarative_is_monotonic` (when it lands) ratchets the
  set at module import time, but this test mutates the dialect's
  re-bound name for the duration of one call only.
  '''
  import srdatalog.ir.dialects.relation.sorted_array as sa_dialect

  saved = sa_dialect.USE_DECLARATIVE
  sa_dialect.USE_DECLARATIVE = frozenset(saved - {mir.Aggregate})
  try:
    yield
  finally:
    sa_dialect.USE_DECLARATIVE = saved


def _insert_into(arity: int = 2) -> mir.InsertInto:
  vars_ = [f'v{i}' for i in range(arity)]
  return mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=vars_,
    index=list(range(arity)),
  )


def _agg_count(
  result_var: str = 'cnt',
  rel_name: str = 'R',
  prefix_vars: list[str] | None = None,
  index: list[int] | None = None,
  handle_start: int = 1,
  version: Version = Version.FULL,
) -> mir.Aggregate:
  '''Build a `mir.Aggregate` with sensible defaults. Mirrors the
  shape `count(cnt, R(x, y))` produces if it ever reached MIR.'''
  return mir.Aggregate(
    result_var=result_var,
    agg_func='AggCount',
    rel_name=rel_name,
    version=version,
    index=index if index is not None else [0, 1],
    prefix_vars=prefix_vars if prefix_vars is not None else [],
    handle_start=handle_start,
  )


def _new_ctx(**kwargs: Any) -> LoweringCtx:
  '''Fresh LoweringCtx — counters reset so both paths bump identically.'''
  return LoweringCtx(output_var='ctx0', **kwargs)


# -----------------------------------------------------------------------------
# 1. Default-shape Aggregate (AggCount, no prefix_vars)
# -----------------------------------------------------------------------------


def test_aggregate_default_count_byte_equivalent_raise():
  '''Default `count(cnt, R(x, y))`-shaped Aggregate. Both paths
  raise the same `ValueError` with text
  `'unsupported inner op: Aggregate'`.'''
  agg = _agg_count()
  ins = _insert_into(1)

  with _force_legacy_branch(), pytest.raises(ValueError) as legacy_exc:
    _lower_inner_chain([agg, ins], _new_ctx())

  with pytest.raises(ValueError) as new_exc:
    _lower_inner_chain([agg, ins], _new_ctx())

  assert str(legacy_exc.value) == str(new_exc.value)
  assert str(legacy_exc.value) == 'unsupported inner op: Aggregate'


# -----------------------------------------------------------------------------
# 2. Aggregate with prefix_vars (join-prefixed shape)
# -----------------------------------------------------------------------------


def test_aggregate_with_prefix_vars_byte_equivalent_raise():
  '''Aggregate carrying join-prefix vars (e.g. `count(cnt, R(x, y))`
  inside a body where `x` is bound). Both paths raise the same
  error — prefix_vars are inspected by `_var_used_in_op` only
  (line 2458), not by the chain dispatcher.'''
  agg = _agg_count(prefix_vars=['x', 'y'])
  ins = _insert_into(1)

  with _force_legacy_branch(), pytest.raises(ValueError) as legacy_exc:
    _lower_inner_chain([agg, ins], _new_ctx())

  with pytest.raises(ValueError) as new_exc:
    _lower_inner_chain([agg, ins], _new_ctx())

  assert str(legacy_exc.value) == str(new_exc.value)


# -----------------------------------------------------------------------------
# 3. Aggregate with a non-default agg_func (custom aggregator)
# -----------------------------------------------------------------------------


def test_aggregate_custom_agg_func_byte_equivalent_raise():
  '''Aggregate with a non-AggCount aggregator (`AggSum`, `MyAgg<int>`,
  etc.). The agg_func is not consulted by either path today.'''
  agg = _agg_count()
  agg = mir.Aggregate(
    result_var=agg.result_var,
    agg_func='AggSum',
    rel_name=agg.rel_name,
    version=agg.version,
    index=agg.index,
    prefix_vars=agg.prefix_vars,
    handle_start=agg.handle_start,
  )
  ins = _insert_into(1)

  with _force_legacy_branch(), pytest.raises(ValueError) as legacy_exc:
    _lower_inner_chain([agg, ins], _new_ctx())

  with pytest.raises(ValueError) as new_exc:
    _lower_inner_chain([agg, ins], _new_ctx())

  assert str(legacy_exc.value) == str(new_exc.value)


# -----------------------------------------------------------------------------
# 4. Aggregate with a non-default version (NEW / DELTA)
# -----------------------------------------------------------------------------


def test_aggregate_version_new_byte_equivalent_raise():
  '''Aggregate over a NEW version of the relation. Version is not
  consulted by either path today.'''
  agg = _agg_count(version=Version.NEW)
  ins = _insert_into(1)

  with _force_legacy_branch(), pytest.raises(ValueError) as legacy_exc:
    _lower_inner_chain([agg, ins], _new_ctx())

  with pytest.raises(ValueError) as new_exc:
    _lower_inner_chain([agg, ins], _new_ctx())

  assert str(legacy_exc.value) == str(new_exc.value)


# -----------------------------------------------------------------------------
# 5. Aggregate as the SOLE chain head (no trailing tail)
# -----------------------------------------------------------------------------


def test_aggregate_no_tail_byte_equivalent_raise():
  '''Aggregate followed by a single InsertInto (minimum legal
  chain — `_lower_inner_chain` requires non-empty `rest`). Both
  paths raise on Aggregate before tail processing.'''
  agg = _agg_count()
  ins = _insert_into(2)

  with _force_legacy_branch(), pytest.raises(ValueError) as legacy_exc:
    _lower_inner_chain([agg, ins], _new_ctx())

  with pytest.raises(ValueError) as new_exc:
    _lower_inner_chain([agg, ins], _new_ctx())

  assert str(legacy_exc.value) == str(new_exc.value)
  assert 'Aggregate' in str(legacy_exc.value)


# -----------------------------------------------------------------------------
# 6. Aggregate followed by Filter (would be a real downstream chain
#    once a real lowering lands)
# -----------------------------------------------------------------------------


def test_aggregate_with_filter_tail_byte_equivalent_raise():
  '''Aggregate followed by a Filter then InsertInto. Both paths
  still raise on the Aggregate head before the tail is considered.
  Future real lowering will keep this dispatch-on-head invariant.'''
  agg = _agg_count()
  filt = mir.Filter(vars=['cnt'], code='return cnt > 0;')
  ins = _insert_into(1)

  with _force_legacy_branch(), pytest.raises(ValueError) as legacy_exc:
    _lower_inner_chain([agg, filt, ins], _new_ctx())

  with pytest.raises(ValueError) as new_exc:
    _lower_inner_chain([agg, filt, ins], _new_ctx())

  assert str(legacy_exc.value) == str(new_exc.value)


# -----------------------------------------------------------------------------
# 7. Direct call to `lower_mir_aggregate_in_chain` matches dispatch
# -----------------------------------------------------------------------------


def test_lower_mir_aggregate_in_chain_raises_unsupported():
  '''Pin the chain-aware entry's exception text directly (independent
  of dispatch routing). The legacy fall-through is
  `f'unsupported inner op: {type(head).__name__}'`.'''
  agg = _agg_count()
  ins = _insert_into(1)

  with pytest.raises(ValueError) as exc:
    lower_mir_aggregate_in_chain(agg, [ins], _new_ctx())
  assert str(exc.value) == 'unsupported inner op: Aggregate'


# -----------------------------------------------------------------------------
# 8. Registry contract — stub asserts on direct call
# -----------------------------------------------------------------------------


def test_lower_mir_aggregate_registry_stub_asserts():
  '''The `@lowering(target=iir.cf, source=mir.Aggregate)` registry
  entry is a stub that asserts on direct invocation — dispatch is
  expected to flow through `_lower_inner_chain` -> the chain-aware
  variant. Mirrors the C5 `lower_tiled_cartesian` split + the other
  Wave 2A migrations.
  '''
  agg = _agg_count()
  ctx = _new_ctx()
  with pytest.raises(AssertionError, match=r'lower_mir_aggregate_in_chain'):
    lower_mir_aggregate(agg, ctx)


def test_lower_mir_aggregate_is_registered_on_sorted_array_dialect():
  '''The `Aggregate` `@lowering` is registered on the
  `relation.sorted_array` dialect. Pins dialect ownership per
  `docs/phase_b_lowering_dispatcher.md` §4 (one `@lowering` per MIR
  op, on the dialect that lowers it).
  '''
  from srdatalog.ir.dialects.relation.sorted_array import DIALECT as SA_DIALECT

  matched = [low for low in SA_DIALECT.lowerings if low.matches is mir.Aggregate]
  assert len(matched) == 1
  assert matched[0].consumes == ('mir',)
  assert 'iir.cf' in matched[0].produces


# -----------------------------------------------------------------------------
# 9. Aggregate is in USE_DECLARATIVE (ratchet contract)
# -----------------------------------------------------------------------------


def test_aggregate_in_use_declarative_ratchet():
  '''`mir.Aggregate` must be in the `USE_DECLARATIVE` set so the
  chain dispatcher routes through the per-op file. The set is
  monotonically growing during Phase B — removing this entry
  requires owner sign-off.'''
  from srdatalog.ir.dialects.relation.sorted_array import USE_DECLARATIVE

  assert mir.Aggregate in USE_DECLARATIVE


# -----------------------------------------------------------------------------
# 10. Full-pipeline shape: _supported_pipeline rejects Aggregate
# -----------------------------------------------------------------------------


def test_aggregate_in_pipeline_rejected_by_supported_predicate():
  '''Sanity: `_supported_pipeline` rejects any pipeline that
  contains an Aggregate, so the chain dispatcher is never reached
  via `compile_pipeline` today. Both legacy and new paths share
  this gate.'''
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _supported_pipeline,
  )

  scan = mir.Scan(
    vars=['x', 'y'],
    rel_name='Src',
    version=Version.FULL,
    index=[0, 1],
    handle_start=0,
  )
  agg = _agg_count()
  ins = mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['cnt'],
    index=[0],
  )

  # Scan-rooted pipeline with Aggregate in the middle: not supported.
  assert not _supported_pipeline([scan, agg, ins])
  # Aggregate-rooted: also not supported.
  assert not _supported_pipeline([agg, ins])


# -----------------------------------------------------------------------------
# 11. Dispatch routing: with USE_DECLARATIVE present the path is
#     the per-op file, not the legacy fall-through
# -----------------------------------------------------------------------------


def test_aggregate_dispatch_routes_through_per_op_file_when_in_use_declarative():
  '''When `mir.Aggregate` is in `USE_DECLARATIVE`, the dispatcher
  routes through `lower_mir_aggregate_in_chain` rather than the
  legacy fall-through. We verify this by monkey-patching the per-op
  file's entry and confirming it is the one that runs.'''
  import srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_aggregate as agg_mod

  marker: list[int] = []
  original = agg_mod.lower_mir_aggregate_in_chain

  def _spy(op: Any, tail: list[Any], ctx: Any) -> Any:
    marker.append(1)
    return original(op, tail, ctx)

  agg_mod.lower_mir_aggregate_in_chain = _spy
  try:
    agg = _agg_count()
    ins = _insert_into(1)
    with pytest.raises(ValueError):
      _lower_inner_chain([agg, ins], _new_ctx())
    assert marker == [1]
  finally:
    agg_mod.lower_mir_aggregate_in_chain = original


def test_aggregate_dispatch_skips_per_op_file_in_legacy_mode():
  '''Sanity inverse: under `_force_legacy_branch`, the per-op file's
  entry is NOT called — the legacy fall-through raises directly.'''
  import srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_aggregate as agg_mod

  marker: list[int] = []
  original = agg_mod.lower_mir_aggregate_in_chain

  def _spy(op: Any, tail: list[Any], ctx: Any) -> Any:
    marker.append(1)
    return original(op, tail, ctx)

  agg_mod.lower_mir_aggregate_in_chain = _spy
  try:
    agg = _agg_count()
    ins = _insert_into(1)
    with _force_legacy_branch(), pytest.raises(ValueError):
      _lower_inner_chain([agg, ins], _new_ctx())
    assert marker == []  # Per-op entry was bypassed.
  finally:
    agg_mod.lower_mir_aggregate_in_chain = original
