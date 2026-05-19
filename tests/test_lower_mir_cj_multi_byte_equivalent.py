'''Byte-equivalence test for Wave 2A B-CJ-multi migration.

Per `docs/phase_b_lowering_dispatcher.md` §4.2 (per-PR acceptance
gate): each per-MIR-op migration ships a
`test_lower_mir_<op>_byte_equivalent` test that runs the migrated
path on every relevant fixture and asserts byte equality with the
legacy `if isinstance(head, mir.X):` branch.

For multi-source ColumnJoin (`len(sources) >= 2`) the relevant
fixtures are:

  - root-position multi-source CJ (the production-load-bearing
    case — every TC-like fixture uses this shape),
  - mid-chain (nested) multi-source CJ — CJ under CJ,
  - 2-source and 3-source variants,
  - prefix-narrowed sources (parent handle aliased from the
    enclosing CJ scope) versus fresh sources,
  - BG-eligible shapes: bare multi-source root CJ with
    `ctx.bg_enabled=True` (the legacy `block_group: bool` dual-
    write path) AND `BlockGroupRoot(inner=multi-source CJ)`
    (the typed-pragma path post C3 materialization),
  - full-pipeline shapes via `compile_pipeline`,
  - Filter / ConstantBind interleaved before / after the CJ.

The test compares two compilation paths:

  - LEGACY: `USE_DECLARATIVE` patched to NOT contain
    `mir.ColumnJoin`, so `_lower_inner_chain` and
    `lower_scan_pipeline` fall into the legacy
    `_lower_nested_cj_multi` / `_lower_root_cj_multi` branches
    directly.

  - NEW: `USE_DECLARATIVE` left alone (`ColumnJoin` IS in the set;
    `_should_use_declarative` no longer gates by source count
    post B-CJ-multi), so the dispatchers route through
    `lower_mir_cj_multi_in_chain` / `lower_mir_cj_multi_root`.

Byte-equivalence is asserted on the rendered IIR text. The chain-
aware variants delegate to the same legacy helpers the legacy
branches called, so equivalence holds by construction; the test
pins it explicitly as the per-PR acceptance gate.

CRITICAL — BlockGroup coexistence: when `ctx.bg_enabled` is True
or `pipeline[0]` is a `BlockGroupRoot` wrap op, the BG variant
(`_lower_root_cj_bg`) is the correct emission target — NOT the
standard `_lower_root_cj_multi`. The dispatch ordering in
`lower_scan_pipeline` keeps both BG paths intact (the
`BlockGroupRoot` unwrap sits BEFORE `_should_use_declarative`;
the bare multi-source CJ path with `ctx.bg_enabled=True` delegates
into `_lower_root_cj_multi`, whose first statement re-dispatches
to `_lower_root_cj_bg`). The BG fixtures here pin both routes.
'''

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

import srdatalog.ir.mir.types as mir
from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit
from srdatalog.ir.dialects.iir.cf import Block as IirBlock
from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
  LoweringCtx,
  _lower_inner_chain,
  _lower_nested_cj_multi,
  _lower_root_cj_bg,
  _lower_root_cj_multi,
  _should_use_declarative,
  _state_key,
  lower_scan_pipeline,
)
from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_cj_multi import (
  lower_mir_cj_multi,
  lower_mir_cj_multi_in_chain,
  lower_mir_cj_multi_root,
)
from srdatalog.ir.hir.types import Version

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _render(op: Any) -> str:
  '''Render an IIR op tree through the CUDA emitter — the same
  surface the byte-equivalence harnesses exercise.'''
  return emit(op, EmitCtx(indent_level=2))


@contextmanager
def _force_legacy_branch():
  '''Temporarily strip `mir.ColumnJoin` from `USE_DECLARATIVE` so
  the dispatchers fall into the legacy multi-source CJ branches
  (`_lower_nested_cj_multi` / `_lower_root_cj_multi`) directly.

  Save / restore as a context manager — `_should_use_declarative`
  re-reads the (mutable) dialect-module attribute on each call, so
  swapping the frozenset for the duration of one block is enough
  to force the legacy path.

  Mirrors the same fixture in `test_lower_mir_cj_single_byte_equivalent.py`.
  '''
  import srdatalog.ir.dialects.relation.sorted_array as sa_dialect

  saved = sa_dialect.USE_DECLARATIVE
  sa_dialect.USE_DECLARATIVE = frozenset(saved - {mir.ColumnJoin})
  try:
    yield
  finally:
    sa_dialect.USE_DECLARATIVE = saved


def _new_ctx(**kwargs: Any) -> LoweringCtx:
  '''Fresh LoweringCtx — counters reset so both paths bump identically.'''
  view_var_names = kwargs.pop('view_var_names', None) or {
    '0': 'v_A_full',
    '1': 'v_B_full',
    '2': 'v_C_full',
    '3': 'v_D_full',
  }
  return LoweringCtx(output_var='ctx0', view_var_names=view_var_names, **kwargs)


def _src(
  rel: str,
  handle_start: int,
  *,
  prefix_vars: tuple[str, ...] = (),
  index: tuple[int, ...] = (0, 1),
  version: Version = Version.FULL,
) -> mir.ColumnSource:
  return mir.ColumnSource(
    rel_name=rel,
    version=version,
    index=list(index),
    prefix_vars=list(prefix_vars),
    handle_start=handle_start,
  )


def _multi_src_cj(
  *sources: mir.ColumnSource,
  var_name: str = 'y',
  handle_start: int = -1,
) -> mir.ColumnJoin:
  return mir.ColumnJoin(
    var_name=var_name,
    sources=list(sources),
    handle_start=handle_start,
  )


def _insert_into(vars_: tuple[str, ...] = ('y',)) -> mir.InsertInto:
  return mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=list(vars_),
    index=list(range(len(vars_))),
  )


# -----------------------------------------------------------------------------
# 1. Mid-chain (nested) multi-source CJ — 2 sources, fresh
# -----------------------------------------------------------------------------


def test_nested_multi_cj_2_sources_fresh_byte_equivalent():
  '''Nested multi-source CJ with two fresh-root sources (no prefix
  narrowing). Both paths emit identical intersect_handles +
  IntersectIter scaffold.
  '''
  cj = _multi_src_cj(_src('A', 0), _src('B', 1), var_name='y')
  ins = _insert_into(('y',))

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([cj, ins], _new_ctx()))
  new_text = _render(_lower_inner_chain([cj, ins], _new_ctx()))

  assert legacy_text == new_text
  assert 'intersect_handles' in new_text


def test_nested_multi_cj_3_sources_byte_equivalent():
  '''Nested multi-source CJ with three sources stress-tests the
  N-way intersect ordering. Both paths must match.
  '''
  cj = _multi_src_cj(_src('A', 0), _src('B', 1), _src('C', 2), var_name='y')
  ins = _insert_into(('y',))

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([cj, ins], _new_ctx()))
  new_text = _render(_lower_inner_chain([cj, ins], _new_ctx()))

  assert legacy_text == new_text


# -----------------------------------------------------------------------------
# 2. Mid-chain multi-source CJ with prefix-narrowed sources
# -----------------------------------------------------------------------------


def test_nested_multi_cj_prefix_narrowed_byte_equivalent():
  '''Nested multi-source CJ where the sources carry `prefix_vars`
  — each one aliases a parent handle from the enclosing scope.
  Both paths emit the same `auto h = parent;` bindings.
  '''
  cj = _multi_src_cj(
    _src('A', 0, prefix_vars=('x',)),
    _src('B', 1, prefix_vars=('x',)),
    var_name='y',
  )
  ins = _insert_into(('x', 'y'))

  parent_key_a = _state_key('A', [0, 1], ['x'], Version.FULL)
  parent_key_b = _state_key('B', [0, 1], ['x'], Version.FULL)

  with _force_legacy_branch():
    ctx = _new_ctx()
    ctx.handle_vars[parent_key_a] = 'parent_h_A'
    ctx.handle_vars[parent_key_b] = 'parent_h_B'
    ctx.bound_vars.append('x')
    legacy_text = _render(_lower_inner_chain([cj, ins], ctx))

  ctx = _new_ctx()
  ctx.handle_vars[parent_key_a] = 'parent_h_A'
  ctx.handle_vars[parent_key_b] = 'parent_h_B'
  ctx.bound_vars.append('x')
  new_text = _render(_lower_inner_chain([cj, ins], ctx))

  assert legacy_text == new_text
  assert 'parent_h_A' in new_text
  assert 'parent_h_B' in new_text


# -----------------------------------------------------------------------------
# 3. Interleaved Filter / ConstantBind around a nested multi-source CJ
# -----------------------------------------------------------------------------


def test_nested_multi_cj_with_trailing_filter_byte_equivalent():
  '''Chain: nested multi-source CJ + Filter + InsertInto. Filter
  body sits inside the CJ's iter loop.'''
  cj = _multi_src_cj(_src('A', 0), _src('B', 1), var_name='y')
  filt = mir.Filter(vars=['y'], code='return y > 0;')
  ins = _insert_into(('y',))

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([cj, filt, ins], _new_ctx()))
  new_text = _render(_lower_inner_chain([cj, filt, ins], _new_ctx()))

  assert legacy_text == new_text
  assert 'if (y > 0)' in new_text


def test_nested_multi_cj_with_trailing_constant_bind_byte_equivalent():
  '''Chain: nested multi-source CJ + ConstantBind + InsertInto.'''
  cj = _multi_src_cj(_src('A', 0), _src('B', 1), var_name='y')
  cbind = mir.ConstantBind(var_name='yy', code='y + 1', deps=['y'])
  ins = _insert_into(('y', 'yy'))

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([cj, cbind, ins], _new_ctx()))
  new_text = _render(_lower_inner_chain([cj, cbind, ins], _new_ctx()))

  assert legacy_text == new_text
  assert 'auto yy = y + 1;' in new_text


# -----------------------------------------------------------------------------
# 4. Root-position multi-source CJ (the production-load-bearing shape)
# -----------------------------------------------------------------------------


def test_root_multi_cj_2_sources_byte_equivalent():
  '''Root-position multi-source CJ — every TC-like fixture uses
  this shape. The new path routes via `lower_mir_cj_multi_root`,
  which delegates to `_lower_root_cj_multi`; the legacy path
  invokes `_lower_root_cj_multi` directly through
  `lower_scan_pipeline`'s isinstance-cascade.
  '''
  cj = _multi_src_cj(_src('A', 0), _src('B', 1), var_name='y')
  ins = _insert_into(('y',))

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([cj, ins], _new_ctx()))
  new_text = _render(lower_scan_pipeline([cj, ins], _new_ctx()))

  assert legacy_text == new_text
  assert 'root_unique_values' in new_text


def test_root_multi_cj_3_sources_byte_equivalent():
  '''Root-position three-source CJ.'''
  cj = _multi_src_cj(_src('A', 0), _src('B', 1), _src('C', 2), var_name='y')
  ins = _insert_into(('y',))

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([cj, ins], _new_ctx()))
  new_text = _render(lower_scan_pipeline([cj, ins], _new_ctx()))

  assert legacy_text == new_text


# -----------------------------------------------------------------------------
# 5. CJ under CJ: root-multi + nested-multi (deepest TC shape)
# -----------------------------------------------------------------------------


def test_root_multi_cj_with_nested_multi_cj_byte_equivalent():
  '''Root multi-source CJ binding `y`, then a nested multi-source
  CJ binding `z` with two fresh sources (no prefix narrowing). The
  deepest shape `_lower_root_cj_multi` -> `_lower_inner_chain` ->
  `_lower_nested_cj_multi` is exercised end-to-end.

  Fresh sources are used because root CJ only registers state keys
  for its own sources (A, B), not for the deeper fresh ones (C, D).
  Prefix-narrowed nested CJ would need parent handles registered by
  matching root sources — the dedicated mid-chain prefix-narrowed
  test exercises that path with pre-populated `ctx.handle_vars`.
  '''
  root_cj = _multi_src_cj(_src('A', 0), _src('B', 1), var_name='y')
  nested_cj = _multi_src_cj(
    _src('C', 2),
    _src('D', 3),
    var_name='z',
  )
  ins = _insert_into(('y', 'z'))

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([root_cj, nested_cj, ins], _new_ctx()))
  new_text = _render(lower_scan_pipeline([root_cj, nested_cj, ins], _new_ctx()))

  assert legacy_text == new_text
  assert 'root_unique_values' in new_text
  assert 'intersect_handles' in new_text


# -----------------------------------------------------------------------------
# 6. BlockGroup coexistence — bare multi CJ with ctx.bg_enabled
# -----------------------------------------------------------------------------


def test_root_multi_cj_with_bg_enabled_routes_to_bg_variant_byte_equivalent():
  '''C3 dual-write path: a bare multi-source `mir.ColumnJoin` with
  `ctx.bg_enabled=True` (the `ep.block_group=True` flag carried by
  `compile_kernel_body`). Both paths reach `_lower_root_cj_multi`
  whose first statement is `if ctx.bg_enabled: return _lower_root_cj_bg(...)`,
  so both emit the BG scaffold (BgRootCjMulti).

  This is the load-bearing dual-write contract — the BG variant
  fires before any non-BG scaffolding runs.
  '''
  cj = _multi_src_cj(_src('A', 0), _src('B', 1), var_name='y')
  ins = _insert_into(('y',))

  with _force_legacy_branch():
    ctx = _new_ctx()
    ctx.bg_enabled = True
    legacy_text = _render(lower_scan_pipeline([cj, ins], ctx))

  ctx = _new_ctx()
  ctx.bg_enabled = True
  new_text = _render(lower_scan_pipeline([cj, ins], ctx))

  assert legacy_text == new_text
  # BG path emits the `BgRootCjMulti` scaffold (binary search +
  # warp redistribution) — a marker that the non-BG `_lower_root_cj_multi`
  # would NOT produce.
  assert 'bg_key_idx' in new_text or 'block_group' in new_text.lower()


# -----------------------------------------------------------------------------
# 7. BlockGroup coexistence — BlockGroupRoot wrap op (typed-pragma path)
# -----------------------------------------------------------------------------


def test_block_group_root_wrap_routes_via_lower_scan_pipeline_byte_equivalent():
  '''C3 typed-pragma path: `pipeline[0]` is `BlockGroupRoot(inner=
  multi-source ColumnJoin)`. `lower_scan_pipeline` recognizes the
  wrap head BEFORE the `_should_use_declarative` check fires, slices
  off `rest`, and dispatches into `_lower_root_cj_bg` directly with
  `ctx.bg_enabled=True`.

  Both LEGACY and NEW must produce the same BG scaffold — the
  BlockGroupRoot unwrap path is invariant under B-CJ-multi (we did
  NOT touch the wrap dispatch in `lower_scan_pipeline`; it sits
  above the new `_should_use_declarative` branch).

  Pins the load-bearing constraint that B-CJ-multi did NOT perturb
  the typed-pragma BG dispatch.
  '''
  inner_cj = _multi_src_cj(_src('A', 0), _src('B', 1), var_name='y')
  wrap = mir.BlockGroupRoot(inner=inner_cj)
  ins = _insert_into(('y',))

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([wrap, ins], _new_ctx()))
  new_text = _render(lower_scan_pipeline([wrap, ins], _new_ctx()))

  assert legacy_text == new_text
  # BG scaffold marker (same as the dual-write test above).
  assert 'bg_key_idx' in new_text or 'block_group' in new_text.lower()


# -----------------------------------------------------------------------------
# 8. Full pipeline via compile_pipeline — root multi CJ + InsertInto
# -----------------------------------------------------------------------------


def test_compile_pipeline_root_multi_cj_byte_equivalent():
  '''End-to-end smoke test: a multi-source root CJ + InsertInto
  via `compile_pipeline`. Both paths produce identical CUDA text.
  '''
  from srdatalog.compile import compile_pipeline

  s1 = _src('A', 0)
  s2 = _src('B', 1)
  cj = _multi_src_cj(s1, s2, var_name='y')
  ins = mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['y'],
    index=[0],
  )
  ep = mir.ExecutePipeline(
    pipeline=[cj, ins],
    source_specs=[s1, s2],
    dest_specs=[ins],
    rule_name='CjMultiRoot',
  )

  with _force_legacy_branch():
    legacy_out = compile_pipeline(ep)
  new_out = compile_pipeline(ep)

  assert legacy_out == new_out
  assert new_out


def test_compile_pipeline_root_multi_cj_with_filter_byte_equivalent():
  '''End-to-end: root multi-source CJ with a trailing Filter before
  the InsertInto.'''
  from srdatalog.compile import compile_pipeline

  s1 = _src('A', 0)
  s2 = _src('B', 1)
  cj = _multi_src_cj(s1, s2, var_name='y')
  filt = mir.Filter(vars=['y'], code='return y > 0;')
  ins = mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['y'],
    index=[0],
  )
  ep = mir.ExecutePipeline(
    pipeline=[cj, filt, ins],
    source_specs=[s1, s2],
    dest_specs=[ins],
    rule_name='CjMultiRootFilter',
  )

  with _force_legacy_branch():
    legacy_out = compile_pipeline(ep)
  new_out = compile_pipeline(ep)

  assert legacy_out == new_out


# -----------------------------------------------------------------------------
# 9. Direct lowering call returns the expected IIR shape
# -----------------------------------------------------------------------------


def test_lower_mir_cj_multi_in_chain_emits_block():
  '''Pin the IIR tree shape: nested multi-source CJ -> a `Block`
  containing alias binds + IntersectIter (or a wrapping
  D2lSegmentLoop for D2L FULL_VER sources).
  '''
  cj = _multi_src_cj(_src('A', 0), _src('B', 1), var_name='y')
  ins = _insert_into(('y',))

  out = lower_mir_cj_multi_in_chain(cj, [ins], _new_ctx())
  assert isinstance(out, IirBlock)


def test_lower_mir_cj_multi_root_emits_block():
  '''Pin the IIR tree shape: root multi-source CJ -> a `Block`
  containing the ParallelFor + GridStrideLoop scaffold.'''
  cj = _multi_src_cj(_src('A', 0), _src('B', 1), var_name='y')
  ins = _insert_into(('y',))

  out = lower_mir_cj_multi_root(cj, [ins], _new_ctx())
  assert isinstance(out, IirBlock)


def test_lower_mir_cj_multi_in_chain_delegates_to_helper():
  '''The chain entry delegates to `_lower_nested_cj_multi` directly
  — the rendered text matches calling the helper.'''
  cj = _multi_src_cj(_src('A', 0), _src('B', 1), var_name='y')
  ins = _insert_into(('y',))

  via_entry = _render(lower_mir_cj_multi_in_chain(cj, [ins], _new_ctx()))
  via_helper = _render(_lower_nested_cj_multi(cj, [ins], _new_ctx()))
  assert via_entry == via_helper


def test_lower_mir_cj_multi_root_delegates_to_helper():
  '''The root entry delegates to `_lower_root_cj_multi` directly.'''
  cj = _multi_src_cj(_src('A', 0), _src('B', 1), var_name='y')
  ins = _insert_into(('y',))

  via_entry = _render(lower_mir_cj_multi_root(cj, [ins], _new_ctx()))
  via_helper = _render(_lower_root_cj_multi(cj, [ins], _new_ctx()))
  assert via_entry == via_helper


def test_lower_mir_cj_multi_root_delegates_to_bg_when_enabled():
  '''When `ctx.bg_enabled=True`, the root entry's delegation flows
  through `_lower_root_cj_multi`'s internal `if ctx.bg_enabled:`
  dispatch into `_lower_root_cj_bg`. Pins the dual-write contract
  from the entry surface.
  '''
  cj = _multi_src_cj(_src('A', 0), _src('B', 1), var_name='y')
  ins = _insert_into(('y',))

  ctx_a = _new_ctx()
  ctx_a.bg_enabled = True
  via_entry = _render(lower_mir_cj_multi_root(cj, [ins], ctx_a))

  ctx_b = _new_ctx()
  ctx_b.bg_enabled = True
  via_helper = _render(_lower_root_cj_bg(cj, [ins], ctx_b))

  assert via_entry == via_helper


def test_lower_mir_cj_multi_in_chain_rejects_single_source():
  '''The chain entry's structural guard fires when a single-source
  CJ slips through — pins that the source-count gate isn't only
  enforced by the caller. Single-source CJ is owned by B-CJ-single
  and must NOT route through this entry.
  '''
  single_cj = _multi_src_cj(_src('A', 0), var_name='y')
  ins = _insert_into(('y',))
  with pytest.raises(AssertionError, match=r'expected at least two ColumnSources'):
    lower_mir_cj_multi_in_chain(single_cj, [ins], _new_ctx())


def test_lower_mir_cj_multi_root_rejects_single_source():
  '''Mirror of `*_in_chain_rejects_single_source` for the root entry.'''
  single_cj = _multi_src_cj(_src('A', 0), var_name='y')
  ins = _insert_into(('y',))
  with pytest.raises(AssertionError, match=r'expected at least two ColumnSources'):
    lower_mir_cj_multi_root(single_cj, [ins], _new_ctx())


# -----------------------------------------------------------------------------
# 10. Registry contract — stub asserts on direct call
# -----------------------------------------------------------------------------


def test_lower_mir_cj_multi_registry_stub_asserts():
  '''The multi-source half of the `@lowering(target=iir.cf, source=
  mir.ColumnJoin)` registry entry is a stub that asserts on direct
  invocation — dispatch is expected to flow through `_lower_inner_chain`
  / `lower_scan_pipeline` -> the chain-aware variant.

  Mirrors the B-CJ-single stub assertion.
  '''
  cj = _multi_src_cj(_src('A', 0), _src('B', 1), var_name='y')
  ctx = _new_ctx()
  with pytest.raises(AssertionError, match=r'lower_mir_cj_multi_in_chain'):
    lower_mir_cj_multi(cj, ctx)


def test_lower_mir_column_join_remains_a_single_registration():
  '''Pin that B-CJ-multi did NOT add a SECOND dialect-level
  `@lowering(mir.ColumnJoin, ...)` registration. The single
  registration covers both single-source and multi-source shapes
  via a shape-aware dispatcher in `_register_passes`.
  '''
  from srdatalog.ir.dialects.relation.sorted_array import DIALECT as SA_DIALECT

  matched = [low for low in SA_DIALECT.lowerings if low.matches is mir.ColumnJoin]
  assert len(matched) == 1
  assert matched[0].consumes == ('mir',)
  assert 'iir.cf' in matched[0].produces


# -----------------------------------------------------------------------------
# 11. _should_use_declarative now accepts both shapes
# -----------------------------------------------------------------------------


def test_should_use_declarative_multi_source_cj_returns_true():
  '''Post-B-CJ-multi the helper no longer gates `mir.ColumnJoin`
  on source count — both shapes return True.'''
  multi_cj = _multi_src_cj(_src('A', 0), _src('B', 1), var_name='y')
  assert _should_use_declarative(multi_cj) is True


def test_should_use_declarative_single_source_cj_still_returns_true():
  '''Regression guard: single-source CJ stays in the new path.'''
  single_cj = _multi_src_cj(_src('A', 0), var_name='z')
  assert _should_use_declarative(single_cj) is True


# -----------------------------------------------------------------------------
# 12. BG end-to-end via compile_kernel_body — load-bearing dual-write
# -----------------------------------------------------------------------------


def test_compile_kernel_body_bg_enabled_byte_equivalent():
  '''The most load-bearing path for the C3 dual-write transition:
  `compile_kernel_body(ep, bg_enabled=True)` threads `bg_enabled`
  through to `LoweringCtx`, which the legacy `_lower_root_cj_multi`
  consults to re-dispatch into `_lower_root_cj_bg`. Pins that the
  new dispatch ordering preserves the legacy BG emission.
  '''
  from srdatalog.compile import compile_kernel_body

  s1 = _src('R', 0)
  s2 = _src('S', 1)
  cj = _multi_src_cj(s1, s2, var_name='x')
  ins = mir.InsertInto(
    rel_name='Out',
    version=Version.NEW,
    vars=['x'],
    index=[0],
  )
  ep = mir.ExecutePipeline(
    pipeline=[cj, ins],
    source_specs=[s1, s2],
    dest_specs=[ins],
    rule_name='BgDualWrite',
    block_group=True,
  )

  with _force_legacy_branch():
    legacy_text = compile_kernel_body(
      ep, is_counting=False, bg_enabled=True, output_var_name='output_ctx_0'
    )
  new_text = compile_kernel_body(
    ep, is_counting=False, bg_enabled=True, output_var_name='output_ctx_0'
  )

  assert legacy_text == new_text
  # BG-specific scaffolding marker — the BG path emits the
  # `bg_*` work-balanced scaffold via `BgRootCjMulti`.
  assert 'bg_key_idx' in new_text or 'cumulative_work' in new_text


# -----------------------------------------------------------------------------
# 13. USE_DECLARATIVE invariants
# -----------------------------------------------------------------------------


def test_use_declarative_contains_column_join_after_b_cj_multi():
  '''Pin the ratchet: `mir.ColumnJoin` remains in `USE_DECLARATIVE`
  after B-CJ-multi (the dispatch surface widened, the ratchet
  membership did not change). All prior Wave 2A migrations remain.
  '''
  from srdatalog.ir.dialects.relation.sorted_array import USE_DECLARATIVE

  assert mir.ColumnJoin in USE_DECLARATIVE
  assert mir.Filter in USE_DECLARATIVE
  assert mir.ConstantBind in USE_DECLARATIVE
  assert mir.InsertInto in USE_DECLARATIVE
  assert mir.Scan in USE_DECLARATIVE
  assert mir.Negation in USE_DECLARATIVE
  assert mir.Aggregate in USE_DECLARATIVE
