'''Byte-equivalence test for Wave 2A B-CJ-single migration.

Per `docs/phase_b_lowering_dispatcher.md` §4.2 (per-PR acceptance
gate): each per-MIR-op migration ships a
`test_lower_mir_<op>_byte_equivalent` test that runs the migrated
path on every relevant fixture and asserts byte equality with the
legacy `if isinstance(head, mir.X):` branch.

For single-source ColumnJoin the relevant fixtures are:

  - the plain nested single-source CJ over a fresh source
    (`prefix_vars=[]`),
  - the prefix-narrowed nested single-source CJ (parent handle
    aliased from the enclosing CJ scope),
  - nesting under both Scan-rooted and multi-source CJ-rooted
    pipelines,
  - chains with Filter / ConstantBind interleaved before / after
    the single-source CJ.

The test compares two compilation paths:

  - LEGACY: `USE_DECLARATIVE` patched to NOT contain
    `mir.ColumnJoin`, so `_lower_inner_chain` falls into the legacy
    `_lower_nested_cj_single` branch directly.

  - NEW: `USE_DECLARATIVE` left alone (`ColumnJoin` IS in the set,
    AND the source-count gate in `_should_use_declarative` matches
    single-source), so `_lower_inner_chain` routes through
    `lower_mir_cj_single_in_chain`.

Byte-equivalence is asserted on the rendered IIR text. Multi-source
ColumnJoin is NOT covered here (it stays on the legacy
`_lower_nested_cj_multi` branch); the next PR (B-CJ-multi) extends
the migration to multi-source and adds the corresponding fixture
coverage. The `test_multi_source_cj_still_uses_legacy_branch`
assertion below pins that B-CJ-single did not perturb multi-source
routing.
'''

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

import srdatalog.ir.mir.types as mir
from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit
from srdatalog.ir.dialects.iir.cf import Block as IirBlock
from srdatalog.ir.dialects.iir.cf import CartesianFlatLoop
from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
  LoweringCtx,
  _lower_inner_chain,
  _lower_nested_cj_single,
  _should_use_declarative,
  _state_key,
)
from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_cj_single import (
  lower_mir_cj_single,
  lower_mir_cj_single_in_chain,
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
  `_lower_inner_chain` falls into the legacy single-source CJ
  branch (`_lower_nested_cj_single` directly).

  Save / restore as a context manager — `_should_use_declarative`
  re-reads the (mutable) dialect-module attribute on each call, so
  swapping the frozenset for the duration of one block is enough
  to force the legacy path.
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
  }
  return LoweringCtx(output_var='ctx0', view_var_names=view_var_names, **kwargs)


def _single_src_cj(
  rel: str = 'C',
  handle_start: int = 2,
  prefix_vars: tuple[str, ...] = (),
  var_name: str = 'z',
  index: tuple[int, ...] = (0, 1),
) -> mir.ColumnJoin:
  return mir.ColumnJoin(
    var_name=var_name,
    sources=[
      mir.ColumnSource(
        rel_name=rel,
        version=Version.FULL,
        index=list(index),
        prefix_vars=list(prefix_vars),
        handle_start=handle_start,
      )
    ],
    handle_start=handle_start,
  )


def _insert_into(vars_: tuple[str, ...] = ('y', 'z')) -> mir.InsertInto:
  return mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=list(vars_),
    index=list(range(len(vars_))),
  )


# -----------------------------------------------------------------------------
# 1. Plain single-source CJ over a fresh source (no prefix)
# -----------------------------------------------------------------------------


def test_cj_single_fresh_source_byte_equivalent():
  '''Single-source nested CJ with empty prefix_vars: emits a fresh
  SaRoot, validity check, lane-strided for-loop binding the value
  via SaGetValAt. Both paths must produce identical rendered text.
  '''
  cj = _single_src_cj()
  ins = _insert_into(('z',))

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([cj, ins], _new_ctx()))
  new_text = _render(_lower_inner_chain([cj, ins], _new_ctx()))

  assert legacy_text == new_text
  # Sanity: the rendered text contains the lane-strided loop shape.
  assert 'thread_rank()' in new_text
  assert '.get_value_at(' in new_text
  # No intersect / cooperative narrowing for the single-source case.
  assert 'intersect' not in new_text
  assert 'child_range(' in new_text  # registered for deeper CJs


# -----------------------------------------------------------------------------
# 2. Prefix-narrowed single-source CJ (parent handle aliased)
# -----------------------------------------------------------------------------


def test_cj_single_prefix_narrowed_byte_equivalent():
  '''Single-source nested CJ with prefix_vars=['y']: aliases a
  parent handle from the enclosing scope (looked up by state key).
  Both paths must emit the same `auto h = parent;` binding.
  '''
  cj = _single_src_cj(prefix_vars=('y',))
  ins = _insert_into(('y', 'z'))

  # Pre-populate the parent handle for both ctx instances.
  parent_key = _state_key('C', [0, 1], ['y'], Version.FULL)

  with _force_legacy_branch():
    ctx = _new_ctx()
    ctx.handle_vars[parent_key] = 'parent_h_C'
    ctx.bound_vars.append('y')
    legacy_text = _render(_lower_inner_chain([cj, ins], ctx))

  ctx = _new_ctx()
  ctx.handle_vars[parent_key] = 'parent_h_C'
  ctx.bound_vars.append('y')
  new_text = _render(_lower_inner_chain([cj, ins], ctx))

  assert legacy_text == new_text
  # The alias bind references the parent handle directly.
  assert 'parent_h_C' in new_text


# -----------------------------------------------------------------------------
# 3. Filter + ConstantBind interleaved with single-source CJ
# -----------------------------------------------------------------------------


def test_cj_single_with_trailing_filter_byte_equivalent():
  '''Chain: single-source CJ + Filter + InsertInto. The Filter
  body sits inside the CJ's for-loop. Both paths must match.
  '''
  cj = _single_src_cj(var_name='z')
  filt = mir.Filter(vars=['z'], code='return z > 0;')
  ins = _insert_into(('z',))

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([cj, filt, ins], _new_ctx()))
  new_text = _render(_lower_inner_chain([cj, filt, ins], _new_ctx()))

  assert legacy_text == new_text
  assert 'if (z > 0)' in new_text


def test_cj_single_with_trailing_constant_bind_byte_equivalent():
  '''Chain: single-source CJ + ConstantBind + InsertInto. The
  ConstantBind emits its `auto var = code;` inside the CJ's
  for-loop body. Both paths must match.
  '''
  cj = _single_src_cj(var_name='z')
  cbind = mir.ConstantBind(var_name='zz', code='z + 1', deps=['z'])
  ins = _insert_into(('z', 'zz'))

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([cj, cbind, ins], _new_ctx()))
  new_text = _render(_lower_inner_chain([cj, cbind, ins], _new_ctx()))

  assert legacy_text == new_text
  assert 'auto zz = z + 1;' in new_text


# -----------------------------------------------------------------------------
# 4. End-to-end pipeline: Scan + single-source CJ + InsertInto
# -----------------------------------------------------------------------------


def test_cj_single_under_scan_root_byte_equivalent():
  '''Smoke test: a Scan-rooted pipeline with a nested single-source
  CJ in the middle slot, fresh source (`prefix_vars=[]` — Scan
  doesn't populate `handle_vars`, so prefix-narrowed sources here
  would have no parent to alias). Renders through `compile_pipeline`
  end-to-end.
  '''
  from srdatalog.compile import compile_pipeline

  scan = mir.Scan(
    vars=['y'],
    rel_name='B',
    version=Version.FULL,
    index=[0],
    handle_start=0,
  )
  cj = _single_src_cj(rel='C', handle_start=1, prefix_vars=(), var_name='z', index=(0,))
  ins = mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['y', 'z'],
    index=[0, 1],
  )
  ep = mir.ExecutePipeline(
    pipeline=[scan, cj, ins],
    source_specs=[scan, cj.sources[0]],
    dest_specs=[ins],
    rule_name='ScanCjSingle',
  )

  with _force_legacy_branch():
    legacy_out = compile_pipeline(ep)
  new_out = compile_pipeline(ep)

  assert legacy_out == new_out
  # Non-empty rendered body.
  assert new_out


# -----------------------------------------------------------------------------
# 5. End-to-end pipeline: multi-source CJ root + single-source nested CJ
# -----------------------------------------------------------------------------


def test_cj_single_under_multi_source_cj_root_byte_equivalent():
  '''Smoke test: a multi-source CJ root + nested single-source CJ
  with empty prefix (fresh source). Pins that the multi-source
  root still flows through the legacy `_lower_root_cj_multi`
  branch (B-CJ-multi territory) while the nested single-source CJ
  is dispatched via the new path.
  '''
  from srdatalog.compile import compile_pipeline

  src_a = mir.ColumnSource(
    rel_name='A',
    version=Version.FULL,
    index=[0, 1],
    prefix_vars=[],
    handle_start=0,
  )
  src_b = mir.ColumnSource(
    rel_name='B',
    version=Version.FULL,
    index=[0, 1],
    prefix_vars=[],
    handle_start=1,
  )
  root_cj = mir.ColumnJoin(var_name='y', sources=[src_a, src_b])
  nested_cj = _single_src_cj(rel='C', handle_start=2, prefix_vars=(), var_name='z', index=(0,))
  ins = mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['y', 'z'],
    index=[0, 1],
  )
  ep = mir.ExecutePipeline(
    pipeline=[root_cj, nested_cj, ins],
    source_specs=[src_a, src_b, nested_cj.sources[0]],
    dest_specs=[ins],
    rule_name='CjMultiCjSingle',
  )

  with _force_legacy_branch():
    legacy_out = compile_pipeline(ep)
  new_out = compile_pipeline(ep)

  assert legacy_out == new_out
  assert new_out


# -----------------------------------------------------------------------------
# 6. Direct lowering call returns the expected IIR shape
# -----------------------------------------------------------------------------


def test_lower_mir_cj_single_in_chain_emits_block_with_for_loop():
  '''Pin the IIR tree shape: single-source CJ -> a Block whose
  last statement is a CartesianFlatLoop binding the per-thread idx
  via lane stride.
  '''
  cj = _single_src_cj(var_name='z')
  ins = _insert_into(('z',))

  out = lower_mir_cj_single_in_chain(cj, [ins], _new_ctx())
  assert isinstance(out, IirBlock)
  # The terminal stmt is the lane-strided for-loop.
  assert any(isinstance(s, CartesianFlatLoop) for s in out.stmts)


def test_lower_mir_cj_single_in_chain_delegates_to_helper():
  '''The chain entry delegates to `_lower_nested_cj_single` directly
  — the rendered text matches calling the helper.
  '''
  cj = _single_src_cj(var_name='z')
  ins = _insert_into(('z',))

  via_entry = _render(lower_mir_cj_single_in_chain(cj, [ins], _new_ctx()))
  via_helper = _render(_lower_nested_cj_single(cj, [ins], _new_ctx()))
  assert via_entry == via_helper


def test_lower_mir_cj_single_in_chain_rejects_multi_source():
  '''The chain entry's structural guard fires when a multi-source
  CJ slips through — pins that the source-count gate isn't only
  enforced by the caller. Multi-source CJ is owned by B-CJ-multi
  (next PR) and must NOT route through this entry.
  '''
  src_a = mir.ColumnSource(
    rel_name='A', version=Version.FULL, index=[0, 1], prefix_vars=[], handle_start=0
  )
  src_b = mir.ColumnSource(
    rel_name='B', version=Version.FULL, index=[0, 1], prefix_vars=[], handle_start=1
  )
  multi_cj = mir.ColumnJoin(var_name='y', sources=[src_a, src_b])
  ins = _insert_into(('y',))
  with pytest.raises(AssertionError, match=r'expected exactly one ColumnSource'):
    lower_mir_cj_single_in_chain(multi_cj, [ins], _new_ctx())


# -----------------------------------------------------------------------------
# 7. Registry contract — stub asserts on direct call
# -----------------------------------------------------------------------------


def test_lower_mir_cj_single_registry_stub_asserts():
  '''The `@lowering(target=iir.cf, source=mir.ColumnJoin)` registry
  entry is a stub that asserts on direct invocation — dispatch is
  expected to flow through `_lower_inner_chain` -> the chain-aware
  variant. Mirrors the B-Filter / B-Scan / B-InsertInto split.
  '''
  cj = _single_src_cj()
  ctx = _new_ctx()
  with pytest.raises(AssertionError, match=r'lower_mir_cj_single_in_chain'):
    lower_mir_cj_single(cj, ctx)


def test_lower_mir_cj_single_is_registered_on_sorted_array_dialect():
  '''The `ColumnJoin` `@lowering` is registered on the
  `relation.sorted_array` dialect. Pins dialect ownership per
  `docs/phase_b_lowering_dispatcher.md` §4 (one `@lowering` per MIR
  op, on the dialect that lowers it). NOTE: this single registration
  covers BOTH single-source (this PR) and multi-source (next PR);
  the shape-aware dispatch lives in `_should_use_declarative`.
  '''
  from srdatalog.ir.dialects.relation.sorted_array import DIALECT as SA_DIALECT

  matched = [low for low in SA_DIALECT.lowerings if low.matches is mir.ColumnJoin]
  assert len(matched) == 1
  assert matched[0].consumes == ('mir',)
  assert 'iir.cf' in matched[0].produces


# -----------------------------------------------------------------------------
# 8. _should_use_declarative shape-gate behavior
# -----------------------------------------------------------------------------


def test_should_use_declarative_single_source_cj_returns_true():
  '''The shape-gate accepts single-source ColumnJoin — the new
  path owns it.'''
  cj = _single_src_cj()
  assert _should_use_declarative(cj) is True


def test_should_use_declarative_multi_source_cj_returns_true_after_b_cj_multi():
  '''B-CJ-multi dropped the source-count guard in `_should_use_declarative`.
  Multi-source ColumnJoin now routes through the new path
  (`lower_mir_cj_multi_in_chain` mid-chain, `lower_mir_cj_multi_root`
  at root) instead of falling through to the legacy
  `_lower_nested_cj_multi` / `_lower_root_cj_multi` branches.

  Historical note: pre-B-CJ-multi this returned False — the helper
  encoded a `len(sources) == 1` gate so only single-source CJ used
  the new path. The new entries delegate to the same legacy helpers,
  so the rendered IIR remains byte-equivalent.
  '''
  src_a = mir.ColumnSource(
    rel_name='A', version=Version.FULL, index=[0, 1], prefix_vars=[], handle_start=0
  )
  src_b = mir.ColumnSource(
    rel_name='B', version=Version.FULL, index=[0, 1], prefix_vars=[], handle_start=1
  )
  multi_cj = mir.ColumnJoin(var_name='y', sources=[src_a, src_b])
  assert _should_use_declarative(multi_cj) is True


def test_should_use_declarative_filter_returns_true():
  '''Sanity: the shape-gate still passes through non-ColumnJoin
  ops in `USE_DECLARATIVE` (B-Filter / B-ConstantBind / B-Scan /
  B-InsertInto rows).'''
  filt = mir.Filter(vars=['x'], code='return x > 0;')
  assert _should_use_declarative(filt) is True


# -----------------------------------------------------------------------------
# 9. USE_DECLARATIVE invariants
# -----------------------------------------------------------------------------


def test_use_declarative_contains_column_join():
  '''Pin the ratchet: after B-CJ-single, `mir.ColumnJoin` appears
  in `USE_DECLARATIVE`. The shape-aware dispatch (`_should_use_declarative`)
  decides per-instance whether to actually route through the new
  path — see the `test_should_use_declarative_*` tests above. The
  monotonic discipline test (when it lands) catches accidental
  removals.
  '''
  from srdatalog.ir.dialects.relation.sorted_array import USE_DECLARATIVE

  assert mir.ColumnJoin in USE_DECLARATIVE
  # B-CJ-single must not regress prior Wave 2A migrations.
  assert mir.Filter in USE_DECLARATIVE
  assert mir.ConstantBind in USE_DECLARATIVE
  assert mir.InsertInto in USE_DECLARATIVE
  assert mir.Scan in USE_DECLARATIVE


# -----------------------------------------------------------------------------
# 10. Multi-source CJ still uses the legacy branch (regression guard)
# -----------------------------------------------------------------------------


def test_multi_source_cj_still_emits_intersect_handles():
  '''Regression guard: with `mir.ColumnJoin` in `USE_DECLARATIVE` AND
  B-CJ-multi's `lower_mir_cj_multi_in_chain` wiring, multi-source CJ
  still emits the `intersect_handles + iter` scaffold by delegating
  to `_lower_nested_cj_multi`. The rendered text is byte-equivalent
  to the legacy path.

  Pre-B-CJ-multi: this test exercised the "multi-source falls
  through to the legacy branch" contract. Post-B-CJ-multi: multi-
  source CJ routes through `lower_mir_cj_multi_in_chain`, which
  delegates to `_lower_nested_cj_multi` — same emission, new
  dispatch surface.
  '''
  src_a = mir.ColumnSource(
    rel_name='A', version=Version.FULL, index=[0, 1], prefix_vars=[], handle_start=0
  )
  src_b = mir.ColumnSource(
    rel_name='B', version=Version.FULL, index=[0, 1], prefix_vars=[], handle_start=1
  )
  multi_cj = mir.ColumnJoin(var_name='y', sources=[src_a, src_b])
  ins = _insert_into(('y',))

  ctx = _new_ctx()
  # No exception: dispatch goes through `_lower_nested_cj_multi`
  # (now via `lower_mir_cj_multi_in_chain`, post-B-CJ-multi).
  out = _lower_inner_chain([multi_cj, ins], ctx)
  rendered = _render(out)
  # The multi-source CJ emits an intersect_handles + iter scaffold —
  # the legacy `_lower_nested_cj_multi` signature.
  assert 'intersect_handles' in rendered
