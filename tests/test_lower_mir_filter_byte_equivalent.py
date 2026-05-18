'''Byte-equivalence test for Wave 2A B-Filter migration.

Per `docs/phase_b_lowering_dispatcher.md` §4.2 (per-PR acceptance
gate): each per-MIR-op migration ships a
`test_lower_mir_<op>_byte_equivalent` test that runs the migrated
path on every relevant fixture and asserts byte equality with the
legacy `if isinstance(head, mir.X):` branch.

For Filter the relevant fixtures are:
  - the plain `if (cond) { body }` shape (no ws/tiled flag set),
  - the WS batched-Cartesian fold (`ws_cartesian_valid_var` set),
  - the tiled-Cartesian ballot fold (`tiled_cartesian_valid_var`
    set),
  - filters with multi-var conditions, return-prefixed code,
    trailing-semicolon code (the three `_filter_expr`
    normalization shapes).

The test compares two compilation paths:
  - LEGACY: `USE_DECLARATIVE` patched to NOT contain `mir.Filter`,
    so `_lower_inner_chain` falls into the imperative branch.
  - NEW: `USE_DECLARATIVE` left alone (Filter IS in the set), so
    `_lower_inner_chain` routes through
    `lower_mir_filter_in_chain`.

Byte-equivalence is asserted on the rendered IIR text (which the
load-bearing harnesses in `tests/test_runner_byte_equivalence.py`
+ `tests/test_byte_equivalence_jit.py` anchor against the legacy
CUDA emit). The 532-fixture harness re-running green under the new
path is the strongest signal; this file adds direct-call coverage
of the corner cases.
'''

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

import srdatalog.ir.mir.types as mir
from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit
from srdatalog.ir.dialects.iir.cf import Block as IirBlock
from srdatalog.ir.dialects.iir.cf import If, RawString
from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
  LoweringCtx,
  _lower_inner_chain,
)
from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_filter import (
  lower_mir_filter,
  lower_mir_filter_in_chain,
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
  '''Temporarily strip `mir.Filter` from `USE_DECLARATIVE` so
  `_lower_inner_chain` falls into the legacy imperative branch.

  Save / restore as a context manager — discipline test
  `test_use_declarative_is_monotonic` (when it lands) ratchets the
  set at module import time, but this test mutates the dialect's
  re-bound name for the duration of one call only.
  '''
  import srdatalog.ir.dialects.relation.sorted_array as sa_dialect

  saved = sa_dialect.USE_DECLARATIVE
  sa_dialect.USE_DECLARATIVE = frozenset(saved - {mir.Filter})
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


def _new_ctx(**kwargs: Any) -> LoweringCtx:
  '''Fresh LoweringCtx — counters reset so both paths bump identically.'''
  return LoweringCtx(output_var='ctx0', **kwargs)


# -----------------------------------------------------------------------------
# 1. Plain `if (cond) { body }` shape
# -----------------------------------------------------------------------------


def test_filter_plain_if_byte_equivalent():
  '''Standard Filter with no ws/tiled flag: emits `if (cond) {body}`.
  Both paths must produce identical rendered text.'''
  filt = mir.Filter(vars=['v0', 'v1'], code='return v0 < v1;')
  ins = _insert_into(2)

  with _force_legacy_branch():
    legacy_op = _lower_inner_chain([filt, ins], _new_ctx())
    legacy_text = _render(legacy_op)

  new_op = _lower_inner_chain([filt, ins], _new_ctx())
  new_text = _render(new_op)

  assert legacy_text == new_text
  # Sanity: the rendered text contains the `if` shape.
  assert 'if (v0 < v1)' in legacy_text


def test_filter_return_prefix_normalization_byte_equivalent():
  '''Filter code starting with `return ` gets the prefix stripped
  by `_filter_expr`. Both paths reuse the same helper, so they
  match.'''
  filt = mir.Filter(vars=['v0'], code='return v0 != 0;')
  ins = _insert_into(1)

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([filt, ins], _new_ctx()))
  new_text = _render(_lower_inner_chain([filt, ins], _new_ctx()))

  assert legacy_text == new_text
  assert 'if (v0 != 0)' in legacy_text
  assert 'return' not in legacy_text


def test_filter_trailing_semicolon_normalization_byte_equivalent():
  '''Filter code ending with `;` gets the semicolon stripped by
  `_filter_expr` (the rendered `if (cond)` has its own delimiters).'''
  filt = mir.Filter(vars=['v0'], code='v0 > 0;')
  ins = _insert_into(1)

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([filt, ins], _new_ctx()))
  new_text = _render(_lower_inner_chain([filt, ins], _new_ctx()))

  assert legacy_text == new_text
  assert 'if (v0 > 0)' in legacy_text


# -----------------------------------------------------------------------------
# 2. ws_cartesian_valid_var fold
# -----------------------------------------------------------------------------


def test_filter_ws_cart_fold_byte_equivalent():
  '''When `ws_cartesian_valid_var` is set, the filter folds into
  `<v> = <v> && (<cond>);` instead of `if (...) {...}`. Both paths
  produce the same Block shape (RawString + body).'''
  filt = mir.Filter(vars=['x', 'y'], code='return x != y;')
  ins = _insert_into(2)

  with _force_legacy_branch():
    legacy_text = _render(
      _lower_inner_chain([filt, ins], _new_ctx(ws_cartesian_valid_var='ws_valid'))
    )
  new_text = _render(_lower_inner_chain([filt, ins], _new_ctx(ws_cartesian_valid_var='ws_valid')))

  assert legacy_text == new_text
  assert 'ws_valid = ws_valid && (x != y);' in legacy_text
  # The whole branch should be fold, not `if`.
  assert 'if (' not in legacy_text


# -----------------------------------------------------------------------------
# 3. tiled_cartesian_valid_var fold
# -----------------------------------------------------------------------------


def test_filter_tiled_cart_fold_byte_equivalent():
  '''When `tiled_cartesian_valid_var` is set, the filter folds into
  `<v> = <v> && (<cond>);` then renders the body unchanged. The body
  in this fixture is a `TiledBallotBlock` (the InsertInto run with
  the tiled flag set).'''
  filt = mir.Filter(vars=['v0', 'v1'], code='return v0 < v1;')
  ins = _insert_into(2)

  with _force_legacy_branch():
    legacy_text = _render(
      _lower_inner_chain([filt, ins], _new_ctx(tiled_cartesian_valid_var='tc_valid'))
    )
  new_text = _render(
    _lower_inner_chain([filt, ins], _new_ctx(tiled_cartesian_valid_var='tc_valid'))
  )

  assert legacy_text == new_text
  assert 'tc_valid = tc_valid && (v0 < v1);' in legacy_text


# -----------------------------------------------------------------------------
# 4. Direct lowering call returns the expected IIR shape
# -----------------------------------------------------------------------------


def test_lower_mir_filter_in_chain_emits_if_shape():
  '''Pin the IIR tree shape directly (independent of byte text):
  plain Filter -> `If(RawString, body)`.'''
  filt = mir.Filter(vars=['v0'], code='return v0 > 0;')
  ins = _insert_into(1)

  out = lower_mir_filter_in_chain(filt, [ins], _new_ctx())
  assert isinstance(out, If)
  assert isinstance(out.cond, RawString)
  assert out.cond.text == 'v0 > 0'


def test_lower_mir_filter_in_chain_emits_fold_block_when_ws_set():
  '''Pin the IIR tree shape directly under ws fold: `Block(RawString,
  body)`. The RawString carries the fold statement.'''
  filt = mir.Filter(vars=['x'], code='x > 0')
  ins = _insert_into(1)

  out = lower_mir_filter_in_chain(filt, [ins], _new_ctx(ws_cartesian_valid_var='vv'))
  assert isinstance(out, IirBlock)
  assert isinstance(out.stmts[0], RawString)
  assert out.stmts[0].text == 'vv = vv && (x > 0);'


# -----------------------------------------------------------------------------
# 5. Registry contract — stub asserts on direct call
# -----------------------------------------------------------------------------


def test_lower_mir_filter_registry_stub_asserts():
  '''The `@lowering(target=iir.cf, source=mir.Filter)` registry
  entry is a stub that asserts on direct invocation — dispatch is
  expected to flow through `_lower_inner_chain` -> the chain-aware
  variant. Mirrors the C5 `lower_tiled_cartesian` split.
  '''
  filt = mir.Filter(vars=['x'], code='x > 0')
  ctx = _new_ctx()
  with pytest.raises(AssertionError, match=r'lower_mir_filter_in_chain'):
    lower_mir_filter(filt, ctx)


def test_lower_mir_filter_is_registered_on_sorted_array_dialect():
  '''The `Filter` `@lowering` is registered on the
  `relation.sorted_array` dialect. Pins dialect ownership per
  `docs/phase_b_lowering_dispatcher.md` §4 (one `@lowering` per MIR
  op, on the dialect that lowers it).
  '''
  from srdatalog.ir.dialects.relation.sorted_array import DIALECT as SA_DIALECT

  matched = [low for low in SA_DIALECT.lowerings if low.matches is mir.Filter]
  assert len(matched) == 1
  assert matched[0].consumes == ('mir',)
  assert 'iir.cf' in matched[0].produces


# -----------------------------------------------------------------------------
# 6. Smoke test through the full compile_pipeline surface
# -----------------------------------------------------------------------------


def test_filter_in_full_pipeline_byte_equivalent():
  '''Smoke test: a Scan + Filter + InsertInto pipeline rendered
  through `compile_pipeline` (the production surface that the byte-
  equivalence harnesses guard) must produce identical CUDA under
  both `USE_DECLARATIVE` states.

  This closes the loop between the direct-call tests above and the
  532-fixture harness — if a divergence ever creeps in, both this
  test and the harness will catch it.
  '''
  from srdatalog.compile import compile_pipeline

  scan = mir.Scan(
    vars=['x', 'y'],
    rel_name='Src',
    version=Version.FULL,
    index=[0, 1],
    handle_start=0,
  )
  filt = mir.Filter(vars=['x', 'y'], code='return x < y;')
  ins = mir.InsertInto(rel_name='Dst', version=Version.NEW, vars=['x', 'y'], index=[0, 1])
  ep = mir.ExecutePipeline(
    pipeline=[scan, filt, ins],
    source_specs=[scan],
    dest_specs=[ins],
    rule_name='FiltRule',
  )

  with _force_legacy_branch():
    legacy_out = compile_pipeline(ep)
  new_out = compile_pipeline(ep)

  assert legacy_out == new_out
  assert 'if (x < y)' in new_out
