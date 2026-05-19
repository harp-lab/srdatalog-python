'''Byte-equivalence test for Wave 2A B-Negation migration.

Per `docs/phase_b_lowering_dispatcher.md` §4.2 (per-PR acceptance
gate): each per-MIR-op migration ships a
`test_lower_mir_<op>_byte_equivalent` test that runs the migrated
path on every relevant fixture and asserts byte equality with the
legacy `if isinstance(head, mir.X):` branch.

For Negation the relevant fixtures (per
`docs/phase_b_lowering_dispatcher.md` §4 row B-Negation,
"hard" — `_lower_negation` interacts with `const_args`,
multi-version, multi-source negation shapes):

  - standard path: a plain Negation following a root Scan; emits
    `Bind(h_handle, SaRoot(...).prefix...)` + `if (!h.valid()) {...}`,
  - standard path with multiple prefix vars: walks the prefix list
    cooperatively (SaPrefCoop) outside any Cartesian,
  - inside-Cartesian path: prefix walked sequentially (SaPrefSeq)
    when `ctx.inside_cartesian` is True,
  - WS valid-var fold (`ws_cartesian_valid_var` set): `<v> = <v>
    && (!h.valid());` instead of an `if`,
  - tiled-Cartesian valid-var fold (`tiled_cartesian_valid_var`
    set): same fold shape,
  - pre-narrow path (`ctx.neg_pre_narrow[src_idx]` registered by a
    surrounding `_lower_nested_cart`): reuses the pre-allocated
    handle; no fresh `Bind` unless `in_cartesian_vars` are present,
  - pre-narrow path with `in_cartesian_vars`: allocates a fresh
    `h_<rel>_neg_<idx>` and chains `SaPrefSeq` calls,
  - N5.4 Nim-broken case (standard-path Negation over D2L FULL_VER):
    legacy raises `NotImplementedError` referencing N5.4 — preserved
    by the migration (per `docs/milestones.md` F5, not a fix),
  - standard-path with `const_args`: legacy raises
    `NotImplementedError('... const_args not yet lowered ...')` —
    preserved.

The test compares two compilation paths:
  - LEGACY: `USE_DECLARATIVE` patched to NOT contain `mir.Negation`,
    so `_lower_inner_chain` falls into the imperative branch.
  - NEW: `USE_DECLARATIVE` left alone (Negation IS in the set), so
    `_lower_inner_chain` routes through
    `lower_mir_negation_in_chain` -> `_lower_negation`.

Byte-equivalence is asserted on the rendered IIR text (which the
load-bearing harnesses in `tests/test_runner_byte_equivalence.py`
+ `tests/test_byte_equivalence_jit.py` anchor against the legacy
CUDA emit). The full-fixture harness re-running green under the
new path is the strongest signal (the production Negation fixtures
live under `tests/fixtures/jit/{lsqb_q9_neg2hop, crdt, doop,
ddisasm, reg_scc, polonius_test}/...`); this file adds direct-call
coverage of the corner cases plus the Nim-broken raises.
'''

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

import srdatalog.ir.mir.types as mir
from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit
from srdatalog.ir.dialects.iir.cf import Bind, If, RawString
from srdatalog.ir.dialects.iir.cf import Block as IirBlock
from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
  LoweringCtx,
  NegPreNarrowInfo,
  _lower_inner_chain,
)
from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_negation import (
  lower_mir_negation,
  lower_mir_negation_in_chain,
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
  '''Temporarily strip `mir.Negation` from `USE_DECLARATIVE` so
  `_lower_inner_chain` falls into the legacy imperative branch.

  Save / restore as a context manager — discipline test
  `test_use_declarative_is_monotonic` (when it lands) ratchets the
  set at module import time, but this test mutates the dialect's
  re-bound name for the duration of one call only.
  '''
  import srdatalog.ir.dialects.relation.sorted_array as sa_dialect

  saved = sa_dialect.USE_DECLARATIVE
  sa_dialect.USE_DECLARATIVE = frozenset(saved - {mir.Negation})
  try:
    yield
  finally:
    sa_dialect.USE_DECLARATIVE = saved


def _insert_into(vars_: list[str]) -> mir.InsertInto:
  return mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=vars_,
    index=list(range(len(vars_))),
  )


def _new_ctx(**kwargs: Any) -> LoweringCtx:
  '''Fresh LoweringCtx — counters reset so both paths bump identically.'''
  return LoweringCtx(output_var='ctx0', **kwargs)


# -----------------------------------------------------------------------------
# 1. Standard path — single prefix var, plain `if (!h.valid())`
# -----------------------------------------------------------------------------


def test_negation_standard_single_prefix_byte_equivalent():
  '''Standard-path Negation with one prefix var: emits a fresh
  `h_<rel>_neg_<idx>` handle bound to SaRoot.prefix_coop(...) and
  an `if (!h.valid())` guard around the body. Both paths must
  produce identical rendered text.'''
  neg = mir.Negation(
    rel_name='Bad',
    version=Version.FULL,
    prefix_vars=['x'],
    index=[0],
    handle_start=1,
  )
  ins = _insert_into(['x'])

  ctx_kw = dict(
    view_var_names={'1': 'view_Bad_0_FULL_VER'},
    bound_vars=['x'],
  )

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([neg, ins], _new_ctx(**ctx_kw)))
  new_text = _render(_lower_inner_chain([neg, ins], _new_ctx(**ctx_kw)))

  assert legacy_text == new_text
  assert 'if (!' in legacy_text
  assert '.valid()' in legacy_text
  assert 'h_Bad_neg_1' in legacy_text


def test_negation_standard_multi_prefix_byte_equivalent():
  '''Multiple prefix vars: each walked cooperatively via SaPrefCoop
  outside a Cart. The handle expression is a chained
  `Root.prefix_coop(x).prefix_coop(y)`.'''
  neg = mir.Negation(
    rel_name='Bad',
    version=Version.FULL,
    prefix_vars=['x', 'y'],
    index=[0, 1],
    handle_start=2,
  )
  ins = _insert_into(['x', 'y'])

  ctx_kw = dict(
    view_var_names={'2': 'view_Bad_0_1_FULL_VER'},
    bound_vars=['x', 'y'],
  )

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([neg, ins], _new_ctx(**ctx_kw)))
  new_text = _render(_lower_inner_chain([neg, ins], _new_ctx(**ctx_kw)))

  assert legacy_text == new_text
  # Two prefix vars rendered.
  assert 'x' in legacy_text and 'y' in legacy_text
  assert 'h_Bad_neg_2' in legacy_text


# -----------------------------------------------------------------------------
# 2. Inside-Cartesian dispatch — SaPrefSeq instead of SaPrefCoop
# -----------------------------------------------------------------------------


def test_negation_inside_cartesian_prefix_seq_byte_equivalent():
  '''When `ctx.inside_cartesian=True`, the prefix walk uses SaPrefSeq
  (sequential) instead of SaPrefCoop (cooperative). Both paths
  produce identical text.'''
  neg = mir.Negation(
    rel_name='Bad',
    version=Version.FULL,
    prefix_vars=['z'],
    index=[0],
    handle_start=3,
  )
  ins = _insert_into(['z'])

  ctx_kw = dict(
    view_var_names={'3': 'view_Bad_0_FULL_VER'},
    bound_vars=['z'],
    inside_cartesian=True,
  )

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([neg, ins], _new_ctx(**ctx_kw)))
  new_text = _render(_lower_inner_chain([neg, ins], _new_ctx(**ctx_kw)))

  assert legacy_text == new_text


# -----------------------------------------------------------------------------
# 3. WS valid-var fold
# -----------------------------------------------------------------------------


def test_negation_ws_valid_fold_byte_equivalent():
  '''When `ws_cartesian_valid_var` is set, the negation folds into
  `<v> = <v> && (!h.valid());` instead of an `if`. Mirrors the
  legacy `ctx.ws_cartesian_valid_var` branch in `_lower_negation`.'''
  neg = mir.Negation(
    rel_name='Bad',
    version=Version.FULL,
    prefix_vars=['x'],
    index=[0],
    handle_start=1,
  )
  ins = _insert_into(['x'])

  ctx_kw = dict(
    view_var_names={'1': 'view_Bad_0_FULL_VER'},
    bound_vars=['x'],
    ws_cartesian_valid_var='ws_valid',
  )

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([neg, ins], _new_ctx(**ctx_kw)))
  new_text = _render(_lower_inner_chain([neg, ins], _new_ctx(**ctx_kw)))

  assert legacy_text == new_text
  assert 'ws_valid = ws_valid && (!' in legacy_text
  # Plain `if (!h.valid())` MUST NOT appear — folded.
  assert 'if (!' not in legacy_text


def test_negation_tiled_valid_fold_byte_equivalent():
  '''When `tiled_cartesian_valid_var` is set, the same fold shape
  applies. Both paths produce identical text.'''
  neg = mir.Negation(
    rel_name='Bad',
    version=Version.FULL,
    prefix_vars=['x'],
    index=[0],
    handle_start=1,
  )
  ins = _insert_into(['x'])

  ctx_kw = dict(
    view_var_names={'1': 'view_Bad_0_FULL_VER'},
    bound_vars=['x'],
    tiled_cartesian_valid_var='tc_valid',
  )

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([neg, ins], _new_ctx(**ctx_kw)))
  new_text = _render(_lower_inner_chain([neg, ins], _new_ctx(**ctx_kw)))

  assert legacy_text == new_text
  assert 'tc_valid = tc_valid && (!' in legacy_text


# -----------------------------------------------------------------------------
# 4. Pre-narrow path (Negation after a Cartesian)
# -----------------------------------------------------------------------------


def test_negation_pre_narrow_no_in_cart_vars_byte_equivalent():
  '''Pre-narrow path with no in-Cartesian vars: reuses the
  pre-allocated handle directly (no fresh `Bind` needed). Mirrors
  the legacy `info.in_cartesian_vars` empty branch.'''
  neg = mir.Negation(
    rel_name='Bad',
    version=Version.FULL,
    prefix_vars=['x'],
    index=[0],
    handle_start=1,
  )
  ins = _insert_into(['x'])

  pre_info = NegPreNarrowInfo(
    var_name='h_Bad_neg_pre_1',
    pre_vars=['x'],
    in_cartesian_vars=[],
    pre_consts=[],
    view_var='view_Bad_0_FULL_VER',
    rel_name='Bad',
  )

  def _ctx_kw():
    return dict(
      view_var_names={'1': 'view_Bad_0_FULL_VER'},
      bound_vars=['x'],
      neg_pre_narrow={1: pre_info},
    )

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([neg, ins], _new_ctx(**_ctx_kw())))
  new_text = _render(_lower_inner_chain([neg, ins], _new_ctx(**_ctx_kw())))

  assert legacy_text == new_text
  # Pre-allocated handle var appears in the validity check.
  assert 'h_Bad_neg_pre_1' in legacy_text


def test_negation_pre_narrow_with_in_cart_vars_byte_equivalent():
  '''Pre-narrow path with `in_cartesian_vars`: allocates a fresh
  `h_<rel>_neg_<idx>` handle and chains SaPrefSeq calls for each
  in-Cartesian var. Both paths produce identical text.'''
  neg = mir.Negation(
    rel_name='Bad',
    version=Version.FULL,
    prefix_vars=['x', 'y'],
    index=[0, 1],
    handle_start=2,
  )
  ins = _insert_into(['x', 'y'])

  pre_info = NegPreNarrowInfo(
    var_name='h_Bad_neg_pre_2',
    pre_vars=['x'],
    in_cartesian_vars=['y'],
    pre_consts=[],
    view_var='view_Bad_0_1_FULL_VER',
    rel_name='Bad',
  )

  def _ctx_kw():
    return dict(
      view_var_names={'2': 'view_Bad_0_1_FULL_VER'},
      bound_vars=['x', 'y'],
      inside_cartesian=True,
      neg_pre_narrow={2: pre_info},
    )

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([neg, ins], _new_ctx(**_ctx_kw())))
  new_text = _render(_lower_inner_chain([neg, ins], _new_ctx(**_ctx_kw())))

  assert legacy_text == new_text
  # Fresh handle bound off of the pre-narrowed one, then validity
  # checked.
  assert 'h_Bad_neg_pre_2' in legacy_text
  assert 'h_Bad_neg_2' in legacy_text


# -----------------------------------------------------------------------------
# 5. N5.4 Nim-broken raise — standard-path Negation over D2L FULL_VER
# -----------------------------------------------------------------------------


def test_negation_n5_4_d2l_full_raises_in_both_paths():
  '''Standard-path Negation over a D2L FULL_VER source raises
  `NotImplementedError` referencing N5.4 (per `docs/milestones.md`
  F5: both Nim and the dialect are broken; deferred). The
  migration preserves the raise byte-for-byte — it is NOT a fix.

  Both the legacy branch AND the new dispatch path must raise
  identically. This pins the Nim-broken contract.
  '''
  neg = mir.Negation(
    rel_name='Bad',
    version=Version.FULL,
    prefix_vars=['x'],
    index=[0],
    handle_start=1,
  )
  ins = _insert_into(['x'])

  ctx_kw = dict(
    view_var_names={'1': 'view_Bad_0_FULL_VER'},
    bound_vars=['x'],
    rel_index_types={'Bad': 'Device2LevelIndex'},
  )

  with _force_legacy_branch(), pytest.raises(NotImplementedError, match=r'N5\.4'):
    _lower_inner_chain([neg, ins], _new_ctx(**ctx_kw))
  with pytest.raises(NotImplementedError, match=r'N5\.4'):
    _lower_inner_chain([neg, ins], _new_ctx(**ctx_kw))


def test_negation_standard_const_args_raises_in_both_paths():
  '''Standard-path Negation with non-empty `const_args` raises
  `NotImplementedError` (only the pre-narrow path handles
  const_args today). Migration preserves the raise.'''
  neg = mir.Negation(
    rel_name='Bad',
    version=Version.FULL,
    prefix_vars=[],
    const_args=[(0, 42)],
    index=[0],
    handle_start=1,
  )
  ins = _insert_into(['x'])

  ctx_kw = dict(
    view_var_names={'1': 'view_Bad_0_FULL_VER'},
    bound_vars=['x'],
  )

  with _force_legacy_branch(), pytest.raises(NotImplementedError, match=r'const_args'):
    _lower_inner_chain([neg, ins], _new_ctx(**ctx_kw))
  with pytest.raises(NotImplementedError, match=r'const_args'):
    _lower_inner_chain([neg, ins], _new_ctx(**ctx_kw))


# -----------------------------------------------------------------------------
# 6. Direct lowering call returns the expected IIR shape
# -----------------------------------------------------------------------------


def test_lower_mir_negation_in_chain_emits_block_with_if():
  '''Pin the IIR tree shape directly: standard-path Negation
  produces a `Block` whose tail (after optional Comments + the
  handle `Bind`) is an `If(RawString("!h_..._neg_1.valid()"),
  body)`. The Block always wraps because `_lower_negation`
  builds a `stmts` list (comments + bind + branch).'''
  neg = mir.Negation(
    rel_name='Bad',
    version=Version.FULL,
    prefix_vars=['x'],
    index=[0],
    handle_start=1,
  )
  ins = _insert_into(['x'])
  ctx = _new_ctx(
    view_var_names={'1': 'view_Bad_0_FULL_VER'},
    bound_vars=['x'],
    debug=False,
  )

  out = lower_mir_negation_in_chain(neg, [ins], ctx)
  assert isinstance(out, IirBlock)
  # First stmt is the Bind, second is the If guard.
  assert isinstance(out.stmts[0], Bind)
  assert out.stmts[0].name.startswith('h_Bad_neg_')
  if_op = out.stmts[1]
  assert isinstance(if_op, If)
  assert isinstance(if_op.cond, RawString)
  assert if_op.cond.text.startswith('!') and '.valid()' in if_op.cond.text


def test_lower_mir_negation_in_chain_emits_fold_under_ws():
  '''Pin the IIR tree shape under WS fold: the validity check
  becomes a `RawString` with the `<v> = <v> && (!h.valid());` text.'''
  neg = mir.Negation(
    rel_name='Bad',
    version=Version.FULL,
    prefix_vars=['x'],
    index=[0],
    handle_start=1,
  )
  ins = _insert_into(['x'])
  ctx = _new_ctx(
    view_var_names={'1': 'view_Bad_0_FULL_VER'},
    bound_vars=['x'],
    ws_cartesian_valid_var='ws_valid',
    debug=False,
  )

  out = lower_mir_negation_in_chain(neg, [ins], ctx)
  assert isinstance(out, IirBlock)
  # One of the stmts is the fold RawString.
  fold_texts = [s.text for s in out.stmts if isinstance(s, RawString)]
  assert any('ws_valid = ws_valid && (!' in t for t in fold_texts)


# -----------------------------------------------------------------------------
# 7. Registry contract — stub asserts on direct call
# -----------------------------------------------------------------------------


def test_lower_mir_negation_registry_stub_asserts():
  '''The `@lowering(target=iir.cf, source=mir.Negation)` registry
  entry is a stub that asserts on direct invocation — dispatch is
  expected to flow through `_lower_inner_chain` -> the chain-aware
  variant. Mirrors the C5 `lower_tiled_cartesian` split.
  '''
  neg = mir.Negation(
    rel_name='Bad',
    version=Version.FULL,
    prefix_vars=['x'],
    index=[0],
    handle_start=1,
  )
  ctx = _new_ctx()
  with pytest.raises(AssertionError, match=r'lower_mir_negation_in_chain'):
    lower_mir_negation(neg, ctx)


def test_lower_mir_negation_is_registered_on_sorted_array_dialect():
  '''The `Negation` `@lowering` is registered on the
  `relation.sorted_array` dialect. Pins dialect ownership per
  `docs/phase_b_lowering_dispatcher.md` §4 (one `@lowering` per
  MIR op, on the dialect that lowers it).
  '''
  from srdatalog.ir.dialects.relation.sorted_array import DIALECT as SA_DIALECT

  matched = [low for low in SA_DIALECT.lowerings if low.matches is mir.Negation]
  assert len(matched) == 1
  assert matched[0].consumes == ('mir',)
  assert 'iir.cf' in matched[0].produces


# -----------------------------------------------------------------------------
# 8. USE_DECLARATIVE invariants
# -----------------------------------------------------------------------------


def test_use_declarative_contains_negation():
  '''Pin the ratchet: after this Wave 2A PR, `mir.Negation` must
  appear in `USE_DECLARATIVE`. The monotonic discipline test (when
  it lands) catches accidental removals.'''
  from srdatalog.ir.dialects.relation.sorted_array import USE_DECLARATIVE

  assert mir.Negation in USE_DECLARATIVE
  # Sanity: the previously-migrated ops are still in the set.
  assert mir.Filter in USE_DECLARATIVE
  assert mir.ConstantBind in USE_DECLARATIVE
  assert mir.InsertInto in USE_DECLARATIVE
  assert mir.Scan in USE_DECLARATIVE


# -----------------------------------------------------------------------------
# 9. Smoke test through the full compile_pipeline surface
# -----------------------------------------------------------------------------


def test_negation_in_full_pipeline_byte_equivalent():
  '''Smoke test: a Scan + Negation + InsertInto pipeline rendered
  through `compile_pipeline` (the production surface that the
  byte-equivalence harnesses guard) must produce identical CUDA
  under both `USE_DECLARATIVE` states.

  This closes the loop between the direct-call tests above and the
  full-fixture harness — if a divergence ever creeps in, both this
  test and the harness will catch it.
  '''
  from srdatalog.compile import compile_pipeline

  scan = mir.Scan(
    vars=['x'],
    rel_name='Src',
    version=Version.FULL,
    index=[0],
    handle_start=0,
  )
  neg = mir.Negation(
    rel_name='Bad',
    version=Version.FULL,
    prefix_vars=['x'],
    index=[0],
    handle_start=1,
  )
  ins = mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['x'],
    index=[0],
  )
  ep = mir.ExecutePipeline(
    pipeline=[scan, neg, ins],
    source_specs=[scan, neg],
    dest_specs=[ins],
    rule_name='NegRule',
  )

  with _force_legacy_branch():
    legacy_out = compile_pipeline(ep)
  new_out = compile_pipeline(ep)

  assert legacy_out == new_out
  # Sanity: the negation `NOT EXISTS` comment / handle name appears.
  assert 'h_Bad_neg_1' in new_out
  assert '!' in new_out and '.valid()' in new_out
