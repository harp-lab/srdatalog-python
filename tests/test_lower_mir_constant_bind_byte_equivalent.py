'''Byte-equivalence test for Wave 2A B-ConstantBind migration.

Per `docs/phase_b_lowering_dispatcher.md` §4.2 (per-PR acceptance
gate): each per-MIR-op migration ships a
`test_lower_mir_<op>_byte_equivalent` test that runs the migrated
path on every relevant fixture and asserts byte equality with the
legacy `if isinstance(head, mir.X):` branch.

For ConstantBind the relevant fixtures are:
  - the trailing rest_op is a `Block` (Filter / nested CJ / multi-
    InsertInto) — the bind statement prepends and the surrounding
    block flattens,
  - the trailing rest_op is a single non-Block op (e.g. a single
    InsertInto) — the bind statement plus the single op wrap in a
    fresh `Block`,
  - the bound var name collides with a C++ keyword — sanitized via
    `_sanitize_var_name` (e.g. `class` -> `class_val`),
  - chains with multiple ConstantBinds in sequence.

The test compares two compilation paths:
  - LEGACY: `USE_DECLARATIVE` patched to NOT contain
    `mir.ConstantBind`, so `_lower_inner_chain` falls into the
    imperative branch.
  - NEW: `USE_DECLARATIVE` left alone (ConstantBind IS in the set),
    so `_lower_inner_chain` routes through
    `lower_mir_constant_bind_in_chain`.

Byte-equivalence is asserted on the rendered IIR text (which the
load-bearing harnesses in `tests/test_runner_byte_equivalence.py`
+ `tests/test_byte_equivalence_jit.py` anchor against the legacy
CUDA emit).
'''

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

import srdatalog.ir.mir.types as mir
from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit
from srdatalog.ir.dialects.iir.cf import Bind, RawString
from srdatalog.ir.dialects.iir.cf import Block as IirBlock
from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
  LoweringCtx,
  _lower_inner_chain,
)
from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_constant_bind import (
  lower_mir_constant_bind,
  lower_mir_constant_bind_in_chain,
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
  '''Temporarily strip `mir.ConstantBind` from `USE_DECLARATIVE` so
  `_lower_inner_chain` falls into the legacy imperative branch.'''
  import srdatalog.ir.dialects.relation.sorted_array as sa_dialect

  saved = sa_dialect.USE_DECLARATIVE
  sa_dialect.USE_DECLARATIVE = frozenset(saved - {mir.ConstantBind})
  try:
    yield
  finally:
    sa_dialect.USE_DECLARATIVE = saved


def _insert_into(arity: int = 1, vars_: list[str] | None = None) -> mir.InsertInto:
  vars_ = vars_ or [f'v{i}' for i in range(arity)]
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
# 1. Single InsertInto rest_op (non-Block path)
# -----------------------------------------------------------------------------


def test_constant_bind_with_single_insert_byte_equivalent():
  '''ConstantBind followed by a single InsertInto: rest_op is the
  Block from the trailing-InsertInto branch (which IS a Block).
  Bind prepends + flatten. Both paths produce identical text.'''
  cb = mir.ConstantBind(var_name='k', code='42', deps=[])
  ins = _insert_into(vars_=['k'])

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([cb, ins], _new_ctx()))
  new_text = _render(_lower_inner_chain([cb, ins], _new_ctx()))

  assert legacy_text == new_text
  assert 'auto k = 42;' in legacy_text


def test_constant_bind_with_filter_then_insert_byte_equivalent():
  '''ConstantBind -> Filter -> InsertInto: rest_op is an `If` (the
  Filter's emission), which is NOT a Block. Bind + If wrap in a
  fresh Block.'''
  cb = mir.ConstantBind(var_name='lo', code='0', deps=[])
  filt = mir.Filter(vars=['v0', 'lo'], code='return v0 > lo;')
  ins = _insert_into(vars_=['v0'])

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([cb, filt, ins], _new_ctx()))
  new_text = _render(_lower_inner_chain([cb, filt, ins], _new_ctx()))

  assert legacy_text == new_text
  assert 'auto lo = 0;' in legacy_text
  assert 'if (v0 > lo)' in legacy_text


# -----------------------------------------------------------------------------
# 2. C++ keyword sanitization
# -----------------------------------------------------------------------------


def test_constant_bind_cpp_keyword_sanitize_byte_equivalent():
  '''A var named `class` collides with C++ and gets a `_val`
  suffix via `_sanitize_var_name`. Both paths reuse the same
  helper, so they match.'''
  cb = mir.ConstantBind(var_name='class', code='7', deps=[])
  ins = _insert_into(vars_=['v0'])

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([cb, ins], _new_ctx()))
  new_text = _render(_lower_inner_chain([cb, ins], _new_ctx()))

  assert legacy_text == new_text
  assert 'auto class_val = 7;' in legacy_text
  # Bare `class =` (the unsuffixed form) MUST NOT appear.
  assert 'auto class =' not in legacy_text


# -----------------------------------------------------------------------------
# 3. Multiple ConstantBinds in sequence
# -----------------------------------------------------------------------------


def test_multiple_constant_binds_byte_equivalent():
  '''Two ConstantBinds in a row — the inner one's flatten-into-Block
  behavior should compose with the outer one's. Both paths
  produce identical output.'''
  cb1 = mir.ConstantBind(var_name='a', code='1', deps=[])
  cb2 = mir.ConstantBind(var_name='b', code='a + 1', deps=['a'])
  ins = _insert_into(vars_=['a'])

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([cb1, cb2, ins], _new_ctx()))
  new_text = _render(_lower_inner_chain([cb1, cb2, ins], _new_ctx()))

  assert legacy_text == new_text
  assert 'auto a = 1;' in legacy_text
  assert 'auto b = a + 1;' in legacy_text


# -----------------------------------------------------------------------------
# 4. Direct lowering call returns the expected IIR shape
# -----------------------------------------------------------------------------


def test_lower_mir_constant_bind_in_chain_emits_block_with_bind():
  '''Pin the IIR tree shape: ConstantBind -> InsertInto produces
  `Block(Bind, *insert_stmts)` — the bind prepends to the inner
  block's statements (flatten).'''
  cb = mir.ConstantBind(var_name='k', code='5', deps=[])
  ins = _insert_into(vars_=['k'])

  out = lower_mir_constant_bind_in_chain(cb, [ins], _new_ctx())
  assert isinstance(out, IirBlock)
  assert isinstance(out.stmts[0], Bind)
  assert out.stmts[0].name == 'k'
  assert isinstance(out.stmts[0].expr, RawString)
  assert out.stmts[0].expr.text == '5'


def test_lower_mir_constant_bind_in_chain_wraps_non_block_rest():
  '''When rest_op is NOT a Block (e.g. Filter's `If`), the bind and
  the single op wrap in a fresh `Block(bind, if_op)`.'''
  cb = mir.ConstantBind(var_name='lo', code='0', deps=[])
  filt = mir.Filter(vars=['v0', 'lo'], code='return v0 > lo;')
  ins = _insert_into(vars_=['v0'])

  out = lower_mir_constant_bind_in_chain(cb, [filt, ins], _new_ctx())
  assert isinstance(out, IirBlock)
  assert isinstance(out.stmts[0], Bind)
  # Second stmt is the If from the filter — not flattened (the rest_op
  # was an If, not a Block).
  assert len(out.stmts) == 2


# -----------------------------------------------------------------------------
# 5. Registry contract — stub asserts on direct call
# -----------------------------------------------------------------------------


def test_lower_mir_constant_bind_registry_stub_asserts():
  '''The `@lowering(target=iir.cf, source=mir.ConstantBind)`
  registry entry is a stub that asserts on direct invocation —
  dispatch is expected to flow through `_lower_inner_chain` -> the
  chain-aware variant. Mirrors the C5 `lower_tiled_cartesian` split.
  '''
  cb = mir.ConstantBind(var_name='k', code='1', deps=[])
  ctx = _new_ctx()
  with pytest.raises(AssertionError, match=r'lower_mir_constant_bind_in_chain'):
    lower_mir_constant_bind(cb, ctx)


def test_lower_mir_constant_bind_is_registered_on_sorted_array_dialect():
  '''The `ConstantBind` `@lowering` is registered on the
  `relation.sorted_array` dialect. Pins dialect ownership per
  `docs/phase_b_lowering_dispatcher.md` §4 (one `@lowering` per
  MIR op, on the dialect that lowers it).
  '''
  from srdatalog.ir.dialects.relation.sorted_array import DIALECT as SA_DIALECT

  matched = [low for low in SA_DIALECT.lowerings if low.matches is mir.ConstantBind]
  assert len(matched) == 1
  assert matched[0].consumes == ('mir',)
  assert 'iir.cf' in matched[0].produces


# -----------------------------------------------------------------------------
# 6. Smoke test through the full compile_pipeline surface
# -----------------------------------------------------------------------------


def test_constant_bind_in_full_pipeline_byte_equivalent():
  '''Smoke test: a Scan + ConstantBind + InsertInto pipeline rendered
  through `compile_pipeline` (the production surface that the byte-
  equivalence harnesses guard) must produce identical CUDA under
  both `USE_DECLARATIVE` states.
  '''
  from srdatalog.compile import compile_pipeline

  scan = mir.Scan(
    vars=['x'],
    rel_name='Src',
    version=Version.FULL,
    index=[0],
    handle_start=0,
  )
  cb = mir.ConstantBind(var_name='k', code='99', deps=[])
  ins = mir.InsertInto(rel_name='Dst', version=Version.NEW, vars=['x', 'k'], index=[0, 1])
  ep = mir.ExecutePipeline(
    pipeline=[scan, cb, ins],
    source_specs=[scan],
    dest_specs=[ins],
    rule_name='CbRule',
  )

  with _force_legacy_branch():
    legacy_out = compile_pipeline(ep)
  new_out = compile_pipeline(ep)

  assert legacy_out == new_out
  assert 'auto k = 99;' in new_out


# -----------------------------------------------------------------------------
# 7. USE_DECLARATIVE invariants
# -----------------------------------------------------------------------------


def test_use_declarative_contains_filter_and_constant_bind():
  '''Pin the ratchet: after this Wave 2A PR, both `mir.Filter` and
  `mir.ConstantBind` must appear in `USE_DECLARATIVE`. Future Wave
  2A PRs add their MIR ops; the monotonic discipline test (when it
  lands) catches accidental removals.'''
  from srdatalog.ir.dialects.relation.sorted_array import USE_DECLARATIVE

  assert mir.Filter in USE_DECLARATIVE
  assert mir.ConstantBind in USE_DECLARATIVE
