'''Tests for the sorted_array terminal-wrap-op registry.

Covers the dispatcher gap revealed by PR #68 (jaccard external plugin
demo): the sorted_array chain dispatcher historically matched only
`mir.InsertInto` as the terminal op in an `ExecutePipeline.pipeline`.
External plugins whose wrap ops appear at the tail of the pipeline
were rejected by `_supported_pipeline`. The fix:

  - A module-level `_TERMINAL_WRAP_OPS: set[type]` registry on the
    dialect (`srdatalog.ir.dialects.relation.sorted_array`).
  - A public `register_terminal_wrap_op(op_type)` helper for external
    plugins to call from their `register(compiler)` callable.
  - `_trailing_inserts` / `_supported_pipeline` consult the registry,
    so any registered wrap op may appear at the tail.
  - `_lower_inner_chain` dispatches a registered wrap op at the head
    through the dialect's `@lowering` registry for that op type.

This test exercises:

  1. Registry shape — pre-populated with built-in wrap ops, idempotent
     re-registration, accepts arbitrary types.
  2. Predicate behaviour — `_supported_pipeline` accepts a synthetic
     EP whose tail is a registered mock wrap op.
  3. End-to-end dispatch — `compile_pipeline(ep)` lowers a synthetic
     EP with a registered mock wrap op at the tail (the integration
     gate from the task brief).

PR #68's jaccard demo is INTENTIONALLY NOT exercised here — that demo
isn't merged into design/redesign-package, so its sources are not
present in the worktree. The synthetic mock wrap op below is enough
to gate the dispatcher contract; the jaccard demo will land once PR
#68 merges and `srdatalog_jaccard.register` adds a
`register_terminal_wrap_op(JaccardIndex)` call.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import pytest

import srdatalog.ir.mir.types as m
from srdatalog.ir.codegen.cuda.api import compile_pipeline
from srdatalog.ir.core import Op
from srdatalog.ir.core.passes import lowering
from srdatalog.ir.dialects.relation.sorted_array import (
  _TERMINAL_WRAP_OPS,
  register_terminal_wrap_op,
)
from srdatalog.ir.dialects.relation.sorted_array import (
  DIALECT as SA_DIALECT,
)
from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
  _is_terminal_op,
  _supported_pipeline,
  _trailing_inserts,
)
from srdatalog.ir.hir.types import Version

# -----------------------------------------------------------------------------
# Synthetic mock wrap op + lowering
# -----------------------------------------------------------------------------
#
# A minimal stand-in for an external plugin's terminal wrap op. The
# dialect knows nothing about `MockExtensionWrap` until the test
# registers it; mirrors what an external plugin's `register(compiler)`
# would do.


@final
@dataclass(frozen=True, slots=True)
class MockExtensionWrap(Op):
  '''Synthetic terminal wrap op for the registry test.

  Wraps an `mir.InsertInto` and is lowered by delegating straight
  back to the dialect's `_lower_insert_into` helper, so the resulting
  IIR is byte-equivalent to a bare `InsertInto` for the same inner
  op. The exact IIR isn't load-bearing here — what matters is that
  the dispatcher routes through the registered `@lowering` instead
  of raising `unsupported inner op`.
  '''

  inner: m.InsertInto


def _lower_mock_extension_wrap(op, ctx):
  '''Mock @lowering body: delegate to the dialect's
  `_lower_insert_into` so the rendered CUDA matches the no-wrap
  emission for the same `InsertInto`.
  '''
  from srdatalog.ir.dialects.iir.cf import Block
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _lower_insert_into,
  )

  stmts = _lower_insert_into(op.inner, ctx)
  return Block(stmts=tuple(stmts))


# Wire the mock wrap op into the dialect ONCE per process — the
# discipline tests in `tests/test_discipline_*.py` assert exactly one
# `@lowering` per op type, so re-registering would trip
# `AmbiguousLowering`. The module-level guard below short-circuits
# re-imports under `pytest`'s plugin discovery.

_MOCK_REGISTERED = False


def _ensure_mock_registered() -> None:
  global _MOCK_REGISTERED
  if _MOCK_REGISTERED:
    return
  # The @lowering decorator appends to DIALECT.lowerings; guard
  # against double-registration (e.g. test re-runs in the same
  # process) by checking first.
  already = any(low.matches is MockExtensionWrap for low in SA_DIALECT.lowerings)
  if not already:
    lowering(
      SA_DIALECT,
      MockExtensionWrap,
      consumes=('mir',),
      produces=('iir.cf',),
    )(_lower_mock_extension_wrap)
  register_terminal_wrap_op(MockExtensionWrap)
  _MOCK_REGISTERED = True


# -----------------------------------------------------------------------------
# 1. Registry shape — built-ins + idempotency + arbitrary types
# -----------------------------------------------------------------------------


def test_builtin_wrap_ops_are_preregistered():
  '''The dialect pre-populates the registry with the three built-in
  C-pragma wrap ops at module-load time. This is the fallback that
  keeps byte-equivalence intact once Phase A3 drops the dual-write
  bool fields.
  '''
  assert m.DedupGate in _TERMINAL_WRAP_OPS
  assert m.WSScope in _TERMINAL_WRAP_OPS
  assert m.FanOut in _TERMINAL_WRAP_OPS


def test_register_terminal_wrap_op_adds_op_type():
  '''An external plugin's wrap op type lands in the registry after
  `register_terminal_wrap_op(...)` is called. Stand-in for the
  jaccard demo's `register(compiler)` callable.
  '''

  @final
  @dataclass(frozen=True, slots=True)
  class _NeverSeenWrap(Op):
    pass

  assert _NeverSeenWrap not in _TERMINAL_WRAP_OPS
  register_terminal_wrap_op(_NeverSeenWrap)
  assert _NeverSeenWrap in _TERMINAL_WRAP_OPS


def test_register_terminal_wrap_op_is_idempotent():
  '''Re-registering the same op type is a no-op (set semantics).
  External plugins may re-run `register(compiler)` across multiple
  Compilers in the same process; the registry must tolerate that
  without growing duplicate entries.
  '''

  @final
  @dataclass(frozen=True, slots=True)
  class _IdempotentWrap(Op):
    pass

  register_terminal_wrap_op(_IdempotentWrap)
  size_after_first = len(_TERMINAL_WRAP_OPS)
  register_terminal_wrap_op(_IdempotentWrap)
  register_terminal_wrap_op(_IdempotentWrap)
  assert len(_TERMINAL_WRAP_OPS) == size_after_first


# -----------------------------------------------------------------------------
# 2. Predicate behaviour — _supported_pipeline / _trailing_inserts /
#    _is_terminal_op accept the mock wrap op
# -----------------------------------------------------------------------------


def _make_scan() -> m.Scan:
  return m.Scan(
    vars=['v0', 'v1'],
    rel_name='Src',
    version=Version.FULL,
    index=[0, 1],
    handle_start=0,
  )


def _make_insert() -> m.InsertInto:
  return m.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['v0', 'v1'],
    index=[0, 1],
  )


def test_is_terminal_op_matches_insert_into():
  assert _is_terminal_op(_make_insert()) is True


def test_is_terminal_op_matches_registered_wrap_op():
  _ensure_mock_registered()
  wrap = MockExtensionWrap(inner=_make_insert())
  assert _is_terminal_op(wrap) is True


def test_is_terminal_op_rejects_unregistered_op():
  scan = _make_scan()
  assert _is_terminal_op(scan) is False


def test_trailing_inserts_collects_mixed_terminal_run():
  '''The trailing-run walker collects bare InsertInto AND registered
  wrap ops. Order is preserved.
  '''
  _ensure_mock_registered()
  insert = _make_insert()
  wrap = MockExtensionWrap(inner=_make_insert())
  pipeline = [_make_scan(), insert, wrap]
  trailing = _trailing_inserts(pipeline)
  assert trailing == [insert, wrap]


def test_supported_pipeline_accepts_wrap_op_terminal():
  '''A pipeline shape `[Scan, MockExtensionWrap]` is structurally
  legal once the mock wrap op is registered. Pre-fix, this returned
  False because the trailing-tail predicate only matched
  `mir.InsertInto`.
  '''
  _ensure_mock_registered()
  pipeline = [_make_scan(), MockExtensionWrap(inner=_make_insert())]
  assert _supported_pipeline(pipeline) is True


def test_supported_pipeline_rejects_unregistered_wrap_at_tail():
  '''Sanity: an unregistered op type at the tail still fails the
  predicate — the registry is the sole extension point.
  '''

  @final
  @dataclass(frozen=True, slots=True)
  class _UnregisteredTail(Op):
    inner: m.InsertInto

  pipeline = [_make_scan(), _UnregisteredTail(inner=_make_insert())]
  assert _supported_pipeline(pipeline) is False


# -----------------------------------------------------------------------------
# 3. End-to-end dispatch via compile_pipeline
# -----------------------------------------------------------------------------


def _ep_with_mock_tail() -> m.ExecutePipeline:
  '''Build a minimal EP whose pipeline ends in a registered mock
  wrap op. Mirrors the shape an external plugin would produce after
  its pragma's materialize handler wraps the bare `InsertInto`.
  '''
  scan = _make_scan()
  insert = _make_insert()
  wrap = MockExtensionWrap(inner=insert)
  return m.ExecutePipeline(
    pipeline=[scan, wrap],
    source_specs=[scan],
    dest_specs=[insert],
    rule_name='MockRule',
  )


def test_compile_pipeline_dispatches_registered_wrap_op_at_tail():
  '''Integration: `compile_pipeline(ep)` lowers a pipeline ending in
  a registered terminal wrap op end-to-end. This is the load-bearing
  assertion from PR #68's contract-gap analysis — pre-fix, this call
  raised `ValueError: lower_scan_pipeline: unsupported pipeline shape`
  (or `unsupported inner op` from `_lower_inner_chain`).

  The emitted CUDA contains the InsertInto's emission (the mock
  lowering delegates to `_lower_insert_into`). Exact byte content
  isn't asserted — that belongs to per-pragma byte-equivalence
  fixtures.
  '''
  _ensure_mock_registered()
  ep = _ep_with_mock_tail()
  out = compile_pipeline(ep)
  # The mock wrap delegates to the bare-InsertInto emission, so the
  # rendered CUDA must contain the standard emit_direct call shape
  # for the inner `InsertInto`. We don't pin the exact text — just
  # that the dispatcher reached the inner emission.
  assert 'emit_direct' in out or 'output' in out
  # The wrap op type itself never appears in rendered CUDA (it is a
  # MIR-layer artifact, lowered before render). Sanity guard.
  assert 'MockExtensionWrap' not in out


def test_compile_pipeline_unregistered_wrap_still_rejected():
  '''The registry is the sole extension point: an EP whose tail is
  an unregistered op type still fails loudly at lowering time,
  matching the pre-fix behaviour. Guards against accidental over-
  acceptance of arbitrary structural shapes.
  '''

  @final
  @dataclass(frozen=True, slots=True)
  class _NotRegisteredAtTail(Op):
    inner: m.InsertInto

  scan = _make_scan()
  insert = _make_insert()
  ep = m.ExecutePipeline(
    pipeline=[scan, _NotRegisteredAtTail(inner=insert)],
    source_specs=[scan],
    dest_specs=[insert],
    rule_name='UnregRule',
  )
  with pytest.raises(ValueError, match=r'unsupported pipeline shape'):
    compile_pipeline(ep)
