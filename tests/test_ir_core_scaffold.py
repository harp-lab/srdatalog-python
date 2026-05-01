'''Smoke tests for the dialect-agnostic IR framework.

Stage 1 gate: register a no-op dialect, run the pass driver, verify
the registry rejects duplicates and exposes registered dialects in
order. These don't exercise lowering/rewrite logic — those arrive
with the first real dialect.
'''

import pytest

from srdatalog.ir_core import (
  Compiler,
  Dialect,
  Lowering,
  Op,
  PassDriver,
  Rewrite,
  Type,
  VerificationError,
)


def test_compiler_starts_empty():
  c = Compiler()
  assert c.dialects == []


def test_register_dialect_records_it():
  c = Compiler()
  d = Dialect(name='test.no_op')
  c.register_dialect(d)
  assert c.dialects == [d]
  assert c.get_dialect('test.no_op') is d


def test_register_preserves_order():
  c = Compiler()
  a = Dialect(name='test.a')
  b = Dialect(name='test.b')
  c_d = Dialect(name='test.c')
  c.register_dialect(a)
  c.register_dialect(b)
  c.register_dialect(c_d)
  assert c.dialects == [a, b, c_d]


def test_duplicate_dialect_rejected():
  c = Compiler()
  c.register_dialect(Dialect(name='test.no_op'))
  with pytest.raises(ValueError, match=r"'test\.no_op' already registered"):
    c.register_dialect(Dialect(name='test.no_op'))


def test_get_unknown_dialect_raises():
  c = Compiler()
  with pytest.raises(KeyError):
    c.get_dialect('nonexistent')


def test_pass_driver_runs_with_no_dialects():
  '''Driver must work with an empty registry — important for Stage 1
  bootstrapping, where no dialects are wired in yet.'''
  driver = PassDriver(Compiler())
  prog = object()
  assert driver.run(prog) is prog


def test_pass_driver_runs_with_no_op_dialect():
  '''Registering a dialect with no passes must not break the driver.'''
  c = Compiler()
  c.register_dialect(Dialect(name='test.no_op'))
  driver = PassDriver(c)
  prog = object()
  assert driver.run(prog) is prog


def test_pass_driver_walks_multiple_dialects():
  '''Multiple registered dialects, each with empty pass lists, must
  all be walked without error.'''
  c = Compiler()
  c.register_dialect(Dialect(name='test.a'))
  c.register_dialect(Dialect(name='test.b'))
  c.register_dialect(Dialect(name='test.c'))
  driver = PassDriver(c)
  assert driver.run('prog-token') == 'prog-token'


def test_dialect_carries_op_and_type_classes():
  '''A dialect can list its Op and Type subclasses; the registry
  preserves them. Real dialects use this for verifier dispatch.'''

  class MyOp(Op):
    pass

  class MyType(Type):
    pass

  d = Dialect(name='test.with_kinds', ops=[MyOp], types=[MyType])
  c = Compiler()
  c.register_dialect(d)
  assert c.get_dialect('test.with_kinds').ops == [MyOp]
  assert c.get_dialect('test.with_kinds').types == [MyType]


def test_lowering_and_rewrite_are_registrable():
  '''Lowerings and Rewrites are values that go into a Dialect's lists.
  This test pins down the constructor shape so dialects have a stable
  API to build against.'''

  class SrcOp(Op):
    pass

  class DstOp(Op):
    pass

  def lower_src(op, ctx):
    return [DstOp()]

  def rewrite_src(op, ctx):
    return [SrcOp()]

  lowering = Lowering(matches=SrcOp, apply=lower_src, name='src->dst')
  rewrite = Rewrite(matches=SrcOp, apply=rewrite_src, name='src-canonical')

  d = Dialect(
    name='test.with_passes',
    ops=[SrcOp, DstOp],
    lowerings=[lowering],
    rewrites=[rewrite],
  )
  c = Compiler()
  c.register_dialect(d)
  registered = c.get_dialect('test.with_passes')
  assert registered.lowerings == [lowering]
  assert registered.rewrites == [rewrite]


def test_verification_error_str():
  '''VerificationError stringifies with the op's class name and the
  message — the format the pass driver will use for diagnostics.'''

  class FakeOp:
    pass

  err = VerificationError(op=FakeOp(), message='wrong shape')
  assert 'FakeOp' in str(err)
  assert 'wrong shape' in str(err)
