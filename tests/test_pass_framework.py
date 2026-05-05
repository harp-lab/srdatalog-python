'''Bundle B framework tests.

Pins the contract for:
  - @lowering / @rewrite / @verifier decorators (S3A.6 part 1)
  - Lowering / Rewrite consumes/produces fields
  - PassDriver.validate_dependencies — raises PassDependencyError
    when a registered pass declares consumes=(D,) and D is not
    in the Compiler's registry
  - PassDriver.verify_all — invokes each dialect's verifier
  - sorted_array's MIR→IIR entry point is registered (S3A.4)
  - every existing dialect ships at least a no-op verifier (S3A.7)
'''

from __future__ import annotations

import pytest

from srdatalog.ir.core import Compiler, Dialect, Op
from srdatalog.ir.core.passes import (
  Lowering,
  PassDependencyError,
  PassDriver,
  Rewrite,
  lowering,
  rewrite,
  verifier,
)

# -----------------------------------------------------------------------------
# Decorators (S3A.6 part 1)
# -----------------------------------------------------------------------------


class _ToyOp(Op):
  pass


def test_lowering_decorator_registers_on_dialect():
  d = Dialect(name='toy.lowering')
  assert d.lowerings == []

  @lowering(d, _ToyOp, consumes=('a',), produces=('b',))
  def _l(op, ctx):
    return [op]

  assert len(d.lowerings) == 1
  l = d.lowerings[0]
  assert isinstance(l, Lowering)
  assert l.matches is _ToyOp
  assert l.consumes == ('a',)
  assert l.produces == ('b',)
  assert l.name == '_l'


def test_rewrite_decorator_registers_on_dialect():
  d = Dialect(name='toy.rewrite')
  assert d.rewrites == []

  @rewrite(d, _ToyOp, name='count_as_product')
  def _r(op, ctx):
    return op

  assert len(d.rewrites) == 1
  r = d.rewrites[0]
  assert isinstance(r, Rewrite)
  assert r.matches is _ToyOp
  assert r.name == 'count_as_product'


def test_verifier_decorator_attaches_to_dialect():
  d = Dialect(name='toy.verifier')
  assert d.verifier is None

  @verifier(d)
  def _v(_):
    return ['err']

  assert d.verifier is _v
  assert d.verifier(None) == ['err']


def test_verifier_decorator_rejects_double_registration():
  d = Dialect(name='toy.verifier.dup')

  @verifier(d)
  def _v1(_):
    return []

  with pytest.raises(ValueError, match='already registered'):

    @verifier(d)
    def _v2(_):
      return []


# -----------------------------------------------------------------------------
# PassDriver.validate_dependencies (S3A.6 part 2)
# -----------------------------------------------------------------------------


def test_pass_driver_passes_with_satisfied_deps():
  c = Compiler()
  source = Dialect(name='toy.source')
  target = Dialect(name='toy.target')

  @lowering(target, _ToyOp, consumes=('toy.source',))
  def _l(op, ctx):
    return [op]

  c.register_dialect(source)
  c.register_dialect(target)

  pd = PassDriver(c)
  pd.validate_dependencies()  # should not raise


def test_pass_driver_raises_on_unmet_dep():
  c = Compiler()
  target = Dialect(name='toy.target.unmet')

  @lowering(target, _ToyOp, consumes=('toy.missing',))
  def _l(op, ctx):
    return [op]

  c.register_dialect(target)

  pd = PassDriver(c)
  with pytest.raises(PassDependencyError) as ei:
    pd.validate_dependencies()
  assert ei.value.missing_dialect == 'toy.missing'
  assert ei.value.in_dialect == 'toy.target.unmet'
  assert 'toy.missing' in str(ei.value)


def test_pass_driver_run_validates_then_verifies():
  '''Smoke: PassDriver.run validates deps then runs verifiers and
  returns prog unchanged.'''
  c = Compiler()
  d = Dialect(name='toy.runner')

  @verifier(d)
  def _v(_):
    return []

  c.register_dialect(d)
  pd = PassDriver(c)
  out = pd.run('hello')
  assert out == 'hello'


# -----------------------------------------------------------------------------
# Production dialect registrations (S3A.4 + S3A.7)
# -----------------------------------------------------------------------------


def _all_production_dialects():
  '''Every framework-tracked dialect in srdatalog.ir.'''
  from srdatalog.ir.dialects.iir.cf import DIALECT as cf_d
  from srdatalog.ir.dialects.parallel.data import DIALECT as pd_d
  from srdatalog.ir.dialects.relation.d2l import DIALECT as d2l_d
  from srdatalog.ir.dialects.relation.sorted_array import DIALECT as sa_d
  from srdatalog.ir.hir import DIALECT as hir_d
  from srdatalog.ir.mir import DIALECT as mir_d

  return [hir_d, mir_d, cf_d, sa_d, d2l_d, pd_d]


def test_every_production_dialect_has_a_verifier():
  '''S3A.7 acceptance gate: every framework-tracked dialect ships at
  least a no-op verifier so PassDriver.verify_all has something to
  call. Real per-dialect invariants land incrementally.'''
  for d in _all_production_dialects():
    assert d.verifier is not None, f'dialect {d.name!r} missing verifier'
    # Smoke: verifier callable + returns iterable
    assert d.verifier(None) == [], f'dialect {d.name!r} verifier should return [] today'


def test_sorted_array_lowering_registered():
  '''S3A.4: sorted_array's MIR→IIR entry is registered as a Lowering.'''
  import srdatalog.ir.mir.types as mir
  from srdatalog.ir.dialects.relation.sorted_array import DIALECT

  matches = [l for l in DIALECT.lowerings if l.matches is mir.ExecutePipeline]
  assert len(matches) == 1, (
    f'expected 1 Lowering matching mir.ExecutePipeline on sorted_array.DIALECT, got {len(matches)}'
  )
  l = matches[0]
  assert l.consumes == ('mir',)
  assert 'iir.cf' in l.produces
  assert 'relation.sorted_array' in l.produces


def test_all_production_dialects_register_with_a_compiler():
  '''Smoke: a fresh Compiler can hold every production dialect; deps
  validate cleanly (sorted_array's consumes=('mir',) is satisfied
  because mir is in the set).'''
  c = Compiler()
  for d in _all_production_dialects():
    c.register_dialect(d)

  pd = PassDriver(c)
  pd.validate_dependencies()  # should not raise
  assert pd.verify_all(None) == []
