'''F1 / Layer 1 foundation tests — `Pass` kinds + `Compiler.run`.

Covers the contract from `docs/compiler_redesign.md` §4 (the three
Pass kinds) and `docs/phase_b_lowering_dispatcher.md` §2 (LoweringPass
algorithm) and `docs/phase_d_hir_passes.md` §2 (ProgramPass contract).

This file is self-contained: synthetic `Op` + `Dialect` fixtures, no
import of any production dialect. F1 ships only the dispatcher
shape; the full 5-field LowerCtx (compiler_redesign.md §5) lands in F3.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from srdatalog.ir.core import (
  AmbiguousLowering,
  Compiler,
  Dialect,
  LoweringMissingError,
  LoweringPass,
  Op,
  Pass,
  PassOrderingError,
  ProgramPass,
  RewritePass,
  program_pass,
)
from srdatalog.ir.core.passes import lowering, rewrite

# -----------------------------------------------------------------------------
# Synthetic op fixtures (no production dialect imports)
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SrcLeaf(Op):
  '''Source-dialect leaf op carrying an integer payload.'''

  value: int


@dataclass(frozen=True, slots=True)
class _SrcWrap(Op):
  '''Source-dialect wrapper holding one inner source op (used to
  exercise explicit ctx.lower(child) recursion).'''

  inner: Any


@dataclass(frozen=True, slots=True)
class _TgtLeaf(Op):
  '''Target-dialect leaf op carrying the lowered payload.'''

  value: int


@dataclass(frozen=True, slots=True)
class _TgtWrap(Op):
  '''Target-dialect wrapper holding one lowered inner op.'''

  inner: Any


@dataclass(frozen=True, slots=True)
class _SameDialectOp(Op):
  '''Op used to exercise RewritePass intra-dialect rewriting. The
  rewrite halves `n` until 0; fixpoint when n=0.'''

  n: int


# -----------------------------------------------------------------------------
# Pass abstractness (T1)
# -----------------------------------------------------------------------------


def test_pass_is_abstract():
  '''Instantiating `Pass(...)` directly fails (abstract base).'''
  with pytest.raises(TypeError, match="abstract"):
    Pass(name='bad')  # type: ignore[abstract]


# -----------------------------------------------------------------------------
# LoweringPass dispatch (T2 - T5)
# -----------------------------------------------------------------------------


def _src_dialect() -> Dialect:
  d = Dialect(name='test.src')
  d.ops = [_SrcLeaf, _SrcWrap]
  return d


def _tgt_dialect() -> Dialect:
  d = Dialect(name='test.tgt')
  d.ops = [_TgtLeaf, _TgtWrap]
  return d


def test_lowering_pass_dispatches_registered_lowering():
  '''A registered @lowering for the source op type is found and
  applied by LoweringPass.apply.'''
  src = _src_dialect()
  tgt = _tgt_dialect()

  @lowering(tgt, _SrcLeaf, consumes=('test.src',), produces=('test.tgt',))
  def _l_leaf(op, ctx):
    return _TgtLeaf(value=op.value * 10)

  c = Compiler()
  c.register_dialect(src)
  c.register_dialect(tgt)

  p = LoweringPass(
    name='src_to_tgt',
    consumes=('test.src',),
    produces=('test.tgt',),
    target_dialect_name='test.tgt',
  )
  out = p.apply(_SrcLeaf(value=3), c)
  assert out == _TgtLeaf(value=30)


def test_lowering_pass_missing_handler_raises():
  '''An op whose type has no registered @lowering raises
  LoweringMissingError at dispatch time.'''
  src = _src_dialect()
  tgt = _tgt_dialect()

  c = Compiler()
  c.register_dialect(src)
  c.register_dialect(tgt)

  p = LoweringPass(
    name='src_to_tgt_empty',
    consumes=('test.src',),
    produces=('test.tgt',),
    target_dialect_name='test.tgt',
  )
  with pytest.raises(LoweringMissingError) as ei:
    p.apply(_SrcLeaf(value=1), c)
  assert ei.value.op_type is _SrcLeaf
  assert ei.value.target_dialect_name == 'test.tgt'


def test_lowering_pass_ambiguous_registration_raises():
  '''Two @lowering for the same source op type on the same target
  dialect raises AmbiguousLowering at apply-time table build.'''
  src = _src_dialect()
  tgt = _tgt_dialect()

  @lowering(tgt, _SrcLeaf)
  def _l1(op, ctx):
    return _TgtLeaf(value=op.value)

  @lowering(tgt, _SrcLeaf)
  def _l2(op, ctx):
    return _TgtLeaf(value=op.value + 1)

  c = Compiler()
  c.register_dialect(src)
  c.register_dialect(tgt)

  p = LoweringPass(
    name='src_to_tgt_amb',
    consumes=('test.src',),
    produces=('test.tgt',),
    target_dialect_name='test.tgt',
  )
  with pytest.raises(AmbiguousLowering) as ei:
    p.apply(_SrcLeaf(value=1), c)
  assert ei.value.source_op_type is _SrcLeaf
  assert ei.value.target_dialect_name == 'test.tgt'


def test_lowering_pass_explicit_recursion():
  '''A lowering that calls ctx.lower(child) recursively works on a
  synthetic op tree (per R1: explicit recursion, not auto-walk).'''
  src = _src_dialect()
  tgt = _tgt_dialect()

  @lowering(tgt, _SrcLeaf)
  def _l_leaf(op, ctx):
    return _TgtLeaf(value=op.value + 100)

  @lowering(tgt, _SrcWrap)
  def _l_wrap(op, ctx):
    # Explicit recursion into the child — no auto-walk.
    return _TgtWrap(inner=ctx.lower(op.inner))

  c = Compiler()
  c.register_dialect(src)
  c.register_dialect(tgt)

  p = LoweringPass(
    name='src_to_tgt_rec',
    consumes=('test.src',),
    produces=('test.tgt',),
    target_dialect_name='test.tgt',
  )
  tree = _SrcWrap(inner=_SrcWrap(inner=_SrcLeaf(value=7)))
  out = p.apply(tree, c)
  assert out == _TgtWrap(inner=_TgtWrap(inner=_TgtLeaf(value=107)))


# -----------------------------------------------------------------------------
# RewritePass (T6)
# -----------------------------------------------------------------------------


def test_rewrite_pass_runs_to_fixpoint():
  '''A registered @rewrite is applied to fixpoint by RewritePass.

  We register a rule that halves `n` (rounding toward zero) on
  _SameDialectOp; starting from n=8, fixpoint is n=0.
  '''
  d = Dialect(name='test.rw')
  d.ops = [_SameDialectOp]

  @rewrite(d, _SameDialectOp)
  def _halve(op, ctx):
    if op.n == 0:
      return None
    return _SameDialectOp(n=op.n // 2)

  c = Compiler()
  c.register_dialect(d)

  p = RewritePass(
    name='halve_to_zero',
    consumes=('test.rw',),
    produces=('test.rw',),
    dialect_name='test.rw',
    until_fixpoint=True,
  )
  out = p.apply(_SameDialectOp(n=8), c)
  assert out == _SameDialectOp(n=0)


def test_rewrite_pass_one_shot_when_no_fixpoint():
  '''until_fixpoint=False applies one bottom-up sweep only.'''
  d = Dialect(name='test.rw.once')
  d.ops = [_SameDialectOp]

  @rewrite(d, _SameDialectOp)
  def _halve(op, ctx):
    if op.n == 0:
      return None
    return _SameDialectOp(n=op.n // 2)

  c = Compiler()
  c.register_dialect(d)

  p = RewritePass(
    name='halve_once',
    consumes=('test.rw.once',),
    produces=('test.rw.once',),
    dialect_name='test.rw.once',
    until_fixpoint=False,
  )
  out = p.apply(_SameDialectOp(n=8), c)
  assert out == _SameDialectOp(n=4)


# -----------------------------------------------------------------------------
# ProgramPass + program_pass decorator (T7, T8)
# -----------------------------------------------------------------------------


def test_program_pass_invokes_fn_once_with_prog_and_compiler():
  '''A ProgramPass with a closure verifies the fn is called once
  with (prog, compiler) and its return is threaded out.'''
  call_log: list[tuple[Any, Any]] = []

  def _fn(prog, compiler):
    call_log.append((prog, compiler))
    return ('transformed', prog)

  c = Compiler()
  p = ProgramPass(name='hello', consumes=(), produces=(), fn=_fn)
  out = p.apply('input', c)

  assert call_log == [('input', c)]
  assert out == ('transformed', 'input')


def test_program_pass_decorator_returns_program_pass_instance():
  '''@program_pass(...) wraps a function as a ProgramPass instance
  with the declared name/consumes/produces.'''

  @program_pass(name='stratify', consumes=('hir',), produces=('hir',))
  def stratify_pass(prog, compiler):
    return prog

  assert isinstance(stratify_pass, ProgramPass)
  assert stratify_pass.name == 'stratify'
  assert stratify_pass.consumes == ('hir',)
  assert stratify_pass.produces == ('hir',)
  # Smoke: invoking it threads the prog through.
  c = Compiler()
  assert stratify_pass.apply('hir-prog', c) == 'hir-prog'


# -----------------------------------------------------------------------------
# Compiler.run pipeline composition (T9)
# -----------------------------------------------------------------------------


def test_compiler_run_pipeline_threads_prog_through_passes():
  '''A pipeline of [LoweringPass, RewritePass, ProgramPass] runs in
  order, threading prog through each.'''
  src = _src_dialect()
  tgt = _tgt_dialect()

  # Tag tgt as also a "rewriting" dialect.
  @lowering(tgt, _SrcLeaf)
  def _l_leaf(op, ctx):
    return _TgtLeaf(value=op.value + 1)

  @rewrite(tgt, _TgtLeaf)
  def _bump_until_5(op, ctx):
    if op.value >= 5:
      return None
    return _TgtLeaf(value=op.value + 1)

  c = Compiler()
  c.register_dialect(src)
  c.register_dialect(tgt)

  trace: list[str] = []

  lp = LoweringPass(
    name='lp',
    consumes=('test.src',),
    produces=('test.tgt',),
    target_dialect_name='test.tgt',
  )
  rp = RewritePass(
    name='rp',
    consumes=('test.tgt',),
    produces=('test.tgt',),
    dialect_name='test.tgt',
    until_fixpoint=True,
  )

  def _final(prog, compiler):
    trace.append(f'final({prog!r})')
    return ('done', prog)

  pp = ProgramPass(name='pp', consumes=('test.tgt',), produces=('test.tgt',), fn=_final)

  out = c.run(_SrcLeaf(value=2), pipeline=[lp, rp, pp])
  # 2 → lower → TgtLeaf(3) → rewrite-to-fixpoint → TgtLeaf(5) → pp → ('done', TgtLeaf(5))
  assert out == ('done', _TgtLeaf(value=5))
  assert trace == ['final(_TgtLeaf(value=5))']


# -----------------------------------------------------------------------------
# Pass ordering validation (T10, T11)
# -----------------------------------------------------------------------------


def test_compiler_run_raises_pass_ordering_error_for_unmet_consumes():
  '''A pipeline whose Pass requires a dialect that no earlier Pass
  produces (and that isn't registered with the Compiler) raises
  PassOrderingError at construction-time validation, before any
  Pass.apply runs.'''
  c = Compiler()  # no dialects registered

  applied: list[str] = []

  def _record(name):
    def _fn(prog, compiler):
      applied.append(name)
      return prog

    return _fn

  p1 = ProgramPass(name='p1', consumes=(), produces=('a',), fn=_record('p1'))
  # p2 requires 'b' but neither registered nor produced earlier.
  p2 = ProgramPass(name='p2', consumes=('b',), produces=(), fn=_record('p2'))

  with pytest.raises(PassOrderingError) as ei:
    c.run('prog', pipeline=[p1, p2])
  assert ei.value.pass_name == 'p2'
  assert ei.value.missing == 'b'
  assert ei.value.position == 1
  # No pass should have been applied.
  assert applied == []


def test_compiler_run_ordering_satisfied_by_compiler_registration():
  '''Pipeline Pass with consumes=('test.src',) is satisfied because
  the dialect is registered on the Compiler at run start.'''
  src = _src_dialect()
  c = Compiler()
  c.register_dialect(src)

  applied: list[str] = []

  def _fn(prog, compiler):
    applied.append('p1')
    return prog

  p1 = ProgramPass(name='p1', consumes=('test.src',), produces=(), fn=_fn)
  out = c.run('hello', pipeline=[p1])
  assert out == 'hello'
  assert applied == ['p1']


def test_compiler_run_ordering_satisfied_by_earlier_pass():
  '''A Pass whose consumes is satisfied by an earlier Pass's produces
  (rather than a registered dialect) passes validation.'''
  c = Compiler()  # no dialects registered

  p1 = ProgramPass(name='p1', consumes=(), produces=('synthetic',), fn=lambda prog, _c: prog)
  p2 = ProgramPass(name='p2', consumes=('synthetic',), produces=(), fn=lambda prog, _c: prog)
  # Should not raise.
  out = c.run('prog', pipeline=[p1, p2])
  assert out == 'prog'
