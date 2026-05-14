'''S4.9 — `PassDriver.apply_rewrites_to_fixpoint` + `verify_renderability`.

Per `docs/ir_dialect_contract.md` §2 and §3. The driver gains:

  * `apply_rewrites_to_fixpoint` — bottom-up + repeat over registered
    `@rewrite` rules until fixpoint.
  * `verify_renderability` — every op surviving fixpoint has a
    registered renderer; loud failure (`UnrenderableOpError`)
    otherwise.

These tests exercise the framework with toy dialects so the cases
stay tight. End-to-end byte-equivalence with the production CUDA
pipeline is covered by the existing test suite (S4.9 is a no-op for
production today: no `@rewrite`s are registered, fixpoint converges
in one pass, and every op is renderable).
'''

from __future__ import annotations

from dataclasses import dataclass

import pytest

from srdatalog.ir.core import (
  Compiler,
  Dialect,
  Op,
  PassDriver,
  RewriteRegistrationConflict,
  UnrenderableOpError,
)
from srdatalog.ir.core.passes import _walk, rewrite

# -----------------------------------------------------------------------------
# Toy ops + dialect for the tests
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Leaf(Op):
  '''A LEAF op — has a renderer, no rewrite.'''

  name: str


@dataclass(frozen=True, slots=True)
class Compound(Op):
  '''A COMPOUND op — has a rewrite that decomposes to a Leaf.'''

  text: str


@dataclass(frozen=True, slots=True)
class Container(Op):
  '''A LEAF op holding a tuple of children.'''

  children: tuple[Op, ...]


@dataclass(frozen=True, slots=True)
class TwoLevel(Op):
  '''A COMPOUND op that rewrites to a Container holding *another*
  Compound — exercises bottom_up + repeat (the inner Compound is
  picked up on the next iteration).'''

  inner_text: str


def _fresh_compiler_with_dialect():
  '''Build a fresh Compiler + a fresh Dialect each test, so registrations
  don't bleed between cases.'''
  d = Dialect(name='toy')
  c = Compiler()
  c.register_dialect(d)
  return c, d


# -----------------------------------------------------------------------------
# _walk
# -----------------------------------------------------------------------------


def test_walk_yields_root_then_descendants_preorder():
  tree = Container(children=(Leaf(name='a'), Container(children=(Leaf(name='b'),))))
  yielded = list(_walk(tree))
  assert yielded == [
    tree,
    tree.children[0],
    tree.children[1],
    tree.children[1].children[0],
  ]


def test_walk_skips_non_op_fields():
  '''Op subclasses can have str / int fields; walk must not recurse
  into them as if they were children.'''
  yielded = list(_walk(Leaf(name='only-name-no-children')))
  assert len(yielded) == 1


def test_walk_descends_into_tuple_fields():
  tree = Container(children=(Leaf(name='a'), Leaf(name='b'), Leaf(name='c')))
  ops = list(_walk(tree))
  assert len(ops) == 4
  assert all(isinstance(o, Op) for o in ops)


def test_walk_on_non_op_returns_empty():
  assert list(_walk('not-an-op')) == []
  assert list(_walk(42)) == []


# -----------------------------------------------------------------------------
# apply_rewrites_to_fixpoint
# -----------------------------------------------------------------------------


def test_fixpoint_is_identity_with_no_rewrites():
  c, _ = _fresh_compiler_with_dialect()
  driver = PassDriver(c)
  tree = Container(children=(Leaf(name='x'),))
  out = driver.apply_rewrites_to_fixpoint(tree)
  assert out is tree


def test_fixpoint_decomposes_compound_to_leaf():
  c, d = _fresh_compiler_with_dialect()

  @rewrite(d, Compound)
  def _decompose(op, _ctx):
    return Leaf(name=op.text)

  driver = PassDriver(c)
  tree = Container(children=(Compound(text='alpha'), Leaf(name='beta')))
  out = driver.apply_rewrites_to_fixpoint(tree)
  assert out == Container(children=(Leaf(name='alpha'), Leaf(name='beta')))


def test_fixpoint_iterates_when_rewrite_emits_compound():
  '''TwoLevel → Container(Compound, ...). The next iteration picks
  up the Compound and decomposes it. Validates bottom_up + repeat.'''
  c, d = _fresh_compiler_with_dialect()

  @rewrite(d, TwoLevel)
  def _level1(op, _ctx):
    return Container(children=(Compound(text=op.inner_text),))

  @rewrite(d, Compound)
  def _level2(op, _ctx):
    return Leaf(name=op.text)

  driver = PassDriver(c)
  out = driver.apply_rewrites_to_fixpoint(TwoLevel(inner_text='x'))
  assert out == Container(children=(Leaf(name='x'),))


def test_fixpoint_raises_on_divergent_rewrite():
  c, d = _fresh_compiler_with_dialect()

  @rewrite(d, Compound)
  def _grow_forever(op, _ctx):
    return Compound(text=op.text + '+')

  driver = PassDriver(c)
  with pytest.raises(RuntimeError, match='did not converge'):
    driver.apply_rewrites_to_fixpoint(Compound(text='x'), max_iters=5)


def test_two_rewrites_for_same_op_type_raises_conflict():
  '''Per contract §2.2: at most one rewrite per op type. The conflict
  is detected at use (build_rewrite_table), not at decoration.'''
  c, d = _fresh_compiler_with_dialect()
  d2 = Dialect(name='toy.other')
  c.register_dialect(d2)

  @rewrite(d, Compound)
  def _r1(op, _ctx):
    return Leaf(name=op.text)

  @rewrite(d2, Compound)
  def _r2(op, _ctx):
    return Leaf(name=op.text + '!')

  driver = PassDriver(c)
  with pytest.raises(RewriteRegistrationConflict) as exc_info:
    driver.apply_rewrites_to_fixpoint(Compound(text='x'))
  assert exc_info.value.op_type is Compound
  assert set(exc_info.value.in_dialects) == {'toy', 'toy.other'}


# -----------------------------------------------------------------------------
# verify_renderability
# -----------------------------------------------------------------------------


def test_verify_renderability_passes_when_all_have_renderers():
  c, _ = _fresh_compiler_with_dialect()
  driver = PassDriver(c)
  renderable = {Leaf, Container}
  errs = driver.verify_renderability(
    Container(children=(Leaf(name='x'), Leaf(name='y'))),
    target='toy',
    has_renderer=lambda t: t in renderable,
  )
  assert errs == []


def test_verify_renderability_reports_missing_renderer():
  c, _ = _fresh_compiler_with_dialect()
  driver = PassDriver(c)
  renderable = {Container, Leaf}
  errs = driver.verify_renderability(
    Container(children=(Compound(text='x'), Leaf(name='y'))),
    target='toy',
    has_renderer=lambda t: t in renderable,
  )
  assert len(errs) == 1
  assert errs[0].op_type is Compound
  assert errs[0].target == 'toy'


def test_verify_renderability_dedupes_by_op_type():
  '''Two Compound nodes in the tree → exactly one error, not two.'''
  c, _ = _fresh_compiler_with_dialect()
  driver = PassDriver(c)
  errs = driver.verify_renderability(
    Container(children=(Compound(text='a'), Compound(text='b'))),
    target='toy',
    has_renderer=lambda t: t is Container,
  )
  assert {e.op_type for e in errs} == {Compound, Leaf} or {e.op_type for e in errs} == {Compound}
  # Container is renderable; Compound and Leaf (if any) each surface once.
  type_counts = {e.op_type: 0 for e in errs}
  for e in errs:
    type_counts[e.op_type] += 1
  assert all(n == 1 for n in type_counts.values())


# -----------------------------------------------------------------------------
# PassDriver.run integration
# -----------------------------------------------------------------------------


def test_run_applies_fixpoint_then_verifies_renderability():
  c, d = _fresh_compiler_with_dialect()

  @rewrite(d, Compound)
  def _decompose(op, _ctx):
    return Leaf(name=op.text)

  driver = PassDriver(c)
  renderable = {Leaf, Container}
  out = driver.run(
    Container(children=(Compound(text='x'),)),
    target='toy',
    has_renderer=lambda t: t in renderable,
  )
  assert out == Container(children=(Leaf(name='x'),))


def test_run_raises_unrenderable_op_error_on_survivor():
  '''A COMPOUND op with no rewrite registered survives fixpoint;
  verify_renderability flags it; run() raises UnrenderableOpError.'''
  c, _ = _fresh_compiler_with_dialect()
  driver = PassDriver(c)
  with pytest.raises(UnrenderableOpError) as exc_info:
    driver.run(
      Container(children=(Compound(text='x'),)),
      target='toy',
      has_renderer=lambda t: t in {Leaf, Container},
    )
  assert any(e.op_type is Compound for e in exc_info.value.errors)


def test_run_target_without_has_renderer_raises_value_error():
  c, _ = _fresh_compiler_with_dialect()
  driver = PassDriver(c)
  with pytest.raises(ValueError, match='has_renderer=None'):
    driver.run(Leaf(name='x'), target='toy')


def test_run_without_target_skips_renderability_check():
  '''Pre-S4.9 callers (no target arg) keep their old behavior.'''
  c, _ = _fresh_compiler_with_dialect()
  driver = PassDriver(c)
  out = driver.run(Container(children=(Compound(text='x'),)))
  # No rewrites registered → tree returned unchanged, no error raised.
  assert isinstance(out, Container)


# -----------------------------------------------------------------------------
# Integration with the production CUDA renderer registry
# -----------------------------------------------------------------------------


def test_has_renderer_accessor_recognizes_real_iir_cf_op():
  '''Smoke check: the cuda render module's `has_renderer` returns True
  for a known LEAF op (iir.cf.Block) and False for a synthetic op
  with no renderer.'''
  from srdatalog.ir.codegen.cuda.render import has_renderer
  from srdatalog.ir.dialects.iir.cf.ops import Block

  assert has_renderer(Block) is True
  assert has_renderer(Compound) is False
