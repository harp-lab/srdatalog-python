'''Tests for the strategy combinators.

The combinators are lifted from Stratego; the tests use small
synthetic Op trees to verify the standard semantics.
'''

from __future__ import annotations

from dataclasses import dataclass

import pytest

from srdatalog.ir.core import (
  Op,
  all_,
  assert_never,
  bottom_up,
  choice,
  fail,
  id_,
  one,
  repeat,
  seq,
  some,
  top_down,
  try_,
)

# -----------------------------------------------------------------------------
# Synthetic op tree for tests
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Lit(Op):
  value: int


@dataclass(frozen=True, slots=True)
class Add(Op):
  left: Op
  right: Op


@dataclass(frozen=True, slots=True)
class Mul(Op):
  left: Op
  right: Op


@dataclass(frozen=True, slots=True)
class Block(Op):
  '''Holds a list of children — exercises list-field traversal.'''

  ops: tuple[Op, ...]


# -----------------------------------------------------------------------------
# Atomic strategies
# -----------------------------------------------------------------------------


def test_id_returns_op_unchanged():
  op = Lit(42)
  assert id_(op) is op


def test_fail_returns_none():
  assert fail(Lit(42)) is None


# -----------------------------------------------------------------------------
# Composition
# -----------------------------------------------------------------------------


def test_try_recovers_from_failure():
  '''try_(fail) succeeds with the original op.'''
  op = Lit(7)
  assert try_(fail)(op) is op


def test_try_passes_through_success():
  '''try_(s) returns s's result when s succeeds.'''
  bump = lambda op: Lit(op.value + 1) if isinstance(op, Lit) else None
  assert try_(bump)(Lit(7)) == Lit(8)


def test_seq_chains_two_strategies():
  bump = lambda op: Lit(op.value + 1) if isinstance(op, Lit) else None
  assert seq(bump, bump)(Lit(0)) == Lit(2)


def test_seq_short_circuits_on_failure():
  bump = lambda op: Lit(op.value + 1) if isinstance(op, Lit) else None
  assert seq(fail, bump)(Lit(0)) is None
  assert seq(bump, fail)(Lit(0)) is None


def test_choice_takes_first_success():
  bump = lambda op: Lit(op.value + 1) if isinstance(op, Lit) else None
  twice = lambda op: Lit(op.value * 2) if isinstance(op, Lit) else None
  assert choice(bump, twice)(Lit(5)) == Lit(6)
  assert choice(fail, twice)(Lit(5)) == Lit(10)


# -----------------------------------------------------------------------------
# Repeat / fixpoint
# -----------------------------------------------------------------------------


def test_repeat_runs_to_fixpoint():
  '''Decrement until 0; should converge.'''
  dec = lambda op: Lit(op.value - 1) if isinstance(op, Lit) and op.value > 0 else None
  assert repeat(dec)(Lit(5)) == Lit(0)


def test_repeat_returns_op_when_strategy_immediately_fails():
  assert repeat(fail)(Lit(5)) == Lit(5)


def test_repeat_raises_on_non_termination():
  '''A strategy that always reports a different op without fixpoint
  must hit the iteration cap.'''
  bump = lambda op: Lit(op.value + 1) if isinstance(op, Lit) else None
  with pytest.raises(RuntimeError, match='did not converge'):
    repeat(bump, max_iters=10)(Lit(0))


# -----------------------------------------------------------------------------
# Traversal
# -----------------------------------------------------------------------------


def test_top_down_visits_root_first():
  '''top_down replaces matches at the root before descending. After
  the replacement, top_down recurses into the *new* op's children.'''
  expr = Add(Lit(1), Lit(2))

  def double_lit(op):
    if isinstance(op, Lit):
      return Lit(op.value * 2)
    return None

  result = top_down(double_lit)(expr)
  assert result == Add(Lit(2), Lit(4))


def test_bottom_up_visits_children_first():
  '''bottom_up rewrites children before the root.'''
  expr = Add(Add(Lit(1), Lit(2)), Lit(3))

  def fold_add(op):
    if isinstance(op, Add) and isinstance(op.left, Lit) and isinstance(op.right, Lit):
      return Lit(op.left.value + op.right.value)
    return None

  result = bottom_up(fold_add)(expr)
  assert result == Lit(6)


def test_top_down_traverses_list_fields():
  '''top_down/bottom_up must recurse into list-of-Op fields.'''
  expr = Block(ops=(Lit(1), Add(Lit(2), Lit(3)), Lit(4)))

  def double_lit(op):
    if isinstance(op, Lit):
      return Lit(op.value * 2)
    return None

  result = top_down(double_lit)(expr)
  assert result == Block(ops=(Lit(2), Add(Lit(4), Lit(6)), Lit(8)))


def test_all_succeeds_when_all_children_match():
  expr = Add(Lit(1), Lit(2))
  bump_lit = lambda op: Lit(op.value + 1) if isinstance(op, Lit) else None
  assert all_(bump_lit)(expr) == Add(Lit(2), Lit(3))


def test_all_fails_when_any_child_fails():
  expr = Add(Lit(1), Add(Lit(2), Lit(3)))
  bump_lit = lambda op: Lit(op.value + 1) if isinstance(op, Lit) else None
  # The right child is an Add, which bump_lit fails on -> all_ fails.
  assert all_(bump_lit)(expr) is None


def test_one_applies_to_first_succeeding_child():
  expr = Add(Lit(1), Lit(2))
  bump_lit = lambda op: Lit(op.value + 1) if isinstance(op, Lit) else None
  # `one` rewrites only one child (the leftmost that succeeds).
  assert one(bump_lit)(expr) == Add(Lit(2), Lit(2))


def test_one_fails_when_no_child_succeeds():
  expr = Add(Lit(1), Lit(2))
  assert one(fail)(expr) is None


def test_some_applies_to_all_succeeding_children():
  expr = Block(ops=(Lit(1), Add(Lit(2), Lit(3)), Lit(4)))
  # bump_lit succeeds on Lits at top level, fails on the inner Add.
  bump_lit = lambda op: Lit(op.value + 1) if isinstance(op, Lit) else None
  result = some(bump_lit)(expr)
  # Lit(1) and Lit(4) bumped; Add untouched (some is non-recursive at this level).
  assert result == Block(ops=(Lit(2), Add(Lit(2), Lit(3)), Lit(5)))


def test_some_fails_when_no_child_succeeds():
  expr = Add(Lit(1), Lit(2))
  assert some(fail)(expr) is None


# -----------------------------------------------------------------------------
# assert_never
# -----------------------------------------------------------------------------


def test_assert_never_raises():
  with pytest.raises(AssertionError, match='Unhandled case'):
    assert_never(Lit(7))


def test_pattern_match_with_assert_never():
  '''Realistic dispatcher pattern: closed sum + match + assert_never.'''

  def evaluate(op: Op) -> int:
    match op:
      case Lit(value=v):
        return v
      case Add(left=l, right=r):
        return evaluate(l) + evaluate(r)
      case Mul(left=l, right=r):
        return evaluate(l) * evaluate(r)
      case Block(ops=ops):
        return sum(evaluate(o) for o in ops)
      case _:
        assert_never(op)

  assert evaluate(Lit(7)) == 7
  assert evaluate(Add(Lit(1), Lit(2))) == 3
  assert evaluate(Mul(Add(Lit(1), Lit(2)), Lit(3))) == 9
