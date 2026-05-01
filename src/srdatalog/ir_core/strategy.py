'''Strategy combinators for IR rewrite walks.

Lifted from Stratego (Visser, *Program Transformation with Stratego/XT*).
The idea: a *strategy* is a partial function `Op -> Op | None` where
`None` means "the strategy didn't apply at this op." Strategies compose
via combinators — `seq`, `choice`, `try_`, `repeat`, `top_down`,
`bottom_up`, `all_`, `one`. The composition is total, deterministic,
and pure.

Why strategies and not ad-hoc walkers:

  - **Composable**: rewrite logic is built from named combinators
    rather than re-implementing tree traversal in every pass.
  - **Pure**: each strategy is a function. No mutable rewriter
    objects. Unlike MLIR's `PatternRewriter`, strategies don't carry
    hidden state.
  - **Confluence-friendly**: applying `top_down(repeat(rule))` is a
    standard idiom; correctness arguments lift directly from the
    term-rewriting literature.

Children of an Op are discovered generically: any field whose value is
an Op or a list/tuple of Ops is considered a child. This is the "scrap
your boilerplate" lesson — generic traversal beats per-op visitor
methods.

References:
  Visser, "Strategic pattern matching", RTA 1999.
  Visser & Benaissa, "A core language for rewriting", FroCoS 1998.
  Lämmel & Peyton Jones, "Scrap your boilerplate", TLDI 2003.
'''

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TypeVar

from srdatalog.ir_core.ops import Op

T = TypeVar('T', bound=Op)

# A strategy maps an Op to a transformed Op or None ("did not apply").
Strategy = Callable[[Op], Op | None]


# -----------------------------------------------------------------------------
# Atomic strategies
# -----------------------------------------------------------------------------


def id_(op: Op) -> Op:
  '''The identity strategy. Always succeeds, returning the op unchanged.'''
  return op


def fail(_op: Op) -> Op | None:
  '''The failing strategy. Always returns None.'''
  return None


# -----------------------------------------------------------------------------
# Composition combinators
# -----------------------------------------------------------------------------


def try_(s: Strategy) -> Strategy:
  '''Apply `s`. If it fails, return the op unchanged. Always succeeds.'''

  def go(op: Op) -> Op:
    r = s(op)
    return op if r is None else r

  return go


def seq(s1: Strategy, s2: Strategy) -> Strategy:
  '''Apply `s1`, then `s2` to the result. Fails if either fails.'''

  def go(op: Op) -> Op | None:
    r1 = s1(op)
    if r1 is None:
      return None
    return s2(r1)

  return go


def choice(s1: Strategy, s2: Strategy) -> Strategy:
  '''Apply `s1`. If it fails, apply `s2` to the original op.'''

  def go(op: Op) -> Op | None:
    r = s1(op)
    return r if r is not None else s2(op)

  return go


def repeat(s: Strategy, max_iters: int = 1024) -> Strategy:
  '''Apply `s` to fixpoint (until it fails or returns the same op).

  `max_iters` guards against non-terminating rewrites; when reached,
  raises RuntimeError. In the standard term-rewriting literature this
  bound is implicit; we make it explicit since Python doesn't have a
  natural confluence proof.
  '''

  def go(op: Op) -> Op:
    for _ in range(max_iters):
      r = s(op)
      if r is None or r == op:
        return op
      op = r
    raise RuntimeError(f'repeat: did not converge in {max_iters} iterations')

  return go


# -----------------------------------------------------------------------------
# Traversal combinators
# -----------------------------------------------------------------------------


def all_(s: Strategy) -> Strategy:
  '''Apply `s` to every immediate child of the op. Fails if `s` fails
  on any child (Stratego semantics: `all` is "all-or-nothing").

  Returns a new op with transformed children, or None if any child
  transformation failed.
  '''

  def go(op: Op) -> Op | None:
    return _map_children(op, s, all_or_nothing=True)

  return go


def one(s: Strategy) -> Strategy:
  '''Apply `s` to exactly one immediate child of the op (the first
  that succeeds). Fails if `s` fails on every child.
  '''

  def go(op: Op) -> Op | None:
    return _map_one_child(op, s)

  return go


def some(s: Strategy) -> Strategy:
  '''Apply `s` to as many immediate children as it succeeds on. Fails
  if `s` fails on every child.
  '''

  def go(op: Op) -> Op | None:
    return _map_some_children(op, s)

  return go


def top_down(s: Strategy) -> Strategy:
  '''Apply `s` at the root, then recursively at every descendant
  (preorder). Equivalent to `try_(seq(s, all_(top_down(s))))`.
  '''

  def go(op: Op) -> Op:
    r = s(op)
    op = op if r is None else r
    children_result = _map_children(op, go, all_or_nothing=False)
    return op if children_result is None else children_result

  return go


def bottom_up(s: Strategy) -> Strategy:
  '''Apply `s` at every descendant, then at the root (postorder).
  Equivalent to `seq(all_(bottom_up(s)), try_(s))`.
  '''

  def go(op: Op) -> Op:
    children_result = _map_children(op, go, all_or_nothing=False)
    op = op if children_result is None else children_result
    r = s(op)
    return op if r is None else r

  return go


# -----------------------------------------------------------------------------
# Internal: child traversal via dataclass field walking
# -----------------------------------------------------------------------------


def _map_children(op: Op, transform: Strategy, *, all_or_nothing: bool) -> Op | None:
  '''Apply `transform` to every immediate Op-valued child of `op`.

  Children are discovered by walking dataclass fields. A field is a
  child if its value is an Op, or a list/tuple/dict whose entries
  contain Ops. Non-Op fields are passed through unchanged.

  When `all_or_nothing=True`, returns None if `transform` fails on
  any child (Stratego `all` semantics).

  When `all_or_nothing=False`, transform-fails are treated as
  identity (Stratego `try_(all)` semantics, used by traversal
  walkers).
  '''
  # Op subclasses are always dataclasses by D4 (see design_principles.md).
  changed = False
  new_values: dict[str, object] = {}
  for f in dataclasses.fields(op):
    val = getattr(op, f.name)
    new_val, child_changed, ok = _transform_field(val, transform, all_or_nothing)
    if not ok:
      return None
    new_values[f.name] = new_val
    changed = changed or child_changed
  if not changed:
    return op
  return dataclasses.replace(op, **new_values)


def _map_one_child(op: Op, transform: Strategy) -> Op | None:
  '''Apply `transform` to exactly one Op-valued child (left-to-right
  first that succeeds). Returns None if no child succeeds.
  '''
  # Op subclasses are always dataclasses by D4 (see design_principles.md).
  for f in dataclasses.fields(op):
    val = getattr(op, f.name)
    if isinstance(val, Op):
      r = transform(val)
      if r is not None and r is not val:
        return dataclasses.replace(op, **{f.name: r})
    elif isinstance(val, (list, tuple)):
      for i, x in enumerate(val):
        if isinstance(x, Op):
          r = transform(x)
          if r is not None and r is not x:
            new_list = list(val)
            new_list[i] = r
            return dataclasses.replace(op, **{f.name: type(val)(new_list)})
  return None


def _map_some_children(op: Op, transform: Strategy) -> Op | None:
  '''Apply `transform` to as many Op-valued children as succeed.
  Returns None if no child succeeds (Stratego `some` semantics).
  '''
  # Op subclasses are always dataclasses by D4 (see design_principles.md).
  any_succeeded = False
  new_values: dict[str, object] = {}
  for f in dataclasses.fields(op):
    val = getattr(op, f.name)
    new_val, ok = _transform_field_some(val, transform)
    new_values[f.name] = new_val
    any_succeeded = any_succeeded or ok
  if not any_succeeded:
    return None
  return dataclasses.replace(op, **new_values)


def _transform_field(
  val: object,
  transform: Strategy,
  all_or_nothing: bool,
) -> tuple[object, bool, bool]:
  '''Apply `transform` to a field value. Returns (new_val, changed, ok).

  `ok` is False only when `all_or_nothing=True` and a child transform
  returned None.
  '''
  if isinstance(val, Op):
    r = transform(val)
    if r is None:
      return val, False, not all_or_nothing
    return r, r is not val, True
  if isinstance(val, list):
    new_list = []
    changed = False
    for x in val:
      nv, ch, ok = _transform_field(x, transform, all_or_nothing)
      if not ok:
        return val, False, False
      new_list.append(nv)
      changed = changed or ch
    return new_list, changed, True
  if isinstance(val, tuple):
    new_tuple = []
    changed = False
    for x in val:
      nv, ch, ok = _transform_field(x, transform, all_or_nothing)
      if not ok:
        return val, False, False
      new_tuple.append(nv)
      changed = changed or ch
    return tuple(new_tuple), changed, True
  return val, False, True


def _transform_field_some(val: object, transform: Strategy) -> tuple[object, bool]:
  '''Helper for `some`: apply `transform`, treat failures as identity,
  but report whether any application succeeded. Returns (new_val, any_ok).
  '''
  if isinstance(val, Op):
    r = transform(val)
    if r is None:
      return val, False
    return r, True
  if isinstance(val, list):
    new_list = []
    any_ok = False
    for x in val:
      nv, ok = _transform_field_some(x, transform)
      new_list.append(nv)
      any_ok = any_ok or ok
    return new_list, any_ok
  if isinstance(val, tuple):
    new_tuple = []
    any_ok = False
    for x in val:
      nv, ok = _transform_field_some(x, transform)
      new_tuple.append(nv)
      any_ok = any_ok or ok
    return tuple(new_tuple), any_ok
  return val, False
