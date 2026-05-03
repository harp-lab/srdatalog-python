'''Discipline tests for the IR framework.

Sentinel tests that catch design-rule regressions before they spread.
The rules they enforce are stated in docs/design_principles.md:

  D1  Op/Type subclasses are pure data. No methods.
  D2  IR is immutable (frozen dataclasses).
  D3  Slots required (no attribute injection at runtime).
  D4  Subclasses are dataclasses (so they round-trip cleanly,
      participate in pattern matching, and have field iteration for
      generic walks).

These tests fail loudly when a future change quietly violates a rule.
'''

from __future__ import annotations

import dataclasses
import inspect

from srdatalog.ir.core import Op, Type
from srdatalog.ir.core import strategy as strategy_mod

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _all_subclasses(cls: type) -> set[type]:
  '''Recursively collect every subclass of `cls` known at this point.'''
  out: set[type] = set()
  pending = list(cls.__subclasses__())
  while pending:
    sub = pending.pop()
    if sub in out:
      continue
    out.add(sub)
    pending.extend(sub.__subclasses__())
  return out


def _user_methods(cls: type) -> list[str]:
  '''Methods defined directly on `cls` that aren't dataclass-generated
  or dunder.'''
  generated = {
    '__init__',
    '__repr__',
    '__eq__',
    '__hash__',
    '__match_args__',
    '__dataclass_fields__',
    '__dataclass_params__',
    '__class_getitem__',
    '__weakref__',
    '__doc__',
    '__module__',
    '__qualname__',
    '__dict__',
    '__annotations__',
    '__slots__',
    '__static_attributes__',
    '__firstlineno__',
  }
  out = []
  for name, val in vars(cls).items():
    if name in generated:
      continue
    if name.startswith('_') and name.endswith('_'):
      # any dunder we didn't list — skip; they're frame-generated
      continue
    if not callable(val):
      continue
    out.append(name)
  return out


def _import_synthetic_subclasses() -> None:
  '''Force-load test modules that define synthetic Op subclasses, so
  `Op.__subclasses__()` enumerates them. Without this, discipline
  tests run before the test files are imported and would see an empty
  subclass set.'''
  import tests.test_ir_core_strategy  # noqa: F401


# -----------------------------------------------------------------------------
# Discipline rules
# -----------------------------------------------------------------------------


def test_op_subclasses_have_no_user_methods():
  '''D1: Op subclasses are pure data — no methods beyond what
  `@dataclass` generates and dunders. Algorithms live in external
  functions, dispatched via `match`.'''
  _import_synthetic_subclasses()
  offenders = []
  for sub in _all_subclasses(Op):
    methods = _user_methods(sub)
    if methods:
      offenders.append((sub.__name__, methods))
  assert not offenders, (
    'Op subclasses must be pure data; found user-defined methods on:\n'
    + '\n'.join(f'  {name}: {ms}' for name, ms in offenders)
    + '\nMove the algorithm to a free function and dispatch with `match`.'
  )


def test_type_subclasses_have_no_user_methods():
  '''D1 applied to Type subclasses.'''
  _import_synthetic_subclasses()
  offenders = []
  for sub in _all_subclasses(Type):
    methods = _user_methods(sub)
    if methods:
      offenders.append((sub.__name__, methods))
  assert not offenders, (
    'Type subclasses must be pure data; found user-defined methods on:\n'
    + '\n'.join(f'  {name}: {ms}' for name, ms in offenders)
  )


def test_op_subclasses_are_dataclasses():
  '''D4: every Op subclass must be a dataclass — needed for field
  iteration in strategy combinators and for sexpr round-trip.'''
  _import_synthetic_subclasses()
  offenders = [sub.__name__ for sub in _all_subclasses(Op) if not dataclasses.is_dataclass(sub)]
  assert not offenders, (
    f'Op subclasses must be @dataclass; non-dataclass subclasses found: {offenders}'
  )


def test_op_subclasses_are_frozen():
  '''D2: Op subclasses must be frozen — IR is immutable.'''
  _import_synthetic_subclasses()
  offenders = []
  for sub in _all_subclasses(Op):
    if not dataclasses.is_dataclass(sub):
      continue
    if not sub.__dataclass_params__.frozen:
      offenders.append(sub.__name__)
  assert not offenders, f'Op subclasses must use @dataclass(frozen=True); non-frozen: {offenders}'


def test_op_subclasses_use_slots():
  '''D3: Op subclasses must use slots — forbids attribute injection.'''
  _import_synthetic_subclasses()
  offenders = []
  for sub in _all_subclasses(Op):
    if not hasattr(sub, '__slots__'):
      offenders.append(sub.__name__)
  assert not offenders, (
    f'Op subclasses must use @dataclass(slots=True); missing __slots__: {offenders}'
  )


def test_op_base_is_frozen_and_slotted():
  '''The bases themselves carry the discipline so subclasses inherit it.'''
  assert dataclasses.is_dataclass(Op)
  assert Op.__dataclass_params__.frozen
  assert hasattr(Op, '__slots__')


def test_type_base_is_frozen_and_slotted():
  assert dataclasses.is_dataclass(Type)
  assert Type.__dataclass_params__.frozen
  assert hasattr(Type, '__slots__')


def test_op_instances_are_immutable():
  '''Functional check that frozen actually works at runtime.'''
  from dataclasses import FrozenInstanceError, dataclass

  @dataclass(frozen=True, slots=True)
  class FrozenOp(Op):
    x: int

  op = FrozenOp(x=1)
  try:
    op.x = 2
  except (FrozenInstanceError, AttributeError):
    return
  raise AssertionError('FrozenOp should not allow attribute assignment')


def test_op_instances_reject_unknown_attrs():
  '''Functional check that slots actually work — adding attributes
  outside the declared slots must fail.'''
  from dataclasses import dataclass

  @dataclass(frozen=True, slots=True)
  class SlottedOp(Op):
    x: int

  op = SlottedOp(x=1)
  try:
    object.__setattr__(op, 'y', 2)
  except AttributeError:
    return
  raise AssertionError('Slotted op should reject unknown attributes')


# -----------------------------------------------------------------------------
# Strategy module discipline
# -----------------------------------------------------------------------------


def test_strategy_combinators_are_module_level_callables():
  '''Strategy combinators are pure functions, not classes. No state,
  no methods, no inheritance.'''
  expected = [
    'id_',
    'fail',
    'try_',
    'seq',
    'choice',
    'repeat',
    'all_',
    'one',
    'some',
    'top_down',
    'bottom_up',
  ]
  for name in expected:
    obj = getattr(strategy_mod, name, None)
    assert obj is not None, f'strategy.{name} missing'
    assert inspect.isfunction(obj), (
      f'strategy.{name} must be a plain function (no classes / partials)'
    )
