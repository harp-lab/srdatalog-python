'''Tests for the typed `Pragma` infrastructure in `core/pragma.py`.

These cover the Layer 1 foundation surface: the base class shape, the
`@pragma_handler` decorator's eager validation, the `get_pragma` /
`has_pragma` lookup helpers, and the registry isolation contract.

Discipline tests over the *full set of known Pragma subclasses* (e.g.
`test_pragma_subclasses_are_frozen_final`) deliberately do NOT live
here — no production Pragma subclass exists yet (those land in Phase
C, per `docs/pragma_as_typed_object.md` §8 and `code_discipline.md`
D13/D14). The discipline tests will arrive with their first targets.

Spec: [`docs/pragma_as_typed_object.md`](../docs/pragma_as_typed_object.md).
'''

from __future__ import annotations

import dataclasses
from dataclasses import FrozenInstanceError, dataclass
from typing import final

import pytest

from srdatalog.ir.core import (
  Pragma,
  PragmaConfigError,
  PragmaCtx,
  UnconsumedPragmaError,
  UnregisteredPragmaError,
  get_pragma,
  has_pragma,
  pragma_handler,
)
from srdatalog.ir.core.pragma import (
  PragmaOrderingError,
  PragmaRegistration,
  get_pragma_registrations,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def isolated_registry(monkeypatch):
  '''Swap `_PRAGMA_REGISTRY` for an empty list so the test's
  registrations don't leak into other tests, and other tests' (or
  module-level) registrations don't leak into this test.

  Tests that exercise `pragma_handler` should always use this fixture.
  '''
  fresh: list[PragmaRegistration] = []
  monkeypatch.setattr('srdatalog.ir.core.pragma._PRAGMA_REGISTRY', fresh)
  return fresh


# -----------------------------------------------------------------------------
# 1. Pragma subclass shape
# -----------------------------------------------------------------------------


def test_pragma_subclass_with_canonical_shape_constructs():
  '''A subclass declared @final + @dataclass(frozen=True, slots=True)
  is the canonical Pragma shape (per discipline D13). Construction
  with field defaults works; equality and hashing are dataclass-
  generated.'''

  @final
  @dataclass(frozen=True, slots=True)
  class MyPragma(Pragma):
    capacity: int = 1024

  p = MyPragma()
  assert p.capacity == 1024

  q = MyPragma(capacity=2048)
  assert q.capacity == 2048

  # Equality is structural (dataclass-generated).
  assert MyPragma(capacity=1024) == MyPragma()
  assert MyPragma(capacity=512) != MyPragma()

  # Frozen dataclasses are hashable when eq=True (the default).
  assert hash(MyPragma()) == hash(MyPragma(capacity=1024))


def test_pragma_subclass_can_have_no_fields():
  '''A pragma carrying only its type as the signal (no config) is
  valid — equivalent to a marker like `WorkStealing` in the spec.'''

  @final
  @dataclass(frozen=True, slots=True)
  class Marker(Pragma):
    pass

  assert Marker() == Marker()


# -----------------------------------------------------------------------------
# 2. __post_init__ validation
# -----------------------------------------------------------------------------


def test_post_init_validation_fires_at_construction():
  '''A Pragma subclass that raises in __post_init__ must surface the
  error at the user's keystroke — at DSL-time — not deep in
  MirPragmaPass.'''

  @final
  @dataclass(frozen=True, slots=True)
  class BoundedPragma(Pragma):
    threshold: float = 0.5

    def __post_init__(self):
      if not 0.0 <= self.threshold <= 1.0:
        raise PragmaConfigError(f'BoundedPragma.threshold must be in [0, 1]; got {self.threshold}')

  # Valid construction returns an instance.
  assert BoundedPragma(threshold=0.7).threshold == 0.7

  # Invalid construction raises at the BoundedPragma(...) call site.
  with pytest.raises(PragmaConfigError, match=r'must be in \[0, 1\]'):
    BoundedPragma(threshold=1.5)


def test_pragma_config_error_is_value_error():
  '''PragmaConfigError subclasses ValueError so existing
  `except ValueError` clauses keep catching it.'''
  assert issubclass(PragmaConfigError, ValueError)


# -----------------------------------------------------------------------------
# 3. @pragma_handler registration
# -----------------------------------------------------------------------------


def test_pragma_handler_registers_a_callable(isolated_registry):
  '''The decorator appends a PragmaRegistration to the module-global
  registry and returns the original function unchanged.'''

  @final
  @dataclass(frozen=True, slots=True)
  class P(Pragma):
    pass

  class FakeMirOp:
    pass

  @pragma_handler(P, on=FakeMirOp)
  def materialize(op, pragma, ctx):
    return op

  regs = get_pragma_registrations()
  assert len(regs) == 1
  reg = regs[0]
  assert reg.pragma_cls is P
  assert reg.on is FakeMirOp
  assert reg.fn is materialize
  assert reg.before == ()
  assert reg.after == ()


def test_pragma_handler_returns_function_unchanged(isolated_registry):
  '''Decoration must not wrap the function; symmetric with
  `@lowering` / `@rewrite` so other decorators stack cleanly.'''

  @final
  @dataclass(frozen=True, slots=True)
  class P(Pragma):
    pass

  class FakeMirOp:
    pass

  def raw(op, pragma, ctx):
    return op

  decorated = pragma_handler(P, on=FakeMirOp)(raw)
  assert decorated is raw


def test_pragma_handler_records_before_after(isolated_registry):
  '''before/after are stored as tuples of Pragma TYPES, not strings.'''

  @final
  @dataclass(frozen=True, slots=True)
  class A(Pragma):
    pass

  @final
  @dataclass(frozen=True, slots=True)
  class B(Pragma):
    pass

  @final
  @dataclass(frozen=True, slots=True)
  class C(Pragma):
    pass

  class FakeMirOp:
    pass

  @pragma_handler(A, on=FakeMirOp, before=(B,), after=(C,))
  def materialize_a(op, pragma, ctx):
    return op

  regs = get_pragma_registrations()
  assert len(regs) == 1
  assert regs[0].before == (B,)
  assert regs[0].after == (C,)


# -----------------------------------------------------------------------------
# 4. @pragma_handler validates pragma_cls is a Pragma subclass
# -----------------------------------------------------------------------------


def test_pragma_handler_rejects_non_pragma_class(isolated_registry):
  '''Passing a class that does not subclass Pragma fails at
  decoration — immediate, traceable.'''

  class NotAPragma:
    pass

  class FakeMirOp:
    pass

  with pytest.raises(TypeError, match=r'must be a Pragma subclass'):

    @pragma_handler(NotAPragma, on=FakeMirOp)  # type: ignore[arg-type]
    def materialize(op, pragma, ctx):
      return op


def test_pragma_handler_rejects_non_class_pragma_cls(isolated_registry):
  '''Passing a non-type (e.g., an instance) for pragma_cls fails.'''

  class FakeMirOp:
    pass

  with pytest.raises(TypeError, match=r'must be a Pragma subclass'):

    @pragma_handler('not-a-class', on=FakeMirOp)  # type: ignore[arg-type]
    def materialize(op, pragma, ctx):
      return op


def test_pragma_handler_rejects_non_type_on(isolated_registry):
  '''The `on` kwarg must be a type. A string (e.g., a "name", as the
  old string-keyed sketch used) fails immediately.'''

  @final
  @dataclass(frozen=True, slots=True)
  class P(Pragma):
    pass

  with pytest.raises(TypeError, match=r'`on` must be a type'):

    @pragma_handler(P, on='mir.ExecutePipeline')  # type: ignore[arg-type]
    def materialize(op, pragma, ctx):
      return op


# -----------------------------------------------------------------------------
# 5. @pragma_handler validates before/after entries are Pragma subclasses
# -----------------------------------------------------------------------------


def test_pragma_handler_rejects_non_pragma_in_before(isolated_registry):
  '''A non-Pragma class in `before` fails at decoration. The string-
  keyed alternative (`before=("count_as_product",)`) was a runtime-
  PragmaOrderingError; now it's a TypeError at import time.'''

  @final
  @dataclass(frozen=True, slots=True)
  class P(Pragma):
    pass

  class NotAPragma:
    pass

  class FakeMirOp:
    pass

  with pytest.raises(TypeError, match=r'before=.*must be a Pragma subclass'):

    @pragma_handler(P, on=FakeMirOp, before=(NotAPragma,))  # type: ignore[arg-type]
    def materialize(op, pragma, ctx):
      return op


def test_pragma_handler_rejects_non_pragma_in_after(isolated_registry):
  '''Same check on `after`.'''

  @final
  @dataclass(frozen=True, slots=True)
  class P(Pragma):
    pass

  class FakeMirOp:
    pass

  with pytest.raises(TypeError, match=r'after=.*must be a Pragma subclass'):

    @pragma_handler(P, on=FakeMirOp, after=('a-string',))  # type: ignore[arg-type]
    def materialize(op, pragma, ctx):
      return op


def test_pragma_handler_rejects_string_in_before(isolated_registry):
  '''Strings in before/after are the legacy mistake — explicitly
  rejected.'''

  @final
  @dataclass(frozen=True, slots=True)
  class P(Pragma):
    pass

  class FakeMirOp:
    pass

  with pytest.raises(TypeError, match=r'before=.*must be a Pragma subclass'):

    @pragma_handler(P, on=FakeMirOp, before=('count_as_product',))  # type: ignore[arg-type]
    def materialize(op, pragma, ctx):
      return op


# -----------------------------------------------------------------------------
# 6. get_pragma / has_pragma
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SyntheticOp:
  '''Top-level synthetic op for the get_pragma / has_pragma tests.

  Not a real `Op` subclass — the lookup helpers only care about the
  `pragmas` attribute.
  '''

  pragmas: tuple[Pragma, ...] = ()


def test_get_pragma_returns_first_matching_instance():
  @final
  @dataclass(frozen=True, slots=True)
  class A(Pragma):
    capacity: int = 1024

  @final
  @dataclass(frozen=True, slots=True)
  class B(Pragma):
    pass

  a = A(capacity=2048)
  b = B()
  op = _SyntheticOp(pragmas=(b, a))

  assert get_pragma(op, A) is a
  assert get_pragma(op, B) is b


def test_get_pragma_returns_none_when_missing():
  @final
  @dataclass(frozen=True, slots=True)
  class A(Pragma):
    pass

  @final
  @dataclass(frozen=True, slots=True)
  class B(Pragma):
    pass

  op = _SyntheticOp(pragmas=(A(),))
  assert get_pragma(op, B) is None


def test_get_pragma_on_op_without_pragmas_attr_returns_none():
  '''An op type that doesn't carry pragmas at all (e.g., an op outside
  the MIR pragma-carrying set) returns None gracefully.'''

  class BareOp:
    pass

  assert get_pragma(BareOp(), Pragma) is None


def test_has_pragma_true_when_present():
  @final
  @dataclass(frozen=True, slots=True)
  class A(Pragma):
    pass

  op = _SyntheticOp(pragmas=(A(),))
  assert has_pragma(op, A) is True


def test_has_pragma_false_when_absent():
  @final
  @dataclass(frozen=True, slots=True)
  class A(Pragma):
    pass

  @final
  @dataclass(frozen=True, slots=True)
  class B(Pragma):
    pass

  op = _SyntheticOp(pragmas=(A(),))
  assert has_pragma(op, B) is False


def test_has_pragma_false_on_op_without_pragmas_attr():
  class BareOp:
    pass

  assert has_pragma(BareOp(), Pragma) is False


def test_get_pragma_returns_first_when_duplicates():
  '''If somehow the pragmas tuple holds multiple instances of the
  same type (a bug elsewhere, but this helper must be defined),
  return the FIRST one — first-wins, matching the spec wording
  "the first pragma of type pragma_cls".'''

  @final
  @dataclass(frozen=True, slots=True)
  class A(Pragma):
    capacity: int = 1024

  first = A(capacity=10)
  second = A(capacity=20)
  op = _SyntheticOp(pragmas=(first, second))
  assert get_pragma(op, A) is first


# -----------------------------------------------------------------------------
# 7. Frozen behavior
# -----------------------------------------------------------------------------


def test_pragma_instance_is_frozen():
  '''Field reassignment on a Pragma instance raises FrozenInstanceError
  (parallel to the Op discipline; pragmas are immutable compile-time
  values).'''

  @final
  @dataclass(frozen=True, slots=True)
  class P(Pragma):
    capacity: int = 1024

  p = P()
  with pytest.raises(FrozenInstanceError):
    p.capacity = 2048  # type: ignore[misc]


def test_pragma_instance_has_no_dict():
  '''slots=True forbids attribute injection at runtime — symmetric
  with the Op discipline rule D3 in the existing core/ tests.

  Note: depending on how slots+frozen interact for an inherited
  class, the rejection comes through as `AttributeError` (slots
  enforcement) OR `FrozenInstanceError` (frozen __setattr__) OR
  `TypeError` (the `super(type, obj)` quirk that fires in some
  closure contexts). Any of the three signals "field injection
  rejected" — the discipline guarantee.
  '''

  @final
  @dataclass(frozen=True, slots=True)
  class P(Pragma):
    pass

  p = P()
  assert not hasattr(p, '__dict__')
  with pytest.raises((AttributeError, FrozenInstanceError, TypeError)):
    p.unknown_attr = 'x'  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# 8. Registry isolation
# -----------------------------------------------------------------------------


def test_isolated_registry_starts_empty(isolated_registry):
  '''Inside the fixture, the registry is empty regardless of any
  module-level registrations elsewhere in the test suite.'''
  assert get_pragma_registrations() == []


def test_isolated_registry_does_not_leak_after_test(isolated_registry):
  '''Registrations made inside the fixture must not survive the test
  — verified by counting registrations in the fresh fixture in this
  test (any leakage from a previous test would make len(...) > 0
  before our own decoration).'''
  assert len(get_pragma_registrations()) == 0

  @final
  @dataclass(frozen=True, slots=True)
  class Local(Pragma):
    pass

  class FakeMirOp:
    pass

  @pragma_handler(Local, on=FakeMirOp)
  def materialize(op, pragma, ctx):
    return op

  assert len(get_pragma_registrations()) == 1


def test_get_pragma_registrations_returns_a_snapshot(isolated_registry):
  '''Mutating the returned list must not affect the underlying
  registry — defensive against accidental clears during a pass.'''

  @final
  @dataclass(frozen=True, slots=True)
  class P(Pragma):
    pass

  class FakeMirOp:
    pass

  @pragma_handler(P, on=FakeMirOp)
  def materialize(op, pragma, ctx):
    return op

  snap = get_pragma_registrations()
  assert len(snap) == 1
  snap.clear()
  assert len(get_pragma_registrations()) == 1  # underlying registry intact


# -----------------------------------------------------------------------------
# Sanity checks on the ancillary surface
# -----------------------------------------------------------------------------


def test_pragma_ctx_carries_compiler():
  '''PragmaCtx is a tiny frozen dataclass; the compiler field is
  the only handle handlers use today.'''
  from srdatalog.ir.core import Compiler

  c = Compiler()
  ctx = PragmaCtx(compiler=c)
  assert ctx.compiler is c

  with pytest.raises(FrozenInstanceError):
    ctx.compiler = Compiler()  # type: ignore[misc]


def test_error_class_hierarchy():
  '''Public error classes have the documented inheritance — keeps
  user-side `except` clauses honest.'''
  assert issubclass(PragmaConfigError, ValueError)
  assert issubclass(UnregisteredPragmaError, TypeError)
  assert issubclass(PragmaOrderingError, Exception)
  assert issubclass(UnconsumedPragmaError, Exception)


def test_pragma_registration_is_frozen_dataclass():
  '''PragmaRegistration is a record consumed by the future
  MirPragmaPass; freezing it prevents accidental mutation while
  iterating.'''
  assert dataclasses.is_dataclass(PragmaRegistration)
  params = PragmaRegistration.__dataclass_params__  # type: ignore[attr-defined]
  assert params.frozen is True
