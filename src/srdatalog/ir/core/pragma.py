'''Typed compile-time pragma objects + `@pragma_handler` registry.

Spec: `docs/pragma_as_typed_object.md`.

A pragma is a typed, structured fact about a rule that the compiler
partially-evaluates by inserting wrap ops into MIR. Pragmas:

  - Are constructed at DSL-time (eager Python type checks via the
    dataclass __init__).
  - Carry structured config (not flat bool/Any).
  - Are consumed once by `MirPragmaPass` — they don't survive past
    it. Their effect is the wrap-op insertion; their evidence
    downstream is the typed op tree, not the pragma itself.
  - Dispatch by type, not string name.

This module is Layer 1 foundation. It defines:

  - `Pragma`             — pure-data base class (subclasses are
                           `@final + @dataclass(frozen=True, slots=True)`,
                           per discipline D13/D14 in
                           `docs/code_discipline.md` §2).
  - `pragma_handler`     — Triton-style decorator that registers a
                           materialization handler for one (Pragma
                           subclass, MIR op type) pair. Symmetric with
                           `@lowering` / `@rewrite` in `passes.py`
                           (per the project memory note
                           `feedback_decorator_registries.md`).
  - `get_pragma` /
    `has_pragma`         — typed lookup helpers (free functions, not
                           methods on Op — symmetric with `Op` being
                           pure data).
  - `PragmaCtx`          — context object passed to handler functions.
  - Error classes        — `PragmaConfigError` (DSL-time validation),
                           `PragmaOrderingError` (`MirPragmaPass`
                           topo-sort cycle), `UnconsumedPragmaError`
                           (a Pragma instance survived
                           `MirPragmaPass`), `UnregisteredPragmaError`
                           (DSL `with_pragma` got a Pragma whose type
                           has no handler).

What does NOT live here:

  - `MirPragmaPass`      — Phase C work; needs the F1 Pass framework
                           first. The registry exposed by
                           `get_pragma_registrations()` is the
                           interface the future pass will consume.
  - Concrete Pragma subclasses (e.g. `DedupHash`, `BlockGroup`) —
    those are plugin modules under `src/srdatalog/pragmas/`; this
    module knows only the base class.

The module-global registry list (`_PRAGMA_REGISTRY`) follows the
existing `@lowering` / `@rewrite` decorator pattern. Phase E may
move to a per-`Compiler` overlay; not now. Tests that need
isolation should monkeypatch `_PRAGMA_REGISTRY`.
'''

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from srdatalog.ir.core.dialect import Compiler


# -----------------------------------------------------------------------------
# Pragma base class
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pragma:
  '''Base class for compile-time pragma objects.

  Subclassing discipline (per `docs/code_discipline.md` D13/D14):

    - Subclass with `@final + @dataclass(frozen=True, slots=True)`.
    - Pure data: no methods other than `__post_init__` for
      cross-field validation. Behavior lives in a separately-
      registered `@pragma_handler` — symmetric with `@lowering` /
      `@rewrite` (every transformation is a registered decoration,
      separate from the data it operates on).
    - Use `__post_init__` to raise `PragmaConfigError` on invalid
      construction; the error fires at DSL-time, at the user's
      keystroke, not deep in `MirPragmaPass`.

  See `docs/pragma_as_typed_object.md` §2 for the full spec.
  '''


# -----------------------------------------------------------------------------
# Error classes
# -----------------------------------------------------------------------------


class PragmaConfigError(ValueError):
  '''Raised by a `Pragma` subclass `__post_init__` on invalid config.

  Subclass of `ValueError` so existing DSL-validation `except
  ValueError` clauses keep catching it.

  Example:

      @final
      @dataclass(frozen=True, slots=True)
      class BlockGroup(Pragma):
          threads_per_warp: int = 32

          def __post_init__(self):
              if self.threads_per_warp not in (16, 32):
                  raise PragmaConfigError(
                      f'BlockGroup.threads_per_warp must be 16 or 32, '
                      f'got {self.threads_per_warp}'
                  )
  '''


class PragmaOrderingError(Exception):
  '''Cycle in `@pragma_handler` `before` / `after` constraints.

  Raised by the future `MirPragmaPass` when its topo-sort over the
  registered pragma types finds a cycle. Carries the cycle as a
  list of `Pragma` subclasses for diagnostics.
  '''


class UnconsumedPragmaError(Exception):
  '''A `Pragma` instance survived `MirPragmaPass`.

  Raised by the future `MirPragmaPass` when, after applying every
  registered `@pragma_handler` in topo-sorted order, some op in the
  tree still has a non-empty `pragmas` field. Indicates either:

    - A `Pragma` subclass with no `@pragma_handler` registration
      (the discipline rule R5 violation).
    - A handler that returned an op without removing the pragma
      instance from `op.pragmas` (the handler bug).

  The error message should list the surviving pragma class and the
  set of registered handlers so the user can spot a missing import
  or typo (did-you-mean, per `docs/pragma_as_typed_object.md` §5).
  '''


class UnregisteredPragmaError(TypeError):
  '''DSL `with_pragma()` got a `Pragma` whose type has no handler.

  Subclass of `TypeError` because the offending arg fails the type-
  level contract: every `Pragma` subclass passed to the DSL must
  have a registered `@pragma_handler` in the active `Compiler`.

  Raised by the future DSL `Rule.with_pragma(p: Pragma)` validator,
  not by this module — listed here so the public surface is
  declared in one place, with the matching error other discipline
  rules will reference.
  '''


# -----------------------------------------------------------------------------
# Handler registration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PragmaCtx:
  '''Context object passed to `@pragma_handler` functions.

  Currently exposes only the active `Compiler` (for cross-registry
  lookups, e.g. asking whether a sibling Pragma type has a
  registration). Additional fields are added by amendment as
  handlers need them; per discipline D10 (parallel rule for
  `LowerCtx`), we keep this object minimal.
  '''

  compiler: Compiler


@dataclass(frozen=True)
class PragmaRegistration:
  '''A registered `@pragma_handler` entry.

  Fields:
    pragma_cls — the `Pragma` subclass this handler claims.
    on        — the MIR op type whose `pragmas` tuple this handler
                inspects (typically `mir.ExecutePipeline`).
    fn        — the materialization callable; signature
                `(op, pragma, ctx: PragmaCtx) -> Op | None`.
    before    — pragma TYPES that must run after this one; consumed
                by `MirPragmaPass`'s topo-sort (per
                `docs/pragma_as_typed_object.md` §5).
    after     — pragma TYPES that must run before this one.

  `before` / `after` reference Pragma classes, not strings — a typo
  is a `NameError` at decoration, not a runtime
  `PragmaOrderingError`.
  '''

  pragma_cls: type[Pragma]
  on: type
  fn: Callable[[Any, Pragma, PragmaCtx], Any]
  before: tuple[type[Pragma], ...] = field(default_factory=tuple)
  after: tuple[type[Pragma], ...] = field(default_factory=tuple)


# Module-global registry. The `@pragma_handler` decorator appends to it
# at import time (so registrations are stable once handler modules are
# loaded). Phase E may swap this for a per-`Compiler` overlay; until
# then, tests that need isolation monkeypatch `_PRAGMA_REGISTRY`.
_PRAGMA_REGISTRY: list[PragmaRegistration] = []


def pragma_handler(
  pragma_cls: type[Pragma],
  *,
  on: type,
  before: tuple[type[Pragma], ...] = (),
  after: tuple[type[Pragma], ...] = (),
) -> Callable[
  [Callable[[Any, Pragma, PragmaCtx], Any]],
  Callable[[Any, Pragma, PragmaCtx], Any],
]:
  '''Decorator: register a materialization handler for `pragma_cls`.

  The decorated function fires during `MirPragmaPass` on every op of
  type `on` whose `pragmas` tuple contains a `pragma_cls` instance.

  Signature of the decorated function:

      def handler(op: Op, pragma: pragma_cls, ctx: PragmaCtx) -> Op | None:

  Returns the transformed op (with the pragma instance removed from
  `op.pragmas`), or `None` to skip (no-op for this pragma).

  Validates at registration:

    - `pragma_cls` is a subclass of `Pragma` (`TypeError` otherwise).
    - `on` is a type (`TypeError` otherwise).
    - Every entry in `before` / `after` is a `Pragma` subclass
      (`TypeError` otherwise).

  These checks fire at import time, before the registry is ever
  read; a typo (`pragma_handler(SomeClassThatIsntAPragma, ...)`)
  cannot reach `MirPragmaPass`.

  Returns the original function unchanged so other decorators can
  stack on top.

  See `docs/pragma_as_typed_object.md` §3.
  '''
  if not isinstance(pragma_cls, type) or not issubclass(pragma_cls, Pragma):
    raise TypeError(f'@pragma_handler: pragma_cls must be a Pragma subclass; got {pragma_cls!r}')
  if not isinstance(on, type):
    raise TypeError(
      f'@pragma_handler({pragma_cls.__name__}, on=...): `on` must be a type; got {on!r}'
    )
  for label, seq in (('before', before), ('after', after)):
    for entry in seq:
      if not isinstance(entry, type) or not issubclass(entry, Pragma):
        raise TypeError(
          f'@pragma_handler({pragma_cls.__name__}, {label}=...): every '
          f'entry must be a Pragma subclass; got {entry!r}'
        )

  def _wrap(
    fn: Callable[[Any, Pragma, PragmaCtx], Any],
  ) -> Callable[[Any, Pragma, PragmaCtx], Any]:
    _PRAGMA_REGISTRY.append(
      PragmaRegistration(
        pragma_cls=pragma_cls,
        on=on,
        fn=fn,
        before=tuple(before),
        after=tuple(after),
      )
    )
    return fn

  return _wrap


def get_pragma_registrations() -> list[PragmaRegistration]:
  '''Snapshot of the module-global registry.

  Returns a fresh `list`, not the underlying mutable backing store,
  so callers cannot mutate the registry by accident. The future
  `MirPragmaPass` calls this once per pass invocation.
  '''
  return list(_PRAGMA_REGISTRY)


# -----------------------------------------------------------------------------
# Lookup helpers
# -----------------------------------------------------------------------------


def get_pragma(op: Any, pragma_cls: type[Pragma]) -> Pragma | None:
  '''Return the first pragma of type `pragma_cls` on `op`, or `None`.

  `op` must have a `pragmas` attribute that is iterable and contains
  `Pragma` instances. Ops that don't carry pragmas return `None`
  (an op without a `pragmas` field is just an op without any
  pragmas, by definition).

  See `docs/pragma_as_typed_object.md` §4.
  '''
  pragmas = getattr(op, 'pragmas', None)
  if pragmas is None:
    return None
  for p in pragmas:
    if isinstance(p, pragma_cls):
      return p
  return None


def has_pragma(op: Any, pragma_cls: type[Pragma]) -> bool:
  '''Return True if `op` carries any pragma of type `pragma_cls`.

  Symmetric with `get_pragma`; ops without a `pragmas` attribute
  return False.

  See `docs/pragma_as_typed_object.md` §4.
  '''
  pragmas = getattr(op, 'pragmas', None)
  if pragmas is None:
    return False
  return any(isinstance(p, pragma_cls) for p in pragmas)


__all__ = [
  'Pragma',
  'PragmaConfigError',
  'PragmaCtx',
  'PragmaOrderingError',
  'PragmaRegistration',
  'UnconsumedPragmaError',
  'UnregisteredPragmaError',
  'get_pragma',
  'get_pragma_registrations',
  'has_pragma',
  'pragma_handler',
]
