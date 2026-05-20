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
    reg = PragmaRegistration(
      pragma_cls=pragma_cls,
      on=on,
      fn=fn,
      before=tuple(before),
      after=tuple(after),
    )

    # PR-P0 back-compat shim (spec § 6.0.0 row 8): if a
    # `_compiler_registration_scope(compiler)` is active, stage the
    # registration into THAT compiler's per-Compiler pragma_handlers
    # list. Otherwise fall back to the legacy module-global registry
    # (existing behavior, deleted in PR-P5).
    #
    # The per-Compiler list lives on `compiler._pragma_handlers`
    # (private attr; the framework owns it). It is populated either
    # directly here (when this decorator runs inside a registration
    # scope) or by `register_pragma_plugin` unpacking the plugin's
    # `pragma_cls` into a synthetic legacy registration. PR-P5
    # collapses both code paths into the typed PragmaPlugin contract.
    from srdatalog.ir.core.dialect import _get_current_compiler

    compiler = _get_current_compiler()
    if compiler is not None:
      handlers: list[PragmaRegistration] = getattr(compiler, '_pragma_handlers', [])
      if not handlers:
        compiler._pragma_handlers = handlers  # type: ignore[attr-defined]
      handlers.append(reg)
    else:
      _PRAGMA_REGISTRY.append(reg)
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


# -----------------------------------------------------------------------------
# PR-P0 back-compat Pass: MaterializePragmaPass
#
# When a `@pragma_handler` is staged into a per-Compiler scope via
# `_compiler_registration_scope`, the framework needs a Pass instance
# that actually runs those handlers as part of `compiler.run(...)`.
# Today's `mir.pragma_pass.MirPragmaPass` is the production handler-
# runner, but it pulls from the module-global registry. The shim below
# is a lightweight Pass that iterates `get_compiler_pragma_handlers`
# (which returns the per-Compiler list when present) and applies each
# handler to matching ops in `prog`.
#
# PR-P5 deletes this shim alongside the module-global registry; by
# then every pragma is a full `PragmaPlugin` with its own typed Pass.
# -----------------------------------------------------------------------------


def _materialize_pragma_pass_factory() -> Any:
  '''Lazily build the `MaterializePragmaPass` class so its
  dependency on `Pass` (from `passes.py`, which imports Compiler from
  `dialect.py`) doesn't introduce a module-load cycle.

  Returns the class; instances are constructed by callers.
  '''
  from dataclasses import dataclass as _dataclass
  from dataclasses import field as _field

  from srdatalog.ir.core.passes import Pass

  @_dataclass(frozen=True)
  class MaterializePragmaPass(Pass):
    '''PR-P0 back-compat shim Pass: materialize one Pragma subclass.

    Runs the registered `@pragma_handler(pragma_cls, on=<op_type>)`
    for `pragma_cls` against every matching op in `prog`. Hand-rolled
    tree walk (no strategy combinator) so the shim has no dependency
    on the IIR-walk infrastructure.

    Each handler signature is `(op, pragma, ctx) -> Op | None`. A
    None return is treated as "this pragma did not apply to this op"
    and leaves the op unchanged; a returned Op replaces the matched
    op in the program tree.
    '''

    name: str = 'materialize_pragma_shim'
    pragma_cls: type[Pragma] | None = _field(default=None)

    def apply(self, prog: Any, compiler: Any) -> Any:
      handlers = get_compiler_pragma_handlers(compiler)
      if self.pragma_cls is not None:
        handlers = [h for h in handlers if h.pragma_cls is self.pragma_cls]
      if not handlers:
        return prog

      ctx = PragmaCtx(compiler=compiler)

      def _apply_handlers_to(op: Any) -> Any:
        for h in handlers:
          if not isinstance(op, h.on):
            continue
          pragmas = getattr(op, 'pragmas', None)
          if pragmas is None:
            continue
          for p in pragmas:
            if isinstance(p, h.pragma_cls):
              new = h.fn(op, p, ctx)
              if new is not None:
                op = new
        return op

      # Walk the tree top-down: visit the root, then recurse into
      # dataclass fields that are Ops or contain Ops. Mirrors the
      # MIR pragma-pass's reach without depending on its dialect
      # specifics.
      def _walk(node: Any) -> Any:
        import dataclasses as _dc

        node = _apply_handlers_to(node) if _is_op_like(node) else node
        if not _dc.is_dataclass(node) or isinstance(node, type):
          return node
        new_fields: dict[str, Any] = {}
        any_changed = False
        for f in _dc.fields(node):
          val = getattr(node, f.name)
          new_val = _walk_value(val)
          new_fields[f.name] = new_val
          if new_val is not val:
            any_changed = True
        if any_changed:
          return _dc.replace(node, **new_fields)
        return node

      def _walk_value(val: Any) -> Any:
        if _is_op_like(val):
          return _walk(val)
        if isinstance(val, list):
          new_list = [_walk_value(x) for x in val]
          return new_list if any(a is not b for a, b in zip(new_list, val)) else val
        if isinstance(val, tuple):
          new_tuple = tuple(_walk_value(x) for x in val)
          return new_tuple if any(a is not b for a, b in zip(new_tuple, val)) else val
        return val

      def _is_op_like(node: Any) -> bool:
        # An Op subclass instance; avoid importing Op at module init
        # by walking MRO names.
        cls = type(node)
        for base in cls.__mro__:
          if base.__name__ == 'Op' and base.__module__ == 'srdatalog.ir.core.ops':
            return True
        return False

      return _walk(prog)

  return MaterializePragmaPass


def get_compiler_pragma_handlers(compiler: Any) -> list[PragmaRegistration]:
  '''PR-P0 back-compat shim (spec § 6.0.0 row 8): snapshot the
  per-Compiler `_pragma_handlers` list staged by
  `@pragma_handler` decorations that ran inside a
  `_compiler_registration_scope(compiler)` block.

  Returns the legacy module-global registrations IF the compiler has
  no per-instance staging (every existing test path), so callers
  written before the per-Compiler shape continue to work unchanged.
  Otherwise returns the per-Compiler staged registrations
  concatenated with the module-globals (the compiler instance is
  authoritative; module-globals are fallback for legacy code paths).

  Used by `MaterializePragmaPass` (a back-compat shim Pass that
  iterates these registrations and applies them). After PR-P5,
  pragmas register exclusively via `PragmaPlugin` and this helper
  is deleted alongside the module-global registry.
  '''
  per_compiler: list[PragmaRegistration] = list(getattr(compiler, '_pragma_handlers', []))
  if per_compiler:
    return per_compiler + list(_PRAGMA_REGISTRY)
  return list(_PRAGMA_REGISTRY)


def MaterializePragmaPass(*args: Any, **kwargs: Any) -> Any:
  '''Public constructor for the back-compat shim Pass (PR-P0).

  Calls `_materialize_pragma_pass_factory()` on first use to avoid
  the module-load cycle with `passes.py`; subsequent calls reuse the
  cached class.
  '''
  global _MATERIALIZE_PRAGMA_PASS_CLS
  if _MATERIALIZE_PRAGMA_PASS_CLS is None:
    _MATERIALIZE_PRAGMA_PASS_CLS = _materialize_pragma_pass_factory()
  return _MATERIALIZE_PRAGMA_PASS_CLS(*args, **kwargs)


_MATERIALIZE_PRAGMA_PASS_CLS: type | None = None


__all__ = [
  'MaterializePragmaPass',
  'Pragma',
  'PragmaConfigError',
  'PragmaCtx',
  'PragmaOrderingError',
  'PragmaRegistration',
  'UnconsumedPragmaError',
  'UnregisteredPragmaError',
  'get_compiler_pragma_handlers',
  'get_pragma',
  'get_pragma_registrations',
  'has_pragma',
  'pragma_handler',
]
