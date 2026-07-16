'''Pass kinds, registration decorators, and driver.

A pass is a transformation on the IR. Two flavors:

  Lowering — matches an op of one dialect, produces ops of another
             (or of a target dialect). Each Lowering carries the op
             class it matches, an `apply` callable that performs the
             transformation, declared `consumes` / `produces` dialect
             names for dependency validation, and a name for diagnostics.

  Rewrite  — matches an op, produces ops of the *same* dialect.
             Used for internal optimizations like the IIR-sorted-array
             count-as-product or hint-narrowing rules.

Per the project memory note (`feedback_decorator_registries.md`),
registration uses Triton-style decorators rather than imperative
`register_X(OpClass, fn)` calls or class-based dispatch:

    from srdatalog.ir.core import lowering, rewrite, verifier
    from srdatalog.ir.dialects.relation.sorted_array import DIALECT

    @lowering(DIALECT, mir.ExecutePipeline,
              consumes=('mir',), produces=('iir.cf', 'relation.sorted_array'))
    def _lower_execute_pipeline(ep, ctx):
        ...

    @rewrite(DIALECT, SaPrefCoop)
    def _hint_introduction(op, ctx):
        ...

    @verifier(DIALECT)
    def _verify_sorted_array(prog):
        return []   # list of VerificationError

The decorators mutate `dialect.lowerings` / `dialect.rewrites` /
`dialect.verifier` in place. Decoration is the only registration
path — there is no imperative API exposed.

The `PassDriver` walks `compiler.dialects` to validate dependencies
(every Lowering's `consumes` must be in the registered dialect set;
otherwise raise `PassDependencyError`). Actual op-level dispatch is
left to callers for now (production code calls lowering functions
directly); the registry exists so external consumers can introspect
"who lowers what" and so future stages can add a tree-walking dispatcher.

See docs/ir_lowering_semantics.md, section 21.
'''

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from srdatalog.ir.core.dialect import Compiler, Dialect


@dataclass
class Lowering:
  '''A lowering rule from one dialect to another.

  Fields:
    matches  — the Op subclass this rule matches (e.g. MirColumnJoin).
    apply    — callable taking (op, context) and returning the
               replacement IR (single op or list of ops; type depends
               on the target dialect's contract).
    name     — short identifier for diagnostics and pass tracing.
    consumes — dialect names whose ops this lowering reads. Used by
               PassDriver to validate that every required dialect is
               registered before the lowering runs.
    produces — dialect names whose ops this lowering emits. Used by
               PassDriver for topological ordering of multi-pass
               pipelines (a pass that produces dialect D must run
               before any pass that consumes D).
  '''

  matches: type
  apply: Callable[[Any, Any], Any]
  name: str = ''
  consumes: tuple[str, ...] = field(default_factory=tuple)
  produces: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class Rewrite:
  '''A rewrite rule within a single dialect.

  Same shape as Lowering, but conventionally produces ops of the same
  dialect as `matches`. `consumes` / `produces` typically equal the
  dialect's own name; PassDriver still uses them for dependency
  validation if a rewrite reads ops from a sibling dialect.
  '''

  matches: type
  apply: Callable[[Any, Any], Any]
  name: str = ''
  consumes: tuple[str, ...] = field(default_factory=tuple)
  produces: tuple[str, ...] = field(default_factory=tuple)


# -----------------------------------------------------------------------------
# Decorator-style registration
# -----------------------------------------------------------------------------


def lowering(
  dialect: Dialect,
  matches: type,
  *,
  consumes: tuple[str, ...] = (),
  produces: tuple[str, ...] = (),
  name: str = '',
) -> Callable[[Callable[[Any, Any], Any]], Callable[[Any, Any], Any]]:
  '''Decorator: wrap fn as a Lowering and register on dialect.lowerings.

  Usage:

      @lowering(MY_DIALECT, mir.SomeOp,
                consumes=('mir',),
                produces=('iir.cf', 'relation.sorted_array'))
      def _lower_some_op(op, ctx):
          return ...

  Returns the original function (so other decorators can stack).
  '''

  def _wrap(fn: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    inst = Lowering(
      matches=matches,
      apply=fn,
      name=name or fn.__name__,
      consumes=tuple(consumes),
      produces=tuple(produces),
    )
    dialect.lowerings.append(inst)
    return fn

  return _wrap


def rewrite(
  dialect: Dialect,
  matches: type,
  *,
  consumes: tuple[str, ...] = (),
  produces: tuple[str, ...] = (),
  name: str = '',
) -> Callable[[Callable[[Any, Any], Any]], Callable[[Any, Any], Any]]:
  '''Decorator: wrap fn as a Rewrite and register on dialect.rewrites.

  Usage:

      @rewrite(MY_DIALECT, SomeOp)
      def _hint_introduction(op, ctx):
          return ...
  '''

  def _wrap(fn: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    inst = Rewrite(
      matches=matches,
      apply=fn,
      name=name or fn.__name__,
      consumes=tuple(consumes),
      produces=tuple(produces),
    )
    dialect.rewrites.append(inst)
    return fn

  return _wrap


def verifier(dialect: Dialect) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
  '''Decorator: register fn as the dialect's verifier.

  Usage:

      @verifier(MY_DIALECT)
      def _verify(prog):
          return []   # list of VerificationError, [] = OK

  Raises ValueError if the dialect already has a verifier registered.
  '''

  def _wrap(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    if dialect.verifier is not None:
      raise ValueError(f'verifier already registered on {dialect.name!r}')
    dialect.verifier = fn
    return fn

  return _wrap


# -----------------------------------------------------------------------------
# Pass dependency error
# -----------------------------------------------------------------------------


class PassDependencyError(Exception):
  '''A registered pass declared `consumes=(D, ...)` but dialect D is
  not registered with the Compiler. Raised by PassDriver.run before
  any pass executes.

  The recommended posture (per docs/stage3a_execution_plan.md §9) is
  loud failure over silent fallback: a pipeline opting out of a
  dialect's passes does so by not registering those passes, not by
  not registering the dialect.
  '''

  def __init__(self, pass_name: str, missing_dialect: str, in_dialect: str) -> None:
    self.pass_name = pass_name
    self.missing_dialect = missing_dialect
    self.in_dialect = in_dialect
    super().__init__(
      f'pass {pass_name!r} in dialect {in_dialect!r} declares '
      f'consumes={missing_dialect!r}, but that dialect is not registered '
      f'with the Compiler. Either register the dialect or unregister the pass.'
    )


# -----------------------------------------------------------------------------
# PassDriver
# -----------------------------------------------------------------------------


class PassDriver:
  '''Runs registered rewrites and lowerings.

  Today the driver does dependency validation (catches "pass P
  consumes dialect D not enabled") and verifier dispatch. Op-level
  dispatch (a tree walker that finds the registered Lowering for
  each op kind and applies it) lands when the first production
  consumer needs it; until then, callers invoke lowering functions
  directly and the registry serves as introspection metadata.

  The driver does not know about specific dialects. New dialects
  participate by being registered; the driver consults the registry.
  '''

  def __init__(self, compiler: Compiler) -> None:
    self._compiler = compiler

  def validate_dependencies(self) -> None:
    '''Check every registered Lowering / Rewrite's `consumes` against
    the registered dialect set. Raises PassDependencyError on the
    first unmet dependency.'''
    registered = {d.name for d in self._compiler.dialects}
    for d in self._compiler.dialects:
      for p in (*d.lowerings, *d.rewrites):
        for needed in p.consumes:
          if needed not in registered:
            raise PassDependencyError(
              pass_name=p.name,
              missing_dialect=needed,
              in_dialect=d.name,
            )

  def verify_all(self, prog: Any) -> list[Any]:
    '''Invoke every registered dialect's verifier on `prog` and
    aggregate the returned VerificationErrors. Returns [] if all
    verifiers pass.'''
    errors: list[Any] = []
    for d in self._compiler.dialects:
      if d.verifier is not None:
        errors.extend(d.verifier(prog))
    return errors

  def run(self, prog: Any) -> Any:
    '''Run all registered passes against `prog`. Returns the (possibly
    transformed) program.

    Validates pass dependencies first; raises PassDependencyError on
    unmet consumes. Then runs verifiers and returns prog unchanged
    (op-level dispatch is caller-driven for now; see class docstring).
    '''
    self.validate_dependencies()
    errors = self.verify_all(prog)
    if errors:
      raise RuntimeError(f'Verification failed: {errors}')
    return prog


__all__ = [
  'Lowering',
  'PassDependencyError',
  'PassDriver',
  'Rewrite',
  'lowering',
  'rewrite',
  'verifier',
]
