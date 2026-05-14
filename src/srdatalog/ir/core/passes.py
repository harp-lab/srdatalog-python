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

import dataclasses
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from srdatalog.ir.core.dialect import Compiler, Dialect
from srdatalog.ir.core.ops import Op
from srdatalog.ir.core.strategy import bottom_up, repeat, try_


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
# Rewrite-to-fixpoint + renderability verification (S4.9)
# Per docs/ir_dialect_contract.md §2 and §3.
# -----------------------------------------------------------------------------


class RewriteRegistrationConflict(Exception):
  '''Two `@rewrite` decorators target the same op type. Per the
  dialect contract (`docs/ir_dialect_contract.md` §2.2), at most one
  rewrite per op type is allowed until an optimizer phase introduces
  ordering metadata. Raised at PassDriver use, not at decoration time
  (decorators don't see sibling dialects).'''

  def __init__(self, op_type: type, in_dialects: list[str]) -> None:
    self.op_type = op_type
    self.in_dialects = in_dialects
    super().__init__(
      f'multiple rewrites registered for op {op_type.__name__}: '
      f'in dialects {in_dialects!r}. Per ir_dialect_contract.md §2.2, '
      f'at most one rewrite per op type. If both are normalizing, '
      f'merge them; if one is optimizing, defer until phase metadata exists.'
    )


@dataclass(frozen=True, slots=True)
class UnrenderableOp:
  '''An op surfaced by `verify_renderability`: present in the IR after
  rewrite-to-fixpoint, but no renderer is registered for the active
  target. Either a missing rewrite (the op should have decomposed) or
  a missing renderer (the op is intended as LEAF but the codegen
  hasn't been wired up).'''

  op_type: type
  target: str

  def __str__(self) -> str:
    return (
      f'no renderer registered for {self.op_type.__name__} '
      f'(module={self.op_type.__module__!r}) on target={self.target!r}'
    )


class UnrenderableOpError(Exception):
  '''`verify_renderability` found one or more ops with no renderer
  registered for the active target. Aggregates an `UnrenderableOp`
  per offending op type.'''

  def __init__(self, errors: list[UnrenderableOp]) -> None:
    self.errors = errors
    summary = '\n'.join(f'  - {e}' for e in errors)
    super().__init__(
      f'verify_renderability found {len(errors)} unrenderable op type(s):\n{summary}'
    )


@dataclass(frozen=True)
class RewriteContext:
  '''Context passed to `Rewrite.apply(op, ctx)` during fixpoint
  rewriting. Carries a back-reference to the Compiler so a rewrite
  can look up sibling dialects if needed.'''

  compiler: Compiler


def _walk(op: Any) -> Iterator[Op]:
  '''Pre-order iterator yielding `op` and every Op-typed descendant.

  Children are discovered the same way `core/strategy.py` does: walk
  dataclass fields, descend into Op-valued fields and into
  list/tuple containers. Non-Op fields are skipped.'''
  if not isinstance(op, Op):
    return
  yield op
  for f in dataclasses.fields(op):
    yield from _walk_value(getattr(op, f.name))


def _walk_value(v: Any) -> Iterator[Op]:
  if isinstance(v, Op):
    yield from _walk(v)
  elif isinstance(v, (list, tuple)):
    for x in v:
      yield from _walk_value(x)


# -----------------------------------------------------------------------------
# PassDriver
# -----------------------------------------------------------------------------


class PassDriver:
  '''Runs registered rewrites, lowerings, and verifiers.

  Responsibilities:

    1. `validate_dependencies` — every Lowering/Rewrite's `consumes`
       names a registered dialect. Loud failure on missing deps.
    2. `apply_rewrites_to_fixpoint` — bottom-up + repeat over all
       registered `@rewrite` rules, until a full pass is a no-op.
       Implements the "rewrite COMPOUND ops away before codegen sees
       them" half of the dialect contract
       (`docs/ir_dialect_contract.md` §2.1).
    3. `verify_renderability` — every op surviving fixpoint has a
       registered renderer for the active target. Loud-failure
       replacement for the implicit "everything in IIR has a
       renderer" assumption (contract §3).
    4. `verify_all` — runs each dialect's verifier.

  The driver does not know about specific dialects or targets. New
  dialects participate by being registered; renderability is checked
  via a caller-supplied `has_renderer` callable (so target-specific
  registries stay out of `core/`). This preserves the
  no-imports-from-dialects invariant in `ir/core/CLAUDE.md`.
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

  def _build_rewrite_table(self) -> dict[type, tuple[Rewrite, str]]:
    '''Index every registered Rewrite by its `matches` op type.

    Returns `{op_type: (rewrite, dialect_name)}`. Raises
    `RewriteRegistrationConflict` if two dialects register a rewrite
    for the same op type (contract §2.2).'''
    table: dict[type, tuple[Rewrite, str]] = {}
    for d in self._compiler.dialects:
      for r in d.rewrites:
        if r.matches in table:
          _existing_rule, existing_dialect = table[r.matches]
          raise RewriteRegistrationConflict(
            op_type=r.matches,
            in_dialects=[existing_dialect, d.name],
          )
        table[r.matches] = (r, d.name)
    return table

  def apply_rewrites_to_fixpoint(self, prog: Any, *, max_iters: int = 1024) -> Any:
    '''Apply registered `@rewrite` rules to `prog` until a full
    bottom-up pass produces no change.

    Bottom-up + repeat semantics: each iteration walks the tree
    postorder, applies the matching rewrite (if any) at each node,
    rebuilds parents with `dataclasses.replace`. Iterates the whole
    pass until fixpoint. Raises `RuntimeError` (from `repeat`) if
    `max_iters` is exceeded — the catch for divergent rewrites.

    No-op when no rewrites are registered (early return).'''
    table = self._build_rewrite_table()
    if not table:
      return prog
    ctx = RewriteContext(compiler=self._compiler)

    def _rewrite_one(op: Op) -> Op | None:
      entry = table.get(type(op))
      if entry is None:
        return None
      rule, _dialect_name = entry
      return rule.apply(op, ctx)

    return repeat(bottom_up(try_(_rewrite_one)), max_iters=max_iters)(prog)

  def verify_renderability(
    self,
    prog: Any,
    *,
    target: str,
    has_renderer: Callable[[type], bool],
  ) -> list[UnrenderableOp]:
    '''Walk `prog` and return one `UnrenderableOp` per op type that
    has no renderer for `target`. Returns `[]` if every op is
    renderable.

    `target` is a label used only in error messages — the actual
    decision is delegated to `has_renderer(op_type) -> bool`. This
    keeps `core/passes.py` decoupled from any specific codegen
    registry (per the `ir/core/CLAUDE.md` no-imports-from-dialects
    invariant).

    Caller wires it up with the target's renderer registry:

        from srdatalog.ir.codegen.cuda.render import has_renderer
        errs = driver.verify_renderability(
            prog, target='cuda', has_renderer=has_renderer,
        )
    '''
    errors: list[UnrenderableOp] = []
    seen: set[type] = set()
    for op in _walk(prog):
      op_type = type(op)
      if op_type in seen:
        continue
      seen.add(op_type)
      if not has_renderer(op_type):
        errors.append(UnrenderableOp(op_type=op_type, target=target))
    return errors

  def verify_all(self, prog: Any) -> list[Any]:
    '''Invoke every registered dialect's verifier on `prog` and
    aggregate the returned VerificationErrors. Returns [] if all
    verifiers pass.'''
    errors: list[Any] = []
    for d in self._compiler.dialects:
      if d.verifier is not None:
        errors.extend(d.verifier(prog))
    return errors

  def run(
    self,
    prog: Any,
    *,
    target: str | None = None,
    has_renderer: Callable[[type], bool] | None = None,
  ) -> Any:
    '''Run all registered passes against `prog`. Returns the
    (possibly transformed) program.

    Pipeline (per `docs/ir_dialect_contract.md` §2):

      1. `validate_dependencies` — catches missing-dialect deps.
      2. `apply_rewrites_to_fixpoint` — decomposes COMPOUND ops.
      3. `verify_renderability` — only when `target` and
         `has_renderer` are both provided. Raises
         `UnrenderableOpError` on closure violation.
      4. `verify_all` — per-dialect verifiers.

    The renderability check is opt-in (target-aware); legacy callers
    that don't pass `target=` keep their pre-S4.9 behavior.'''
    self.validate_dependencies()
    prog = self.apply_rewrites_to_fixpoint(prog)
    if target is not None:
      if has_renderer is None:
        raise ValueError(
          "PassDriver.run: target=given but has_renderer=None — provide "
          "the target's renderer-check callable so verify_renderability "
          'can run.'
        )
      rerrs = self.verify_renderability(prog, target=target, has_renderer=has_renderer)
      if rerrs:
        raise UnrenderableOpError(rerrs)
    errors = self.verify_all(prog)
    if errors:
      raise RuntimeError(f'Verification failed: {errors}')
    return prog


__all__ = [
  'Lowering',
  'PassDependencyError',
  'PassDriver',
  'Rewrite',
  'RewriteContext',
  'RewriteRegistrationConflict',
  'UnrenderableOp',
  'UnrenderableOpError',
  'lowering',
  'rewrite',
  'verifier',
]
