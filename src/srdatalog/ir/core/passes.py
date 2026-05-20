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

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from srdatalog.ir.core.dialect import Compiler, Dialect
from srdatalog.ir.core.strategy import bottom_up, repeat


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


# -----------------------------------------------------------------------------
# Pass kinds (Layer 1 / F1)
#
# Per docs/compiler_redesign.md §4 (the three Pass kinds) and
# docs/phase_b_lowering_dispatcher.md §2 (LoweringPass algorithm), the
# framework recognizes exactly three kinds of `Pass`:
#
#   LoweringPass  — cross-dialect, source-driven, per-op dispatch
#                    (HIR → MIR, MIR → IIR).
#   RewritePass   — intra-dialect, op-driven, per-op dispatch
#                    (pragma materialization, MIR opts, IIR canonicalization).
#   ProgramPass   — whole-program transformation (HIR planning).
#
# `Pass.apply(prog, compiler)` returns the (possibly transformed) prog.
# `Compiler.run(prog, pipeline=[...])` drives a list of `Pass` instances.
#
# This is Layer 1 foundation. The full LowerCtx (5-field per
# compiler_redesign.md §5) lands in F3; F1 ships only the minimum
# dispatcher needed to wire up the three Pass kinds and prove the
# pipeline shape.
# -----------------------------------------------------------------------------


class PassOrderingError(Exception):
  '''A Pass's `consumes` references a dialect that no earlier pass
  in the pipeline has produced (and that isn't a Compiler-registered
  dialect at run start).

  Raised at construction-time validation in `Compiler.run`, before any
  pass executes. Mirrors `PassDependencyError` but operates on a
  pipeline list rather than the registered dialect set.
  '''

  def __init__(self, pass_name: str, missing: str, position: int) -> None:
    self.pass_name = pass_name
    self.missing = missing
    self.position = position
    super().__init__(
      f'pass {pass_name!r} at pipeline position {position} '
      f'requires dialect {missing!r} which no earlier pass produces '
      f"and which isn't registered with the Compiler. Either register "
      f'the dialect or insert a pass that produces it earlier.'
    )


class AmbiguousLowering(Exception):
  '''Two `@lowering` registrations target the same source op type
  for the same target dialect.

  Raised by `LoweringPass._build_table` at `apply()` time, before
  any op is dispatched.
  '''

  def __init__(self, source_op_type: type, target_dialect_name: str) -> None:
    self.source_op_type = source_op_type
    self.target_dialect_name = target_dialect_name
    super().__init__(
      f'multiple @lowering registrations match source op type '
      f'{source_op_type.__name__!r} for target dialect '
      f'{target_dialect_name!r}; expected exactly one.'
    )


class LoweringMissingError(Exception):
  '''No `@lowering` registered for `type(op)` targeting this
  LoweringPass's `target_dialect_name`.

  Raised by the dispatcher at `ctx.lower(op)` time. Per
  `phase_b_lowering_dispatcher.md` §2, this is the runtime safety
  net; discipline test R4 catches it earlier at registration time
  for built-in ops.
  '''

  def __init__(self, op_type: type, target_dialect_name: str) -> None:
    self.op_type = op_type
    self.target_dialect_name = target_dialect_name
    super().__init__(
      f'no @lowering registered for source op type {op_type.__name__!r} '
      f'targeting dialect {target_dialect_name!r}.'
    )


@dataclass(frozen=True)
class Pass(ABC):
  '''Abstract base for every compile-pipeline step.

  Three concrete kinds: `LoweringPass`, `RewritePass`, `ProgramPass`.
  `Pass.apply(prog, compiler)` returns the (possibly transformed) prog.

  Per `docs/compiler_redesign.md` §4: every step in every
  `Compiler.run` pipeline is one of these three kinds; nothing else.

  `consumes` / `produces` (str-typed, dialect names) declare dialect
  dependencies for the dialect-level ordering validation done by
  `Compiler.run` at construction time (per the R4 research report and
  `phase_b_lowering_dispatcher.md` §5). They are advisory data, not
  enforced inside `apply`.

  `consumes_ops` / `produces_ops` / `preserves` (PR-P0, spec §
  3.2.1.3) declare op-type-level dependencies for the LLVM-style
  topological scheduler. A pass that `produces_ops=(BgRoot, ...)`
  must run before any pass that `consumes_ops=(BgRoot, ...)`. The
  scheduler (`topo_sort_passes`) is consulted by
  `Compiler.register_pragma_plugin` when a plugin contributes new
  passes; the existing dialect-name `consumes` / `produces` remain
  the surface for legacy pipelines built by hand.

  Naming note: the spec example in § 3.2.1.3 / § 6.0.0 lists Pass
  fields as `produces` / `consumes` (typed as op classes). The
  pre-PR-P0 framework already uses those names for str-typed dialect
  IDs (`tuple[str, ...]`); renaming would break every existing pass
  subclass and every pipeline that constructs them. The op-typed
  fields are introduced under suffixed names (`produces_ops`,
  `consumes_ops`, `preserves`) so the two ordering surfaces coexist
  cleanly during the per-pragma migration phases (PR-P1..P5).
  '''

  name: str
  consumes: tuple[str, ...] = ()
  produces: tuple[str, ...] = ()
  consumes_ops: tuple[type, ...] = ()
  produces_ops: tuple[type, ...] = ()
  preserves: tuple[type, ...] = ()

  @abstractmethod
  def apply(self, prog: Any, compiler: Any) -> Any:
    '''Run this pass against `prog`. Returns the (possibly
    transformed) prog. Implementations are pure functions: no
    Compiler mutation, no global state.'''
    ...


@dataclass(frozen=True)
class LoweringPass(Pass):
  '''Cross-dialect, source-driven, per-op dispatch.

  Walks the input tree via the dispatcher; for each op, looks up the
  registered `@lowering(target=self.target_dialect_name,
  source=type(op))` and applies it.

  Per the R1 design report:

    - **Source-driven dispatch.** The dispatch table is built at
      `apply()` time, keyed on op type. One `LoweringPass` instance
      = one source-to-target direction.
    - **Explicit `ctx.lower(child)` recursion.** No auto-walk.
      Lowerings call `ctx.lower(child)` themselves when they want
      to recurse into children. This mirrors MLIR's
      `ConversionPattern` and explicitly rejects the `bottom_up`-
      style auto-walk; lowerings that want auto-walk implement it
      themselves.

  F3 promotes the per-call ``ctx`` to the public 5-field `LowerCtx`
  (see `srdatalog.ir.core.lower_ctx`). The dispatch table is
  attached to the ctx via ``object.__setattr__`` — this is the one
  permitted use of that escape hatch on `LowerCtx` (framework infra,
  NOT a D18 transition shim; documented at the call site below).
  '''

  target_dialect_name: str = ''

  def apply(self, prog: Any, compiler: Any) -> Any:
    # Local imports avoid the framework-init cycle between
    # passes.py and lower_ctx.py.
    from srdatalog.ir.core.lower_ctx import LowerCtx, NameGen, ViewLayout

    table = self._build_table(compiler)
    ctx = LowerCtx(
      compiler=compiler,
      name_gen=NameGen(),
      view_layout=ViewLayout(),  # F5 will populate from prog
      plugin_registry=None,  # F4-shape; F5 wires up
      target=self.target_dialect_name,
    )
    # Framework infra: attach the per-call dispatch table to the
    # frozen ctx. This is the SOLE permitted use of object.__setattr__
    # on LowerCtx — it threads the LoweringPass-built table into the
    # ctx without growing LowerCtx beyond its D10-pinned 5 fields.
    # NOT counted in the D18 transitional-state ratchet (D18 covers
    # shims on frozen Op subclasses + DEPRECATED fields + module-
    # global mutable registries; framework dispatch wiring is none of
    # those).
    object.__setattr__(ctx, '_table', table)
    return ctx.lower(prog)

  def _build_table(self, compiler: Any) -> dict[type, Lowering]:
    '''Build the per-pass dispatch table mapping source op type to
    registered Lowering for `self.target_dialect_name`.

    Raises `AmbiguousLowering` if two registrations claim the same
    source op type for this target.

    A Lowering registered on dialect D targets D — i.e., it appears
    in `D.lowerings` and produces ops in D's vocabulary. So we walk
    all dialects and select Lowerings that belong to the target
    dialect (registered on it).
    '''
    table: dict[type, Lowering] = {}
    for d in compiler.dialects:
      if d.name != self.target_dialect_name:
        continue
      for low in d.lowerings:
        if low.matches in table:
          raise AmbiguousLowering(low.matches, self.target_dialect_name)
        table[low.matches] = low
    return table


@dataclass(frozen=True)
class RewritePass(Pass):
  '''Intra-dialect, op-driven, per-op dispatch (fixpoint optional).

  Walks the input tree; for each op, looks up registered
  `@rewrite(self.dialect_name, type(op))` and applies it. The
  result must be in the same dialect (or in declared
  `consumes`/`produces` dialects).

  Implemented on top of `strategy.bottom_up` + `strategy.repeat`:
  one bottom-up sweep applies every applicable rewrite at every op;
  with `until_fixpoint=True` (default), `repeat` re-runs until the
  tree stops changing.

  Per `compiler_redesign.md` §4: used for MIR pragma materialization
  (one-shot), MIR opts (R1–R5, fixpoint), IIR canonicalization
  (fixpoint).
  '''

  dialect_name: str = ''
  until_fixpoint: bool = True

  def apply(self, prog: Any, compiler: Any) -> Any:
    rewrites_by_type = self._build_table(compiler)
    if not rewrites_by_type:
      return prog

    def _rewrite_one(op: Any) -> Any | None:
      rs = rewrites_by_type.get(type(op))
      if not rs:
        return None
      changed_any = False
      for r in rs:
        new = r.apply(op, compiler)
        if new is not None and new is not op:
          op = new
          changed_any = True
      return op if changed_any else None

    sweep = bottom_up(_rewrite_one)
    if self.until_fixpoint:
      driver = repeat(sweep)
      return driver(prog)
    return sweep(prog)

  def _build_table(self, compiler: Any) -> dict[type, list[Rewrite]]:
    '''Build a dispatch table mapping op type to the list of
    Rewrites registered on the named dialect that match it.

    A dialect may register multiple rewrites for the same op type
    (unlike Lowerings). They are tried in registration order in one
    bottom-up sweep.
    '''
    table: dict[type, list[Rewrite]] = {}
    for d in compiler.dialects:
      if d.name != self.dialect_name:
        continue
      for r in d.rewrites:
        table.setdefault(r.matches, []).append(r)
    return table


@dataclass(frozen=True)
class ProgramPass(Pass):
  '''Whole-program transformation. Per `docs/phase_d_hir_passes.md`
  §2 (ProgramPass contract).

  Used for HIR planning passes that operate on `HirProgram` as a
  unit (stratify, semi-naive variant generation, plan, index
  selection, ...).

  HIR types are mutable planning records (not Op-subclassed), so
  they can't be walked by `LoweringPass` / `RewritePass` per-op
  dispatch. `ProgramPass` is the explicit escape hatch for
  whole-program transformations.

  The function is called as `fn(prog, compiler)` and may mutate or
  replace the program. Per the redesign §4: used ONLY for HIR
  planning; MIR / IIR transformations must be `LoweringPass` or
  `RewritePass`.
  '''

  fn: Callable[[Any, Any], Any] = field(default=lambda prog, _c: prog)

  def apply(self, prog: Any, compiler: Any) -> Any:
    return self.fn(prog, compiler)


class PassCycleError(Exception):
  '''Cycle in the op-type-level produces/consumes graph across the
  passes registered with a `Compiler` (PR-P0, spec § 3.2.1.3).

  Raised by `topo_sort_passes` when the LLVM-style scheduler cannot
  produce a linear order. Carries the cycle as a list of pass names
  for diagnostics.

  Distinct from `PassOrderingError` (pipeline-position-based dialect
  dependency error). `PassCycleError` is the op-type-graph error
  raised during plugin registration; `PassOrderingError` is the
  pipeline-construction error raised at `Compiler.run` time.
  '''

  def __init__(self, cycle: list[str]) -> None:
    self.cycle = cycle
    super().__init__(
      f'pass produces/consumes graph has a cycle involving: {cycle}. '
      f'A pass that consumes an op type must run after every pass that '
      f'produces it; declare an alternative ordering via `preserves` or '
      f'split the cyclic responsibility across separate passes. See '
      f'spec § 3.2.1.3 (LLVM-style topo-sort).'
    )


def topo_sort_passes(passes: list[Pass]) -> list[Pass]:
  '''Topologically sort `passes` by op-type-level produces/consumes.

  LLVM legacy-pass-manager shape (spec § 11.3): a pass that
  `produces_ops=(X, ...)` must run before any pass that
  `consumes_ops=(X, ...)`. Passes that share neither relation may run
  in any order; the scheduler picks a deterministic order (stable on
  the input list, lex-tiebreak on pass name when sets allow it).

  Returns a new list of `Pass` instances in topo-sorted order. Raises
  `PassCycleError` if no linear order exists.

  Self-edges (a pass that both produces and consumes the same op
  type) are tolerated — the produces side is interpreted as "after
  this pass, instances of X exist", and the consumes side as "this
  pass may read X". A pass legitimately may produce and consume the
  same type (the materialization passes do exactly that).
  '''
  # Map op type -> indices of passes that produce that type.
  producers: dict[type, list[int]] = {}
  for i, p in enumerate(passes):
    for op_type in p.produces_ops:
      producers.setdefault(op_type, []).append(i)

  # Build edges: producer index -> consumer index.
  out_edges: dict[int, set[int]] = {i: set() for i in range(len(passes))}
  in_degree: dict[int, int] = dict.fromkeys(range(len(passes)), 0)
  for j, p in enumerate(passes):
    for op_type in p.consumes_ops:
      for i in producers.get(op_type, []):
        if i == j:
          # Self-edge tolerated; see docstring.
          continue
        if j not in out_edges[i]:
          out_edges[i].add(j)
          in_degree[j] += 1

  # Kahn: pop lex-smallest (by pass name, then original index) zero-
  # in-degree node, repeat. Stable on inputs where the graph permits.
  def _sort_key(i: int) -> tuple[str, int]:
    return (passes[i].name, i)

  ready = sorted([i for i, d in in_degree.items() if d == 0], key=_sort_key)
  order: list[int] = []
  while ready:
    i = ready.pop(0)
    order.append(i)
    new_ready: list[int] = []
    for j in sorted(out_edges[i], key=_sort_key):
      in_degree[j] -= 1
      if in_degree[j] == 0:
        new_ready.append(j)
    ready = sorted(set(ready) | set(new_ready), key=_sort_key)

  if len(order) != len(passes):
    remaining = sorted(set(range(len(passes))) - set(order))
    raise PassCycleError([passes[i].name for i in remaining])
  return [passes[i] for i in order]


def program_pass(
  *,
  name: str,
  consumes: tuple[str, ...] = (),
  produces: tuple[str, ...] = (),
) -> Callable[[Callable[[Any, Any], Any]], ProgramPass]:
  '''Decorator: wrap a function as a `ProgramPass` instance.

  Usage:

      @program_pass(name="stratify",
                    consumes=("hir",),
                    produces=("hir",))
      def stratify_pass(prog, compiler):
          ...   # imperative HIR transformation
          return prog

  The decorated name is replaced by the `ProgramPass` instance, so
  it can be inserted directly into a pipeline list. Per
  `phase_d_hir_passes.md` §3.1.
  '''

  def _wrap(fn: Callable[[Any, Any], Any]) -> ProgramPass:
    return ProgramPass(
      name=name,
      consumes=tuple(consumes),
      produces=tuple(produces),
      fn=fn,
    )

  return _wrap


__all__ = [
  'AmbiguousLowering',
  'Lowering',
  'LoweringMissingError',
  'LoweringPass',
  'Pass',
  'PassCycleError',
  'PassDependencyError',
  'PassDriver',
  'PassOrderingError',
  'ProgramPass',
  'Rewrite',
  'RewritePass',
  'lowering',
  'program_pass',
  'rewrite',
  'topo_sort_passes',
  'verifier',
]
