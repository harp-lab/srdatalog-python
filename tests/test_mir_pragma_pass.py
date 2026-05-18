'''Tests for `srdatalog.ir.mir.pragma_pass.MirPragmaPass` (Phase C1).

Covers:
  - No-op behavior on EPs without pragmas (the C1 default state).
  - Hand-built pragma round-trip: register a handler, attach the
    pragma to an EP, run the pass, observe the wrap-op insertion +
    the cleared `pragmas` field.
  - Topo-sort ordering via `before` / `after` declarations.
  - Cycle detection -> `PragmaOrderingError`.
  - Unregistered pragma -> `UnregisteredPragmaError` (with a
    did-you-mean hint).
  - Buggy handler that doesn't strip its pragma ->
    `UnconsumedPragmaError`.
  - Driving via `Compiler.run` end-to-end (pre-flight ordering,
    `consumes='mir'` -> `produces='mir'`).
  - Discipline test scaffold: every EP has empty `pragmas` after
    `MirPragmaPass` (trivially satisfied today; first real gate
    when C2 lands).

Spec: docs/phase_c_pragma_materialization.md §3 + §7.
'''

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import final

import pytest

from srdatalog.ir.core import Compiler, Dialect
from srdatalog.ir.core.pragma import (
  Pragma,
  PragmaOrderingError,
  PragmaRegistration,
  UnconsumedPragmaError,
  UnregisteredPragmaError,
  pragma_handler,
)
from srdatalog.ir.hir.types import Version
from srdatalog.ir.mir.pragma_pass import MirPragmaPass
from srdatalog.ir.mir.types import (
  Block,
  ExecutePipeline,
  FixpointPlan,
  InsertInto,
  Program,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def isolated_registry(monkeypatch):
  '''Snapshot + restore the module-global pragma registry.

  Mirrors the fixture in tests/test_core_pragma.py so this test
  module can register handlers without leaking into others (and
  vice versa).
  '''
  fresh: list[PragmaRegistration] = []
  monkeypatch.setattr('srdatalog.ir.core.pragma._PRAGMA_REGISTRY', fresh)
  return fresh


@pytest.fixture
def mir_compiler():
  '''Compiler with the MIR dialect registered (no others).

  Sufficient for C1's `MirPragmaPass` driver tests; future phases
  will add the parallel.* sub-dialects via plugin discovery.
  '''
  from srdatalog.ir.mir import DIALECT as MIR_DIALECT

  c = Compiler()
  c.register_dialect(MIR_DIALECT)
  return c


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _ep(*, rule_name: str = 'r', pragmas: tuple[Pragma, ...] = ()) -> ExecutePipeline:
  '''Build a minimal ExecutePipeline with the given pragmas.

  C1 attaches typed `Pragma` instances directly onto the
  `pragmas` tuple; the field's static annotation still says
  `tuple[tuple[str, Any], ...]` (until A3 retypes it), but Python
  accepts the typed shape at runtime — that is the whole point of
  the migration.
  '''
  return ExecutePipeline(
    pipeline=[],
    source_specs=[],
    dest_specs=[],
    rule_name=rule_name,
    pragmas=pragmas,  # type: ignore[arg-type]
  )


def _strip_pragma(ep: ExecutePipeline, instance: Pragma) -> ExecutePipeline:
  '''Return `ep` with `instance` removed from `ep.pragmas`.'''
  return dataclasses.replace(
    ep,
    pragmas=tuple(p for p in ep.pragmas if p is not instance),
  )


# -----------------------------------------------------------------------------
# 1. No-op smoke
# -----------------------------------------------------------------------------


def test_noop_on_empty_pragmas_returns_identity_ep(isolated_registry, mir_compiler):
  '''An EP with `pragmas=()` is returned by-identity (the bottom_up
  walker only replaces nodes whose `_materialize` returns non-None).
  '''
  ep = _ep()
  pas = MirPragmaPass()
  out = pas.apply(ep, mir_compiler)
  assert out is ep


def test_noop_on_program_with_many_eps(isolated_registry, mir_compiler):
  '''A nested Program containing several EPs (all `pragmas=()`)
  walks cleanly and returns by-identity.'''
  prog = Program(
    steps=[
      (
        FixpointPlan(
          instructions=[_ep(rule_name='a'), _ep(rule_name='b')],
        ),
        True,
      ),
      (Block(instructions=[_ep(rule_name='c')]), False),
    ]
  )
  out = MirPragmaPass().apply(prog, mir_compiler)
  assert out is prog


# -----------------------------------------------------------------------------
# 2. Hand-built pragma round-trip
# -----------------------------------------------------------------------------


@final
@dataclass(frozen=True, slots=True)
class _TagPragma(Pragma):
  '''Test-local pragma: handler appends `tag` to the EP's rule_name
  and strips the instance.'''

  tag: str = 'X'


def test_handler_consumes_pragma_and_clears_field(isolated_registry, mir_compiler):
  '''A registered handler fires, returns a transformed EP, and the
  `pragmas` field is empty afterwards.'''

  @pragma_handler(_TagPragma, on=ExecutePipeline)
  def _handle(op, pragma, ctx):
    return dataclasses.replace(
      _strip_pragma(op, pragma),
      rule_name=f'{op.rule_name}+{pragma.tag}',
    )

  ep = _ep(rule_name='r', pragmas=(_TagPragma(tag='hello'),))
  out = MirPragmaPass().apply(ep, mir_compiler)
  assert out.rule_name == 'r+hello'
  assert out.pragmas == ()


def test_handler_fires_inside_nested_program(isolated_registry, mir_compiler):
  '''Handler discovery walks nested EPs (FixpointPlan -> EP, Block ->
  EP). Both the inner EPs get materialized; the outer Program is
  rebuilt with the transformed children.'''

  @pragma_handler(_TagPragma, on=ExecutePipeline)
  def _handle(op, pragma, ctx):
    return dataclasses.replace(
      _strip_pragma(op, pragma),
      rule_name=f'{op.rule_name}!',
    )

  inner_a = _ep(rule_name='a', pragmas=(_TagPragma(),))
  inner_b = _ep(rule_name='b')  # no pragma, untouched
  prog = Program(
    steps=[(FixpointPlan(instructions=[inner_a, inner_b]), True)],
  )

  out = MirPragmaPass().apply(prog, mir_compiler)
  fp_out = out.steps[0][0]
  assert fp_out.instructions[0].rule_name == 'a!'
  assert fp_out.instructions[0].pragmas == ()
  assert fp_out.instructions[1] is inner_b


# -----------------------------------------------------------------------------
# 3. Topo-sort ordering
# -----------------------------------------------------------------------------


@final
@dataclass(frozen=True, slots=True)
class _PragmaA(Pragma):
  pass


@final
@dataclass(frozen=True, slots=True)
class _PragmaB(Pragma):
  pass


def test_before_constraint_orders_handlers(isolated_registry, mir_compiler):
  '''`@pragma_handler(A, before=(B,))` => A's handler runs before
  B's. We tag the rule_name in fire-order; the final string
  encodes the sequence.'''

  @pragma_handler(_PragmaA, on=ExecutePipeline, before=(_PragmaB,))
  def _ha(op, pragma, ctx):
    return dataclasses.replace(
      _strip_pragma(op, pragma),
      rule_name=op.rule_name + 'A',
    )

  @pragma_handler(_PragmaB, on=ExecutePipeline)
  def _hb(op, pragma, ctx):
    return dataclasses.replace(
      _strip_pragma(op, pragma),
      rule_name=op.rule_name + 'B',
    )

  # Attach both pragmas in the OPPOSITE order to the desired fire
  # order — proves the topo-sort, not the iteration order, drives
  # the dispatch.
  ep = _ep(rule_name='', pragmas=(_PragmaB(), _PragmaA()))
  out = MirPragmaPass().apply(ep, mir_compiler)
  assert out.rule_name == 'AB'
  assert out.pragmas == ()


def test_after_constraint_orders_handlers(isolated_registry, mir_compiler):
  '''`@pragma_handler(B, after=(A,))` => B's handler runs after A's.
  Symmetric to `before`; same expected order.'''

  @pragma_handler(_PragmaA, on=ExecutePipeline)
  def _ha(op, pragma, ctx):
    return dataclasses.replace(
      _strip_pragma(op, pragma),
      rule_name=op.rule_name + 'A',
    )

  @pragma_handler(_PragmaB, on=ExecutePipeline, after=(_PragmaA,))
  def _hb(op, pragma, ctx):
    return dataclasses.replace(
      _strip_pragma(op, pragma),
      rule_name=op.rule_name + 'B',
    )

  ep = _ep(rule_name='', pragmas=(_PragmaA(), _PragmaB()))
  out = MirPragmaPass().apply(ep, mir_compiler)
  assert out.rule_name == 'AB'


def test_cycle_in_before_after_raises_pragma_ordering_error(isolated_registry, mir_compiler):
  '''A.before=(B,) AND B.before=(A,) => topo-sort cycle => raise
  PragmaOrderingError naming both classes.'''

  @pragma_handler(_PragmaA, on=ExecutePipeline, before=(_PragmaB,))
  def _ha(op, pragma, ctx):
    return _strip_pragma(op, pragma)

  @pragma_handler(_PragmaB, on=ExecutePipeline, before=(_PragmaA,))
  def _hb(op, pragma, ctx):
    return _strip_pragma(op, pragma)

  ep = _ep(pragmas=(_PragmaA(), _PragmaB()))
  with pytest.raises(PragmaOrderingError, match=r'cycle'):
    MirPragmaPass().apply(ep, mir_compiler)


# -----------------------------------------------------------------------------
# 4. Unregistered pragma
# -----------------------------------------------------------------------------


@final
@dataclass(frozen=True, slots=True)
class _NotRegistered(Pragma):
  pass


def test_unregistered_pragma_raises_with_did_you_mean(isolated_registry, mir_compiler):
  '''A pragma instance whose type has no registered handler triggers
  `UnregisteredPragmaError` at pre-flight; the error message lists
  registered handler names so the user can spot a typo / missing
  import.'''

  @pragma_handler(_PragmaA, on=ExecutePipeline)
  def _ha(op, pragma, ctx):
    return _strip_pragma(op, pragma)

  ep = _ep(pragmas=(_NotRegistered(),))
  with pytest.raises(UnregisteredPragmaError) as ei:
    MirPragmaPass().apply(ep, mir_compiler)
  msg = str(ei.value)
  assert '_NotRegistered' in msg
  assert '_PragmaA' in msg  # did-you-mean hint lists known handlers
  assert 'Did you mean' in msg


def test_unregistered_pragma_error_is_typeerror(isolated_registry, mir_compiler):
  '''Per `core.pragma` docstring: UnregisteredPragmaError IS a
  TypeError, so existing `except TypeError` catches it.'''
  ep = _ep(pragmas=(_NotRegistered(),))
  with pytest.raises(TypeError):
    MirPragmaPass().apply(ep, mir_compiler)


# -----------------------------------------------------------------------------
# 5. Buggy (non-stripping) handler -> UnconsumedPragmaError
# -----------------------------------------------------------------------------


def test_handler_that_doesnt_strip_raises_unconsumed(isolated_registry, mir_compiler):
  '''A handler that returns the EP unchanged (or mutates other
  fields but forgets to drop its own Pragma instance) leaves
  `pragmas` non-empty after dispatch -> UnconsumedPragmaError.'''

  @pragma_handler(_PragmaA, on=ExecutePipeline)
  def _bad_handler(op, pragma, ctx):
    # Bug: returns the EP with the pragma still attached.
    return dataclasses.replace(op, rule_name=op.rule_name + '?')

  ep = _ep(pragmas=(_PragmaA(),))
  with pytest.raises(UnconsumedPragmaError, match=r'survived MirPragmaPass'):
    MirPragmaPass().apply(ep, mir_compiler)


def test_handler_returning_none_also_triggers_unconsumed(isolated_registry, mir_compiler):
  '''Per spec §2 the handler may return None to skip; that leaves the
  pragma on the EP, which the post-flight check then reports as an
  UnconsumedPragmaError. The two paths converge on the same error so
  no path silently survives.'''

  @pragma_handler(_PragmaA, on=ExecutePipeline)
  def _skip_handler(op, pragma, ctx):
    return None

  ep = _ep(pragmas=(_PragmaA(),))
  with pytest.raises(UnconsumedPragmaError):
    MirPragmaPass().apply(ep, mir_compiler)


# -----------------------------------------------------------------------------
# 6. Driven via Compiler.run
# -----------------------------------------------------------------------------


def test_pass_runs_via_compiler_run_pipeline(isolated_registry, mir_compiler):
  '''MirPragmaPass declares consumes=('mir',) and produces=('mir',).
  With the MIR dialect registered, `Compiler.run(prog,
  pipeline=[MirPragmaPass()])` passes pre-flight ordering and
  threads the prog through.'''

  @pragma_handler(_PragmaA, on=ExecutePipeline)
  def _ha(op, pragma, ctx):
    return dataclasses.replace(
      _strip_pragma(op, pragma),
      rule_name=op.rule_name + '!',
    )

  ep = _ep(rule_name='r', pragmas=(_PragmaA(),))
  out = mir_compiler.run(ep, pipeline=[MirPragmaPass()])
  assert out.rule_name == 'r!'
  assert out.pragmas == ()


def test_pass_default_metadata_advertises_mir_dialect():
  '''The Pass declares the right `name`, `consumes`, `produces`,
  `dialect_name` so test_pass_kinds-style introspection picks it up.
  '''
  p = MirPragmaPass()
  assert p.name == 'mir_pragma_materialization'
  assert p.consumes == ('mir',)
  assert p.produces == ('mir',)
  assert p.dialect_name == 'mir'
  assert p.until_fixpoint is False


# -----------------------------------------------------------------------------
# 7. Discipline scaffold (Spec §7)
# -----------------------------------------------------------------------------


def test_pragmas_empty_after_materialization(isolated_registry, mir_compiler):
  '''Discipline test (Spec §7): every EP has `pragmas == ()` after
  `MirPragmaPass`. C1 satisfies this trivially because every EP
  starts with `pragmas == ()`; the test is the gate for C2's first
  migration -> any EP that arrives at `MirPragmaPass` carrying a
  typed Pragma whose handler is missing or buggy fails this gate.

  We use a representative MIR program: a Program with a
  FixpointPlan + Block, several EPs, and a non-EP wrap (InsertInto
  has no `pragmas`, must walk past it cleanly).
  '''
  prog = Program(
    steps=[
      (
        FixpointPlan(
          instructions=[
            _ep(rule_name='a'),
            _ep(rule_name='b'),
          ],
        ),
        True,
      ),
      (
        Block(
          instructions=[
            _ep(rule_name='c'),
            InsertInto(
              rel_name='R',
              version=Version.FULL,
              vars=[],
              index=[],
            ),
          ],
        ),
        False,
      ),
    ]
  )

  out = MirPragmaPass().apply(prog, mir_compiler)

  # Walk every EP in the output and assert pragmas == ().
  def _walk_eps(node, found):
    if isinstance(node, ExecutePipeline):
      found.append(node)
    elif isinstance(node, Program):
      for step, _ in node.steps:
        _walk_eps(step, found)
    elif isinstance(node, FixpointPlan) or isinstance(node, Block):
      for instr in node.instructions:
        _walk_eps(instr, found)

  found: list[ExecutePipeline] = []
  _walk_eps(out, found)
  assert len(found) == 3
  for ep in found:
    assert ep.pragmas == ()


# -----------------------------------------------------------------------------
# 8. Misc / surface checks
# -----------------------------------------------------------------------------


def test_pass_skips_handlers_for_other_op_types(isolated_registry, mir_compiler):
  '''A `@pragma_handler` registered with `on=SomeOtherOp` is ignored
  by MirPragmaPass — the pass only dispatches handlers whose `on`
  is `ExecutePipeline`. Otherwise an unrelated registration could
  fire on every EP and corrupt the dispatch.'''

  class _OtherOp:
    pass

  @pragma_handler(_PragmaA, on=_OtherOp)
  def _ha(op, pragma, ctx):  # pragma: no cover — must not fire
    raise AssertionError('handler should not fire for non-ExecutePipeline')

  # An EP with the same Pragma type, but the handler is for a
  # different op kind. The pre-flight then sees no registered
  # handler for ExecutePipeline + _PragmaA and raises.
  ep = _ep(pragmas=(_PragmaA(),))
  with pytest.raises(UnregisteredPragmaError):
    MirPragmaPass().apply(ep, mir_compiler)


def test_dialect_registration_is_used_by_compiler_run(mir_compiler):
  '''Smoke: the mir_compiler fixture has the mir dialect registered
  (so `Compiler.run` pre-flight on `consumes=('mir',)` succeeds).
  '''
  names = [d.name for d in mir_compiler.dialects]
  assert 'mir' in names


def test_dialect_unregistered_run_raises_pass_ordering_error(isolated_registry):
  '''Without the MIR dialect registered, `Compiler.run` pre-flight
  rejects MirPragmaPass on `consumes=('mir',)` — the same gate the
  rest of the F1 framework enforces.'''
  from srdatalog.ir.core.passes import PassOrderingError

  empty = Compiler()
  with pytest.raises(PassOrderingError):
    empty.run(_ep(), pipeline=[MirPragmaPass()])


def test_external_dialect_test_helper_used():
  '''Sanity that the Dialect import path used elsewhere works (the
  C2 PR will register its handler from a dialect's __init__).'''
  d = Dialect(name='test.dummy')
  assert d.name == 'test.dummy'
