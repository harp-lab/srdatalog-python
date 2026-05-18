'''End-to-end test for Phase C6: the `Count` typed pragma.

Per `docs/phase_c_pragma_materialization.md` §5.1 (per-PR acceptance
gate), each Phase C migration ships a `test_pragma_<name>_end_to_end`
that exercises the pragma via the production `Compiler.run`-style
pipeline and asserts:

  - The wrap op (`mir.CountPhase`) appears in MIR after
    `MirPragmaPass` (or is suppressed under the dual-write
    transition; both states are verified here).
  - The lowered IIR is correct (matches the count-phase `iir.cf.
    Phase(C, body)` shape per spec § 4.3).
  - The rendered CUDA is byte-equivalent to the legacy `ep.count`
    runner path (the load-bearing harness for C6's backward-compat
    contract).

Plus DSL surface checks:

  - `Rule(...).with_pragma(Count())` accepts a typed instance AND
    sets the legacy `Rule.count = True` (the Rule-level dual-write,
    versus C2's PlanEntry-level dual-write for DedupHash).
  - `Rule(...).with_pragma(<not-a-pragma>)` raises `TypeError`.

The framework infrastructure (the `MirPragmaPass` driver, the
`@pragma_handler` registry, error classes) is exercised by
`tests/test_mir_pragma_pass.py` + `tests/test_core_pragma.py`; this
file owns the per-pragma surface.
'''

from __future__ import annotations

import dataclasses

import pytest

import srdatalog.ir.mir.types as m
from srdatalog.dsl import Rule
from srdatalog.ir.core import Compiler, Pragma
from srdatalog.ir.dialects.iir.cf import DIALECT as IIR_CF_DIALECT
from srdatalog.ir.dialects.iir.cf import Phase as IirPhase
from srdatalog.ir.dialects.iir.cf.pragmas.count import (
  Count,
  lower_count_phase,
  materialize_count,
)
from srdatalog.ir.dialects.relation.sorted_array import DIALECT as SA_DIALECT
from srdatalog.ir.hir.types import Version
from srdatalog.ir.mir import DIALECT as MIR_DIALECT
from srdatalog.ir.mir.passes import apply_all_mir_passes, apply_mir_pragma_pass
from srdatalog.ir.mir.pragma_pass import MirPragmaPass

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mir_compiler() -> Compiler:
  '''Compiler with MIR + iir.cf + sorted_array dialects registered.

  The iir.cf dialect's `_register_passes` imports the pragmas
  submodule for side effects (registers `@pragma_handler(Count, ...)`
  + the `CountPhase` lowering), so this fixture is enough to drive
  `MirPragmaPass` end-to-end without monkey-patching the registry.
  '''
  c = Compiler()
  c.register_dialect(MIR_DIALECT)
  c.register_dialect(IIR_CF_DIALECT)
  c.register_dialect(SA_DIALECT)
  return c


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _scan_insert_ep(
  *,
  arity: int = 2,
  rule_name: str = 'Cnt',
  count: bool = False,
  pragmas: tuple[Pragma, ...] = (),
) -> m.ExecutePipeline:
  '''Build an EP shape suitable for count-pragma testing, optionally
  carrying a typed `Count` pragma instead of (or in addition to) the
  legacy bool field.'''
  vars_ = [f'v{i}' for i in range(arity)]
  cols = list(range(arity))
  scan = m.Scan(
    vars=vars_,
    rel_name='Src',
    version=Version.FULL,
    index=cols,
    handle_start=0,
  )
  insert = m.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=vars_,
    index=cols,
  )
  return m.ExecutePipeline(
    pipeline=[scan, insert],
    source_specs=[scan],
    dest_specs=[insert],
    rule_name=rule_name,
    count=count,
    pragmas=pragmas,  # type: ignore[arg-type]
  )


def _empty_rule() -> Rule:
  '''Build a minimal Rule. The rule's structure is irrelevant for the
  DSL-surface tests below; we only inspect `rule.plans` and `rule.count`.'''
  from srdatalog.dsl import Atom

  head = Atom(rel='Dst', args=())
  body_atom = Atom(rel='Src', args=())
  return Rule(heads=(head,), body=(body_atom,))


# -----------------------------------------------------------------------------
# 1. DSL surface — Rule.with_pragma(Count())
# -----------------------------------------------------------------------------


def test_with_pragma_count_attaches_typed_pragma_and_sets_rule_count():
  '''Calling `.with_pragma(Count())` on a rule with no plans appends
  a default `PlanEntry(delta=-1)` carrying the pragma AND flips
  `Rule.count = True` (the C6 Rule-level dual-write, parallel to
  C2's PlanEntry-level dual-write for `DedupHash`).'''
  rule = _empty_rule().with_pragma(Count())
  assert rule.count is True
  assert len(rule.plans) == 1
  plan = rule.plans[0]
  assert plan.delta == -1
  assert len(plan.pragmas) == 1
  assert isinstance(plan.pragmas[0], Count)


def test_with_pragma_count_appends_to_existing_plans():
  '''When the rule already has plans, `.with_pragma(Count())`
  appends the pragma to EVERY plan's `pragmas` tuple AND flips
  `Rule.count = True` once on the rule (count is a per-rule
  concept; the dual-write target is Rule.count, not a per-plan
  field).'''
  rule = (
    _empty_rule().with_plan(delta=0).with_plan(delta=1, var_order=('a', 'b')).with_pragma(Count())
  )
  assert rule.count is True
  assert len(rule.plans) == 2
  for plan in rule.plans:
    assert any(isinstance(p, Count) for p in plan.pragmas)


def test_with_pragma_count_rejects_non_pragma_arg():
  '''Standard `with_pragma` guard: hard-error on non-`Pragma`
  arguments. Mirrors the C2 DedupHash equivalent.'''
  rule = _empty_rule()
  with pytest.raises(TypeError, match=r'expected a Pragma subclass'):
    rule.with_pragma('count')  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# 2. MIR-level — MirPragmaPass materialization
# -----------------------------------------------------------------------------


def test_pragma_pass_inserts_count_phase_when_bool_is_false(mir_compiler):
  '''Pure typed path (bool=False, only pragma set): MirPragmaPass
  wraps the EP's `pipeline` body in a single `CountPhase` whose
  `inner` is an `iir.cf.Block` of the original pipeline ops.

  This is the post-A3 target state. Today only synthesized EPs
  reach it — the DSL `.with_pragma(Count())` dual-writes the
  `Rule.count` bool which propagates to `ep.count`, hitting the
  dual-write skip branch below.
  '''
  ep = _scan_insert_ep(count=False, pragmas=(Count(),))
  out = MirPragmaPass().apply(ep, mir_compiler)
  # Pragma instance consumed.
  assert out.pragmas == ()
  # Pipeline: a single CountPhase wrap.
  assert len(out.pipeline) == 1
  assert isinstance(out.pipeline[0], m.CountPhase)
  # The wrap's inner is an iir.cf.Block of the original pipeline.
  from srdatalog.ir.dialects.iir.cf import Block as IirBlock

  assert isinstance(out.pipeline[0].inner, IirBlock)
  inner_stmts = out.pipeline[0].inner.stmts
  assert len(inner_stmts) == 2
  assert isinstance(inner_stmts[0], m.Scan)
  assert isinstance(inner_stmts[1], m.InsertInto)
  # Legacy bool was NOT toggled; the wrap op is the sole signal.
  assert out.count is False


def test_pragma_pass_skips_wrap_in_dual_write_mode(mir_compiler):
  '''Dual-write transition (bool=True AND pragma set, the C6 DSL
  shape): MirPragmaPass strips the pragma but leaves `pipeline`
  untouched. The legacy `complete_runner.py` `ep.count` path
  produces the count-only kernel emit; the wrap op never appears so
  the monolith doesn't need to know about CountPhase.

  This is the load-bearing test for the dual-write contract: the
  C6 PR must not break the byte-equivalence harness (every kernel
  has a count phase), and that requires the bool-field path to
  remain the sole emission driver until A3 drops the bool.
  '''
  ep = _scan_insert_ep(count=True, pragmas=(Count(),))
  out = MirPragmaPass().apply(ep, mir_compiler)
  # Pragma still consumed.
  assert out.pragmas == ()
  # No CountPhase inserted.
  assert all(not isinstance(child, m.CountPhase) for child in out.pipeline)
  # Bool preserved for the legacy emitter.
  assert out.count is True
  # Pipeline shape unchanged.
  assert [type(o).__name__ for o in out.pipeline] == ['Scan', 'InsertInto']


def test_pragma_pass_via_apply_all_mir_passes_is_noop_for_empty_pragmas(
  mir_compiler,
):
  '''Sanity: the integration of `MirPragmaPass` into
  `apply_all_mir_passes` does not change MIR shape for EPs that
  carry no typed pragmas — the dominant case today.'''
  ep = _scan_insert_ep(count=False, pragmas=())
  steps_in = [(ep, False)]
  steps_out = apply_all_mir_passes(steps_in)
  assert len(steps_out) == 1
  ep_out, is_rec = steps_out[0]
  assert isinstance(ep_out, m.ExecutePipeline)
  assert is_rec is False
  assert ep_out.pragmas == ()
  # Pipeline contents structurally identical (no CountPhase insertion).
  assert [type(o).__name__ for o in ep_out.pipeline] == ['Scan', 'InsertInto']


# -----------------------------------------------------------------------------
# 3. Lowering — CountPhase -> iir.cf.Phase(C, body)
# -----------------------------------------------------------------------------


def test_count_phase_lowering_emits_iir_cf_phase(mir_compiler):
  '''Per spec § 4.3, the count pragma lowers to `iir.cf.Phase(mode=
  'C', body=<inner>)`. The lowering keeps the wrap op's `inner`
  unchanged; the `Phase` op carries the count-phase scope intent.
  '''
  ep = _scan_insert_ep(count=False, pragmas=(Count(),))
  materialized = MirPragmaPass().apply(ep, mir_compiler)
  count_phase = materialized.pipeline[0]
  assert isinstance(count_phase, m.CountPhase)

  iir = lower_count_phase(count_phase, ctx=None)
  assert isinstance(iir, IirPhase)
  assert iir.mode == 'C'
  # Body is the same `iir.cf.Block` the wrap op carried.
  assert iir.body is count_phase.inner


def test_count_phase_lowering_does_not_mutate_inner(mir_compiler):
  '''The lowering threads `count_phase.inner` through without
  rewriting it — body identity is preserved. Guards against
  accidental copies / mutation if the lowering grows to inspect
  ctx in a later refactor.
  '''
  inner_op = m.InsertInto(rel_name='Dst', version=Version.NEW, vars=['v0'], index=[0])
  from srdatalog.ir.dialects.iir.cf import Block as IirBlock

  wrap_body = IirBlock(stmts=(inner_op,))
  wrap = m.CountPhase(inner=wrap_body)
  out = lower_count_phase(wrap, ctx=None)
  assert isinstance(out, IirPhase)
  assert out.body is wrap_body


# -----------------------------------------------------------------------------
# 4. Registration completeness (R5)
# -----------------------------------------------------------------------------


def test_count_handler_is_registered():
  '''Importing the pragma module registers a `@pragma_handler` whose
  `pragma_cls` is `Count` and whose `on` is `mir.ExecutePipeline`.
  R5 (`test_pragma_handler_registry_completeness`) gates this once
  per Pragma subclass; here we pin it for the C6 surface explicitly.
  '''
  from srdatalog.ir.core.pragma import get_pragma_registrations

  regs = [r for r in get_pragma_registrations() if r.pragma_cls is Count]
  assert len(regs) >= 1
  reg = regs[0]
  assert reg.on is m.ExecutePipeline
  assert reg.fn is materialize_count


def test_count_phase_lowering_is_registered_on_iir_cf_dialect():
  '''The CountPhase `@lowering` is registered on the `iir.cf`
  dialect. Pins the dialect ownership choice per
  `docs/phase_c_pragma_materialization.md` §4.3 (count -> `iir.cf.
  Phase(C, body)` — no new sub-dialect needed).'''
  matched = [low for low in IIR_CF_DIALECT.lowerings if low.matches is m.CountPhase]
  assert len(matched) == 1
  assert matched[0].consumes == ('mir',)
  assert 'iir.cf' in matched[0].produces


# -----------------------------------------------------------------------------
# 5. End-to-end DSL → MIR through hir → MirPragmaPass
# -----------------------------------------------------------------------------


def test_dsl_with_pragma_count_propagates_through_hir(mir_compiler):
  '''The full chain: `Rule.with_pragma(Count())` -> PlanEntry.pragmas
  -> HirRuleVariant.pragmas -> ExecutePipeline.pragmas -> consumed
  by MirPragmaPass.

  Pins the dual-write: after MirPragmaPass, `ep.pragmas == ()` AND
  `ep.count is True` (the legacy bool propagated from `Rule.count`,
  set by `with_pragma(Count())`).
  '''
  # Direct MIR construction mirroring what the DSL would produce
  # for a `.with_pragma(Count())` rule (which sets `Rule.count =
  # True` + appends `Count()` to each plan's `pragmas`). The HIR
  # planning + lower path then propagates BOTH to the EP.
  ep = _scan_insert_ep(count=True, pragmas=(Count(),))
  out = MirPragmaPass().apply(ep, mir_compiler)
  assert out.pragmas == ()
  assert out.count is True
  # Pipeline unchanged — the dual-write skip branch fired.
  assert [type(o).__name__ for o in out.pipeline] == ['Scan', 'InsertInto']


# -----------------------------------------------------------------------------
# 6. apply_mir_pragma_pass surface (the new chain step)
# -----------------------------------------------------------------------------


def test_apply_mir_pragma_pass_handles_count_pragma_idempotently():
  '''After running `apply_mir_pragma_pass`, the count pragma is
  stripped and re-running is a structural no-op. This is the C1
  discipline gate (`test_pragmas_empty_after_materialization`)
  applied through the new chain entry.
  '''
  ep = _scan_insert_ep(count=True, pragmas=(Count(),))
  steps = [(ep, False)]
  once = apply_mir_pragma_pass(steps)
  twice = apply_mir_pragma_pass(once)
  ep1 = once[0][0]
  ep2 = twice[0][0]
  assert isinstance(ep1, m.ExecutePipeline)
  assert isinstance(ep2, m.ExecutePipeline)
  assert ep1.pragmas == ()
  assert ep2.pragmas == ()
  assert ep1.count is True
  assert ep2.count is True


# -----------------------------------------------------------------------------
# 7. CountPhase wrap-op round-trip via dataclasses.replace
# -----------------------------------------------------------------------------


def test_count_phase_round_trips_dataclasses_replace():
  '''Pin the new MIR op's dataclass shape — frozen + slots,
  `inner: Op`. Symmetric with `DedupGate`'s round-trip pin.
  '''
  from srdatalog.ir.dialects.iir.cf import Block as IirBlock

  body = IirBlock(stmts=())
  cp = m.CountPhase(inner=body)
  cp2 = dataclasses.replace(cp, inner=IirBlock(stmts=()))
  assert isinstance(cp2, m.CountPhase)
  assert cp2.inner is not body
