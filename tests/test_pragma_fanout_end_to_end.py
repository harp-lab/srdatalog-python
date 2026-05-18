'''End-to-end test for Phase C6: the `FanOut` typed pragma.

Per `docs/phase_c_pragma_materialization.md` §5.1 (per-PR acceptance
gate), each Phase C migration ships a `test_pragma_<name>_end_to_end`
that exercises the pragma via the production `Compiler.run`-style
pipeline and asserts:

  - The wrap op (`mir.FanOut`) appears in MIR after `MirPragmaPass`
    (or is suppressed under the dual-write transition; both states
    are verified here).
  - The lowered IIR is correct (matches the non-fanout leaf-emission
    shape; runner-side fan-out scheduling stays in
    `complete_runner.py`, out of scope for C6 per the task constraint).
  - The rendered output is byte-equivalent to the legacy
    `ep.use_fan_out` runner path.

Plus DSL surface checks:

  - `Rule(...).with_pragma(FanOut())` accepts a typed instance.
  - `Rule(...).with_pragma(<not-a-pragma>)` raises `TypeError`.
  - The DSL dual-writes BOTH the typed pragma onto the rule's
    plan(s) AND the matching legacy `fanout=True` PlanEntry field
    (which propagates to `ep.use_fan_out` via `variant.fanout`).

The framework infrastructure (the `MirPragmaPass` driver, the
`@pragma_handler` registry, error classes) is exercised by
`tests/test_mir_pragma_pass.py` + `tests/test_core_pragma.py`; this
file owns the per-pragma surface.
'''

from __future__ import annotations

import dataclasses

import pytest

import srdatalog.ir.mir.types as m
from srdatalog.dsl import PlanEntry, Rule
from srdatalog.ir.core import Compiler, Pragma
from srdatalog.ir.dialects.iir.cf import Block as IirBlock
from srdatalog.ir.dialects.relation.sorted_array import DIALECT as SA_DIALECT
from srdatalog.ir.dialects.relation.sorted_array.pragmas.fanout import (
  FanOut,
  lower_fan_out,
  materialize_fanout,
)
from srdatalog.ir.hir.types import Version
from srdatalog.ir.mir import DIALECT as MIR_DIALECT
from srdatalog.ir.mir.passes import apply_all_mir_passes, apply_mir_pragma_pass
from srdatalog.ir.mir.pragma_pass import MirPragmaPass

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mir_compiler() -> Compiler:
  '''Compiler with MIR + sorted_array dialects registered.

  The sorted_array dialect's `_register_passes` imports the pragmas
  submodule for side effects (registers `@pragma_handler(FanOut, ...)`
  + the `mir.FanOut` lowering), so this fixture is enough to drive
  `MirPragmaPass` end-to-end without monkey-patching the registry.
  '''
  c = Compiler()
  c.register_dialect(MIR_DIALECT)
  c.register_dialect(SA_DIALECT)
  return c


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _scan_insert_ep(
  *,
  arity: int = 2,
  rule_name: str = 'Fan',
  use_fan_out: bool = False,
  pragmas: tuple[Pragma, ...] = (),
) -> m.ExecutePipeline:
  '''Build an EP shape suitable for fanout-pragma testing, optionally
  carrying a typed `FanOut` pragma instead of (or in addition to) the
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
    use_fan_out=use_fan_out,
    pragmas=pragmas,  # type: ignore[arg-type]
  )


def _empty_rule() -> Rule:
  '''Build a minimal Rule. The rule's structure is irrelevant for the
  DSL-surface tests below; we only inspect `rule.plans`.'''
  from srdatalog.dsl import Atom

  head = Atom(rel='Dst', args=())
  body_atom = Atom(rel='Src', args=())
  return Rule(heads=(head,), body=(body_atom,))


# -----------------------------------------------------------------------------
# 1. DSL surface — Rule.with_pragma(FanOut())
# -----------------------------------------------------------------------------


def test_with_pragma_fanout_attaches_typed_pragma_and_sets_plan_fanout():
  '''Calling `.with_pragma(FanOut())` on a rule with no plans
  appends a default `PlanEntry(delta=-1)` carrying the pragma AND
  `fanout=True` (the PlanEntry-level dual-write, parallel to C2's
  DedupHash `dedup_hash=True` dual-write).'''
  rule = _empty_rule().with_pragma(FanOut())
  assert len(rule.plans) == 1
  plan = rule.plans[0]
  assert plan.delta == -1
  assert plan.fanout is True
  assert len(plan.pragmas) == 1
  assert isinstance(plan.pragmas[0], FanOut)


def test_with_pragma_fanout_appends_to_existing_plans():
  '''When the rule already has plans, `.with_pragma(FanOut())`
  appends the pragma to EVERY plan's `pragmas` tuple AND sets each
  plan's `fanout=True` (per-variant dual-write). This matches the
  per-rule semantic intent of `with_pragma`.'''
  rule = (
    _empty_rule().with_plan(delta=0).with_plan(delta=1, var_order=('a', 'b')).with_pragma(FanOut())
  )
  assert len(rule.plans) == 2
  for plan in rule.plans:
    assert plan.fanout is True
    assert any(isinstance(p, FanOut) for p in plan.pragmas)


def test_with_pragma_fanout_rejects_non_pragma_arg():
  '''Standard `with_pragma` guard: hard-error on non-`Pragma`
  arguments. Mirrors the C2 DedupHash equivalent.'''
  rule = _empty_rule()
  with pytest.raises(TypeError, match=r'expected a Pragma subclass'):
    rule.with_pragma('fanout')  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# 2. MIR-level — MirPragmaPass materialization
# -----------------------------------------------------------------------------


def test_pragma_pass_inserts_fan_out_when_bool_is_false(mir_compiler):
  '''Pure typed path (bool=False, only pragma set): MirPragmaPass
  wraps every `InsertInto` in `pipeline` with `mir.FanOut`.

  This is the post-A3 target state. Today only synthesized EPs
  reach it — the DSL `.with_pragma(FanOut())` dual-writes the
  `fanout` field on each plan, which propagates to
  `ep.use_fan_out` via `variant.fanout`, hitting the dual-write
  skip branch below.
  '''
  ep = _scan_insert_ep(use_fan_out=False, pragmas=(FanOut(),))
  out = MirPragmaPass().apply(ep, mir_compiler)
  # Pragma instance consumed.
  assert out.pragmas == ()
  # Pipeline: Scan + mir.FanOut(InsertInto).
  assert len(out.pipeline) == 2
  assert isinstance(out.pipeline[0], m.Scan)
  assert isinstance(out.pipeline[1], m.FanOut)
  assert isinstance(out.pipeline[1].inner, m.InsertInto)
  # Legacy bool was NOT toggled; the wrap op is the sole signal.
  assert out.use_fan_out is False


def test_pragma_pass_skips_wrap_in_dual_write_mode(mir_compiler):
  '''Dual-write transition (bool=True AND pragma set, the C6 DSL
  shape): MirPragmaPass strips the pragma but leaves `pipeline`
  untouched. The legacy `complete_runner.py` `ep.use_fan_out` path
  produces the fan-out runner emit; the wrap op never appears so
  the monolith doesn't need to know about `mir.FanOut`.

  This is the load-bearing test for the dual-write contract: the
  C6 PR must not break the byte-equivalence harness, and that
  requires the bool-field path to remain the sole emission driver
  until A3 drops the bool.
  '''
  ep = _scan_insert_ep(use_fan_out=True, pragmas=(FanOut(),))
  out = MirPragmaPass().apply(ep, mir_compiler)
  # Pragma still consumed.
  assert out.pragmas == ()
  # No mir.FanOut inserted.
  assert all(not isinstance(child, m.FanOut) for child in out.pipeline)
  # Bool preserved for the legacy emitter.
  assert out.use_fan_out is True
  # Pipeline shape unchanged.
  assert [type(o).__name__ for o in out.pipeline] == ['Scan', 'InsertInto']


def test_pragma_pass_via_apply_all_mir_passes_is_noop_for_empty_pragmas(
  mir_compiler,
):
  '''Sanity: the integration of `MirPragmaPass` into
  `apply_all_mir_passes` does not change MIR shape for EPs that
  carry no typed pragmas — the dominant case today.'''
  ep = _scan_insert_ep(use_fan_out=False, pragmas=())
  steps_in = [(ep, False)]
  steps_out = apply_all_mir_passes(steps_in)
  assert len(steps_out) == 1
  ep_out, is_rec = steps_out[0]
  assert isinstance(ep_out, m.ExecutePipeline)
  assert is_rec is False
  assert ep_out.pragmas == ()
  assert [type(o).__name__ for o in ep_out.pipeline] == ['Scan', 'InsertInto']


# -----------------------------------------------------------------------------
# 3. Lowering — mir.FanOut -> IIR delegates to _lower_insert_into
# -----------------------------------------------------------------------------


def test_fan_out_lowering_emits_block_wrapping_insert_emission(mir_compiler):
  '''The fanout lowering delegates to the legacy `_lower_insert_into`
  helper without touching ctx (no `is_counting` / `dedup_hash`
  toggling), so the kernel-body IIR is byte-equivalent to the
  non-fanout leaf-emission shape. The runner-side scheduling is
  what differs for fan-out rules, and that lives in
  `complete_runner.py` (out of scope for C6).
  '''
  ep = _scan_insert_ep(use_fan_out=False, pragmas=(FanOut(),))
  materialized = MirPragmaPass().apply(ep, mir_compiler)
  fan_out = materialized.pipeline[1]
  assert isinstance(fan_out, m.FanOut)

  # Direct-call the lowering against a minimally-populated
  # LoweringCtx (matches the legacy compile path's setup for a
  # 2-arity scan rule). We just need enough state for
  # `_lower_insert_into` not to raise on missing fields.
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import LoweringCtx

  ctx = LoweringCtx(
    view_var_names={'0': 'view_Src_0_FULL'},
    is_counting=False,
    output_var='output',
  )
  iir = lower_fan_out(fan_out, ctx)
  # Block-wrapped: the lowering uses iir.cf.Block to package the
  # emission stmts list returned by _lower_insert_into.
  assert isinstance(iir, IirBlock)
  # Stmts non-empty — the InsertInto emit produced something.
  assert len(iir.stmts) >= 1


def test_fan_out_lowering_byte_equivalent_to_legacy_insert(mir_compiler):
  '''Stronger byte-equivalence claim: the fanout-wrapped emission
  is structurally identical to the non-fanout InsertInto emission
  when both are run through the same LoweringCtx. The fanout
  pragma is a runtime scheduling hint (runner-side), so the
  kernel-body IIR must NOT differ.
  '''
  from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    LoweringCtx,
    _lower_insert_into,
  )

  insert = m.InsertInto(rel_name='Dst', version=Version.NEW, vars=['v0', 'v1'], index=[0, 1])
  fan_out = m.FanOut(inner=insert)

  ctx_legacy = LoweringCtx(
    view_var_names={'0': 'view_Src_0_FULL'},
    is_counting=False,
    output_var='output',
  )
  ctx_typed = LoweringCtx(
    view_var_names={'0': 'view_Src_0_FULL'},
    is_counting=False,
    output_var='output',
  )

  legacy_stmts = _lower_insert_into(insert, ctx_legacy)
  legacy_text = emit(IirBlock(stmts=tuple(legacy_stmts)), EmitCtx(indent_level=4))

  typed_iir = lower_fan_out(fan_out, ctx_typed)
  typed_text = emit(typed_iir, EmitCtx(indent_level=4))

  assert legacy_text == typed_text


# -----------------------------------------------------------------------------
# 4. Registration completeness (R5)
# -----------------------------------------------------------------------------


def test_fan_out_handler_is_registered():
  '''Importing the pragma module registers a `@pragma_handler` whose
  `pragma_cls` is `FanOut` and whose `on` is `mir.ExecutePipeline`.
  R5 (`test_pragma_handler_registry_completeness`) gates this once
  per Pragma subclass; here we pin it for the C6 surface explicitly.
  '''
  from srdatalog.ir.core.pragma import get_pragma_registrations

  regs = [r for r in get_pragma_registrations() if r.pragma_cls is FanOut]
  assert len(regs) >= 1
  reg = regs[0]
  assert reg.on is m.ExecutePipeline
  assert reg.fn is materialize_fanout


def test_fan_out_lowering_is_registered_on_sorted_array_dialect():
  '''The `mir.FanOut` `@lowering` is registered on the
  `relation.sorted_array` dialect. Pins the dialect ownership choice
  per `docs/phase_c_pragma_materialization.md` §4.3 (`fanout` stays
  in sorted_array — no new sub-dialect needed; the runner-side
  scheduling is orthogonal and lives in `complete_runner.py`).'''
  matched = [low for low in SA_DIALECT.lowerings if low.matches is m.FanOut]
  assert len(matched) == 1
  assert matched[0].consumes == ('mir',)
  assert 'iir.cf' in matched[0].produces


# -----------------------------------------------------------------------------
# 5. apply_mir_pragma_pass surface (the new chain step)
# -----------------------------------------------------------------------------


def test_apply_mir_pragma_pass_handles_fanout_pragma_idempotently():
  '''After running `apply_mir_pragma_pass`, the fanout pragma is
  stripped and re-running is a structural no-op. This is the C1
  discipline gate (`test_pragmas_empty_after_materialization`)
  applied through the new chain entry.
  '''
  ep = _scan_insert_ep(use_fan_out=True, pragmas=(FanOut(),))
  steps = [(ep, False)]
  once = apply_mir_pragma_pass(steps)
  twice = apply_mir_pragma_pass(once)
  ep1 = once[0][0]
  ep2 = twice[0][0]
  assert isinstance(ep1, m.ExecutePipeline)
  assert isinstance(ep2, m.ExecutePipeline)
  assert ep1.pragmas == ()
  assert ep2.pragmas == ()
  assert ep1.use_fan_out is True
  assert ep2.use_fan_out is True


# -----------------------------------------------------------------------------
# 6. PlanEntry pragma field round-trip
# -----------------------------------------------------------------------------


def test_plan_entry_fanout_round_trip():
  '''`dataclasses.replace(plan_entry, pragmas=(...), fanout=True)`
  preserves the rest of the entry. Symmetric with how
  `Rule.with_pragma(FanOut())` builds new plans.'''
  pe = PlanEntry(delta=0, var_order=('a', 'b'))
  pe2 = dataclasses.replace(pe, pragmas=(FanOut(),), fanout=True)
  assert pe2.delta == 0
  assert pe2.var_order == ('a', 'b')
  assert pe2.fanout is True
  assert isinstance(pe2.pragmas[0], FanOut)


def test_fan_out_wrap_op_round_trips_dataclasses_replace():
  '''Pin the new MIR op's dataclass shape — frozen + slots,
  `inner: InsertInto`. Symmetric with `DedupGate`'s round-trip pin.
  '''
  insert = m.InsertInto(rel_name='Dst', version=Version.NEW, vars=['v0'], index=[0])
  fo = m.FanOut(inner=insert)
  insert2 = m.InsertInto(rel_name='Dst2', version=Version.NEW, vars=['v0'], index=[0])
  fo2 = dataclasses.replace(fo, inner=insert2)
  assert isinstance(fo2, m.FanOut)
  assert fo2.inner.rel_name == 'Dst2'
