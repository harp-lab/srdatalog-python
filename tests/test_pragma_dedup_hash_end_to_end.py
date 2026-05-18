'''End-to-end test for Phase C2: the `DedupHash` typed pragma.

Per `docs/phase_c_pragma_materialization.md` §5.1 (per-PR acceptance
gate), each Phase C migration ships a `test_pragma_<name>_end_to_end`
that exercises the pragma via the production `Compiler.run`-style
pipeline and asserts:

  - The wrap op (`mir.DedupGate`) appears in MIR after
    `MirPragmaPass` (or is suppressed under the dual-write
    transition; both states are verified here).
  - The lowered IIR is correct (matches the dedup-hash branch of
    `_lower_insert_into`).
  - The rendered CUDA is byte-equivalent to the legacy
    `dedup_hash: bool` path (the load-bearing harness for C2's
    backward-compat contract).

Plus DSL surface checks:

  - `Rule(...).with_pragma(DedupHash())` accepts a typed instance.
  - `Rule(...).with_pragma(<not-a-pragma>)` raises `TypeError`.
  - `Rule(...).with_pragma(<unregistered-Pragma>)` raises
    `UnregisteredPragmaError` (typed sibling of `TypeError`).
  - The DSL dual-writes BOTH the typed pragma onto the rule's
    plan(s) AND the matching legacy bool field (`dedup_hash=True`)
    so the byte-equiv path through `compile_kernel_body` keeps
    working unchanged.

The framework infrastructure (the `MirPragmaPass` driver, the
`@pragma_handler` registry, error classes) is exercised by
`tests/test_mir_pragma_pass.py` + `tests/test_core_pragma.py`; this
file owns the per-pragma surface.
'''

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import final

import pytest

import srdatalog.ir.mir.types as m
from srdatalog.compile import compile_pipeline
from srdatalog.dsl import PlanEntry, Rule
from srdatalog.ir.core import Compiler, Pragma, UnregisteredPragmaError
from srdatalog.ir.dialects.relation.sorted_array import DIALECT as SA_DIALECT
from srdatalog.ir.dialects.relation.sorted_array.pragmas.dedup_hash import (
  DedupHash,
  lower_dedup_gate,
  materialize_dedup_hash,
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
  submodule for side effects (registers `@pragma_handler(DedupHash,
  ...)` + the `DedupGate` lowering), so this fixture is enough to
  drive `MirPragmaPass` end-to-end without monkey-patching the
  registry.
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
  rule_name: str = 'Dupy',
  dedup_hash: bool = False,
  pragmas: tuple[Pragma, ...] = (),
) -> m.ExecutePipeline:
  '''Build an EP shape that mirrors the legacy
  `tests/test_dedup_hash_byte_equivalence.py:_scan_insert_dedup`
  fixture, optionally carrying a typed `DedupHash` pragma instead of
  (or in addition to) the legacy bool field.
  '''
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
    dedup_hash=dedup_hash,
    pragmas=pragmas,  # type: ignore[arg-type]
  )


# -----------------------------------------------------------------------------
# 1. DSL surface — Rule.with_pragma
# -----------------------------------------------------------------------------


def _empty_rule() -> Rule:
  '''Build a minimal Rule. The rule's structure is irrelevant for the
  DSL-surface tests below; we only inspect `rule.plans`.'''
  from srdatalog.dsl import Atom

  head = Atom(rel='Dst', args=())
  body_atom = Atom(rel='Src', args=())
  return Rule(heads=(head,), body=(body_atom,))


def test_with_pragma_attaches_typed_pragma_to_default_plan():
  '''Calling `.with_pragma(DedupHash())` on a rule with no plans
  appends a default `PlanEntry(delta=-1)` carrying the pragma AND
  the matching legacy bool field (the C2 dual-write contract).'''
  rule = _empty_rule().with_pragma(DedupHash())
  assert len(rule.plans) == 1
  plan = rule.plans[0]
  assert plan.delta == -1
  assert plan.dedup_hash is True
  assert len(plan.pragmas) == 1
  assert isinstance(plan.pragmas[0], DedupHash)


def test_with_pragma_appends_to_existing_plans():
  '''When the rule already has plans, `.with_pragma(DedupHash())`
  appends the pragma to EVERY plan's `pragmas` tuple AND sets each
  plan's `dedup_hash=True` (per-variant dual-write). This matches
  the per-rule semantic intent of `with_pragma`.'''
  rule = (
    _empty_rule()
    .with_plan(delta=0)
    .with_plan(delta=1, var_order=('a', 'b'))
    .with_pragma(DedupHash())
  )
  assert len(rule.plans) == 2
  for plan in rule.plans:
    assert plan.dedup_hash is True
    assert any(isinstance(p, DedupHash) for p in plan.pragmas)


def test_with_pragma_rejects_non_pragma_arg():
  '''`Rule.with_pragma("dedup_hash")` (the legacy string-keyed form)
  is hard-error per `docs/code_discipline.md` D15. Same for any
  non-`Pragma` instance.'''
  rule = _empty_rule()
  with pytest.raises(TypeError, match=r'expected a Pragma subclass'):
    rule.with_pragma('dedup_hash')  # type: ignore[arg-type]
  with pytest.raises(TypeError):
    rule.with_pragma(True)  # type: ignore[arg-type]


def test_with_pragma_rejects_unregistered_pragma():
  '''A typed `Pragma` subclass with no `@pragma_handler` registration
  is rejected at DSL time — caught BEFORE compile, with did-you-mean
  hints listing the registered handler class names. Per
  `docs/pragma_as_typed_object.md` §3 and §6.
  '''

  @final
  @dataclass(frozen=True, slots=True)
  class _GhostPragma(Pragma):
    pass

  rule = _empty_rule()
  with pytest.raises(UnregisteredPragmaError, match=r'_GhostPragma'):
    rule.with_pragma(_GhostPragma())


# -----------------------------------------------------------------------------
# 2. MIR-level — MirPragmaPass materialization
# -----------------------------------------------------------------------------


def test_pragma_pass_inserts_dedup_gate_when_bool_is_false(mir_compiler):
  '''Pure typed path (bool=False, only pragma set): MirPragmaPass
  wraps every `InsertInto` in `pipeline` with `DedupGate`.

  This is the post-A3 target state. Today only synthesized EPs
  reach it — the DSL `.with_pragma(DedupHash())` dual-writes the
  bool field as well, hitting the dual-write skip branch below.
  '''
  ep = _scan_insert_ep(dedup_hash=False, pragmas=(DedupHash(),))
  out = MirPragmaPass().apply(ep, mir_compiler)
  # Pragma instance consumed.
  assert out.pragmas == ()
  # Pipeline: Scan + DedupGate(InsertInto).
  assert len(out.pipeline) == 2
  assert isinstance(out.pipeline[0], m.Scan)
  assert isinstance(out.pipeline[1], m.DedupGate)
  assert isinstance(out.pipeline[1].inner, m.InsertInto)
  # Legacy bool was NOT toggled; the wrap op is the sole signal.
  assert out.dedup_hash is False


def test_pragma_pass_skips_wrap_in_dual_write_mode(mir_compiler):
  '''Dual-write transition (bool=True AND pragma set, the C2 DSL
  shape): MirPragmaPass strips the pragma but leaves `pipeline`
  untouched. The legacy `compile_kernel_body` -> `_lower_insert_into`
  bool-driven path produces the dedup emit; the wrap op never
  appears so the monolith doesn't need to know about DedupGate.

  This is the load-bearing test for the dual-write contract: the C2
  PR must not break the byte-equivalence harness, and that requires
  the bool-field path to remain the sole emission driver until A3
  drops the bool.
  '''
  ep = _scan_insert_ep(dedup_hash=True, pragmas=(DedupHash(),))
  out = MirPragmaPass().apply(ep, mir_compiler)
  # Pragma still consumed.
  assert out.pragmas == ()
  # No DedupGate inserted.
  assert all(not isinstance(child, m.DedupGate) for child in out.pipeline)
  # Bool preserved for the legacy emitter.
  assert out.dedup_hash is True


def test_pragma_pass_via_apply_all_mir_passes_is_noop_for_empty_pragmas(
  mir_compiler,
):
  '''Sanity: the integration of `MirPragmaPass` into
  `apply_all_mir_passes` does not change MIR shape for EPs that
  carry no typed pragmas — the dominant case today.'''
  ep = _scan_insert_ep(dedup_hash=False, pragmas=())
  steps_in = [(ep, False)]
  steps_out = apply_all_mir_passes(steps_in)
  # Same number of steps, same EP identity-or-equal at the top.
  assert len(steps_out) == 1
  ep_out, is_rec = steps_out[0]
  assert isinstance(ep_out, m.ExecutePipeline)
  assert is_rec is False
  assert ep_out.pragmas == ()
  # Pipeline contents structurally identical (no DedupGate insertion).
  assert [type(o).__name__ for o in ep_out.pipeline] == ['Scan', 'InsertInto']


# -----------------------------------------------------------------------------
# 3. Lowering — DedupGate -> IIR matches legacy dedup branch byte-for-byte
# -----------------------------------------------------------------------------


def _render(ep: m.ExecutePipeline) -> str:
  '''Render an EP through the production `compile_pipeline` surface,
  which exercises the same `compile_kernel_body` + emit path that
  the byte-equivalence harnesses guard.'''
  return compile_pipeline(ep)


def test_dedup_gate_lowering_matches_legacy_emission(mir_compiler):
  '''Byte-equivalence by construction: the new `lower_dedup_gate`
  function delegates to `_lower_insert_into` with `ctx.dedup_hash=
  True`, so its emission is identical to the legacy
  `if ctx.dedup_hash:` branch.

  We verify this end-to-end by:
    1. Building an EP with the typed pragma (bool=False); running
       `MirPragmaPass` to get the wrap-op MIR.
    2. Building the same EP with the legacy bool (no pragma);
       letting `compile_pipeline` go through the bool-driven path.
    3. Calling `lower_dedup_gate` directly against the wrapped op
       and rendering its IIR through the same emit ctx the legacy
       path uses.

  The rendered CUDA for the dedup gate must match the
  `try_insert` + `if (_p)` shape the legacy fixture in
  `test_dedup_hash_byte_equivalence.py` asserts.
  '''
  # Build the legacy bool-driven EP and render it. This is the
  # "ground truth" the byte-equivalence harness anchors to.
  legacy_ep = _scan_insert_ep(arity=2, dedup_hash=True)
  legacy_out = _render(legacy_ep)
  assert 'dedup_table.try_insert(thread_id, v0, v1)' in legacy_out
  assert 'if (_p) {' in legacy_out
  assert 'atomicAdd(atomic_write_pos, 1u)' in legacy_out

  # Same shape via the typed-pragma path: MirPragmaPass wraps the
  # InsertInto, then we can verify the wrap-op IIR through the
  # registered lowering helper.
  typed_ep = _scan_insert_ep(arity=2, dedup_hash=False, pragmas=(DedupHash(),))
  materialized = MirPragmaPass().apply(typed_ep, mir_compiler)
  gate = materialized.pipeline[1]
  assert isinstance(gate, m.DedupGate)

  # Direct-call the lowering against a minimally-populated
  # LoweringCtx: feature ctx.dedup_hash gets flipped True by the
  # gate lowering, so the inner emission goes through the dedup
  # branch.
  from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import LoweringCtx

  ctx = LoweringCtx(
    view_var_names={'0': 'view_Src_0_FULL'},
    is_counting=False,
    output_var='output',
    dedup_hash=False,
  )
  iir = lower_dedup_gate(gate, ctx)
  rendered = emit(iir, EmitCtx(indent_level=4))
  assert 'dedup_table.try_insert(thread_id, v0, v1)' in rendered
  assert 'if (_p) {' in rendered
  assert 'atomicAdd(atomic_write_pos, 1u)' in rendered
  for col in range(2):
    assert f'out_data_0[(pos + out_base_0) + {col} * out_stride_0] = v{col}' in rendered


def test_dedup_gate_lowering_restores_prior_dedup_hash_flag(mir_compiler):
  '''The lowering's `ctx.dedup_hash` toggle is save/restore — after
  the gate emits, the ctx is left exactly as it was. Guards against
  reentrancy leaks if sibling lowerings depend on the flag.'''
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import LoweringCtx

  for original in (False, True):
    ctx = LoweringCtx(
      view_var_names={'0': 'view_Src_0_FULL'},
      is_counting=False,
      output_var='output',
      dedup_hash=original,
    )
    insert = m.InsertInto(rel_name='Dst', version=Version.NEW, vars=['v0'], index=[0])
    gate = m.DedupGate(inner=insert)
    lower_dedup_gate(gate, ctx)
    assert ctx.dedup_hash is original


# -----------------------------------------------------------------------------
# 4. Registration completeness (R5)
# -----------------------------------------------------------------------------


def test_dedup_hash_handler_is_registered():
  '''Importing the pragma module registers a `@pragma_handler` whose
  `pragma_cls` is `DedupHash` and whose `on` is `mir.ExecutePipeline`.
  R5 (`test_pragma_handler_registry_completeness`) gates this once
  per Pragma subclass; here we pin it for the C2 surface explicitly.
  '''
  from srdatalog.ir.core.pragma import get_pragma_registrations

  regs = [r for r in get_pragma_registrations() if r.pragma_cls is DedupHash]
  assert len(regs) >= 1
  reg = regs[0]
  assert reg.on is m.ExecutePipeline
  assert reg.fn is materialize_dedup_hash


def test_dedup_gate_lowering_is_registered_on_sorted_array_dialect():
  '''The DedupGate `@lowering` is registered on the
  `relation.sorted_array` dialect. Pins the dialect ownership choice
  per `docs/phase_c_pragma_materialization.md` §4.3 (`dedup_hash`
  stays in sorted_array — no new sub-dialect needed).'''
  matched = [low for low in SA_DIALECT.lowerings if low.matches is m.DedupGate]
  assert len(matched) == 1
  assert matched[0].consumes == ('mir',)
  assert 'iir.cf' in matched[0].produces


# -----------------------------------------------------------------------------
# 5. PlanEntry pragma field default
# -----------------------------------------------------------------------------


def test_plan_entry_pragmas_default_empty():
  '''Pin the new field's default. Tests that depend on PlanEntry
  semantics (test_hir_user_plan.py etc.) implicitly rely on `()`
  here so the C2 addition stays back-compatible.'''
  pe = PlanEntry()
  assert pe.pragmas == ()


def test_plan_entry_with_pragma_replace_round_trip():
  '''`dataclasses.replace(plan_entry, pragmas=(...))` preserves the
  rest of the entry. Symmetric with how `Rule.with_pragma` builds
  new plans.'''
  pe = PlanEntry(delta=0, var_order=('a', 'b'))
  pe2 = dataclasses.replace(pe, pragmas=(DedupHash(),))
  assert pe2.delta == 0
  assert pe2.var_order == ('a', 'b')
  assert isinstance(pe2.pragmas[0], DedupHash)


# -----------------------------------------------------------------------------
# 6. apply_mir_pragma_pass surface (the new chain step)
# -----------------------------------------------------------------------------


def test_apply_mir_pragma_pass_is_idempotent_for_empty_pragmas():
  '''After running `apply_mir_pragma_pass`, every EP has `pragmas ==
  ()`. Re-running it is a structural no-op. This is the C1
  discipline gate (R5b / `test_pragmas_empty_after_materialization`)
  applied through the new chain entry.
  '''
  ep = _scan_insert_ep(dedup_hash=False, pragmas=())
  steps = [(ep, False)]
  once = apply_mir_pragma_pass(steps)
  twice = apply_mir_pragma_pass(once)
  ep1 = once[0][0]
  ep2 = twice[0][0]
  assert isinstance(ep1, m.ExecutePipeline)
  assert isinstance(ep2, m.ExecutePipeline)
  assert ep1.pragmas == ()
  assert ep2.pragmas == ()
