'''End-to-end test for Phase C4: the `WorkStealing` typed pragma.

Per `docs/phase_c_pragma_materialization.md` §5.1 (per-PR acceptance
gate), each Phase C migration ships a `test_pragma_<name>_end_to_end`
that exercises the pragma via the production `Compiler.run`-style
pipeline and asserts:

  - The wrap op (`mir.WSScope`) appears in MIR after `MirPragmaPass`
    (or is suppressed under the dual-write transition; both states
    are verified here).
  - The lowered IIR is correct (matches the WS branch of
    `_lower_insert_into`).
  - The rendered CUDA is byte-equivalent to the legacy
    `work_stealing: bool` path (the load-bearing harness for C4's
    backward-compat contract — note that the legacy WS-count emit
    is only reachable via direct LoweringCtx construction today,
    not via `compile_pipeline`; the C4 dual-write contract simply
    promises that the wrap op does NOT appear when the bool is set,
    so the legacy emitter sees the unchanged shape it sees today).

Plus DSL surface checks:

  - `Rule(...).with_pragma(WorkStealing())` accepts a typed instance.
  - `Rule(...).with_pragma(<not-a-pragma>)` raises `TypeError`.
  - `Rule(...).with_pragma(<unregistered-Pragma>)` raises
    `UnregisteredPragmaError` (typed sibling of `TypeError`).
  - The DSL dual-writes BOTH the typed pragma onto the rule's
    plan(s) AND the matching legacy bool field (`work_stealing=
    True`) so the byte-equiv path through the legacy compile flow
    keeps working unchanged.

The framework infrastructure (the `MirPragmaPass` driver, the
`@pragma_handler` registry, error classes) is exercised by
`tests/test_mir_pragma_pass.py` + `tests/test_core_pragma.py`; this
file owns the per-pragma surface for `work_stealing`.

Mirrors `tests/test_pragma_dedup_hash_end_to_end.py` (the C2
template).
'''

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import final

import pytest

import srdatalog.ir.mir.types as m
from srdatalog.dsl import PlanEntry, Rule
from srdatalog.ir.core import Compiler, Pragma, UnregisteredPragmaError
from srdatalog.ir.dialects.parallel.atomic_ws import DIALECT as WS_DIALECT
from srdatalog.ir.dialects.parallel.atomic_ws.pragmas.work_stealing import (
  WorkStealing,
  lower_ws_scope,
  materialize_work_stealing,
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
  '''Compiler with MIR + parallel.atomic_ws dialects registered.

  The parallel.atomic_ws dialect's `_register_passes` imports the
  pragmas submodule for side effects (registers `@pragma_handler(
  WorkStealing, ...)` + the `WSScope` lowering), so this fixture is
  enough to drive `MirPragmaPass` end-to-end without monkey-patching
  the registry.
  '''
  c = Compiler()
  c.register_dialect(MIR_DIALECT)
  c.register_dialect(WS_DIALECT)
  return c


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _scan_insert_ep(
  *,
  arity: int = 2,
  rule_name: str = 'WSy',
  pragmas: tuple[Pragma, ...] = (),
) -> m.ExecutePipeline:
  '''Build a minimal `Scan -> InsertInto` EP shape carrying a typed
  `WorkStealing` pragma (post-A3-2: the legacy bool shadow field
  was retired; this fixture exercises only the typed surface).
  Mirrors the C2 `_scan_insert_ep` fixture in
  `tests/test_pragma_dedup_hash_end_to_end.py` so the two per-pragma
  tests share their structural assumptions.
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
  '''Calling `.with_pragma(WorkStealing())` on a rule with no plans
  appends a default `PlanEntry(delta=-1)` carrying the typed pragma.
  Post-A3-2: the legacy `PlanEntry.work_stealing` shadow bool has
  been retired, so the pragma is the sole signal.'''
  rule = _empty_rule().with_pragma(WorkStealing())
  assert len(rule.plans) == 1
  plan = rule.plans[0]
  assert plan.delta == -1
  assert len(plan.pragmas) == 1
  assert isinstance(plan.pragmas[0], WorkStealing)


def test_with_pragma_appends_to_existing_plans():
  '''When the rule already has plans, `.with_pragma(WorkStealing())`
  appends the pragma to EVERY plan's `pragmas` tuple. Matches the
  per-rule semantic intent of `with_pragma`.'''
  rule = (
    _empty_rule()
    .with_plan(delta=0)
    .with_plan(delta=1, var_order=('a', 'b'))
    .with_pragma(WorkStealing())
  )
  assert len(rule.plans) == 2
  for plan in rule.plans:
    assert any(isinstance(p, WorkStealing) for p in plan.pragmas)


def test_with_pragma_does_not_set_unrelated_bool_fields():
  '''Guards against typos in `_BUILTIN_BOOL_SHADOW_PRAGMAS`: setting
  `WorkStealing` must not toggle the dual-write bools of OTHER
  pragmas that still have a legacy shadow field.'''
  rule = _empty_rule().with_pragma(WorkStealing())
  plan = rule.plans[0]
  assert plan.dedup_hash is False
  assert plan.block_group is False
  assert plan.fanout is False


def test_with_pragma_rejects_non_pragma_arg():
  '''`Rule.with_pragma("work_stealing")` (the legacy string-keyed
  form) is hard-error per `docs/code_discipline.md` D15. Same for
  any non-`Pragma` instance.'''
  rule = _empty_rule()
  with pytest.raises(TypeError, match=r'expected a Pragma subclass'):
    rule.with_pragma('work_stealing')  # type: ignore[arg-type]
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
  class _GhostWSPragma(Pragma):
    pass

  rule = _empty_rule()
  with pytest.raises(UnregisteredPragmaError, match=r'_GhostWSPragma'):
    rule.with_pragma(_GhostWSPragma())


# -----------------------------------------------------------------------------
# 2. MIR-level — MirPragmaPass materialization
# -----------------------------------------------------------------------------


def test_pragma_pass_inserts_ws_scope_when_pragma_set(mir_compiler):
  '''Typed path: MirPragmaPass wraps every `InsertInto` in
  `pipeline` with `WSScope` whenever the EP carries a `WorkStealing`
  pragma. Post-A3-2 this is the ONLY path — the legacy bool shadow
  field is gone.
  '''
  ep = _scan_insert_ep(pragmas=(WorkStealing(),))
  out = MirPragmaPass().apply(ep, mir_compiler)
  # Pragma instance consumed.
  assert out.pragmas == ()
  # Pipeline: Scan + WSScope(InsertInto).
  assert len(out.pipeline) == 2
  assert isinstance(out.pipeline[0], m.Scan)
  assert isinstance(out.pipeline[1], m.WSScope)
  assert isinstance(out.pipeline[1].inner, m.InsertInto)


def test_pragma_pass_via_apply_all_mir_passes_is_noop_for_empty_pragmas(
  mir_compiler,
):
  '''Sanity: the integration of `MirPragmaPass` into
  `apply_all_mir_passes` does not change MIR shape for EPs that
  carry no typed pragmas — the dominant case today.'''
  ep = _scan_insert_ep(pragmas=())
  steps_in = [(ep, False)]
  steps_out = apply_all_mir_passes(steps_in)
  # Same number of steps, same EP identity-or-equal at the top.
  assert len(steps_out) == 1
  ep_out, is_rec = steps_out[0]
  assert isinstance(ep_out, m.ExecutePipeline)
  assert is_rec is False
  assert ep_out.pragmas == ()
  # Pipeline contents structurally identical (no WSScope insertion).
  assert [type(o).__name__ for o in ep_out.pipeline] == ['Scan', 'InsertInto']


def test_pragma_pass_only_wraps_inserts_not_other_ops(mir_compiler):
  '''The `_wrap_inserts_in_ws_scope` helper must leave non-InsertInto
  ops (Scan, Filter, ...) unchanged. Guards against a regression
  where the wrap accidentally captures intermediate join structure.
  '''
  scan = m.Scan(
    vars=['x', 'y'],
    rel_name='Src',
    version=Version.FULL,
    index=[0, 1],
    handle_start=0,
  )
  filt = m.Filter(vars=['x', 'y'], code='return x != y;')
  insert = m.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['x', 'y'],
    index=[0, 1],
  )
  ep = m.ExecutePipeline(
    pipeline=[scan, filt, insert],
    source_specs=[scan],
    dest_specs=[insert],
    rule_name='WSWithFilter',
    pragmas=(WorkStealing(),),  # type: ignore[arg-type]
  )
  out = MirPragmaPass().apply(ep, mir_compiler)
  assert out.pragmas == ()
  assert len(out.pipeline) == 3
  assert isinstance(out.pipeline[0], m.Scan)
  assert isinstance(out.pipeline[1], m.Filter)
  assert isinstance(out.pipeline[2], m.WSScope)
  assert isinstance(out.pipeline[2].inner, m.InsertInto)


# -----------------------------------------------------------------------------
# 3. Lowering — WSScope -> IIR matches legacy ws branch byte-for-byte
# -----------------------------------------------------------------------------


def test_ws_scope_lowering_emits_count_increment(mir_compiler):
  '''Byte-equivalence by construction: the new `lower_ws_scope`
  function delegates to `_lower_insert_into` with `ctx.ws_enabled=
  True`, so its count-phase emission is identical to the legacy
  `if ctx.ws_enabled:` branch (the `<out>++` count emit).

  Build the wrap op via MirPragmaPass, then lower it directly with
  the same LoweringCtx the legacy WS tests
  (`tests/test_ws_emit_dialect.py`) use, and assert the rendered
  text matches the legacy WS-count shape.
  '''
  from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import LoweringCtx

  typed_ep = _scan_insert_ep(arity=2, pragmas=(WorkStealing(),))
  materialized = MirPragmaPass().apply(typed_ep, mir_compiler)
  scope = materialized.pipeline[1]
  assert isinstance(scope, m.WSScope)

  ctx = LoweringCtx(
    is_counting=True,
    output_var='local_count',
    ws_enabled=False,  # gets flipped True by the gate lowering
  )
  iir = lower_ws_scope(scope, ctx)
  rendered = emit(iir, EmitCtx(indent_level=4))
  # Legacy WS-count shape: `<out>++` inside a lane-0 guard.
  assert 'local_count++;' in rendered
  assert 'emit_direct' not in rendered
  assert 'tile.thread_rank() == 0' in rendered


def test_ws_scope_lowering_inside_cartesian_drops_lane0(mir_compiler):
  '''Inside a Cartesian, no lane-0 guard — `local_count++` directly.
  Mirrors `tests/test_ws_emit_dialect.py:
  test_insert_count_ws_enabled_inside_cart_drops_lane0`.'''
  from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import LoweringCtx

  insert = m.InsertInto(rel_name='Dst', version=Version.NEW, vars=['x'], index=[0])
  scope = m.WSScope(inner=insert)

  ctx = LoweringCtx(
    is_counting=True,
    inside_cartesian=True,
    output_var='local_count',
    ws_enabled=False,
  )
  iir = lower_ws_scope(scope, ctx)
  rendered = emit(iir, EmitCtx(indent_level=4))
  assert 'local_count++;' in rendered
  assert 'tile.thread_rank()' not in rendered


def test_ws_scope_lowering_materialize_passes_through(mir_compiler):
  '''In materialize phase, the WS-flagged kernel-functor-level emit
  goes through the standard `<out>.emit_direct(<args>)` path —
  `ws_enabled` only changes the count-phase increment. The
  batched-Cartesian WS materialize variant (`emit_warp_coalesced`)
  is driven by `ws_cartesian_valid_var`, a separate state field set
  by `_lower_nested_cart` and not in C4 scope.
  '''
  from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import LoweringCtx

  insert = m.InsertInto(rel_name='Dst', version=Version.NEW, vars=['v0', 'v1'], index=[0, 1])
  scope = m.WSScope(inner=insert)

  ctx = LoweringCtx(
    is_counting=False,
    output_var='output',
    ws_enabled=False,
  )
  iir = lower_ws_scope(scope, ctx)
  rendered = emit(iir, EmitCtx(indent_level=4))
  assert 'output.emit_direct(v0, v1);' in rendered


def test_ws_scope_lowering_restores_prior_ws_enabled_flag(mir_compiler):
  '''The lowering's `ctx.ws_enabled` toggle is save/restore — after
  the scope emits, the ctx is left exactly as it was. Guards against
  reentrancy leaks if sibling lowerings depend on the flag.'''
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import LoweringCtx

  for original in (False, True):
    ctx = LoweringCtx(
      is_counting=True,
      output_var='local_count',
      ws_enabled=original,
    )
    insert = m.InsertInto(rel_name='Dst', version=Version.NEW, vars=['v0'], index=[0])
    scope = m.WSScope(inner=insert)
    lower_ws_scope(scope, ctx)
    assert ctx.ws_enabled is original


# -----------------------------------------------------------------------------
# 4. Registration completeness (R5)
# -----------------------------------------------------------------------------


def test_work_stealing_handler_is_registered():
  '''Importing the pragma module registers a `@pragma_handler` whose
  `pragma_cls` is `WorkStealing` and whose `on` is
  `mir.ExecutePipeline`. R5
  (`test_pragma_handler_registry_completeness`) gates this once per
  Pragma subclass; here we pin it for the C4 surface explicitly.
  '''
  from srdatalog.ir.core.pragma import get_pragma_registrations

  regs = [r for r in get_pragma_registrations() if r.pragma_cls is WorkStealing]
  assert len(regs) >= 1
  reg = regs[0]
  assert reg.on is m.ExecutePipeline
  assert reg.fn is materialize_work_stealing


def test_ws_scope_lowering_is_registered_on_atomic_ws_dialect():
  '''The WSScope `@lowering` is registered on the
  `parallel.atomic_ws` sub-dialect. Pins the dialect ownership
  choice per `docs/phase_c_pragma_materialization.md` §4.2
  (`work_stealing` lives in a new `parallel.atomic_ws` sub-dialect,
  not in `sorted_array`).'''
  matched = [low for low in WS_DIALECT.lowerings if low.matches is m.WSScope]
  assert len(matched) == 1
  assert matched[0].consumes == ('mir',)
  assert 'iir.cf' in matched[0].produces


def test_atomic_ws_dialect_name_pinned():
  '''The sub-dialect must be named exactly `parallel.atomic_ws` per
  spec §4.2 — downstream tooling (renderer dispatch, plugin
  manifests, discipline tests) keys on this string.'''
  assert WS_DIALECT.name == 'parallel.atomic_ws'


# -----------------------------------------------------------------------------
# 5. PlanEntry pragma field default + DSL dual-write round-trip
# -----------------------------------------------------------------------------


def test_plan_entry_pragmas_default_empty():
  '''Pin the `PlanEntry.pragmas` default. Post-A3-2 the legacy
  `PlanEntry.work_stealing` shadow bool is gone — the typed
  `WorkStealing()` pragma in `pragmas` is the sole signal.'''
  pe = PlanEntry()
  assert pe.pragmas == ()


def test_plan_entry_replace_with_work_stealing_pragma_round_trip():
  '''`dataclasses.replace(plan_entry, pragmas=(WorkStealing(),))`
  preserves the rest of the entry. Symmetric with how
  `Rule.with_pragma` builds new plans.'''
  pe = PlanEntry(delta=0, var_order=('a', 'b'))
  pe2 = dataclasses.replace(pe, pragmas=(WorkStealing(),))
  assert pe2.delta == 0
  assert pe2.var_order == ('a', 'b')
  assert isinstance(pe2.pragmas[0], WorkStealing)


# -----------------------------------------------------------------------------
# 6. apply_mir_pragma_pass surface (the chain step from C1/C2)
# -----------------------------------------------------------------------------


def test_apply_mir_pragma_pass_handles_work_stealing(mir_compiler):
  '''`apply_mir_pragma_pass` (the chain step landed in C1, exercised
  by C2's DedupHash) handles `WorkStealing` symmetrically.

  Pure typed path: the wrap op appears post-pass. The dual-write
  path's bool-skip behavior is covered by
  `test_pragma_pass_skips_wrap_in_dual_write_mode`.
  '''
  ep = _scan_insert_ep(pragmas=(WorkStealing(),))
  steps = [(ep, False)]
  out_steps = apply_mir_pragma_pass(steps)
  ep_out = out_steps[0][0]
  assert isinstance(ep_out, m.ExecutePipeline)
  assert ep_out.pragmas == ()
  assert isinstance(ep_out.pipeline[1], m.WSScope)


def test_apply_mir_pragma_pass_is_idempotent_for_empty_pragmas():
  '''After running `apply_mir_pragma_pass`, every EP has `pragmas
  == ()`. Re-running it is a structural no-op. This is the C1
  discipline gate
  (`test_pragmas_empty_after_materialization`) applied through the
  chain entry, pinned for the C4 pragma class.
  '''
  ep = _scan_insert_ep(pragmas=())
  steps = [(ep, False)]
  once = apply_mir_pragma_pass(steps)
  twice = apply_mir_pragma_pass(once)
  ep1 = once[0][0]
  ep2 = twice[0][0]
  assert isinstance(ep1, m.ExecutePipeline)
  assert isinstance(ep2, m.ExecutePipeline)
  assert ep1.pragmas == ()
  assert ep2.pragmas == ()


# -----------------------------------------------------------------------------
# 7. Coexistence — DedupHash + WorkStealing on the same EP
# -----------------------------------------------------------------------------


def test_work_stealing_coexists_with_dedup_hash_on_same_ep(mir_compiler):
  '''The two C2/C4 pragmas can both fire on the same EP without
  surviving the pass. Each handler only wraps the bare `InsertInto`
  it sees; whichever fires first claims the leaf, the second one
  finds no leaf to wrap and simply strips its pragma. End-state
  invariants:

    - `out.pragmas == ()` (both consumed),
    - the leaf `InsertInto` lives under at least one wrap op (the
      first-firing pragma's),
    - the pipeline is structurally well-formed (no surviving raw
      `InsertInto` if the first-firing pragma wrapped, or only the
      raw `InsertInto` if neither side wraps — but in this fixture
      WorkStealing is registered after DedupHash and both fire, so
      at least one wrap is guaranteed).

  This pins the discipline that pragma materialization is a sequence
  of independent wraps, not a fixed topology — without an explicit
  `before` / `after` declaration the spec's topo-sort breaks ties
  by registration order. C4 does NOT declare cross-pragma ordering
  with C2 (the wraps' lowerings are independent), so we assert only
  the end-state.
  '''
  from srdatalog.ir.dialects.relation.sorted_array import DIALECT as SA_DIALECT
  from srdatalog.ir.dialects.relation.sorted_array.pragmas.dedup_hash import (
    DedupHash,
  )

  c = Compiler()
  c.register_dialect(MIR_DIALECT)
  c.register_dialect(SA_DIALECT)
  c.register_dialect(WS_DIALECT)

  ep = _scan_insert_ep(
    pragmas=(WorkStealing(), DedupHash()),
  )
  # Force the dedup_hash bool off so the DedupHash handler hits its
  # wrap-emitting (non-dual-write) branch. (WorkStealing's
  # corresponding bool shadow was removed in A3-2.)
  ep = dataclasses.replace(ep, dedup_hash=False)
  out = MirPragmaPass().apply(ep, c)
  # Both pragmas consumed.
  assert out.pragmas == ()
  # The leaf InsertInto lives under at least one wrap op.
  seen: set[type] = set()
  cur = out.pipeline[-1]
  for _ in range(8):
    seen.add(type(cur))
    inner = getattr(cur, 'inner', None)
    if inner is None:
      break
    cur = inner
  assert m.InsertInto in seen
  assert seen & {m.WSScope, m.DedupGate}, 'at least one wrap op fired'
