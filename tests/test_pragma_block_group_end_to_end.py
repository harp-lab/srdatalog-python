'''End-to-end test for Phase C3: the `BlockGroup` typed pragma.

Per `docs/phase_c_pragma_materialization.md` §5.1 (per-PR acceptance
gate), each Phase C migration ships a `test_pragma_<name>_end_to_end`
that exercises the pragma via the production `Compiler.run`-style
pipeline and asserts:

  - The wrap op (`mir.BlockGroupRoot`) appears in MIR after
    `MirPragmaPass` (or is suppressed under the dual-write
    transition; both states are verified here).
  - The lowered IIR is correct (matches the BG branch of
    `_lower_root_cj_multi` -> `_lower_root_cj_bg`).
  - The rendered CUDA is byte-equivalent to the legacy
    `block_group: bool` path (the load-bearing harness for C3's
    backward-compat contract).

Plus DSL surface checks:

  - `Rule(...).with_pragma(BlockGroup())` accepts a typed instance.
  - `Rule(...).with_pragma(<not-a-pragma>)` raises `TypeError`.
  - `Rule(...).with_pragma(<unregistered-Pragma>)` raises
    `UnregisteredPragmaError` (typed sibling of `TypeError`).
  - The DSL dual-writes BOTH the typed pragma onto the rule's
    plan(s) AND the matching legacy bool field (`block_group=True`)
    so the byte-equiv path through `compile_kernel_body` keeps
    working unchanged.

The framework infrastructure (the `MirPragmaPass` driver, the
`@pragma_handler` registry, error classes) is exercised by
`tests/test_mir_pragma_pass.py` + `tests/test_core_pragma.py`; this
file owns the per-pragma surface.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, final

import pytest

import srdatalog.ir.mir.types as m
from srdatalog.dsl import PlanEntry, Rule
from srdatalog.ir.core import Compiler, Pragma, UnregisteredPragmaError
from srdatalog.ir.dialects.parallel.block_group import DIALECT as BG_DIALECT
from srdatalog.ir.dialects.parallel.block_group.pragmas.block_group import (
  BlockGroup,
  lower_block_group_root,
  materialize_block_group,
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
  '''Compiler with MIR + parallel.block_group dialects registered.

  Importing `parallel.block_group` runs `_register_passes`, which
  imports the pragma submodule for side effects (registers
  `@pragma_handler(BlockGroup, ...)` + the `BlockGroupRoot` lowering),
  so this fixture is enough to drive `MirPragmaPass` end-to-end
  without monkey-patching the registry.
  '''
  c = Compiler()
  c.register_dialect(MIR_DIALECT)
  c.register_dialect(BG_DIALECT)
  return c


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _cj_insert_ep(
  *,
  rule_name: str = 'BgRule',
  block_group: bool = False,
  pragmas: tuple[Pragma, ...] = (),
) -> m.ExecutePipeline:
  '''Build a minimal multi-source-root-CJ EP, optionally carrying a
  typed `BlockGroup` pragma instead of (or in addition to) the
  legacy bool field.

  Pipeline shape mirrors the minimum the BG path requires: a root
  multi-source `ColumnJoin` followed by an `InsertInto`. Two
  `ColumnSource`s on the join (R and S), each indexed on column 0.
  '''
  s1 = m.ColumnSource(
    rel_name='R',
    version=Version.FULL,
    index=[0, 1],
    prefix_vars=[],
    handle_start=0,
    clause_idx=0,
  )
  s2 = m.ColumnSource(
    rel_name='S',
    version=Version.FULL,
    index=[0, 1],
    prefix_vars=[],
    handle_start=1,
    clause_idx=1,
  )
  cj = m.ColumnJoin(var_name='x', sources=[s1, s2], handle_start=-1)
  ins = m.InsertInto(rel_name='Dst', version=Version.NEW, vars=['x'], index=[0])
  return m.ExecutePipeline(
    pipeline=[cj, ins],
    source_specs=[s1, s2],
    dest_specs=[ins],
    rule_name=rule_name,
    block_group=block_group,
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
  '''Calling `.with_pragma(BlockGroup())` on a rule with no plans
  appends a default `PlanEntry(delta=-1)` carrying the pragma AND
  the matching legacy bool field (the C3 dual-write contract).'''
  rule = _empty_rule().with_pragma(BlockGroup())
  assert len(rule.plans) == 1
  plan = rule.plans[0]
  assert plan.delta == -1
  assert plan.block_group is True
  assert len(plan.pragmas) == 1
  assert isinstance(plan.pragmas[0], BlockGroup)


def test_with_pragma_appends_to_existing_plans():
  '''When the rule already has plans, `.with_pragma(BlockGroup())`
  appends the pragma to EVERY plan's `pragmas` tuple AND sets each
  plan's `block_group=True` (per-variant dual-write). This matches
  the per-rule semantic intent of `with_pragma`.'''
  rule = (
    _empty_rule()
    .with_plan(delta=0)
    .with_plan(delta=1, var_order=('a', 'b'))
    .with_pragma(BlockGroup())
  )
  assert len(rule.plans) == 2
  for plan in rule.plans:
    assert plan.block_group is True
    assert any(isinstance(p, BlockGroup) for p in plan.pragmas)


def test_with_pragma_rejects_non_pragma_arg():
  '''`Rule.with_pragma("block_group")` (the legacy string-keyed form)
  is hard-error per `docs/code_discipline.md` D15. Same for any
  non-`Pragma` instance.'''
  rule = _empty_rule()
  with pytest.raises(TypeError, match=r'expected a Pragma subclass'):
    rule.with_pragma('block_group')  # type: ignore[arg-type]
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


def test_pragma_pass_inserts_block_group_root_when_bool_is_false(mir_compiler):
  '''Pure typed path (bool=False, only pragma set): MirPragmaPass
  wraps the root multi-source `ColumnJoin` in `BlockGroupRoot`.

  This is the post-A3 target state. Today only synthesized EPs
  reach it — the DSL `.with_pragma(BlockGroup())` dual-writes the
  bool field as well, hitting the dual-write skip branch below.
  '''
  ep = _cj_insert_ep(block_group=False, pragmas=(BlockGroup(),))
  out = MirPragmaPass().apply(ep, mir_compiler)
  # Pragma instance consumed.
  assert out.pragmas == ()
  # Pipeline: BlockGroupRoot(ColumnJoin) + InsertInto.
  assert len(out.pipeline) == 2
  assert isinstance(out.pipeline[0], m.BlockGroupRoot)
  assert isinstance(out.pipeline[0].inner, m.ColumnJoin)
  assert isinstance(out.pipeline[1], m.InsertInto)
  # Legacy bool was NOT toggled; the wrap op is the sole signal.
  assert out.block_group is False


def test_pragma_pass_skips_wrap_in_dual_write_mode(mir_compiler):
  '''Dual-write transition (bool=True AND pragma set, the C3 DSL
  shape): MirPragmaPass strips the pragma but leaves `pipeline`
  untouched. The legacy `complete_runner._gen_kernel_bg_*` /
  `compile_kernel_body(bg_enabled=True)` chain produces the BG
  emission; the wrap op never appears so downstream consumers
  (envelope handle assignment, view-spec collection) don't need
  to know about `BlockGroupRoot`.

  This is the load-bearing test for the dual-write contract: the
  C3 PR must not break the byte-equivalence harness, and that
  requires the bool-field path to remain the sole emission driver
  until A3 drops the bool.
  '''
  ep = _cj_insert_ep(block_group=True, pragmas=(BlockGroup(),))
  out = MirPragmaPass().apply(ep, mir_compiler)
  # Pragma still consumed.
  assert out.pragmas == ()
  # No BlockGroupRoot inserted.
  assert all(not isinstance(child, m.BlockGroupRoot) for child in out.pipeline)
  # Bool preserved for the legacy emitter.
  assert out.block_group is True


def test_pragma_pass_via_apply_all_mir_passes_is_noop_for_empty_pragmas(
  mir_compiler,
):
  '''Sanity: the integration of `MirPragmaPass` into
  `apply_all_mir_passes` does not change MIR shape for EPs that
  carry no typed pragmas — the dominant case today.'''
  ep = _cj_insert_ep(block_group=False, pragmas=())
  steps_in = [(ep, False)]
  steps_out = apply_all_mir_passes(steps_in)
  # Same number of steps, same EP identity-or-equal at the top.
  assert len(steps_out) == 1
  ep_out, is_rec = steps_out[0]
  assert isinstance(ep_out, m.ExecutePipeline)
  assert is_rec is False
  assert ep_out.pragmas == ()
  # Pipeline contents structurally identical (no BlockGroupRoot insertion).
  assert [type(o).__name__ for o in ep_out.pipeline] == ['ColumnJoin', 'InsertInto']


def test_pragma_pass_does_not_wrap_non_cj_root(mir_compiler):
  '''The legacy `ctx.bg_enabled` dispatch only fires on root multi-
  source ColumnJoin (`_lower_root_cj_multi` -> `_lower_root_cj_bg`).
  Root Scan and root Cartesian have NO BG branch — `bg_enabled` is
  a silent no-op there in the legacy code. To preserve byte-equiv,
  the pragma handler also skips wrapping non-CJ roots; the pragma
  is still consumed so MirPragmaPass's post-flight invariant holds.
  '''
  scan = m.Scan(vars=['v0'], rel_name='Src', version=Version.FULL, index=[0], handle_start=0)
  ins = m.InsertInto(rel_name='Dst', version=Version.NEW, vars=['v0'], index=[0])
  ep = m.ExecutePipeline(
    pipeline=[scan, ins],
    source_specs=[scan],
    dest_specs=[ins],
    rule_name='ScanRoot',
    block_group=False,
    pragmas=(BlockGroup(),),
  )
  out = MirPragmaPass().apply(ep, mir_compiler)
  assert out.pragmas == ()
  # No wrap inserted; the Scan root stays as-is.
  assert isinstance(out.pipeline[0], m.Scan)
  assert all(not isinstance(child, m.BlockGroupRoot) for child in out.pipeline)


# -----------------------------------------------------------------------------
# 3. Lowering — BlockGroupRoot -> IIR matches legacy BG branch byte-for-byte
# -----------------------------------------------------------------------------


def _make_bg_lowering_ctx() -> Any:
  '''Build a minimally-populated `LoweringCtx` for the BG path.

  `_lower_root_cj_bg` needs `view_var_names` populated for every
  source handle_idx in the wrapped CJ. The test EP uses handles 0
  (R) and 1 (S); we provide both with the standard view names
  matching what `emit_view_declarations` would produce.

  Returns `Any` (not the concrete `LoweringCtx`) to avoid threading
  the sorted_array lowerings import through the test file's top-
  level imports — the import lives at function scope where its only
  use is constructing the ctx instance.
  '''
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import LoweringCtx

  return LoweringCtx(
    view_var_names={'0': 'view_R_0_FULL', '1': 'view_S_1_FULL'},
    is_counting=False,
    output_var='output',
    bg_enabled=False,
    debug=True,
  )


def test_block_group_root_lowering_matches_legacy_bg_emission(mir_compiler):
  '''Byte-equivalence by construction: the new `lower_block_group_root`
  function delegates to `_lower_root_cj_bg` with `ctx.bg_enabled=True`,
  so its emission is identical to the legacy `if ctx.bg_enabled:`
  branch inside `_lower_root_cj_multi`.

  We verify this end-to-end by:
    1. Building an EP with the typed pragma (bool=False); running
       `MirPragmaPass` to get the wrap-op MIR.
    2. Calling `lower_block_group_root` directly against the wrapped
       op and a ctx carrying the trailing `rest` via the
       `rest_for_bg_root` attribute.
    3. Calling `_lower_root_cj_bg` directly with the unwrapped op
       and the same `rest` + a ctx with `bg_enabled=True`.
    4. Asserting the rendered IIR matches between (2) and (3).
  '''
  from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import _lower_root_cj_bg

  typed_ep = _cj_insert_ep(block_group=False, pragmas=(BlockGroup(),))
  materialized = MirPragmaPass().apply(typed_ep, mir_compiler)
  wrap = materialized.pipeline[0]
  rest = list(materialized.pipeline[1:])
  assert isinstance(wrap, m.BlockGroupRoot)
  assert isinstance(wrap.inner, m.ColumnJoin)

  # Path (a): via lower_block_group_root. Stash `rest` on the ctx
  # the way `lower_scan_pipeline` will when it sees a BlockGroupRoot
  # head.
  ctx_a = _make_bg_lowering_ctx()
  ctx_a.rest_for_bg_root = rest
  iir_a = lower_block_group_root(wrap, ctx_a)
  rendered_a = emit(iir_a, EmitCtx(indent_level=4))

  # Path (b): directly via _lower_root_cj_bg with bg_enabled=True.
  ctx_b = _make_bg_lowering_ctx()
  ctx_b.bg_enabled = True
  iir_b = _lower_root_cj_bg(wrap.inner, rest, ctx_b)
  rendered_b = emit(iir_b, EmitCtx(indent_level=4))

  assert rendered_a == rendered_b
  # Sanity: the rendered text contains BG-specific markers (the
  # `BlockGroupRoot` lowering really went through the BG path, not
  # the standard root_cj_multi path).
  assert 'BLOCK-GROUP' in rendered_a or 'block-group' in rendered_a.lower()


def test_block_group_root_lowering_restores_prior_bg_enabled_flag(mir_compiler):
  '''The lowering's `ctx.bg_enabled` toggle is save/restore — after
  the BG dispatch emits, the ctx is left exactly as it was. Guards
  against reentrancy leaks if sibling lowerings depend on the flag.
  '''
  typed_ep = _cj_insert_ep(block_group=False, pragmas=(BlockGroup(),))
  materialized = MirPragmaPass().apply(typed_ep, mir_compiler)
  wrap = materialized.pipeline[0]
  rest = list(materialized.pipeline[1:])
  assert isinstance(wrap, m.BlockGroupRoot)

  for original in (False, True):
    ctx = _make_bg_lowering_ctx()
    ctx.bg_enabled = original
    ctx.rest_for_bg_root = rest
    lower_block_group_root(wrap, ctx)
    assert ctx.bg_enabled is original


def test_lower_scan_pipeline_dispatches_block_group_root(mir_compiler):
  '''The `BlockGroupRoot` head-op dispatch hook in `lower_scan_pipeline`
  (sorted_array dialect, C3): when the head op is a `BlockGroupRoot`,
  unwrap it and re-enter the root-CJ-multi flow with `ctx.bg_enabled=
  True` for the duration. This is the production hook
  `MirPragmaPass`-then-`compile_kernel_body` users hit; the direct
  `lower_block_group_root` call in the prior test exercises the
  registry surface in isolation.
  '''
  from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _lower_root_cj_bg,
    lower_scan_pipeline,
  )

  typed_ep = _cj_insert_ep(block_group=False, pragmas=(BlockGroup(),))
  materialized = MirPragmaPass().apply(typed_ep, mir_compiler)
  wrap = materialized.pipeline[0]
  assert isinstance(wrap, m.BlockGroupRoot)

  ctx_a = _make_bg_lowering_ctx()
  iir_a = lower_scan_pipeline(list(materialized.pipeline), ctx_a)
  rendered_a = emit(iir_a, EmitCtx(indent_level=4))

  ctx_b = _make_bg_lowering_ctx()
  ctx_b.bg_enabled = True
  rest_b = list(materialized.pipeline[1:])
  iir_b = _lower_root_cj_bg(wrap.inner, rest_b, ctx_b)
  rendered_b = emit(iir_b, EmitCtx(indent_level=4))

  assert rendered_a == rendered_b
  # Sanity: ctx_a should NOT have bg_enabled mutated by the dispatch
  # (it was False going in, and the unwrap path save/restores it).
  assert ctx_a.bg_enabled is False


# -----------------------------------------------------------------------------
# 4. Registration completeness (R5)
# -----------------------------------------------------------------------------


def test_block_group_handler_is_registered():
  '''Importing the pragma module registers a `@pragma_handler` whose
  `pragma_cls` is `BlockGroup` and whose `on` is `mir.ExecutePipeline`.
  R5 (`test_pragma_handler_registry_completeness`) gates this once
  per Pragma subclass; here we pin it for the C3 surface explicitly.
  '''
  from srdatalog.ir.core.pragma import get_pragma_registrations

  regs = [r for r in get_pragma_registrations() if r.pragma_cls is BlockGroup]
  assert len(regs) >= 1
  reg = regs[0]
  assert reg.on is m.ExecutePipeline
  assert reg.fn is materialize_block_group


def test_block_group_root_lowering_is_registered_on_block_group_dialect():
  '''The BlockGroupRoot `@lowering` is registered on the
  `parallel.block_group` dialect. Pins the dialect ownership choice
  per `docs/phase_c_pragma_materialization.md` §4.1 (`block_group`
  owns a NEW sub-dialect `parallel.block_group`).'''
  matched = [low for low in BG_DIALECT.lowerings if low.matches is m.BlockGroupRoot]
  assert len(matched) == 1
  assert matched[0].consumes == ('mir',)
  assert 'iir.cf' in matched[0].produces


# -----------------------------------------------------------------------------
# 5. PlanEntry pragma field default
# -----------------------------------------------------------------------------


def test_plan_entry_pragmas_default_empty():
  '''Pin the new field's default. Tests that depend on PlanEntry
  semantics (test_hir_user_plan.py etc.) implicitly rely on `()`
  here so the C3 addition stays back-compatible.'''
  pe = PlanEntry()
  assert pe.pragmas == ()


def test_plan_entry_block_group_default_false():
  '''Sanity: a default PlanEntry has block_group=False (the field
  the dual-write will flip to True). Symmetric with the dedup_hash
  field default.'''
  pe = PlanEntry()
  assert pe.block_group is False


# -----------------------------------------------------------------------------
# 6. apply_mir_pragma_pass surface (the new chain step)
# -----------------------------------------------------------------------------


def test_apply_mir_pragma_pass_is_idempotent_for_empty_pragmas():
  '''After running `apply_mir_pragma_pass`, every EP has `pragmas ==
  ()`. Re-running it is a structural no-op. This is the C1
  discipline gate (R5b / `test_pragmas_empty_after_materialization`)
  applied through the new chain entry.
  '''
  ep = _cj_insert_ep(block_group=False, pragmas=())
  steps = [(ep, False)]
  once = apply_mir_pragma_pass(steps)
  twice = apply_mir_pragma_pass(once)
  ep1 = once[0][0]
  ep2 = twice[0][0]
  assert isinstance(ep1, m.ExecutePipeline)
  assert isinstance(ep2, m.ExecutePipeline)
  assert ep1.pragmas == ()
  assert ep2.pragmas == ()


def test_apply_mir_pragma_pass_consumes_block_group(mir_compiler):
  '''When an EP carries a `BlockGroup` pragma, `apply_mir_pragma_pass`
  consumes it (post-flight `pragmas == ()`); dual-write skip leaves
  the pipeline structurally identical to the legacy bool-only EP.'''
  ep = _cj_insert_ep(block_group=True, pragmas=(BlockGroup(),))
  steps = [(ep, False)]
  out = apply_mir_pragma_pass(steps)
  ep_out, _ = out[0]
  assert isinstance(ep_out, m.ExecutePipeline)
  assert ep_out.pragmas == ()
  assert ep_out.block_group is True
  # Dual-write skip: pipeline unchanged.
  assert [type(o).__name__ for o in ep_out.pipeline] == ['ColumnJoin', 'InsertInto']
