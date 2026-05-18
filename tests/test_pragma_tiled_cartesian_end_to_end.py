'''End-to-end test for Phase C5: the `TiledCartesian` typed pragma.

Per `docs/phase_c_pragma_materialization.md` §5.1 (per-PR acceptance
gate), each Phase C migration ships a `test_pragma_<name>_end_to_end`
that exercises the pragma via the production `Compiler.run`-style
pipeline and asserts:

  - The wrap op (`mir.TiledCartesian`) appears in MIR after
    `MirPragmaPass` (or is suppressed under the dual-write
    transition; both states are verified here).
  - The lowered IIR is correct (matches the tiled-cartesian branch
    of `_lower_nested_cart` -> `_lower_nested_cart_tiled`).
  - The rendered CUDA is byte-equivalent to the legacy
    `ctx.tiled_cartesian: bool` path (the load-bearing harness for
    C5's backward-compat contract).

Plus DSL surface checks:

  - `Rule(...).with_pragma(TiledCartesian())` accepts a typed instance.
  - `Rule(...).with_pragma(<not-a-pragma>)` raises `TypeError`.
  - `Rule(...).with_pragma(<unregistered-Pragma>)` raises
    `UnregisteredPragmaError` (typed sibling of `TypeError`).
  - `tiled_cartesian` is NOT a per-rule PlanEntry bool today (the
    runner drives eligibility from pipeline shape via
    `has_tiled_cartesian_eligible`), so `with_pragma` only sets the
    typed `pragmas` tuple — there is no PlanEntry bool to mirror.

The framework infrastructure (the `MirPragmaPass` driver, the
`@pragma_handler` registry, error classes) is exercised by
`tests/test_mir_pragma_pass.py` + `tests/test_core_pragma.py`; this
file owns the per-pragma surface.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import pytest

import srdatalog.ir.mir.types as m
from srdatalog.dsl import Rule
from srdatalog.ir.core import Compiler, Pragma, UnregisteredPragmaError
from srdatalog.ir.dialects.relation.sorted_array import DIALECT as SA_DIALECT
from srdatalog.ir.dialects.relation.sorted_array.pragmas.tiled_cartesian import (
  TiledCartesian,
  lower_tiled_cartesian,
  lower_tiled_cartesian_in_chain,
  materialize_tiled_cartesian,
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
  submodule for side effects (registers `@pragma_handler(
  TiledCartesian, ...)` + the `mir.TiledCartesian` lowering), so
  this fixture is enough to drive `MirPragmaPass` end-to-end without
  monkey-patching the registry.
  '''
  c = Compiler()
  c.register_dialect(MIR_DIALECT)
  c.register_dialect(SA_DIALECT)
  return c


# -----------------------------------------------------------------------------
# Helpers — synthetic tiled-eligible EP
# -----------------------------------------------------------------------------


def _eligible_cart_ep(
  *,
  tiled_cartesian: bool = False,
  pragmas: tuple[Pragma, ...] = (),
) -> m.ExecutePipeline:
  '''Build a Scan + 2-source/1-var-per-source CartesianJoin +
  InsertInto pipeline — the canonical tiled-eligible shape per
  `pipeline_utils.has_tiled_cartesian_eligible`.

  The Scan binds `x` from `Src` then the nested Cartesian binds `y`
  from `R` and `z` from `S`, emitting `Dst(x, y, z)`. Sources have
  no prefix vars, so the Cartesian uses fresh roots (no state-key
  reuse). Handles are pre-assigned to mirror the post-
  `assign_handle_positions_in_ep` shape `compile_pipeline` produces.
  '''
  # Handles match the post-`assign_handle_positions` shape: Scan
  # claims slot 0 then bumps; CartesianJoin captures `handle =
  # offset_box[0]` (here 1) BEFORE recursing into sources, then
  # the sources claim slots 1 and 2 in order — so Cart.handle_start
  # aliases its first source's handle, matching the legacy
  # convention. Re-running `assign_handle_positions` on this fixture
  # is idempotent.
  scan = m.Scan(
    vars=['x'],
    rel_name='Src',
    version=Version.FULL,
    index=[0],
    handle_start=0,
  )
  src_r = m.ColumnSource(
    rel_name='R',
    version=Version.FULL,
    index=[0],
    prefix_vars=[],
    handle_start=1,
  )
  src_s = m.ColumnSource(
    rel_name='S',
    version=Version.FULL,
    index=[0],
    prefix_vars=[],
    handle_start=2,
  )
  cart = m.CartesianJoin(
    vars=['y', 'z'],
    sources=[src_r, src_s],
    var_from_source=[['y'], ['z']],
    handle_start=1,
  )
  insert = m.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['x', 'y', 'z'],
    index=[0, 1, 2],
  )
  return m.ExecutePipeline(
    pipeline=[scan, cart, insert],
    source_specs=[scan, src_r, src_s],
    dest_specs=[insert],
    rule_name='TC',
    tiled_cartesian=tiled_cartesian,
    pragmas=pragmas,  # type: ignore[arg-type]
  )


def _ineligible_cart_ep(
  *,
  pragmas: tuple[Pragma, ...] = (),
) -> m.ExecutePipeline:
  '''Build a Scan + 3-source CartesianJoin + InsertInto pipeline —
  NOT tiled-eligible (eligibility requires exactly 2 sources). Used
  to verify the wrap step leaves non-eligible Cartesians alone.
  '''
  scan = m.Scan(
    vars=['x'],
    rel_name='Src',
    version=Version.FULL,
    index=[0],
    handle_start=0,
  )
  srcs = [
    m.ColumnSource(
      rel_name=name,
      version=Version.FULL,
      index=[0],
      prefix_vars=[],
      handle_start=1 + i,
    )
    for i, name in enumerate(['R', 'S', 'T'])
  ]
  cart = m.CartesianJoin(
    vars=['y', 'z', 'w'],
    sources=srcs,
    var_from_source=[['y'], ['z'], ['w']],
    handle_start=1,
  )
  insert = m.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['x', 'y', 'z', 'w'],
    index=[0, 1, 2, 3],
  )
  return m.ExecutePipeline(
    pipeline=[scan, cart, insert],
    source_specs=[scan, *srcs],
    dest_specs=[insert],
    rule_name='TCIneligible',
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
  '''Calling `.with_pragma(TiledCartesian())` on a rule with no
  plans appends a default `PlanEntry(delta=-1)` carrying the
  pragma. There is NO matching legacy PlanEntry bool field (unlike
  `DedupHash` -> `dedup_hash`), so the bool-shadow map returns `{}`
  and only the typed `pragmas` tuple gets populated.'''
  rule = _empty_rule().with_pragma(TiledCartesian())
  assert len(rule.plans) == 1
  plan = rule.plans[0]
  assert plan.delta == -1
  assert len(plan.pragmas) == 1
  assert isinstance(plan.pragmas[0], TiledCartesian)


def test_with_pragma_appends_to_existing_plans():
  '''When the rule already has plans, `.with_pragma(TiledCartesian())`
  appends the pragma to EVERY plan's `pragmas` tuple. Matches the
  per-rule semantic intent of `with_pragma` ("this rule, every
  variant"), symmetric with C2.'''
  rule = (
    _empty_rule()
    .with_plan(delta=0)
    .with_plan(delta=1, var_order=('a', 'b'))
    .with_pragma(TiledCartesian())
  )
  assert len(rule.plans) == 2
  for plan in rule.plans:
    assert any(isinstance(p, TiledCartesian) for p in plan.pragmas)


def test_with_pragma_rejects_non_pragma_arg():
  '''Legacy string-keyed form is hard-error per
  `docs/code_discipline.md` D15. Same for any non-`Pragma`
  instance.'''
  rule = _empty_rule()
  with pytest.raises(TypeError, match=r'expected a Pragma subclass'):
    rule.with_pragma('tiled_cartesian')  # type: ignore[arg-type]
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


def test_pragma_pass_inserts_tiled_cartesian_when_bool_is_false(mir_compiler):
  '''Pure typed path (bool=False, only pragma set): MirPragmaPass
  wraps every eligible nested `CartesianJoin` in `pipeline` with
  `mir.TiledCartesian`.

  This is the post-A3 target state. Synthesized EPs reach it
  directly; the DSL `.with_pragma(TiledCartesian())` adds only the
  typed pragma today (no PlanEntry bool exists to dual-write), but
  whether the EP-level bool ends up True depends on HIR -> MIR
  lowering choices outside C5's scope.
  '''
  ep = _eligible_cart_ep(tiled_cartesian=False, pragmas=(TiledCartesian(),))
  out = MirPragmaPass().apply(ep, mir_compiler)
  # Pragma instance consumed.
  assert out.pragmas == ()
  # Pipeline: Scan + TiledCartesian(CartesianJoin) + InsertInto.
  assert len(out.pipeline) == 3
  assert isinstance(out.pipeline[0], m.Scan)
  assert isinstance(out.pipeline[1], m.TiledCartesian)
  assert isinstance(out.pipeline[1].inner, m.CartesianJoin)
  assert isinstance(out.pipeline[2], m.InsertInto)
  # Legacy bool was NOT toggled; the wrap op is the sole signal.
  assert out.tiled_cartesian is False


def test_pragma_pass_skips_wrap_in_dual_write_mode(mir_compiler):
  '''Dual-write transition (bool=True AND pragma set): MirPragmaPass
  strips the pragma but leaves `pipeline` untouched. The legacy
  runner-driven path (`complete_runner.py:has_tiled_cartesian_eligible`
  -> `LoweringCtx.tiled_cartesian=True` -> `_lower_nested_cart` ->
  `_lower_nested_cart_tiled`) produces the tiled emit; the wrap op
  never appears so the monolith doesn't need to know about
  `mir.TiledCartesian`.

  This is the load-bearing test for the dual-write contract: the C5
  PR must not break the byte-equivalence harness, and that requires
  the legacy path to remain the sole emission driver whenever the
  bool is set.
  '''
  ep = _eligible_cart_ep(tiled_cartesian=True, pragmas=(TiledCartesian(),))
  out = MirPragmaPass().apply(ep, mir_compiler)
  # Pragma still consumed.
  assert out.pragmas == ()
  # No TiledCartesian wrap inserted.
  assert all(not isinstance(child, m.TiledCartesian) for child in out.pipeline)
  # Bool preserved for the legacy emitter.
  assert out.tiled_cartesian is True


def test_pragma_pass_skips_ineligible_cartesian(mir_compiler):
  '''A 3-source CartesianJoin (or any shape that
  `pipeline_utils.has_tiled_cartesian_eligible` rejects) is NOT
  wrapped. Mirrors `_is_eligible_cart` predicate; protects against
  emitting `TiledCartesian` around a Cartesian whose body the
  tiled smem dispatch can't accommodate.
  '''
  ep = _ineligible_cart_ep(pragmas=(TiledCartesian(),))
  out = MirPragmaPass().apply(ep, mir_compiler)
  assert out.pragmas == ()
  # The 3-source Cartesian passes through unchanged.
  assert all(not isinstance(child, m.TiledCartesian) for child in out.pipeline)
  assert isinstance(out.pipeline[1], m.CartesianJoin)
  assert len(out.pipeline[1].sources) == 3


def test_pragma_pass_via_apply_all_mir_passes_is_noop_for_empty_pragmas(
  mir_compiler,
):
  '''Sanity: the integration of `MirPragmaPass` into
  `apply_all_mir_passes` does not change MIR shape for EPs that
  carry no typed pragmas — the dominant case today.'''
  ep = _eligible_cart_ep(tiled_cartesian=False, pragmas=())
  steps_in = [(ep, False)]
  steps_out = apply_all_mir_passes(steps_in)
  assert len(steps_out) == 1
  ep_out, is_rec = steps_out[0]
  assert isinstance(ep_out, m.ExecutePipeline)
  assert is_rec is False
  assert ep_out.pragmas == ()
  # Pipeline contents structurally identical (no wrap insertion).
  assert [type(o).__name__ for o in ep_out.pipeline] == [
    'Scan',
    'CartesianJoin',
    'InsertInto',
  ]


# -----------------------------------------------------------------------------
# 3. Lowering — TiledCartesian -> IIR matches legacy tiled branch byte-for-byte
# -----------------------------------------------------------------------------


def _render_body(ep: m.ExecutePipeline, *, tiled_cartesian: bool) -> str:
  '''Render an EP through the production `compile_kernel_body` surface,
  which exercises the same `lower_scan_pipeline` -> dialect path the
  byte-equivalence harnesses guard. `tiled_cartesian` controls the
  legacy ctx flag (the runner-side signal); set independently of
  whether the EP carries the typed pragma.'''
  from srdatalog.compile import compile_kernel_body

  return compile_kernel_body(
    ep,
    is_counting=False,
    slot_mode='handle_idx',
    tiled_cartesian=tiled_cartesian,
  )


def test_tiled_cartesian_lowering_matches_legacy_emission(mir_compiler):
  '''Byte-equivalence by construction: the new wrap-op dispatch in
  `_lower_inner_chain` delegates to `_lower_nested_cart_tiled` with
  `ctx.tiled_cartesian=True`, so its emission is identical to the
  legacy `if _tiled_cart_eligible(...):` branch.

  We verify this end-to-end by:
    1. Building an EP with the legacy bool (no pragma); rendering
       via `compile_kernel_body(tiled_cartesian=True)` — the
       "ground truth" the byte-equivalence harness anchors to.
    2. Building the same EP with the typed pragma (bool=False);
       running `MirPragmaPass` to insert the wrap op, then
       rendering via `compile_kernel_body(tiled_cartesian=False)` —
       the legacy ctx flag is off, so emission comes purely from
       the wrap-op dispatch.

  The rendered CUDA must match — same tile_total var, tc_valid_<n>
  ballot, SaTiledCartesian2D structure.
  '''
  # 1. Ground truth: legacy ctx-driven path on a pragma-free EP.
  legacy_ep = _eligible_cart_ep(tiled_cartesian=False, pragmas=())
  legacy_out = _render_body(legacy_ep, tiled_cartesian=True)
  assert 'tile_total' in legacy_out
  assert 'tc_valid_' in legacy_out
  assert 'sa_tiled_cartesian_2d' not in legacy_out  # rendered as inline C++

  # 2. Typed-pragma path: pragma instance materializes the wrap op,
  # then `_lower_inner_chain` dispatches the wrap to the same
  # `_lower_nested_cart_tiled` helper.
  typed_ep = _eligible_cart_ep(tiled_cartesian=False, pragmas=(TiledCartesian(),))
  materialized = MirPragmaPass().apply(typed_ep, mir_compiler)
  assert isinstance(materialized.pipeline[1], m.TiledCartesian)
  typed_out = _render_body(materialized, tiled_cartesian=False)

  assert typed_out == legacy_out


def test_tiled_cartesian_lowering_restores_prior_flag(mir_compiler):
  '''The chain-aware lowering's `ctx.tiled_cartesian` toggle is
  save/restore — after the wrap emits, the ctx is left exactly as
  it was. Guards against reentrancy leaks if sibling dispatches
  depend on the flag.'''
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import LoweringCtx

  ep = _eligible_cart_ep(tiled_cartesian=False, pragmas=(TiledCartesian(),))
  materialized = MirPragmaPass().apply(ep, mir_compiler)
  wrap = materialized.pipeline[1]
  assert isinstance(wrap, m.TiledCartesian)
  insert = materialized.pipeline[2]
  assert isinstance(insert, m.InsertInto)

  for original in (False, True):
    ctx = LoweringCtx(
      view_var_names={
        '0': 'view_Src_0_FULL',
        '1': 'view_R_0_FULL',
        '2': 'view_S_0_FULL',
      },
      is_counting=False,
      output_var='output',
      tiled_cartesian=original,
    )
    lower_tiled_cartesian_in_chain(wrap, [insert], ctx)
    assert ctx.tiled_cartesian is original


# -----------------------------------------------------------------------------
# 4. Registration completeness (R5)
# -----------------------------------------------------------------------------


def test_tiled_cartesian_handler_is_registered():
  '''Importing the pragma module registers a `@pragma_handler` whose
  `pragma_cls` is `TiledCartesian` and whose `on` is
  `mir.ExecutePipeline`. R5
  (`test_pragma_handler_registry_completeness`) gates this once per
  Pragma subclass; here we pin it for the C5 surface explicitly.
  '''
  from srdatalog.ir.core.pragma import get_pragma_registrations

  regs = [r for r in get_pragma_registrations() if r.pragma_cls is TiledCartesian]
  assert len(regs) >= 1
  reg = regs[0]
  assert reg.on is m.ExecutePipeline
  assert reg.fn is materialize_tiled_cartesian


def test_tiled_cartesian_lowering_is_registered_on_sorted_array_dialect():
  '''The `mir.TiledCartesian` `@lowering` is registered on the
  `relation.sorted_array` dialect. Pins the dialect ownership choice
  per `docs/phase_c_pragma_materialization.md` §4.3
  (`tiled_cartesian` stays in sorted_array — no new sub-dialect
  needed because the emission uses the existing
  `SaTiledCartesian2D` IIR op).'''
  matched = [low for low in SA_DIALECT.lowerings if low.matches is m.TiledCartesian]
  assert len(matched) == 1
  assert matched[0].consumes == ('mir',)
  assert 'iir.cf' in matched[0].produces


def test_tiled_cartesian_framework_entry_asserts_on_direct_call():
  '''`lower_tiled_cartesian` is a framework-registry stub: real
  dispatch goes through `_lower_inner_chain` ->
  `lower_tiled_cartesian_in_chain` so the trailing chain is in
  scope. Direct invocation of the stub raises a structural
  assertion to surface refactors that bypass the chain dispatch.
  '''
  wrap = m.TiledCartesian(
    inner=m.CartesianJoin(
      vars=['y', 'z'],
      sources=[
        m.ColumnSource(rel_name='R', version=Version.FULL, index=[0]),
        m.ColumnSource(rel_name='S', version=Version.FULL, index=[0]),
      ],
      var_from_source=[['y'], ['z']],
    )
  )
  with pytest.raises(AssertionError, match=r'dispatch goes through'):
    lower_tiled_cartesian(wrap, None)


# -----------------------------------------------------------------------------
# 5. apply_mir_pragma_pass surface
# -----------------------------------------------------------------------------


def test_apply_mir_pragma_pass_is_idempotent_for_empty_pragmas():
  '''After running `apply_mir_pragma_pass`, every EP has `pragmas ==
  ()`. Re-running it is a structural no-op. Same discipline gate
  (R5b / `test_pragmas_empty_after_materialization`) the C2 test
  exercises, applied through the new chain entry for C5.
  '''
  ep = _eligible_cart_ep(tiled_cartesian=False, pragmas=())
  steps = [(ep, False)]
  once = apply_mir_pragma_pass(steps)
  twice = apply_mir_pragma_pass(once)
  ep1 = once[0][0]
  ep2 = twice[0][0]
  assert isinstance(ep1, m.ExecutePipeline)
  assert isinstance(ep2, m.ExecutePipeline)
  assert ep1.pragmas == ()
  assert ep2.pragmas == ()


def test_apply_mir_pragma_pass_materializes_wrap(mir_compiler):
  '''Sanity: piping through `apply_mir_pragma_pass` produces the
  same wrap-op insertion the per-EP `MirPragmaPass.apply` does.
  '''
  ep = _eligible_cart_ep(tiled_cartesian=False, pragmas=(TiledCartesian(),))
  steps = [(ep, False)]
  out = apply_mir_pragma_pass(steps)
  ep_out = out[0][0]
  assert isinstance(ep_out, m.ExecutePipeline)
  assert ep_out.pragmas == ()
  assert isinstance(ep_out.pipeline[1], m.TiledCartesian)
