'''Byte-equivalence test for Wave 2A B-InsertInto migration.

Per `docs/phase_b_lowering_dispatcher.md` §4.2 (per-PR acceptance
gate): each per-MIR-op migration ships a
`test_lower_mir_<op>_byte_equivalent` test that runs the migrated
path on every relevant fixture and asserts byte equality with the
legacy `if isinstance(head, mir.X):` branch.

`mir.InsertInto` is the TERMINAL op in every `ExecutePipeline.
pipeline`, so its lowering is exercised by every emitting kernel.
Beyond the standard shape coverage, this test pins COEXISTENCE
with the C2/C4/C6 typed-pragma wrap ops (`DedupGate`, `WSScope`,
`mir.FanOut`) that each wrap an `InsertInto`. Those wrap ops are
NOT in `USE_DECLARATIVE` — their lowerings live in the pragma
modules and delegate back into the same legacy
`_lower_insert_into` helper that B-InsertInto's chain entry calls.
The tests below verify both paths produce identical output for the
same `InsertInto`.

For InsertInto the relevant fixtures are:

  - the plain `emit_direct(...)` shape (no dedup_hash / ws / tiled
    flag set),
  - the count phase variant (`is_counting=True`, emits
    `emit_direct()` with no args, lane-zero guarded outside Cart),
  - the dedup_hash gate variant (`dedup_hash=True`, emits
    `try_insert(...) + if (_p) {...}` around an atomic-add write),
  - the work-stealing count variant (`is_counting=True` +
    `ws_enabled=True`, emits `<out>++`),
  - the tiled-Cartesian ballot variant
    (`tiled_cartesian_valid_var` set, emits a `TiledBallotBlock`),
  - the WS Cartesian batched-valid variant
    (`ws_cartesian_valid_var` set, emits `emit_warp_coalesced(...)`),
  - multi-head trailing-InsertInto runs (two or more InsertIntos
    in sequence at the chain tail),
  - dedup-hash and work-stealing wrap ops dispatched through the
    typed-pragma lowering vs the legacy bool field.

The test compares two compilation paths:

  - LEGACY: `USE_DECLARATIVE` patched to NOT contain
    `mir.InsertInto`, so `_lower_inner_chain` falls into the legacy
    imperative branch (the original `mir.InsertInto` head case below
    the `USE_DECLARATIVE` dispatch).
  - NEW: `USE_DECLARATIVE` left alone (InsertInto IS in the set), so
    `_lower_inner_chain` routes through
    `lower_mir_insert_into_in_chain`.

Byte-equivalence is asserted on the rendered IIR text (which the
load-bearing harnesses in `tests/test_runner_byte_equivalence.py`
+ `tests/test_byte_equivalence_jit.py` anchor against the legacy
CUDA emit). The 532-fixture harness re-running green under the new
path is the strongest signal; this file adds direct-call coverage
of the corner cases.
'''

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

import srdatalog.ir.mir.types as mir
from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit
from srdatalog.ir.dialects.iir.cf import Block as IirBlock
from srdatalog.ir.dialects.iir.cf import TiledBallotBlock
from srdatalog.ir.dialects.parallel.atomic_ws.pragmas.work_stealing import (
  lower_ws_scope,
)
from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
  LoweringCtx,
  _lower_inner_chain,
  _lower_insert_into,
)
from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_insert_into import (
  lower_mir_insert_into,
  lower_mir_insert_into_in_chain,
)
from srdatalog.ir.dialects.relation.sorted_array.pragmas.dedup_hash import (
  lower_dedup_gate,
)
from srdatalog.ir.dialects.relation.sorted_array.pragmas.fanout import (
  lower_fan_out,
)
from srdatalog.ir.hir.types import Version

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _render(op: Any) -> str:
  '''Render an IIR op tree through the CUDA emitter — the same
  surface the byte-equivalence harnesses exercise.'''
  return emit(op, EmitCtx(indent_level=2))


@contextmanager
def _force_legacy_branch():
  '''Temporarily strip `mir.InsertInto` from `USE_DECLARATIVE` so
  `_lower_inner_chain` falls into the legacy imperative branch.

  Save / restore as a context manager — discipline test
  `test_use_declarative_is_monotonic` (when it lands) ratchets the
  set at module import time, but this test mutates the dialect's
  re-bound name for the duration of one call only.
  '''
  import srdatalog.ir.dialects.relation.sorted_array as sa_dialect

  saved = sa_dialect.USE_DECLARATIVE
  sa_dialect.USE_DECLARATIVE = frozenset(saved - {mir.InsertInto})
  try:
    yield
  finally:
    sa_dialect.USE_DECLARATIVE = saved


def _insert_into(
  arity: int = 2,
  rel_name: str = 'Dst',
  vars_: list[str] | None = None,
) -> mir.InsertInto:
  vars_ = vars_ or [f'v{i}' for i in range(arity)]
  return mir.InsertInto(
    rel_name=rel_name,
    version=Version.NEW,
    vars=vars_,
    index=list(range(len(vars_))),
  )


def _new_ctx(**kwargs: Any) -> LoweringCtx:
  '''Fresh LoweringCtx — counters reset so both paths bump identically.'''
  return LoweringCtx(output_var='ctx0', **kwargs)


# -----------------------------------------------------------------------------
# 1. Plain emit_direct shape
# -----------------------------------------------------------------------------


def test_insert_into_plain_emit_direct_byte_equivalent():
  '''Standard InsertInto with no special flags set: emits
  `LaneZeroGuard { ctx0.emit_direct(v0, v1); }`. Both paths must
  produce identical rendered text.'''
  ins = _insert_into(2)

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([ins], _new_ctx()))
  new_text = _render(_lower_inner_chain([ins], _new_ctx()))

  assert legacy_text == new_text
  assert 'ctx0.emit_direct(v0, v1);' in legacy_text


def test_insert_into_inside_cartesian_no_lane_guard_byte_equivalent():
  '''Inside a Cartesian, the lane-zero guard is dropped — every
  thread emits cooperatively. Both paths must agree.'''
  ins = _insert_into(2)

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([ins], _new_ctx(inside_cartesian=True)))
  new_text = _render(_lower_inner_chain([ins], _new_ctx(inside_cartesian=True)))

  assert legacy_text == new_text
  assert 'ctx0.emit_direct(v0, v1);' in legacy_text
  # No lane-zero guard inside Cart.
  assert 'thread_rank()' not in legacy_text


# -----------------------------------------------------------------------------
# 2. Count phase variants
# -----------------------------------------------------------------------------


def test_insert_into_count_phase_byte_equivalent():
  '''Count phase: emits `ctx0.emit_direct()` (no args), lane-zero
  guarded.'''
  ins = _insert_into(2)

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([ins], _new_ctx(is_counting=True)))
  new_text = _render(_lower_inner_chain([ins], _new_ctx(is_counting=True)))

  assert legacy_text == new_text
  assert 'ctx0.emit_direct();' in legacy_text


def test_insert_into_ws_count_phase_byte_equivalent():
  '''Work-stealing count phase: emits `ctx0++` (per-thread local
  counter) instead of `emit_direct()`, still lane-zero guarded
  outside Cart.'''
  ins = _insert_into(2)

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([ins], _new_ctx(is_counting=True, ws_enabled=True)))
  new_text = _render(_lower_inner_chain([ins], _new_ctx(is_counting=True, ws_enabled=True)))

  assert legacy_text == new_text
  assert 'ctx0++;' in legacy_text


# -----------------------------------------------------------------------------
# 3. Dedup hash gate variant
# -----------------------------------------------------------------------------


def test_insert_into_dedup_hash_byte_equivalent():
  '''Dedup gate: emits
  `{ bool _p = dedup_table.try_insert(thread_id, v0, v1);
     if (_p) { atomicAdd...; out_data_0[...] = vN; ... } }`.'''
  ins = _insert_into(2)

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([ins], _new_ctx(dedup_hash=True)))
  new_text = _render(_lower_inner_chain([ins], _new_ctx(dedup_hash=True)))

  assert legacy_text == new_text
  assert 'dedup_table.try_insert(thread_id, v0, v1)' in legacy_text
  assert 'atomicAdd(atomic_write_pos, 1u);' in legacy_text


# -----------------------------------------------------------------------------
# 4. Tiled-Cartesian ballot variant (TiledBallotBlock)
# -----------------------------------------------------------------------------


def test_insert_into_tiled_ballot_byte_equivalent():
  '''When `tiled_cartesian_valid_var` is set, the trailing
  InsertInto run is rendered as a single `TiledBallotBlock`. Both
  paths produce the same IIR block.'''
  ins = _insert_into(2)

  with _force_legacy_branch():
    legacy_op = _lower_inner_chain([ins], _new_ctx(tiled_cartesian_valid_var='tc_valid'))
    legacy_text = _render(legacy_op)
  new_op = _lower_inner_chain([ins], _new_ctx(tiled_cartesian_valid_var='tc_valid'))
  new_text = _render(new_op)

  assert legacy_text == new_text
  # Both legacy and new must produce the TiledBallotBlock form.
  assert isinstance(legacy_op, TiledBallotBlock)
  assert isinstance(new_op, TiledBallotBlock)


def test_insert_into_multi_head_tiled_ballot_byte_equivalent():
  '''Two trailing InsertIntos under tiled-Cartesian: both contribute
  entries to a single `TiledBallotBlock`. Both paths must agree.'''
  ins1 = _insert_into(2, rel_name='Dst1')
  ins2 = _insert_into(2, rel_name='Dst2', vars_=['v0', 'v1'])

  with _force_legacy_branch():
    legacy_text = _render(
      _lower_inner_chain([ins1, ins2], _new_ctx(tiled_cartesian_valid_var='tc_valid'))
    )
  new_text = _render(
    _lower_inner_chain([ins1, ins2], _new_ctx(tiled_cartesian_valid_var='tc_valid'))
  )

  assert legacy_text == new_text


# -----------------------------------------------------------------------------
# 5. WS Cartesian batched-valid variant (emit_warp_coalesced)
# -----------------------------------------------------------------------------


def test_insert_into_ws_cart_batched_byte_equivalent():
  '''When `ws_cartesian_valid_var` is set, materialize phase emits
  `ctx0.emit_warp_coalesced(tile, vv, v0, v1);` — cooperative warp
  write, no lane-0 guard.'''
  ins = _insert_into(2)

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([ins], _new_ctx(ws_cartesian_valid_var='vv')))
  new_text = _render(_lower_inner_chain([ins], _new_ctx(ws_cartesian_valid_var='vv')))

  assert legacy_text == new_text
  assert 'ctx0.emit_warp_coalesced(tile, vv, v0, v1);' in legacy_text


# -----------------------------------------------------------------------------
# 6. Multi-head trailing-InsertInto run (concat into single Block)
# -----------------------------------------------------------------------------


def test_insert_into_multi_head_run_byte_equivalent():
  '''Two trailing InsertIntos in sequence (multi-head rule): each
  contributes its own stmts to a single concatenated `Block`. Both
  paths must produce identical text.'''
  ins1 = _insert_into(2, rel_name='Dst1', vars_=['v0', 'v1'])
  ins2 = _insert_into(2, rel_name='Dst2', vars_=['v0', 'v1'])

  with _force_legacy_branch():
    legacy_text = _render(_lower_inner_chain([ins1, ins2], _new_ctx()))
  new_text = _render(_lower_inner_chain([ins1, ins2], _new_ctx()))

  assert legacy_text == new_text


def test_insert_into_multi_head_count_skip_byte_equivalent():
  '''Multi-head count phase with a secondary output flagged
  `__skip_counting__`: the secondary emits only a comment (or
  nothing) and the primary contributes the count emit. Both paths
  must agree.'''
  ins1 = _insert_into(2, rel_name='Dst1', vars_=['v0', 'v1'])
  ins2 = _insert_into(2, rel_name='Dst2', vars_=['v0', 'v1'])
  overrides = {'Dst2': '__skip_counting__'}

  with _force_legacy_branch():
    legacy_text = _render(
      _lower_inner_chain([ins1, ins2], _new_ctx(is_counting=True, output_var_overrides=overrides))
    )
  new_text = _render(
    _lower_inner_chain([ins1, ins2], _new_ctx(is_counting=True, output_var_overrides=overrides))
  )

  assert legacy_text == new_text
  assert 'ctx0.emit_direct();' in legacy_text


# -----------------------------------------------------------------------------
# 7. Direct lowering call returns the expected IIR shape
# -----------------------------------------------------------------------------


def test_lower_mir_insert_into_in_chain_emits_block():
  '''Pin the IIR tree shape directly (independent of byte text):
  plain InsertInto -> `Block(<stmts...>)`.'''
  ins = _insert_into(2)

  out = lower_mir_insert_into_in_chain(ins, [], _new_ctx())
  assert isinstance(out, IirBlock)
  assert len(out.stmts) > 0


def test_lower_mir_insert_into_in_chain_emits_tiled_ballot_when_set():
  '''Pin the IIR tree shape under tiled-Cartesian:
  `TiledBallotBlock` instead of a plain `Block`.'''
  ins = _insert_into(2)

  out = lower_mir_insert_into_in_chain(ins, [], _new_ctx(tiled_cartesian_valid_var='tc'))
  assert isinstance(out, TiledBallotBlock)


def test_lower_mir_insert_into_in_chain_rejects_non_insert_tail():
  '''A non-InsertInto in the trailing slice is a structural error
  — the legacy `_lower_inner_chain` raises the same ValueError.'''
  ins = _insert_into(2)
  filt = mir.Filter(vars=['v0'], code='return v0 > 0;')

  with pytest.raises(ValueError, match=r'pure InsertInto tail'):
    lower_mir_insert_into_in_chain(ins, [filt], _new_ctx())


# -----------------------------------------------------------------------------
# 8. Registry contract — stub asserts on direct call
# -----------------------------------------------------------------------------


def test_lower_mir_insert_into_registry_stub_asserts():
  '''The `@lowering(target=iir.cf, source=mir.InsertInto)` registry
  entry is a stub that asserts on direct invocation — dispatch is
  expected to flow through `_lower_inner_chain` -> the chain-aware
  variant. Mirrors the C5 `lower_tiled_cartesian` split.
  '''
  ins = _insert_into(2)
  ctx = _new_ctx()
  with pytest.raises(AssertionError, match=r'lower_mir_insert_into_in_chain'):
    lower_mir_insert_into(ins, ctx)


def test_lower_mir_insert_into_is_registered_on_sorted_array_dialect():
  '''The `InsertInto` `@lowering` is registered on the
  `relation.sorted_array` dialect. Pins dialect ownership per
  `docs/phase_b_lowering_dispatcher.md` §4 (one `@lowering` per
  MIR op, on the dialect that lowers it).
  '''
  from srdatalog.ir.dialects.relation.sorted_array import DIALECT as SA_DIALECT

  matched = [low for low in SA_DIALECT.lowerings if low.matches is mir.InsertInto]
  assert len(matched) == 1
  assert matched[0].consumes == ('mir',)
  assert 'iir.cf' in matched[0].produces


# -----------------------------------------------------------------------------
# 9. Coexistence with C2 DedupGate typed-pragma wrap
# -----------------------------------------------------------------------------


def test_dedup_gate_wrapped_insert_into_byte_equivalent_to_legacy_dedup():
  '''C2 coexistence: a `DedupGate(inner=InsertInto)` lowered via
  `lower_dedup_gate` (the typed-pragma path) produces IIR that
  renders byte-equivalent to a bare `InsertInto` lowered via the
  legacy `dedup_hash=True` path.

  The DedupGate wrap op is NOT in `USE_DECLARATIVE` (it lives
  outside the chain dispatcher; the `MirPragmaPass` rewrites the
  pipeline to replace `InsertInto` with `DedupGate(inner=...)`),
  but its lowering delegates to the same `_lower_insert_into`
  helper that B-InsertInto's chain entry calls. Both paths
  therefore funnel through the same dedup branch.
  '''
  ins = _insert_into(2)
  gate = mir.DedupGate(inner=ins)

  # Path A: bare InsertInto via the legacy `dedup_hash=True` ctx
  # field, dispatched via `_lower_inner_chain` (which now routes
  # through the new B-InsertInto chain entry).
  legacy_dedup_text = _render(_lower_inner_chain([ins], _new_ctx(dedup_hash=True)))

  # Path B: typed `DedupGate` wrap op lowered directly via
  # `lower_dedup_gate` (the C2 pragma path).
  dedup_gate_op = lower_dedup_gate(gate, _new_ctx())
  dedup_gate_text = _render(dedup_gate_op)

  assert legacy_dedup_text == dedup_gate_text


def test_dedup_gate_lowering_unaffected_by_use_declarative_toggle():
  '''Toggling `mir.InsertInto` in/out of `USE_DECLARATIVE` does NOT
  change DedupGate's lowering output — the wrap op's `@lowering`
  rule never goes through `_lower_inner_chain`, so the
  `USE_DECLARATIVE` ratchet doesn't affect it.
  '''
  ins = _insert_into(2)
  gate = mir.DedupGate(inner=ins)

  on_text = _render(lower_dedup_gate(gate, _new_ctx()))
  with _force_legacy_branch():
    off_text = _render(lower_dedup_gate(gate, _new_ctx()))

  assert on_text == off_text


# -----------------------------------------------------------------------------
# 10. Coexistence with C4 WSScope typed-pragma wrap
# -----------------------------------------------------------------------------


def test_ws_scope_wrapped_insert_into_byte_equivalent_to_legacy_ws():
  '''C4 coexistence: a `WSScope(inner=InsertInto)` lowered via
  `lower_ws_scope` (the typed-pragma path) produces IIR that
  renders byte-equivalent to a bare `InsertInto` lowered via the
  legacy `ws_enabled=True` path, in count phase.
  '''
  ins = _insert_into(2)
  scope = mir.WSScope(inner=ins)

  legacy_ws_text = _render(_lower_inner_chain([ins], _new_ctx(is_counting=True, ws_enabled=True)))
  ws_scope_op = lower_ws_scope(scope, _new_ctx(is_counting=True))
  ws_scope_text = _render(ws_scope_op)

  assert legacy_ws_text == ws_scope_text


def test_ws_scope_lowering_unaffected_by_use_declarative_toggle():
  '''Toggling `mir.InsertInto` in/out of `USE_DECLARATIVE` does NOT
  change WSScope's lowering output.'''
  ins = _insert_into(2)
  scope = mir.WSScope(inner=ins)

  on_text = _render(lower_ws_scope(scope, _new_ctx(is_counting=True)))
  with _force_legacy_branch():
    off_text = _render(lower_ws_scope(scope, _new_ctx(is_counting=True)))

  assert on_text == off_text


# -----------------------------------------------------------------------------
# 11. Coexistence with C6 FanOut typed-pragma wrap
# -----------------------------------------------------------------------------


def test_fan_out_wrapped_insert_into_byte_equivalent_to_bare():
  '''C6 coexistence: a `mir.FanOut(inner=InsertInto)` lowered via
  `lower_fan_out` (the typed-pragma path) produces IIR that
  renders byte-equivalent to a bare `InsertInto` lowered via the
  legacy chain (no ctx flag flip — FanOut's runner-side scheduling
  is what effects the work-stealing; the kernel-body IIR is
  identical to the non-fanout shape).
  '''
  ins = _insert_into(2)
  fan = mir.FanOut(inner=ins)

  legacy_bare_text = _render(_lower_inner_chain([ins], _new_ctx()))
  fan_out_op = lower_fan_out(fan, _new_ctx())
  fan_out_text = _render(fan_out_op)

  assert legacy_bare_text == fan_out_text


def test_fan_out_lowering_unaffected_by_use_declarative_toggle():
  '''Toggling `mir.InsertInto` in/out of `USE_DECLARATIVE` does NOT
  change FanOut's lowering output.'''
  ins = _insert_into(2)
  fan = mir.FanOut(inner=ins)

  on_text = _render(lower_fan_out(fan, _new_ctx()))
  with _force_legacy_branch():
    off_text = _render(lower_fan_out(fan, _new_ctx()))

  assert on_text == off_text


# -----------------------------------------------------------------------------
# 12. Smoke test through the full compile_pipeline surface
# -----------------------------------------------------------------------------


def test_insert_into_in_full_pipeline_byte_equivalent():
  '''Smoke test: a Scan + InsertInto pipeline rendered through
  `compile_pipeline` (the production surface that the byte-
  equivalence harnesses guard) must produce identical CUDA under
  both `USE_DECLARATIVE` states.

  This closes the loop between the direct-call tests above and the
  529-fixture harness — if a divergence ever creeps in, both this
  test and the harness will catch it.
  '''
  from srdatalog.compile import compile_pipeline

  scan = mir.Scan(
    vars=['x', 'y'],
    rel_name='Src',
    version=Version.FULL,
    index=[0, 1],
    handle_start=0,
  )
  ins = mir.InsertInto(rel_name='Dst', version=Version.NEW, vars=['x', 'y'], index=[0, 1])
  ep = mir.ExecutePipeline(
    pipeline=[scan, ins],
    source_specs=[scan],
    dest_specs=[ins],
    rule_name='InsertRule',
  )

  with _force_legacy_branch():
    legacy_out = compile_pipeline(ep)
  new_out = compile_pipeline(ep)

  assert legacy_out == new_out


def test_insert_into_full_pipeline_with_filter_and_constant_bind_byte_equivalent():
  '''Smoke test: Scan + Filter + ConstantBind + InsertInto — all
  four Wave 2A migrated ops in one pipeline. Verifies the
  USE_DECLARATIVE dispatch composes correctly across multiple
  migrated ops.'''
  from srdatalog.compile import compile_pipeline

  scan = mir.Scan(
    vars=['x', 'y'],
    rel_name='Src',
    version=Version.FULL,
    index=[0, 1],
    handle_start=0,
  )
  filt = mir.Filter(vars=['x', 'y'], code='return x < y;')
  cb = mir.ConstantBind(var_name='k', code='1', deps=[])
  ins = mir.InsertInto(rel_name='Dst', version=Version.NEW, vars=['x', 'y', 'k'], index=[0, 1, 2])
  ep = mir.ExecutePipeline(
    pipeline=[scan, filt, cb, ins],
    source_specs=[scan],
    dest_specs=[ins],
    rule_name='MixRule',
  )

  with _force_legacy_branch():
    legacy_out = compile_pipeline(ep)
  new_out = compile_pipeline(ep)

  assert legacy_out == new_out
  assert 'if (x < y)' in new_out
  assert 'auto k = 1;' in new_out


# -----------------------------------------------------------------------------
# 13. USE_DECLARATIVE invariants
# -----------------------------------------------------------------------------


def test_use_declarative_contains_filter_constant_bind_and_insert_into():
  '''Pin the ratchet: after this Wave 2A PR, `mir.Filter`,
  `mir.ConstantBind`, and `mir.InsertInto` must all appear in
  `USE_DECLARATIVE`. Future Wave 2A PRs add their MIR ops; the
  monotonic discipline test (when it lands) catches accidental
  removals.'''
  from srdatalog.ir.dialects.relation.sorted_array import USE_DECLARATIVE

  assert mir.Filter in USE_DECLARATIVE
  assert mir.ConstantBind in USE_DECLARATIVE
  assert mir.InsertInto in USE_DECLARATIVE


# -----------------------------------------------------------------------------
# 14. Direct-call sanity: legacy helper still produces stmts list
# -----------------------------------------------------------------------------


def test_legacy_lower_insert_into_helper_still_callable():
  '''The B-InsertInto migration explicitly does NOT touch the
  legacy `_lower_insert_into` helper — the C2/C4/C6 typed-pragma
  lowerings depend on it. Pin that it still returns a list of
  Ops.'''
  ins = _insert_into(2)
  stmts = _lower_insert_into(ins, _new_ctx())
  assert isinstance(stmts, list)
  assert len(stmts) > 0
