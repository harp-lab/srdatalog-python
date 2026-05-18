'''Byte-equivalence test for Wave 2A B-Scan migration.

Per `docs/phase_b_lowering_dispatcher.md` §4.2 (per-PR acceptance
gate): each per-MIR-op migration ships a
`test_lower_mir_<op>_byte_equivalent` test that runs the migrated
path on every relevant fixture and asserts byte equality with the
legacy `if isinstance(head, mir.X):` branch.

For Scan the relevant fixtures are:
  - the M1 plain Scan -> InsertInto pipeline (no middle ops),
  - the M2 Scan -> Filter -> InsertInto chain,
  - the M2 Scan -> ConstantBind -> InsertInto chain,
  - multi-var scans (binding 1, 2, 3 columns) — the per-column Bind
    statements + GridStrideLoop scaffold,
  - count phase (`is_counting=True`) where the var-elision shortcut
    drops unused binds,
  - debug mode toggling the leading Comment stmts,
  - chains driving the body through nested middle ops (Filter +
    ConstantBind composed) — exercises the counter save/restore
    around `_lower_inner_chain`.

The test compares two compilation paths:
  - LEGACY: `USE_DECLARATIVE` patched to NOT contain `mir.Scan`,
    so `lower_scan_pipeline` falls into the imperative
    `_lower_root_scan(head, rest, ctx)` branch.
  - NEW: `USE_DECLARATIVE` left alone (Scan IS in the set), so
    `lower_scan_pipeline` routes through
    `lower_mir_scan_in_chain`.

Byte-equivalence is asserted on the rendered IIR text (which the
load-bearing harnesses in `tests/test_runner_byte_equivalence.py`
+ `tests/test_byte_equivalence_jit.py` anchor against the legacy
CUDA emit). The 532-fixture harness re-running green under the new
path is the strongest signal; this file adds direct-call coverage
of the corner cases.

Root-position note (per `docs/phase_b_lowering_dispatcher.md` §4
guidance — "if a Scan never appears mid-chain (only as root)"):
the dispatch site is `lower_scan_pipeline` (NOT
`_lower_inner_chain`). The chain-aware variant
`lower_mir_scan_in_chain` takes `rest` = the trailing pipeline
ops (middle + InsertIntos) — same signature shape as
`_lower_root_scan`. The `_in_chain` suffix is kept for naming
consistency across all Wave 2A per-op files.
'''

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

import srdatalog.ir.mir.types as mir
from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit
from srdatalog.ir.dialects.iir.cf import Block as IirBlock
from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
  LoweringCtx,
  lower_scan_pipeline,
)
from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_scan import (
  lower_mir_scan,
  lower_mir_scan_in_chain,
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
  '''Temporarily strip `mir.Scan` from `USE_DECLARATIVE` so
  `lower_scan_pipeline` falls into the legacy imperative branch
  (`_lower_root_scan`).

  Save / restore as a context manager — discipline test
  `test_use_declarative_is_monotonic` (when it lands) ratchets the
  set at module import time, but this test mutates the dialect's
  re-bound name for the duration of one call only.
  '''
  import srdatalog.ir.dialects.relation.sorted_array as sa_dialect

  saved = sa_dialect.USE_DECLARATIVE
  sa_dialect.USE_DECLARATIVE = frozenset(saved - {mir.Scan})
  try:
    yield
  finally:
    sa_dialect.USE_DECLARATIVE = saved


def _scan(
  vars_: list[str] | None = None,
  rel_name: str = 'Src',
  handle_start: int = 0,
) -> mir.Scan:
  vars_ = vars_ or ['v0', 'v1']
  return mir.Scan(
    vars=vars_,
    rel_name=rel_name,
    version=Version.FULL,
    index=list(range(len(vars_))),
    handle_start=handle_start,
  )


def _insert_into(
  arity: int = 2,
  vars_: list[str] | None = None,
  rel_name: str = 'Dst',
) -> mir.InsertInto:
  vars_ = vars_ or [f'v{i}' for i in range(arity)]
  return mir.InsertInto(
    rel_name=rel_name,
    version=Version.NEW,
    vars=vars_,
    index=list(range(len(vars_))),
  )


def _new_ctx(
  view_var_names: dict[str, str] | None = None,
  **kwargs: Any,
) -> LoweringCtx:
  '''Fresh LoweringCtx — counters reset so both paths bump identically.

  Scan needs `view_var_names` populated for its handle_start; default
  wires handle 0 to `view_src` to match the simple Scan fixtures
  below.
  '''
  if view_var_names is None:
    view_var_names = {'0': 'view_src'}
  return LoweringCtx(view_var_names=view_var_names, output_var='ctx0', **kwargs)


# -----------------------------------------------------------------------------
# 1. M1 plain Scan -> InsertInto
# -----------------------------------------------------------------------------


def test_scan_m1_plain_byte_equivalent():
  '''Plain M1: Scan(2 vars) directly into InsertInto. Both paths
  must produce identical rendered text — the GridStrideLoop scaffold
  + two Bind stmts + the InsertInto write.'''
  scan = _scan(vars_=['v0', 'v1'])
  ins = _insert_into(2)

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([scan, ins], _new_ctx()))
  new_text = _render(lower_scan_pipeline([scan, ins], _new_ctx()))

  assert legacy_text == new_text
  # Sanity: the rendered text contains the root-scan scaffold
  # (HandleType bind + degree() + a per-column get_value).
  assert 'HandleType(' in legacy_text
  assert '.degree()' in legacy_text
  assert 'view_src.get_value(' in legacy_text


def test_scan_single_var_byte_equivalent():
  '''Single-var scan: just one Bind in the body. Counter trajectory
  matches because both paths delegate to the same `_lower_root_scan`.'''
  scan = _scan(vars_=['x'])
  ins = _insert_into(vars_=['x'])

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([scan, ins], _new_ctx()))
  new_text = _render(lower_scan_pipeline([scan, ins], _new_ctx()))

  assert legacy_text == new_text


def test_scan_three_var_byte_equivalent():
  '''Three-var scan: three Bind stmts, exercises the per-column loop
  in `_lower_root_scan`.'''
  scan = _scan(vars_=['a', 'b', 'c'])
  ins = _insert_into(vars_=['a', 'b', 'c'])

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([scan, ins], _new_ctx()))
  new_text = _render(lower_scan_pipeline([scan, ins], _new_ctx()))

  assert legacy_text == new_text


# -----------------------------------------------------------------------------
# 2. M2 Scan + middle ops
# -----------------------------------------------------------------------------


def test_scan_with_filter_byte_equivalent():
  '''M2 Scan -> Filter -> InsertInto: the Filter goes through
  `_lower_inner_chain` (which itself may dispatch through
  `USE_DECLARATIVE` for Filter). Both paths must agree on the full
  rendered output.'''
  scan = _scan(vars_=['v0', 'v1'])
  filt = mir.Filter(vars=['v0', 'v1'], code='return v0 < v1;')
  ins = _insert_into(2)

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([scan, filt, ins], _new_ctx()))
  new_text = _render(lower_scan_pipeline([scan, filt, ins], _new_ctx()))

  assert legacy_text == new_text
  assert 'if (v0 < v1)' in legacy_text


def test_scan_with_constant_bind_byte_equivalent():
  '''M2 Scan -> ConstantBind -> InsertInto: the ConstantBind goes
  through `_lower_inner_chain`. Both paths must agree.'''
  scan = _scan(vars_=['x'])
  cb = mir.ConstantBind(var_name='k', code='99', deps=[])
  ins = _insert_into(vars_=['x', 'k'])

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([scan, cb, ins], _new_ctx()))
  new_text = _render(lower_scan_pipeline([scan, cb, ins], _new_ctx()))

  assert legacy_text == new_text
  assert 'auto k = 99;' in legacy_text


def test_scan_with_filter_and_constant_bind_byte_equivalent():
  '''M2 Scan -> ConstantBind -> Filter -> InsertInto: exercises the
  composed middle-chain through `_lower_inner_chain`. Counter
  trajectory + body shape must match.'''
  scan = _scan(vars_=['v0', 'v1'])
  cb = mir.ConstantBind(var_name='lo', code='0', deps=[])
  filt = mir.Filter(vars=['v0', 'lo'], code='return v0 > lo;')
  ins = _insert_into(2)

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([scan, cb, filt, ins], _new_ctx()))
  new_text = _render(lower_scan_pipeline([scan, cb, filt, ins], _new_ctx()))

  assert legacy_text == new_text


# -----------------------------------------------------------------------------
# 3. Count phase + var-elision
# -----------------------------------------------------------------------------


def test_scan_count_phase_byte_equivalent():
  '''In count phase, vars not referenced by the body are elided from
  the Bind list (Nim's `varName notin body` substring check). Both
  paths must implement the same elision.'''
  scan = _scan(vars_=['v0', 'v1'])
  # body references only v0 — v1 should be elided in count phase.
  ins = _insert_into(vars_=['v0'])

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([scan, ins], _new_ctx(is_counting=True)))
  new_text = _render(lower_scan_pipeline([scan, ins], _new_ctx(is_counting=True)))

  assert legacy_text == new_text


# -----------------------------------------------------------------------------
# 4. Debug-mode Comments
# -----------------------------------------------------------------------------


def test_scan_debug_off_byte_equivalent():
  '''With `debug=False`, the leading `Root Scan: ...` and `MIR: ...`
  Comment stmts are dropped. Both paths must agree.'''
  scan = _scan(vars_=['v0', 'v1'])
  ins = _insert_into(2)

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([scan, ins], _new_ctx(debug=False)))
  new_text = _render(lower_scan_pipeline([scan, ins], _new_ctx(debug=False)))

  assert legacy_text == new_text
  assert 'Root Scan:' not in legacy_text


# -----------------------------------------------------------------------------
# 5. Direct lowering call returns the expected IIR shape
# -----------------------------------------------------------------------------


def test_lower_mir_scan_in_chain_returns_block():
  '''Pin the IIR tree shape directly (independent of byte text):
  root-scan emission is wrapped in a `Block` of scaffold stmts +
  ParallelFor.'''
  scan = _scan(vars_=['v0', 'v1'])
  ins = _insert_into(2)

  out = lower_mir_scan_in_chain(scan, [ins], _new_ctx())
  assert isinstance(out, IirBlock)
  # The Block contains the scaffold: at minimum the handle Bind, the
  # IfReturnIfNot validity check, the degree Bind, a BlankLine, and
  # the ParallelFor. Without debug Comments, that's >= 5 stmts.
  assert len(out.stmts) >= 5


def test_lower_mir_scan_in_chain_delegates_to_legacy():
  '''The chain-aware variant delegates to `_lower_root_scan`
  byte-for-byte. Verify by comparing the IIR text produced both
  ways from a freshly-reset ctx.'''
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
    _lower_root_scan,
  )

  scan = _scan(vars_=['v0', 'v1'])
  ins = _insert_into(2)

  via_chain = _render(lower_mir_scan_in_chain(scan, [ins], _new_ctx()))
  via_legacy = _render(_lower_root_scan(scan, [ins], _new_ctx()))

  assert via_chain == via_legacy


# -----------------------------------------------------------------------------
# 6. Registry contract — stub asserts on direct call
# -----------------------------------------------------------------------------


def test_lower_mir_scan_registry_stub_asserts():
  '''The `@lowering(target=iir.cf, source=mir.Scan)` registry entry
  is a stub that asserts on direct invocation — dispatch is
  expected to flow through `lower_scan_pipeline` -> the chain-aware
  variant. Mirrors the B-Filter / B-ConstantBind / C5
  `lower_tiled_cartesian` split.
  '''
  scan = _scan(vars_=['v0'])
  ctx = _new_ctx()
  with pytest.raises(AssertionError, match=r'lower_mir_scan_in_chain'):
    lower_mir_scan(scan, ctx)


def test_lower_mir_scan_is_registered_on_sorted_array_dialect():
  '''The `Scan` `@lowering` is registered on the
  `relation.sorted_array` dialect. Pins dialect ownership per
  `docs/phase_b_lowering_dispatcher.md` §4 (one `@lowering` per MIR
  op, on the dialect that lowers it).
  '''
  from srdatalog.ir.dialects.relation.sorted_array import DIALECT as SA_DIALECT

  matched = [low for low in SA_DIALECT.lowerings if low.matches is mir.Scan]
  assert len(matched) == 1
  assert matched[0].consumes == ('mir',)
  assert 'iir.cf' in matched[0].produces


# -----------------------------------------------------------------------------
# 7. Smoke test through the full compile_pipeline surface
# -----------------------------------------------------------------------------


def test_scan_in_full_pipeline_byte_equivalent():
  '''Smoke test: a Scan + InsertInto pipeline rendered through
  `compile_pipeline` (the production surface that the byte-
  equivalence harnesses guard) must produce identical CUDA under
  both `USE_DECLARATIVE` states.

  This closes the loop between the direct-call tests above and the
  532-fixture harness — if a divergence ever creeps in, both this
  test and the harness will catch it.
  '''
  from srdatalog.ir.codegen.cuda.api import compile_pipeline

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
    rule_name='ScanRule',
  )

  with _force_legacy_branch():
    legacy_out = compile_pipeline(ep)
  new_out = compile_pipeline(ep)

  assert legacy_out == new_out


def test_scan_with_filter_in_full_pipeline_byte_equivalent():
  '''Smoke test: a Scan + Filter + InsertInto pipeline through the
  full `compile_pipeline` surface. Exercises both the Scan
  migration and the Filter migration interacting.
  '''
  from srdatalog.ir.codegen.cuda.api import compile_pipeline

  scan = mir.Scan(
    vars=['x', 'y'],
    rel_name='Src',
    version=Version.FULL,
    index=[0, 1],
    handle_start=0,
  )
  filt = mir.Filter(vars=['x', 'y'], code='return x < y;')
  ins = mir.InsertInto(rel_name='Dst', version=Version.NEW, vars=['x', 'y'], index=[0, 1])
  ep = mir.ExecutePipeline(
    pipeline=[scan, filt, ins],
    source_specs=[scan],
    dest_specs=[ins],
    rule_name='ScanFiltRule',
  )

  with _force_legacy_branch():
    legacy_out = compile_pipeline(ep)
  new_out = compile_pipeline(ep)

  assert legacy_out == new_out
  assert 'if (x < y)' in new_out


# -----------------------------------------------------------------------------
# 8. USE_DECLARATIVE invariants
# -----------------------------------------------------------------------------


def test_use_declarative_contains_scan():
  '''Pin the ratchet: after this Wave 2A PR, `mir.Scan` must appear
  in `USE_DECLARATIVE` alongside `mir.Filter` and `mir.ConstantBind`.
  Future Wave 2A PRs add their MIR ops; the monotonic discipline
  test (when it lands) catches accidental removals.'''
  from srdatalog.ir.dialects.relation.sorted_array import USE_DECLARATIVE

  assert mir.Scan in USE_DECLARATIVE
  assert mir.Filter in USE_DECLARATIVE
  assert mir.ConstantBind in USE_DECLARATIVE
