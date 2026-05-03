'''N5.3 + N5.4 — D2L edge cases.

  - **N5.4 (Scan over D2L FULL_VER)** — matches Nim: emits a
    single-view scan (NO segment-loop wrap). Nim's `jitRootScan`
    (codegen/target_jit/jit_root.nim:61-126) does the same.
    Both are technically incorrect on FULL_VER's HEAD/FULL split
    but byte-equivalent. See docs/milestones.md "Nim-reference
    audit" for the gap.

  - **N5.4 (standard-path Negation over D2L FULL_VER)** — guarded:
    raises `NotImplementedError`. Nim ALSO has no segment-loop wrap
    here (`jitNegation` at jit_scan_negation.nim:142-187). Both
    ends broken; defer.

  - **N5.3 (single-source nested ColumnJoin)** — guarded by
    `_supported_pipeline`: the predicate requires `len(sources) >=
    2` for nested CJs; single-source raises `ValueError("unsupported
    pipeline shape ...")`. Nim emits seg-loop wrap for this shape
    (jit_instructions.nim:42-143); real gap, defer until workload.

When N5.3 / N5.4 (Negation) land, replace the corresponding guards
with byte-equivalence tests against checked-in goldens.
'''

from __future__ import annotations

import re

import pytest

import srdatalog.ir.mir.types as m
from srdatalog.compile import compile_kernel_body, compile_pipeline
from srdatalog.ir.hir.types import Version

# -----------------------------------------------------------------------------
# N5.4 — root Scan over D2L FULL_VER emits a single-view scan (no segment-loop
# wrap), matching Nim's jitRootScan. Both ends share the FULL_VER HEAD/FULL
# semantic gap; preserving byte-equivalence is the priority.
# -----------------------------------------------------------------------------


def _scan_d2l_full_ep() -> m.ExecutePipeline:
  scan = m.Scan(
    vars=['x'], rel_name='Src', version=Version.FULL,
    index=[0], handle_start=0,
  )
  insert = m.InsertInto(
    rel_name='Dst', version=Version.NEW, vars=['x'], index=[0],
  )
  return m.ExecutePipeline(
    pipeline=[scan, insert],
    source_specs=[scan],
    dest_specs=[insert],
    rule_name='ScanD2L',
  )


def test_n5_4_scan_d2l_full_no_segment_loop_matches_nim():
  '''Scan over a D2L FULL_VER source emits the single-view shape, NOT
  a segment-loop wrap — matching Nim's `jitRootScan`. Pin against
  silent re-introduction of the over-implementation that previously
  diverged from Nim.'''
  ep = _scan_d2l_full_ep()
  out = compile_kernel_body(
    ep,
    is_counting=False,
    rel_index_types={'Src': 'Device2LevelIndex'},
  )
  # No segment loop — Nim doesn't emit one for root Scan over D2L.
  assert '_seg_' not in out, f'Unexpected segment loop in root Scan emit:\n{out}'
  # Validity check is `return` (no per-segment continue path).
  assert re.search(r'if \(!root_handle_\d+\.valid\(\)\) return;', out)


def test_n5_4_scan_dsai_emits_single_view():
  '''Non-D2L scan also emits the single-view shape (control case —
  same emit shape as the D2L FULL_VER case above).'''
  ep = _scan_d2l_full_ep()
  out = compile_kernel_body(
    ep,
    is_counting=False,
    rel_index_types={},  # default = DSAI single-view
  )
  assert '_seg_' not in out
  assert re.search(r'if \(!root_handle_\d+\.valid\(\)\) return;', out)


# -----------------------------------------------------------------------------
# N5.4 — standard-path Negation over D2L FULL_VER raises with N5.4 reference.
# -----------------------------------------------------------------------------


def test_n5_4_negation_d2l_full_raises():
  '''Synthetic: root multi-source CJ, then Negation over a D2L
  FULL_VER source (standard path — no preceding Cart). The naive
  segment-loop wrap is unsafe for antijoin (body should fire only
  when prefix is missing from BOTH segments). Deferred.'''
  src_a = m.ColumnSource(
    rel_name='A', version=Version.FULL, index=[0, 1],
    prefix_vars=[], handle_start=0,
  )
  src_b = m.ColumnSource(
    rel_name='B', version=Version.FULL, index=[0, 1],
    prefix_vars=[], handle_start=1,
  )
  root_cj = m.ColumnJoin(
    var_name='y', sources=[src_a, src_b], handle_start=0,
  )
  neg = m.Negation(
    rel_name='Neg', version=Version.FULL, prefix_vars=['y'],
    index=[0], handle_start=2,
  )
  insert = m.InsertInto(
    rel_name='Dst', version=Version.NEW, vars=['y'], index=[0],
  )
  ep = m.ExecutePipeline(
    pipeline=[root_cj, neg, insert],
    source_specs=[src_a, src_b, neg],
    dest_specs=[insert],
    rule_name='NegD2L',
  )
  with pytest.raises(NotImplementedError) as excinfo:
    compile_kernel_body(
      ep,
      is_counting=False,
      rel_index_types={'Neg': 'Device2LevelIndex'},
    )
  msg = str(excinfo.value)
  assert 'N5.4' in msg
  assert 'Neg' in msg


# -----------------------------------------------------------------------------
# N5.3 — single-source nested CJ rejected by `_supported_pipeline`.
# -----------------------------------------------------------------------------


def test_n5_3_single_source_nested_cj_raises():
  '''A pipeline whose nested CJ has `len(sources) == 1` is rejected
  by `_supported_pipeline` with a clear ValueError naming the shape.

  When N5.3 lands, this test should flip — the pipeline will compile
  and a byte-equivalence test will replace this guard.
  '''
  src_a = m.ColumnSource(
    rel_name='A', version=Version.FULL, index=[0, 1],
    prefix_vars=[], handle_start=0,
  )
  src_b = m.ColumnSource(
    rel_name='B', version=Version.FULL, index=[0, 1],
    prefix_vars=[], handle_start=1,
  )
  root_cj = m.ColumnJoin(
    var_name='y', sources=[src_a, src_b], handle_start=0,
  )
  src_c = m.ColumnSource(
    rel_name='C', version=Version.FULL, index=[0, 1],
    prefix_vars=['y'], handle_start=2,
  )
  nested_cj = m.ColumnJoin(
    var_name='z', sources=[src_c], handle_start=2,
  )
  insert = m.InsertInto(
    rel_name='Dst', version=Version.NEW, vars=['y', 'z'], index=[0, 1],
  )
  ep = m.ExecutePipeline(
    pipeline=[root_cj, nested_cj, insert],
    source_specs=[src_a, src_b, src_c],
    dest_specs=[insert],
    rule_name='CjSingle',
  )
  with pytest.raises(ValueError, match=r'unsupported pipeline shape'):
    compile_pipeline(ep)
