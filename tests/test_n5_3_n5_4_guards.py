'''N5.3 + N5.4 — D2L edge cases.

  - **N5.4 (Scan over D2L FULL_VER)** — implemented: the dialect
    wraps the scan body in a `D2lSegmentLoop(declare=True)` so HEAD
    and FULL segments are visited in turn with the view variable
    shadowed per segment. Structural test below pins the emit shape.

  - **N5.4 (standard-path Negation over D2L FULL_VER)** — guarded:
    raises `NotImplementedError` with an N5.4 reference. The
    naive segment-loop wrap is unsafe for antijoin semantics
    (the body should fire only when the prefix is missing from
    BOTH segments, not from each segment independently). A correct
    implementation needs a cross-segment accumulator; deferred
    until a workload demands it.

  - **N5.3 (single-source nested ColumnJoin)** — guarded by
    `_supported_pipeline`: the predicate requires `len(sources) >=
    2` for nested CJs; single-source raises `ValueError("unsupported
    pipeline shape ...")`.

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
# N5.4 — root Scan over D2L FULL_VER emits a segment-loop wrap.
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


def test_n5_4_scan_d2l_full_emits_segment_loop():
  '''Scan over a D2L FULL_VER source emits the segment-loop wrap +
  per-segment view shadowing + `continue` (not `return`) on
  per-segment validity failure.'''
  ep = _scan_d2l_full_ep()
  out = compile_kernel_body(
    ep,
    is_counting=False,
    rel_index_types={'Src': 'Device2LevelIndex'},
  )
  # Segment loop opens with `for (int _seg_<n>` (counter-bumped name).
  assert re.search(r'for \(int _seg_\d+ = 0; _seg_\d+ < 2;', out)
  # Per-segment view rebinding via `auto view_<rel>_... = views[<base> + _seg_<n>]`.
  assert re.search(r'auto view_Src_0_FULL_VER = views\[\d+ \+ _seg_\d+\];', out)
  # Inside the segment loop, the validity check is `continue` (skip
  # to next segment), NOT `return`.
  assert 'continue' in out
  # The legacy `return` path used outside D2L should NOT appear.
  m_seg_return = re.search(r'_seg_\d+ < 2[\s\S]*?return\s*;', out)
  assert m_seg_return is None, f'Found unexpected `return` inside segment loop: {m_seg_return.group(0)!r}'


def test_n5_4_scan_dsai_does_not_emit_segment_loop():
  '''Sanity: a non-D2L scan compiles without the segment-loop wrap
  and uses `return` for the validity check (not `continue`).'''
  ep = _scan_d2l_full_ep()
  out = compile_kernel_body(
    ep,
    is_counting=False,
    rel_index_types={},  # default = DSAI single-view
  )
  assert '_seg_' not in out
  # DSAI scan emits `if (!root_handle_<n>.valid()) return;`
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
