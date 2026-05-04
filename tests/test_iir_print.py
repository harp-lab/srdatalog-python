'''S3A.1 — Print_i smoke tests.

For each IIR op class, construct an instance and assert:
  1. `print_iir(op)` returns a non-empty string.
  2. Output parens are balanced (well-formed s-expression).
  3. Output starts with `(<op-name-kebab-case>` after stripping indent.

Plus dispatcher tests:
  - Every IIR op class is covered by exactly one dialect's `OPS`.
  - Cross-dialect trees (iir.cf.Block containing sorted_array ops) render.
  - Unknown op raises TypeError with a helpful message.

Frozen-golden equality tests are deferred until the lowering pipeline
starts producing IIR snapshots (S3A.2 + later) — at that point each
real lowering output gets pinned to a `tests/fixtures/iir/*.sexpr`.
'''

from __future__ import annotations

import pytest

from srdatalog.ir.core import Op
from srdatalog.ir.dialects.iir.cf.ops import (
  AddCount,
  Bind,
  BlankLine,
  Block,
  Cartesian2DDecompose,
  CartesianFlatLoop,
  CartesianNDecompose,
  Comment,
  GridStrideLoop,
  If,
  IfContinueIfNot,
  IfReturnIfNot,
  IndentBlock,
  IntersectIter,
  LaneZeroGuard,
  OuterAnchor,
  ParallelFor,
  Phase,
  RawString,
  TiledBallotBlock,
  VarRef,
  WriteOutput,
)
from srdatalog.ir.dialects.parallel.data.block_group import BgRootCjMulti, BgSourceSpec
from srdatalog.ir.dialects.relation.d2l.ops import D2lSegmentLoop
from srdatalog.ir.dialects.relation.sorted_array.ops import (
  SaChildRange,
  SaDegree,
  SaGetVal,
  SaGetValAt,
  SaGetValAtPos,
  SaHint,
  SaIterators,
  SaPrefCoop,
  SaPrefSeq,
  SaRoot,
  SaTiledCartesian2D,
  SaValid,
)
from srdatalog.ir.print_iir import print_iir


def _check(s: str, *, starts_with: str) -> None:
  assert s, 'empty output'
  stripped = s.lstrip()
  assert stripped.startswith(f'({starts_with}'), (
    f"expected '({starts_with}' prefix, got: {stripped[:80]!r}"
  )
  open_count = s.count('(')
  close_count = s.count(')')
  assert open_count == close_count, (
    f'unbalanced parens: {open_count} "(" vs {close_count} ")" in:\n{s}'
  )


# A common dummy expr/body op used to fill required Op-typed fields.
_DUMMY = Comment(text='dummy')
_DUMMY_VARREF = VarRef(name='x')


# -----------------------------------------------------------------------------
# Per-op smoke (one parametrize case per op class)
# -----------------------------------------------------------------------------


_IIR_CF_CASES = [
  # name (id) , op instance , expected starts_with token
  ('Block-empty', Block(stmts=()), 'block'),
  ('Block-with-stmts', Block(stmts=(_DUMMY, _DUMMY)), 'block'),
  ('IndentBlock', IndentBlock(extra=2, stmts=(_DUMMY,)), 'indent-block'),
  ('IndentBlock-empty', IndentBlock(extra=1, stmts=()), 'indent-block'),
  ('BlankLine', BlankLine(), 'blank-line'),
  ('Bind', Bind(name='x', expr=_DUMMY_VARREF, type_decl='auto'), 'bind'),
  ('VarRef', VarRef(name='y'), 'var-ref'),
  ('IfReturnIfNot', IfReturnIfNot(cond=_DUMMY_VARREF), 'if-return-if-not'),
  ('IfContinueIfNot', IfContinueIfNot(cond=_DUMMY_VARREF), 'if-continue-if-not'),
  (
    'CartesianFlatLoop',
    CartesianFlatLoop(
      idx_var='i',
      bound_var='n',
      lane_var='lane',
      group_size_var='gs',
      body=_DUMMY,
    ),
    'cartesian-flat-loop',
  ),
  (
    'Cartesian2DDecompose',
    Cartesian2DDecompose(
      major_var='m',
      idx0_var='i0',
      idx1_var='i1',
      flat_idx_var='fi',
      deg0_var='d0',
      deg1_var='d1',
    ),
    'cartesian-2d-decompose',
  ),
  (
    'CartesianNDecompose',
    CartesianNDecompose(
      flat_idx_var='fi',
      idx_vars=('i0', 'i1', 'i2'),
      deg_vars=('d0', 'd1', 'd2'),
    ),
    'cartesian-n-decompose',
  ),
  (
    'IntersectIter',
    IntersectIter(
      intersect_var='ix',
      iter_var='it',
      iterator_exprs=(_DUMMY,),
      value_var='v',
      body=_DUMMY,
    ),
    'intersect-iter',
  ),
  ('If', If(cond=_DUMMY_VARREF, body=_DUMMY), 'if'),
  (
    'GridStrideLoop',
    GridStrideLoop(idx_name='gsi', bound=_DUMMY_VARREF, body=_DUMMY),
    'grid-stride-loop',
  ),
  ('ParallelFor', ParallelFor(strategy='warp_strided', body=_DUMMY), 'parallel-for'),
  ('Phase', Phase(mode='M', body=_DUMMY), 'phase'),
  ('LaneZeroGuard', LaneZeroGuard(body=_DUMMY), 'lane-zero-guard'),
  ('WriteOutput-empty', WriteOutput(output_var='out', values=()), 'write-output'),
  (
    'WriteOutput-values',
    WriteOutput(output_var='out', values=(_DUMMY_VARREF, _DUMMY_VARREF)),
    'write-output',
  ),
  ('AddCount', AddCount(output_var='out', delta=_DUMMY_VARREF), 'add-count'),
  ('Comment', Comment(text='hello world'), 'comment'),
  ('RawString', RawString(text='// raw\ntext "quotes"'), 'raw-string'),
  ('TiledBallotBlock-empty', TiledBallotBlock(valid_var='v', outputs=()), 'tiled-ballot-block'),
  (
    'TiledBallotBlock-with-outputs',
    TiledBallotBlock(
      valid_var='v',
      outputs=(
        (0, ('a', 'b'), '0|debug'),
        (1, ('c',), '1|other'),
      ),
    ),
    'tiled-ballot-block',
  ),
  ('OuterAnchor', OuterAnchor(body=_DUMMY), 'outer-anchor'),
]

_SORTED_ARRAY_CASES = [
  ('SaRoot', SaRoot(view_name='view_R_0'), 'sa-root'),
  ('SaValid', SaValid(handle_name='h_0'), 'sa-valid'),
  ('SaDegree', SaDegree(handle_name='h_0'), 'sa-degree'),
  (
    'SaGetVal',
    SaGetVal(view_name='view_R_0', col=1, idx_var_name='idx'),
    'sa-get-val',
  ),
  (
    'SaGetValAt',
    SaGetValAt(handle_name='h', view_name='view_R_0', idx_var_name='idx'),
    'sa-get-val-at',
  ),
  ('SaHint', SaHint(lo_var='lo', hi_var='hi', depth=0), 'sa-hint'),
  (
    'SaPrefCoop',
    SaPrefCoop(parent=SaRoot(view_name='view_R_0'), key_var='k', view_name='view_R_0'),
    'sa-pref-coop',
  ),
  (
    'SaPrefSeq',
    SaPrefSeq(parent=SaRoot(view_name='view_R_0'), key_var='k', view_name='view_R_0'),
    'sa-pref-seq',
  ),
  (
    'SaIterators',
    SaIterators(handle_name='h', view_name='view_R_0'),
    'sa-iterators',
  ),
  (
    'SaChildRange',
    SaChildRange(
      handle_name='h',
      pos_expr='pos',
      key_var='k',
      view_name='view_R_0',
    ),
    'sa-child-range',
  ),
  (
    'SaGetValAtPos',
    SaGetValAtPos(
      view_name='view_R_0',
      col=2,
      handle_name='h',
      idx_var_name='idx',
    ),
    'sa-get-val-at-pos',
  ),
  (
    'SaTiledCartesian2D',
    SaTiledCartesian2D(
      view_var0='v0',
      view_var1='v1',
      handle_var0='h0',
      handle_var1='h1',
      col0=0,
      col1=1,
      var_name0='x',
      var_name1='y',
      lane_var='lane',
      group_size_var='gs',
      total_var='tot',
      degree_var0='d0',
      degree_var1='d1',
      flat_idx_var='fi',
      t0_base='t0b',
      t1_base='t1b',
      t0_len='t0l',
      t1_len='t1l',
      tile_total='tt',
      batch_var='bv',
      valid_var='vv',
      fb_batch_var='fbb',
      major_var='mv',
      idx0_var='i0',
      idx1_var='i1',
      body=_DUMMY,
    ),
    'sa-tiled-cartesian-2d',
  ),
]

_D2L_CASES = [
  (
    'D2lSegmentLoop-rebind',
    D2lSegmentLoop(
      seg_var='_seg',
      view_var='view_R',
      base_slot=4,
      view_count=2,
      declare=False,
      local_view_var='',
      body=_DUMMY,
    ),
    'd2l-segment-loop',
  ),
  (
    'D2lSegmentLoop-declare',
    D2lSegmentLoop(
      seg_var='_seg',
      view_var='view_R',
      base_slot=4,
      view_count=2,
      declare=True,
      local_view_var='view_R_local',
      body=_DUMMY,
    ),
    'd2l-segment-loop',
  ),
]

_PARALLEL_DATA_CASES = [
  (
    'BgRootCjMulti',
    BgRootCjMulti(
      var_name='v',
      is_counting=False,
      key_idx_var='ki',
      root_val_var='rv',
      hint_lo='lo',
      hint_hi='hi',
      sources=(
        BgSourceSpec(
          rel_name='Edge',
          view_var='view_Edge_0',
          handle_var='h_Edge',
          view_count=1,
          base_slot=0,
          index_type='SRDatalog::GPU::DeviceSortedArrayIndex',
        ),
        BgSourceSpec(
          rel_name='Path',
          view_var='view_Path_0',
          handle_var='h_Path',
          view_count=2,
          base_slot=1,
          index_type='SRDatalog::GPU::Device2LevelIndex',
        ),
      ),
      body=_DUMMY,
    ),
    'bg-root-cj-multi',
  ),
]


_ALL_CASES = (
  _IIR_CF_CASES + _SORTED_ARRAY_CASES + _D2L_CASES + _PARALLEL_DATA_CASES
)


@pytest.mark.parametrize(
  'op,starts_with',
  [(op, sw) for (_id, op, sw) in _ALL_CASES],
  ids=[id_ for (id_, _op, _sw) in _ALL_CASES],
)
def test_iir_op_prints_well_formed(op: Op, starts_with: str) -> None:
  out = print_iir(op)
  _check(out, starts_with=starts_with)


# -----------------------------------------------------------------------------
# Dispatcher coverage
# -----------------------------------------------------------------------------


def test_dialect_OPS_are_disjoint() -> None:
  '''No op class should appear in two dialects' OPS — that would mean
  the dispatch order silently picks one (a P1 / extensibility hazard).'''
  from srdatalog.ir.dialects.iir.cf.print import OPS as cf_OPS
  from srdatalog.ir.dialects.parallel.data.print import OPS as pd_OPS
  from srdatalog.ir.dialects.relation.d2l.print import OPS as d2l_OPS
  from srdatalog.ir.dialects.relation.sorted_array.print import OPS as sa_OPS

  by_dialect = {
    'iir.cf': set(cf_OPS),
    'sorted_array': set(sa_OPS),
    'd2l': set(d2l_OPS),
    'parallel.data': set(pd_OPS),
  }
  names = list(by_dialect)
  for i, n1 in enumerate(names):
    for n2 in names[i + 1 :]:
      overlap = by_dialect[n1] & by_dialect[n2]
      assert not overlap, f'op classes in both {n1} and {n2}: {overlap}'


def test_dispatcher_covers_every_test_case() -> None:
  '''Every op tested above should resolve through the dispatcher.
  Primary value: catches a future op added to a dialect's `ops.py` but
  forgotten in `print.py`'s `OPS` tuple.'''
  for _id, op, _sw in _ALL_CASES:
    out = print_iir(op)
    assert out, f'{type(op).__name__} produced empty output'


def test_unknown_op_raises_typeerror() -> None:
  '''Sanity: a bogus op (not in any dialect's OPS) raises TypeError
  with a helpful message — the "did the dispatcher get out of sync" signal.'''

  class _NotARealOp:
    pass

  with pytest.raises(TypeError, match='unknown IIR op'):
    print_iir(_NotARealOp())  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Cross-dialect tree
# -----------------------------------------------------------------------------


def test_cross_dialect_tree_renders() -> None:
  '''A realistic tree mixes ops from multiple dialects: an iir.cf.Block
  wraps a d2l.D2lSegmentLoop wrapping an iir.cf.IntersectIter whose
  iterator_exprs are sorted_array.SaIterators. All four dialects in
  one tree.'''
  tree = Block(
    stmts=(
      D2lSegmentLoop(
        seg_var='_seg',
        view_var='view_R',
        base_slot=0,
        view_count=2,
        declare=False,
        local_view_var='',
        body=IntersectIter(
          intersect_var='ix',
          iter_var='it',
          iterator_exprs=(
            SaIterators(handle_name='h_a', view_name='view_a'),
            SaIterators(handle_name='h_b', view_name='view_b'),
          ),
          value_var='v',
          body=Comment(text='inner body'),
        ),
      ),
      BgRootCjMulti(
        var_name='vv',
        is_counting=True,
        key_idx_var='ki',
        root_val_var='rv',
        hint_lo='lo',
        hint_hi='hi',
        sources=(
          BgSourceSpec(
            rel_name='R',
            view_var='vR',
            handle_var='hR',
            view_count=1,
            base_slot=0,
            index_type='DSAI',
          ),
        ),
        body=Comment(text='bg body'),
      ),
    )
  )
  out = print_iir(tree)
  _check(out, starts_with='block')
  assert 'd2l-segment-loop' in out
  assert 'intersect-iter' in out
  assert 'sa-iterators' in out
  assert 'bg-root-cj-multi' in out
  assert 'bg-source-spec' in out
  assert 'comment' in out
