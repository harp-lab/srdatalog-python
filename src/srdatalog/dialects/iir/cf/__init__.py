'''iir_cf — control-flow IR ops.

Cross-cutting structural ops shared across data-structure dialects:
sequencing (Block), binding (Bind, VarRef), branching/early-return,
parallel scaffolding (ParallelFor, GridStrideLoop, Phase), and
output emission (WriteOutput, AddCount).

These don't belong to any single data-structure dialect — they're the
compositional glue that the IIR-sorted-array, IIR-LSM, etc., dialects
hang their operations from.

See docs/ir_lowering_semantics.md §8 for the node set rationale and
docs/stage2_emitter_audit.md for the emission patterns these lower to.
'''

from __future__ import annotations

from srdatalog.dialects.iir.cf.ops import (
  AddCount,
  Bind,
  BlankLine,
  Block,
  Cartesian2DDecompose,
  CartesianFlatLoop,
  Comment,
  GridStrideLoop,
  If,
  IfContinueIfNot,
  IfReturnIfNot,
  IndentBlock,
  IntersectIter,
  LaneZeroGuard,
  ParallelFor,
  Phase,
  RawString,
  VarRef,
  WriteOutput,
)
from srdatalog.ir_core import Dialect

DIALECT = Dialect(
  name='iir.cf',
  ops=[
    AddCount,
    BlankLine,
    Block,
    Bind,
    Cartesian2DDecompose,
    CartesianFlatLoop,
    Comment,
    GridStrideLoop,
    If,
    IfContinueIfNot,
    IfReturnIfNot,
    IndentBlock,
    IntersectIter,
    LaneZeroGuard,
    ParallelFor,
    Phase,
    RawString,
    VarRef,
    WriteOutput,
  ],
)

__all__ = [
  'DIALECT',
  'AddCount',
  'BlankLine',
  'Bind',
  'Block',
  'Cartesian2DDecompose',
  'CartesianFlatLoop',
  'Comment',
  'GridStrideLoop',
  'If',
  'IfContinueIfNot',
  'IfReturnIfNot',
  'IndentBlock',
  'IntersectIter',
  'LaneZeroGuard',
  'ParallelFor',
  'Phase',
  'RawString',
  'VarRef',
  'WriteOutput',
]
