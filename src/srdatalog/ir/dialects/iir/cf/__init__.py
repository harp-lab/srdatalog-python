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

from srdatalog.ir.core import Dialect
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

DIALECT = Dialect(
  name='iir.cf',
  ops=[
    AddCount,
    BlankLine,
    Block,
    Bind,
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
  'CartesianNDecompose',
  'Comment',
  'GridStrideLoop',
  'If',
  'IfContinueIfNot',
  'IfReturnIfNot',
  'IndentBlock',
  'IntersectIter',
  'LaneZeroGuard',
  'OuterAnchor',
  'ParallelFor',
  'Phase',
  'RawString',
  'TiledBallotBlock',
  'VarRef',
  'WriteOutput',
]


# Verifier scaffolding — control-flow invariants (well-formed Block
# nesting, OuterAnchor inside D2lSegmentLoop scope, etc.) land
# incrementally as we encode them.
def _register_passes() -> None:
  from typing import Any

  import srdatalog.ir.mir.types as mir
  from srdatalog.ir.core.passes import lowering, verifier

  # C6 (per docs/phase_c_pragma_materialization.md §4.3): the
  # `Count` pragma's MIR wrap op `CountPhase` lowers via the rule
  # registered here. Body lives in the pragma module so the wrap op,
  # the @pragma_handler, and the @lowering for the wrap op are all
  # co-located. Importing the module also runs the @pragma_handler
  # registration as a side effect (the side-effect is what gates DSL
  # acceptance of `with_pragma(Count())`).
  from srdatalog.ir.dialects.iir.cf.pragmas.count import (
    lower_count_phase,
  )

  @lowering(
    DIALECT,
    mir.CountPhase,
    consumes=('mir',),
    produces=('iir.cf',),
  )
  def lower_mir_count_phase(op: Any, ctx: Any) -> Any:
    return lower_count_phase(op, ctx)

  @verifier(DIALECT)
  def _verify(_prog):
    return []


_register_passes()
