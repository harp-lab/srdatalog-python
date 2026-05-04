'''Mid-level IR (MIR): types, passes, emit.

Submodules:
  - ``types``  — the MIR ADT (Scan, ColumnJoin, CartesianJoin, etc.)
  - ``passes`` — MIR optimization passes (clause reordering, etc.)
  - ``emit``   — S-expression printer for golden-diff tests

A `DIALECT` catalog is also exposed (see `srdatalog.ir.core.dialect`)
so MIR participates in the framework registry alongside `iir.cf`,
`relation.sorted_array`, etc. The MIR types here are not yet
`Op`-subclassed or frozen — that's the next standardization step.
The catalog is purely metadata today.
'''

from __future__ import annotations

from srdatalog.ir.core import Dialect
from srdatalog.ir.mir.types import (
  Aggregate,
  BalancedScan,
  Block,
  CartesianJoin,
  CheckSize,
  ClearRelation,
  ColumnJoin,
  ColumnSource,
  ComputeDelta,
  ComputeDeltaIndex,
  ConstantBind,
  CreateFlatView,
  ExecutePipeline,
  Filter,
  FixpointPlan,
  GatherColumn,
  InjectCppHook,
  InnerPipeline,
  InsertInto,
  MergeIndex,
  MergeRelation,
  Negation,
  ParallelGroup,
  PositionedExtract,
  PostStratumReconstructInternCols,
  ProbeJoin,
  Program,
  RebuildIndex,
  RebuildIndexFromIndex,
  Scan,
)

DIALECT = Dialect(
  name='mir',
  ops=[
    # Leaf / pipeline ops
    Aggregate,
    CartesianJoin,
    ColumnJoin,
    ColumnSource,
    ConstantBind,
    CreateFlatView,
    Filter,
    GatherColumn,
    InnerPipeline,
    InsertInto,
    Negation,
    PositionedExtract,
    ProbeJoin,
    Scan,
    # Fixpoint maintenance
    CheckSize,
    ClearRelation,
    ComputeDelta,
    ComputeDeltaIndex,
    MergeIndex,
    MergeRelation,
    RebuildIndex,
    RebuildIndexFromIndex,
    # Structural
    BalancedScan,
    Block,
    ExecutePipeline,
    FixpointPlan,
    InjectCppHook,
    ParallelGroup,
    PostStratumReconstructInternCols,
    Program,
  ],
)
