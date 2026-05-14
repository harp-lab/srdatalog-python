'''relation.sorted_array dialect.

Index-aware ops for sorted-array relation storage. Currently covers
the M1-M3 subset:

  M1: SaRoot, SaValid, SaDegree, SaGetVal, SaGetValAt.
  M3: SaHint, SaPrefCoop, SaIterators, SaChildRange.

Planned (M4+): SaPref (for nested CJ), SaExists (for negation),
SaValues, SaPrefLb (lower-bound prefix).

See docs/ir_lowering_semantics.md §10 for the lowering rules and
docs/stage2_emitter_audit.md §6 for the plugin-dispatched expression
shapes the target lowering produces.
'''

from __future__ import annotations

from srdatalog.ir.core import Dialect
from srdatalog.ir.dialects.relation.sorted_array.ops import (
  DedupTryInsert,
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
  SaValid,
)
from srdatalog.ir.dialects.relation.sorted_array.types import (
  SaHandle,
  SaView,
)

DIALECT = Dialect(
  name='relation.sorted_array',
  types=[SaHandle, SaView],
  ops=[
    DedupTryInsert,
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
    SaValid,
  ],
)

__all__ = [
  'DIALECT',
  'DedupTryInsert',
  'SaChildRange',
  'SaDegree',
  'SaGetVal',
  'SaGetValAt',
  'SaGetValAtPos',
  'SaHandle',
  'SaHint',
  'SaIterators',
  'SaPrefCoop',
  'SaPrefSeq',
  'SaRoot',
  'SaValid',
  'SaView',
]


# ---------------------------------------------------------------------------
# Pass registration (S3A.4)
# ---------------------------------------------------------------------------
#
# Wires the MIR→IIR entry point (`lower_scan_pipeline`) into the
# framework registry. Production code today still calls the lowering
# directly via `compile_kernel_body`; this registration makes it
# discoverable via PassDriver and pins its (consumes, produces) for
# dependency validation.
#
# The body of `lower_scan_pipeline` is unchanged — this is a thin
# adapter that takes a MIR ExecutePipeline and forwards its `pipeline`
# list to the existing function. Future stages may move callers onto
# the registry-driven dispatch path; for now both paths coexist.


def _register_passes() -> None:
  import srdatalog.ir.mir.types as mir
  from srdatalog.ir.core.passes import lowering, rewrite, verifier
  from srdatalog.ir.dialects.iir.cf import Bind, BracedBlock, If, VarRef
  from srdatalog.ir.dialects.iir.expr import MemberCall
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import lower_scan_pipeline

  @lowering(
    DIALECT,
    mir.ExecutePipeline,
    consumes=('mir',),
    produces=('iir.cf', 'relation.sorted_array', 'relation.d2l', 'parallel.data'),
  )
  def lower_execute_pipeline(ep, ctx):
    return lower_scan_pipeline(ep.pipeline, ctx)

  # COMPOUND op decomposition (S4.6b) — per docs/ir_dialect_contract.md
  # §1, COMPOUND ops have no direct renderer; they expand into LEAF
  # ops here, before the codegen tree-walk sees them.
  @rewrite(
    DIALECT,
    DedupTryInsert,
    consumes=('relation.sorted_array',),
    produces=('iir.cf', 'iir.expr'),
  )
  def _decompose_dedup_try_insert(op, _ctx):
    return BracedBlock(
      stmts=(
        Bind(
          name='_p',
          type_decl='bool',
          expr=MemberCall(
            obj=VarRef(name='dedup_table'),
            method='try_insert',
            args=(VarRef(name='thread_id'), *op.args),
          ),
        ),
        If(cond=VarRef(name='_p'), body=op.then_body),
      )
    )

  # Verifier scaffolding — per-op invariants (D9: SaHint inside
  # IterURV scope, etc.) land incrementally as we encode them.
  @verifier(DIALECT)
  def _verify(_prog):
    return []


_register_passes()
