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
  from srdatalog.ir.core.passes import lowering, verifier
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import lower_scan_pipeline

  @lowering(
    DIALECT,
    mir.ExecutePipeline,
    consumes=('mir',),
    produces=('iir.cf', 'relation.sorted_array', 'relation.d2l', 'parallel.data'),
  )
  def lower_execute_pipeline(ep, ctx):
    return lower_scan_pipeline(ep.pipeline, ctx)

  # C2 (per docs/phase_c_pragma_materialization.md §2.1): the
  # `DedupHash` pragma's MIR wrap op `DedupGate` lowers via the rule
  # registered here. Body lives in the pragma module so the wrap op,
  # the @pragma_handler, and the @lowering for the wrap op are all
  # co-located. Importing the module also runs the @pragma_handler
  # registration as a side effect.
  from typing import Any

  from srdatalog.ir.dialects.relation.sorted_array.pragmas.dedup_hash import (
    lower_dedup_gate,
  )

  @lowering(
    DIALECT,
    mir.DedupGate,
    consumes=('mir',),
    produces=('iir.cf', 'relation.sorted_array'),
  )
  def lower_mir_dedup_gate(op: Any, ctx: Any) -> Any:
    return lower_dedup_gate(op, ctx)

  # C6 (per docs/phase_c_pragma_materialization.md §4.3): the
  # `FanOut` pragma's MIR wrap op `mir.FanOut` lowers via the rule
  # registered here. Same co-location pattern as `DedupGate` above.
  # Importing the module also runs the @pragma_handler registration
  # as a side effect (the side-effect is what gates DSL acceptance
  # of `with_pragma(FanOut())`).
  from srdatalog.ir.dialects.relation.sorted_array.pragmas.fanout import (
    lower_fan_out,
  )

  @lowering(
    DIALECT,
    mir.FanOut,
    consumes=('mir',),
    produces=('iir.cf', 'relation.sorted_array'),
  )
  def lower_mir_fan_out(op: Any, ctx: Any) -> Any:
    return lower_fan_out(op, ctx)

  # C5 (per docs/phase_c_pragma_materialization.md §4.3): the
  # `TiledCartesian` pragma's MIR wrap op `mir.TiledCartesian` lowers
  # via the rule registered here. The dispatch entry point asserts
  # because the actual emission needs the trailing chain from
  # `_lower_inner_chain` — see the pragma module's
  # `lower_tiled_cartesian_in_chain` docstring for the split rationale.
  # This rule pins the dialect's ownership of `TiledCartesian` for
  # the registry-completeness discipline test.
  from srdatalog.ir.dialects.relation.sorted_array.pragmas.tiled_cartesian import (
    lower_tiled_cartesian,
  )

  @lowering(
    DIALECT,
    mir.TiledCartesian,
    consumes=('mir',),
    produces=('iir.cf', 'relation.sorted_array'),
  )
  def lower_mir_tiled_cartesian(op: Any, ctx: Any) -> Any:
    return lower_tiled_cartesian(op, ctx)

  # Verifier scaffolding — per-op invariants (D9: SaHint inside
  # IterURV scope, etc.) land incrementally as we encode them.
  @verifier(DIALECT)
  def _verify(_prog):
    return []


_register_passes()
