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

from typing import Any

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
  'register',
]


# Verifier scaffolding — control-flow invariants (well-formed Block
# nesting, OuterAnchor inside D2lSegmentLoop scope, etc.) land
# incrementally as we encode them.
def _register_passes() -> None:
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


# ---------------------------------------------------------------------------
# Phase E plugin entry point
# ---------------------------------------------------------------------------
#
# `register(compiler)` is the callable invoked by F4's plugin discovery
# (`Compiler.with_default_plugins()`) AND by explicit user-driven
# `compiler.register_plugin(...)` calls. Wired in `pyproject.toml` under
# `[project.entry-points."srdatalog.plugins"]` as `iir_cf`.
#
# Coexistence with legacy direct-import: the module-level
# `_register_passes()` call above still runs on first import, so the
# dialect's `lowerings` / `verifier` are populated regardless of whether
# anyone goes through `register(compiler)`. Python's module cache makes
# `_register_passes()` itself naturally idempotent (decorators only
# fire on first import). What `register(compiler)` adds on top is the
# per-Compiler `register_dialect(DIALECT)` step that F4's plugin
# loader needs to attribute ownership of the dialect to this plugin
# and populate the running Compiler's registry.
#
# Re-running `register(compiler)` on the same Compiler is safe: F4's
# `register_plugin` (see `core/dialect.py`) short-circuits on
# already-loaded plugin names.


def register(compiler: Any) -> None:
  '''Plugin entry point — register the `iir.cf` dialect on the given
  Compiler.

  Lowerings / pragmas were wired by `_register_passes()` at module-
  import time (a one-shot side effect; Python's module cache makes it
  naturally idempotent). This callable only performs the per-Compiler
  step: `compiler.register_dialect(DIALECT)`.

  Idempotent: F4's `Compiler.register_plugin` short-circuits
  re-registration of the same plugin name.
  '''
  compiler.register_dialect(DIALECT)


# Plugin metadata read by F4's topo-sort + conflict detection
# (see `core/plugin.py` — `_plugin_attr` reads these).
#
# `plugin_name` — what F4 records as the loaded-plugin identifier.
#   Matches the entry-point name `iir_cf` declared in `pyproject.toml`.
# `provides` — the dialect name this plugin contributes. Other plugins
#   may eventually `requires=('iir.cf',)` to load after; see the note
#   in `relation/sorted_array/__init__.py` re: bootstrap ordering.
# `requires` — dialects that must be loaded first. Empty: `iir.cf` is
#   the structural-glue dialect — it depends on nothing.
register.plugin_name = 'iir_cf'  # type: ignore[attr-defined]
register.provides = ('iir.cf',)  # type: ignore[attr-defined]
register.requires = ()  # type: ignore[attr-defined]
