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

from typing import Any

import srdatalog.ir.mir.types as _mir_for_use_declarative
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

# ---------------------------------------------------------------------------
# USE_DECLARATIVE — Phase B migration ratchet (Wave 2A)
# ---------------------------------------------------------------------------
#
# Per docs/phase_b_lowering_dispatcher.md §5: MIR op types listed here
# are dispatched via the new per-op `@lowering` rules (in
# `lowerings/lower_mir_<op>.py`) rather than the legacy imperative
# `if isinstance(head, mir.X):` branches in `_lower_inner_chain`. The
# set is monotonically growing during Phase B — Wave 2A PRs add one
# entry per migrated op type. Layer 3 cleanup deletes both the set
# and the legacy branches once every MIR op is migrated.
#
# DISCIPLINE: removing an entry from this set requires owner
# sign-off (see code_discipline.md D12). The
# `test_use_declarative_is_monotonic` discipline test guards this
# property at CI time once it lands.

USE_DECLARATIVE: frozenset[type] = frozenset(
  {
    _mir_for_use_declarative.Filter,
    _mir_for_use_declarative.ConstantBind,
    _mir_for_use_declarative.InsertInto,
    _mir_for_use_declarative.Scan,
    # B-CJ-single + B-CJ-multi (per docs/phase_b_lowering_dispatcher.md
    # §4 rows B-CJ-single + B-CJ-multi, orders 5 + 6): `mir.ColumnJoin`
    # entry covers BOTH shapes after B-CJ-multi. The chain dispatcher
    # in `lowerings/__init__.py:_lower_inner_chain` routes single-
    # source via `lower_mir_cj_single_in_chain` and multi-source via
    # `lower_mir_cj_multi_in_chain`; the root dispatcher in
    # `lower_scan_pipeline` routes multi-source via
    # `lower_mir_cj_multi_root` (root-position single-source is not
    # a supported pipeline shape today).
    _mir_for_use_declarative.ColumnJoin,
    _mir_for_use_declarative.Negation,
    _mir_for_use_declarative.Aggregate,
    # B-Cart (per docs/phase_b_lowering_dispatcher.md §4 row B-Cart,
    # difficulty `hard`, order 7): adds `mir.CartesianJoin` to the
    # ratchet. The new `@lowering`-dispatch path owns BOTH the root
    # position (dispatched from `lower_scan_pipeline`) and the mid-
    # chain position (dispatched from `_lower_inner_chain`); the
    # `_should_use_declarative` helper has no source-count / shape
    # guard for Cart (unlike B-CJ-single, which gates on source
    # count). The C5 `TiledCartesian` typed-pragma wrap op is a
    # DIFFERENT MIR op type (`mir.TiledCartesian`) and continues to
    # route through `lower_tiled_cartesian_in_chain` independently —
    # see `lowerings/lower_mir_cart.py` for the coexistence rationale.
    _mir_for_use_declarative.CartesianJoin,
  }
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
  'USE_DECLARATIVE',
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
  'register',
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

  # Wave 2A / B-ExecutePipeline (per docs/phase_b_lowering_dispatcher.md
  # §4 row B-ExecutePipeline, order 10, difficulty `medium`):
  # `mir.ExecutePipeline` is the TOP-level structural wrap of every
  # kernel body. Unlike the chain-aware ops (Filter / ConstantBind /
  # Scan / Negation / Aggregate) which need the trailing `tail` from
  # the chain dispatcher and split into a stub + an `_in_chain`
  # variant, EP is self-contained: its `.pipeline` field carries the
  # full op sequence. The canonical entry in `lower_mir_execute_pipeline.py`
  # is therefore directly invocable — no chain-aware split.
  #
  # The registered entry below is a 1-line delegate; the body lives in
  # the sibling module. NOT added to USE_DECLARATIVE (EP sits OUTSIDE
  # `_lower_inner_chain` / `lower_scan_pipeline` — that set governs
  # chain / root dispatch inside the kernel body, not the top-level
  # wrap that opens it).
  from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_execute_pipeline import (
    lower_mir_execute_pipeline,
  )

  @lowering(
    DIALECT,
    mir.ExecutePipeline,
    consumes=('mir',),
    produces=('iir.cf', 'relation.sorted_array', 'relation.d2l', 'parallel.data'),
  )
  def _lower_mir_execute_pipeline_registered(op: Any, ctx: Any) -> Any:
    return lower_mir_execute_pipeline(op, ctx)

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

  # Wave 2A / B-Filter (per docs/phase_b_lowering_dispatcher.md §4
  # row B-Filter): `mir.Filter` migrates to a standalone `@lowering`
  # registration in its own file under
  # `lowerings/lower_mir_filter.py`. The registered stub asserts on
  # direct invocation because the chain-aware variant
  # (`lower_mir_filter_in_chain`) needs the trailing `tail` from
  # `_lower_inner_chain` — same split rationale as
  # `lower_tiled_cartesian` above. The `USE_DECLARATIVE` ratchet
  # below routes chain dispatch through the new path while keeping
  # this registration as the dialect-ownership contract.
  from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_filter import (
    lower_mir_filter,
  )

  @lowering(
    DIALECT,
    mir.Filter,
    consumes=('mir',),
    produces=('iir.cf',),
  )
  def _lower_mir_filter_registered(op: Any, ctx: Any) -> Any:
    return lower_mir_filter(op, ctx)

  # Wave 2A / B-ConstantBind (per docs/phase_b_lowering_dispatcher.md
  # §4 row B-ConstantBind): identical shape to B-Filter above.
  from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_constant_bind import (
    lower_mir_constant_bind,
  )

  @lowering(
    DIALECT,
    mir.ConstantBind,
    consumes=('mir',),
    produces=('iir.cf',),
  )
  def _lower_mir_constant_bind_registered(op: Any, ctx: Any) -> Any:
    return lower_mir_constant_bind(op, ctx)

  # Wave 2A / B-InsertInto (per docs/phase_b_lowering_dispatcher.md
  # §4 row B-InsertInto): identical shape to B-Filter / B-ConstantBind
  # above. `mir.InsertInto` is the terminal op in every
  # `ExecutePipeline.pipeline`, so the chain-aware variant
  # (`lower_mir_insert_into_in_chain`) handles the trailing
  # multi-head InsertInto run as a single unit. The C2 / C4 / C6
  # typed-pragma wrap ops (`DedupGate`, `WSScope`, `mir.FanOut`)
  # are NOT in `USE_DECLARATIVE` and have their own `@lowering`
  # rules that delegate back into the legacy `_lower_insert_into`
  # helper — same helper that this migration's chain entry calls,
  # so byte-equivalence holds across all four entry points.
  from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_insert_into import (
    lower_mir_insert_into,
  )

  @lowering(
    DIALECT,
    mir.InsertInto,
    consumes=('mir',),
    produces=('iir.cf',),
  )
  def _lower_mir_insert_into_registered(op: Any, ctx: Any) -> Any:
    return lower_mir_insert_into(op, ctx)

  # Wave 2A / B-Scan (per docs/phase_b_lowering_dispatcher.md §4 row
  # B-Scan, difficulty `medium`): unlike Filter / ConstantBind which
  # live mid-chain in `_lower_inner_chain`, `mir.Scan` is a ROOT-
  # position op dispatched from `lower_scan_pipeline`. The split-with-
  # stub pattern is the same — the chain-aware variant
  # (`lower_mir_scan_in_chain`) needs the trailing pipeline `rest`
  # from `lower_scan_pipeline`, so the `@lowering`-registered stub
  # asserts on direct invocation. The `USE_DECLARATIVE` ratchet
  # routes root dispatch through the new path while keeping this
  # registration as the dialect-ownership contract.
  from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_scan import (
    lower_mir_scan,
  )

  @lowering(
    DIALECT,
    mir.Scan,
    consumes=('mir',),
    produces=('iir.cf',),
  )
  def _lower_mir_scan_registered(op: Any, ctx: Any) -> Any:
    return lower_mir_scan(op, ctx)

  # Wave 2A / B-CJ-single + B-CJ-multi (per docs/phase_b_lowering_dispatcher.md
  # §4 rows B-CJ-single + B-CJ-multi, orders 5 + 6): `mir.ColumnJoin`
  # gets ONE dialect-level `@lowering(target=iir.cf, source=mir.ColumnJoin)`
  # registration that covers both source-count shapes for discipline
  # ownership. The shape-aware dispatcher below picks the right per-op
  # stub by `len(op.sources)`:
  #
  #   - `len(sources) == 1` (B-CJ-single, PR #59): `lower_mir_cj_single`
  #   - `len(sources) >= 2` (B-CJ-multi, this PR): `lower_mir_cj_multi`
  #
  # Both stubs assert on direct registry invocation — the actual
  # dispatch flows through `_lower_inner_chain` (mid-chain) and
  # `lower_scan_pipeline` (root) in `lowerings/__init__.py`, which
  # call the chain-aware variants (`lower_mir_cj_single_in_chain`,
  # `lower_mir_cj_multi_in_chain`, `lower_mir_cj_multi_root`) with
  # the surrounding pipeline `tail` / `rest` in scope. The
  # `USE_DECLARATIVE` ratchet routes both shapes through the new
  # path while this registration pins the dialect-ownership
  # contract.
  from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_cj_multi import (
    lower_mir_cj_multi,
  )
  from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_cj_single import (
    lower_mir_cj_single,
  )

  @lowering(
    DIALECT,
    mir.ColumnJoin,
    consumes=('mir',),
    produces=('iir.cf',),
  )
  def _lower_mir_column_join_registered(op: Any, ctx: Any) -> Any:
    # Shape-aware dispatch by source count. Both stubs assert (the
    # real entries are the chain-aware variants in their respective
    # `lower_mir_cj_*.py` modules) — this dispatcher exists so the
    # single dialect-level `@lowering` registration produces a
    # consistent error message regardless of which shape was passed.
    if len(op.sources) == 1:
      return lower_mir_cj_single(op, ctx)
    return lower_mir_cj_multi(op, ctx)

  # Wave 2A / B-Negation (per docs/phase_b_lowering_dispatcher.md §4
  # row B-Negation, difficulty `hard`, order 9): `mir.Negation`
  # migrates to a standalone `@lowering` registration in its own
  # file under `lowerings/lower_mir_negation.py`. The registered stub
  # asserts on direct invocation because the chain-aware variant
  # (`lower_mir_negation_in_chain`) needs the trailing `tail` from
  # `_lower_inner_chain` plus the surrounding `ctx.neg_pre_narrow`
  # registration set up by `_lower_nested_cart` — same split
  # rationale as Filter / ConstantBind / Scan above. The chain
  # variant delegates straight back into the legacy `_lower_negation`
  # helper so byte-equivalence holds by construction (including the
  # N5.4 Nim-broken raise documented in `docs/milestones.md` F5).
  from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_negation import (
    lower_mir_negation,
  )

  @lowering(
    DIALECT,
    mir.Negation,
    consumes=('mir',),
    produces=('iir.cf',),
  )
  def _lower_mir_negation_registered(op: Any, ctx: Any) -> Any:
    return lower_mir_negation(op, ctx)

  # Wave 2A / B-Aggregate (per docs/phase_b_lowering_dispatcher.md
  # §4 row B-Aggregate, difficulty `hard`, order 8): identical
  # split-with-stub shape to B-Filter / B-ConstantBind /
  # B-InsertInto above. `mir.Aggregate` is a structurally-supported
  # MIR op type (see runtime/generalized_datalog/mir_def.h for the
  # C++ counterpart) but the Python lowering pipeline never
  # constructs one today — the Nim reference also drops AggClause
  # during HIR -> MIR lowering, so the legacy `_lower_inner_chain`
  # has no dedicated branch and Aggregate falls through to the
  # terminal `raise ValueError('unsupported inner op: ...')`. The
  # registration here pins dialect ownership for the Phase B
  # sign-off bullet "every concrete MIR op has an `@lowering`";
  # `lower_mir_aggregate_in_chain` raises an identical
  # `ValueError` so byte-equivalence holds against the legacy
  # fall-through. See `lowerings/lower_mir_aggregate.py` for the
  # full upgrade-path rationale and the `USE_DECLARATIVE` ratchet
  # below routes chain dispatch through the new path while keeping
  # this registration as the dialect-ownership contract.
  from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_aggregate import (
    lower_mir_aggregate,
  )

  @lowering(
    DIALECT,
    mir.Aggregate,
    consumes=('mir',),
    produces=('iir.cf',),
  )
  def _lower_mir_aggregate_registered(op: Any, ctx: Any) -> Any:
    return lower_mir_aggregate(op, ctx)

  # Wave 2A / B-Cart (per docs/phase_b_lowering_dispatcher.md §4 row
  # B-Cart, difficulty `hard`, order 7): `mir.CartesianJoin` migrates
  # to a standalone `@lowering` registration in its own file under
  # `lowerings/lower_mir_cart.py`. The registered stub asserts on
  # direct invocation because the position-aware variants
  # (`lower_mir_cart_root` from `lower_scan_pipeline`,
  # `lower_mir_cart_in_chain` from `_lower_inner_chain`) need the
  # trailing pipeline / chain — same split-with-stub rationale as
  # B-Scan / B-Negation. The `USE_DECLARATIVE` ratchet above routes
  # BOTH the root and mid-chain dispatch through the new path while
  # keeping this registration as the dialect-ownership contract.
  #
  # Coexistence with C5 (`mir.TiledCartesian` wrap op): the wrap op
  # is a DIFFERENT MIR op type and has its own `@lowering`
  # registration (above, `lower_mir_tiled_cartesian`). The C5
  # dispatch in `_lower_inner_chain` matches `mir.TiledCartesian`
  # by isinstance and never falls through to the bare
  # `mir.CartesianJoin` branch — both branches coexist independently.
  # See `lowerings/lower_mir_cart.py` module docstring for the full
  # coexistence rationale + the C5 dual-write transition handling.
  from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_cart import (
    lower_mir_cart,
  )

  @lowering(
    DIALECT,
    mir.CartesianJoin,
    consumes=('mir',),
    produces=('iir.cf',),
  )
  def _lower_mir_cart_registered(op: Any, ctx: Any) -> Any:
    return lower_mir_cart(op, ctx)

  # Verifier scaffolding — per-op invariants (D9: SaHint inside
  # IterURV scope, etc.) land incrementally as we encode them.
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
# `[project.entry-points."srdatalog.plugins"]`.
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
  '''Plugin entry point — register the `relation.sorted_array` dialect
  on the given Compiler.

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
#   Matches the entry-point name `sorted_array` declared in
#   `pyproject.toml`.
# `provides` — the dialect names this plugin contributes. Other
#   plugins may `requires=('relation.sorted_array',)` to load after.
# `requires` — dialects that must be loaded first. Empty for now: the
#   `iir.cf` lowerings emitted by `_register_passes` are looked up by
#   op-type during MIR→IIR; nothing here actually mutates `iir.cf`'s
#   registry, so there is no bootstrap-time ordering requirement.
#   (When `iir.cf` itself ships as a discoverable plugin, add it here.)
register.plugin_name = 'sorted_array'  # type: ignore[attr-defined]
register.provides = ('relation.sorted_array',)  # type: ignore[attr-defined]
register.requires = ()  # type: ignore[attr-defined]
