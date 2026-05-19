'''parallel.atomic_ws — atomic work-stealing parallelism strategy.

Per `docs/phase_c_pragma_materialization.md` §4.2, this sub-dialect
houses the work-stealing-specific ops + lowerings + render. It is
introduced by Phase C4 to host the `WorkStealing` typed pragma's
materialization surface (see `pragmas/work_stealing.py`).

The atomic-work-stealing strategy targets unbalanced joins where the
default warp-strided dispatch leaves most warps idle waiting on a few
hot root keys. The runner queues work items in a WCOJTask board;
warps that drain their initial slice steal from neighbours via
atomics on the board's head/tail counters. Kernel functors emit a
per-thread `local_count` increment (count phase) and warp-coalesced
writes (materialize phase) instead of the standard
`<out>.emit_direct()` shape, so the runner's aggregation step can
fold per-thread counts and re-batch writes.

C4 scope (per spec §5, PR row C4): only the kernel-functor-level
WS emit variants migrate to typed pragma + wrap op. The runner-side
scaffolding (WCOJTask queue construction, steal-loop emission) was
historically gated by the deprecated `ExecutePipeline.work_stealing:
bool` field; A3-2 retired that field, leaving the C4 lowering as
the sole driver for the kernel-functor-level emit. Runner-side
WS scaffolding integration is a separate follow-up that consumes
the typed `WorkStealing` pragma (or its post-pass `WSScope` wrap
op) directly.

Ops currently registered: none — the only WS-specific op today is
the MIR wrap op `mir.WSScope`, which lives next to `DedupGate` in
`mir/types.py` (matching the C2 placement convention). The dialect's
`lowerings` list carries the `@lowering(target=iir.cf, source=
mir.WSScope)` rule registered in `_register_passes()` below.
'''

from __future__ import annotations

from typing import Any

from srdatalog.ir.core import Dialect

DIALECT = Dialect(
  name='parallel.atomic_ws',
  ops=[],
)


def _register_passes() -> None:
  '''Register the `WSScope` lowering + verifier on this dialect.

  Mirrors the `sorted_array._register_passes` pattern (per C2). The
  `pragmas/work_stealing.py` module owns the lowering body and the
  `@pragma_handler(WorkStealing, on=ExecutePipeline)` materialization
  callback; importing it here runs both registrations as side
  effects.
  '''
  import srdatalog.ir.mir.types as mir
  from srdatalog.ir.core.passes import lowering, verifier
  from srdatalog.ir.dialects.parallel.atomic_ws.pragmas.work_stealing import (
    lower_ws_scope,
  )

  @lowering(
    DIALECT,
    mir.WSScope,
    consumes=('mir',),
    produces=('iir.cf', 'relation.sorted_array'),
  )
  def lower_mir_ws_scope(op: Any, ctx: Any) -> Any:
    return lower_ws_scope(op, ctx)

  # Verifier scaffolding — WS-specific invariants land incrementally
  # as we encode them (e.g. WSScope.inner must be an InsertInto).
  @verifier(DIALECT)
  def _verify(_prog):
    return []


_register_passes()


__all__ = ['DIALECT', 'register']


# ---------------------------------------------------------------------------
# Phase E plugin entry point
# ---------------------------------------------------------------------------
#
# `register(compiler)` is the callable invoked by F4's plugin discovery
# (`Compiler.with_default_plugins()`) AND by explicit user-driven
# `compiler.register_plugin(...)` calls. Wired in `pyproject.toml` under
# `[project.entry-points."srdatalog.plugins"]` as `parallel_atomic_ws`.
#
# Coexistence with legacy direct-import: the module-level
# `_register_passes()` call above still runs on first import, so the
# dialect's `@lowering(target=DIALECT, source=mir.WSScope)` rule and
# the `@pragma_handler(WorkStealing, on=ExecutePipeline)` callback
# (registered as a side effect of importing
# `pragmas.work_stealing` from inside `_register_passes`) populate
# their respective registries regardless of whether anyone goes
# through `register(compiler)`. Python's module cache makes
# `_register_passes()` itself naturally idempotent (decorators only
# fire on first import). What `register(compiler)` adds on top is the
# per-Compiler `register_dialect(DIALECT)` step that F4's plugin
# loader needs to attribute ownership of the dialect to this plugin
# and populate the running Compiler's registry.
#
# Re-running `register(compiler)` on the same Compiler is safe: F4's
# `register_plugin` (see `core/plugin.py`) short-circuits on
# already-loaded plugin names.


def register(compiler: Any) -> None:
  '''Plugin entry point — register the `parallel.atomic_ws` dialect
  on the given Compiler.

  The `WSScope` lowering + `WorkStealing` pragma handler were wired
  by `_register_passes()` at module-import time (a one-shot side
  effect; Python's module cache makes it naturally idempotent). This
  callable only performs the per-Compiler step:
  `compiler.register_dialect(DIALECT)`.

  Idempotent: F4's `Compiler.register_plugin` short-circuits
  re-registration of the same plugin name.
  '''
  compiler.register_dialect(DIALECT)


# Plugin metadata read by F4's topo-sort + conflict detection
# (see `core/plugin.py` — `_plugin_attr` reads these).
#
# `plugin_name` — what F4 records as the loaded-plugin identifier.
#   Matches the entry-point name `parallel_atomic_ws` declared in
#   `pyproject.toml`.
# `provides` — the dialect name this plugin contributes. Pinned to
#   the `parallel.atomic_ws` string above and to the spec
#   (`docs/phase_c_pragma_materialization.md` §4.2).
# `requires` — dialects that must be loaded first. Empty: the
#   `@lowering` rule registered by `_register_passes` consumes
#   `mir` (which is registered unconditionally by core) and produces
#   `iir.cf` / `relation.sorted_array` — both produces-relationships
#   are per-pass tags, not load-time bootstrap dependencies, so no
#   `requires` arrow points out of here.
register.plugin_name = 'parallel_atomic_ws'  # type: ignore[attr-defined]
register.provides = ('parallel.atomic_ws',)  # type: ignore[attr-defined]
register.requires = ()  # type: ignore[attr-defined]
