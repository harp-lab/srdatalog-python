'''Data-parallelism strategy dialects.

Each strategy decides how `ParallelFor` distributes work across
threads/warps/blocks. A pipeline picks one at lowering time
based on its workload characteristics (uniform vs skewed,
fixed-shape vs dynamic).

Currently registered ops:
  - BgRootCjMulti — block-group root multi-source ColumnJoin
'''

from __future__ import annotations

from typing import Any

from srdatalog.ir.core import Dialect
from srdatalog.ir.dialects.parallel.data.block_group import BgRootCjMulti

DIALECT = Dialect(
  name='parallel.data',
  ops=[BgRootCjMulti],
)


# Verifier scaffolding — block-group / parallel-strategy invariants
# land incrementally as we encode them.
def _register_passes() -> None:
  from srdatalog.ir.core.passes import verifier

  @verifier(DIALECT)
  def _verify(_prog):
    return []


_register_passes()


__all__ = ['DIALECT', 'BgRootCjMulti', 'register']


# ---------------------------------------------------------------------------
# Phase E plugin entry point
# ---------------------------------------------------------------------------
#
# `register(compiler)` is the callable invoked by F4's plugin discovery
# (`Compiler.with_default_plugins()`) AND by explicit user-driven
# `compiler.register_plugin(...)` calls. Wired in `pyproject.toml` under
# `[project.entry-points."srdatalog.plugins"]` as `parallel_data`.
#
# Coexistence with legacy direct-import: the module-level
# `_register_passes()` call above still runs on first import, so the
# dialect's `verifier` is populated regardless of whether anyone goes
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
  '''Plugin entry point — register the `parallel.data` dialect on the
  given Compiler.

  Verifier scaffolding was wired by `_register_passes()` at module-
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
#   Matches the entry-point name `parallel_data` declared in
#   `pyproject.toml`.
# `provides` — the dialect name this plugin contributes. Other plugins
#   may eventually `requires=('parallel.data',)` to load after; the
#   `relation.sorted_array` lowerings already produce ops into this
#   dialect (see its `lower_execute_pipeline` `produces` tuple) but
#   that produces-relationship is a per-pass tag, not a load-time
#   dependency, so no `requires` arrow points here today.
# `requires` — dialects that must be loaded first. Empty: the only
#   ops registered here (`BgRootCjMulti`) are emitted by other
#   dialects' lowerings — nothing this plugin's `_register_passes`
#   does mutates another dialect's registry, so there is no
#   bootstrap-time ordering requirement.
register.plugin_name = 'parallel_data'  # type: ignore[attr-defined]
register.provides = ('parallel.data',)  # type: ignore[attr-defined]
register.requires = ()  # type: ignore[attr-defined]
