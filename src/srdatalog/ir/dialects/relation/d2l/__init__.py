'''relation.d2l — Device2LevelIndex dialect.

Today this module owns:
  - `view_count(spec)` and `view_counts_for_specs` — view-slot
    accounting used by `compile.compile_kernel_body` to populate
    `view_counts` for the dialect's view declarations.
  - `D2lSegmentLoop` op (in `ops.py`) — the IIR segment-loop wrapper
    that lets nested CJ pipelines visit both HEAD and FULL segments
    of a D2L FULL_VER source.

Legacy-comment header (kept for the design narrative):

A 2-level index: each relation has a HEAD segment (smaller, dense) and
a FULL segment (the full sorted array). FULL_VER reads expose both
segments as separate views in `views[]`; iteration over the source
visits both segments in turn (HEAD first, FULL second).

Versions:
  - DELTA_VER, NEW_VER:  1 view (FULL only — incremental data)
  - FULL_VER:             2 views (HEAD + FULL)

Compared to `relation.sorted_array` (single-view-per-source DSAI):

  - View-slot accounting: a FULL_VER D2L source occupies 2 contiguous
    slots in `views[]`. `view_count(version) = 2 if version == FULL_VER
    else 1`. The dialect's `emit_view_declarations` honors per-spec
    `view_counts` to compute correct positional slots.

  - Nested CJ segment loop: a nested ColumnJoin whose RHS source is a
    D2L FULL_VER source must wrap its body in
    `for (_nseg = 0; _nseg < 2; ++_nseg) { view_X = views[base+_nseg];
    <cj body> }` so the join visits both segments. This op (planned:
    `D2lSegmentLoop`) is the missing piece keeping D2L pipelines on
    the legacy fallback today.

  - BG histogram segment-aware degree: BG histogram non-first sources
    accumulate degree across both segments before validity checks.
    Lives in `codegen.cuda.render.parallel_data.emit_bg_histogram_kernel`
    (which already handles `view_count > 1`).

Planned ops (N5.x):
  D2lRoot(view_full, view_head)         dual-segment root handle
  D2lPrefix(parent, key)                 cooperative narrow + segment
  D2lSegmentLoop(seg_var, view, body)    iterate over HEAD/FULL
  D2lDualSegmentDegree(handle, view_count)  segment-aware degree sum

The legacy `plugin_view_count` is the underlying source for
`view_count` until we expand the dialect to host its own ops + emit.
'''

from __future__ import annotations

from typing import Any

from srdatalog.ir.codegen.cuda.envelope import ViewSpec
from srdatalog.ir.core import Dialect

# Side-effect import: registers D2L's CUDA plugin with
# `codegen.cuda.plugin` so `plugin_view_count` etc. dispatch correctly
# for relations declared `index_type="...Device2LevelIndex"`.
from srdatalog.ir.dialects.relation.d2l import cuda as _cuda
from srdatalog.ir.dialects.relation.d2l.ops import D2lSegmentLoop

DIALECT = Dialect(
  name='relation.d2l',
  ops=[D2lSegmentLoop],
)


def view_count(version: str, index_type: str) -> int:
  '''Number of physical view slots a (version, index_type) pair
  consumes. D2L FULL_VER = 2; DSAI / D2L DELTA / NEW = 1.

  Delegates to the legacy `plugin_view_count` for the actual
  resolution. Once the relation-dialect registry hosts D2L ops,
  the dispatch shifts here and the legacy plugin module retires.
  '''
  from srdatalog.ir.codegen.cuda.plugin import plugin_view_count

  return plugin_view_count(version, index_type)


def view_counts_for_specs(
  view_specs: list[ViewSpec],
  rel_index_types: dict[str, str],
) -> list[int]:
  '''Per-spec view_count for a pipeline's view specs. Pass the result
  to `emit_view_declarations(..., view_counts=...)` so positional
  slots advance by the right amount per spec.
  '''
  return [view_count(sp.version, rel_index_types.get(sp.rel_name, '')) for sp in view_specs]


__all__ = [
  'DIALECT',
  'D2lSegmentLoop',
  'register',
  'view_count',
  'view_counts_for_specs',
]


# Verifier scaffolding — D2L invariants (segment_depth coherence, etc.)
# land incrementally as we encode them.
def _register_passes() -> None:
  from srdatalog.ir.core.passes import verifier

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
# `[project.entry-points."srdatalog.plugins"]` as `d2l`.
#
# Coexistence with legacy direct-import: the module-level
# `_register_passes()` call above still runs on first import, so the
# dialect's `verifier` is populated regardless of whether anyone goes
# through `register(compiler)`. The other side-effect of importing this
# module — `from . import cuda as _cuda`, which registers the D2L
# `IndexPlugin` with `codegen.cuda.plugin._PLUGIN_REGISTRY` — also runs
# at import time and is naturally idempotent (Python's module cache
# guarantees a single import per process). What `register(compiler)`
# adds on top is the per-Compiler `register_dialect(DIALECT)` step that
# F4's plugin loader needs to attribute ownership of the dialect to
# this plugin and populate the running Compiler's registry.
#
# Re-running `register(compiler)` on the same Compiler is safe: F4's
# `register_plugin` (see `core/dialect.py`) short-circuits on
# already-loaded plugin names.


def register(compiler: Any) -> None:
  '''Plugin entry point — register the `relation.d2l` dialect on the
  given Compiler.

  Verifier was wired by `_register_passes()` and the CUDA `IndexPlugin`
  registration ran via `from . import cuda` — both at module-import
  time (one-shot side effects; Python's module cache makes them
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
#   Matches the entry-point name `d2l` declared in `pyproject.toml`.
# `provides` — the dialect name this plugin contributes.
# `requires` — dialects that must be loaded first. Empty: although
#   `sorted_array`'s `lower_execute_pipeline` declares
#   `produces=('iir.cf', 'relation.sorted_array', 'relation.d2l', ...)`,
#   the D2L dialect itself depends on nothing at registration time —
#   the cross-dialect production ordering is enforced by the
#   pass-driver's `consumes`/`produces` walk, not by plugin load order.
register.plugin_name = 'd2l'  # type: ignore[attr-defined]
register.provides = ('relation.d2l',)  # type: ignore[attr-defined]
register.requires = ()  # type: ignore[attr-defined]
