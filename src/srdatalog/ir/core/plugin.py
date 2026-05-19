'''Plugin discovery + registration infrastructure.

Spec: `docs/phase_e_plugin_extensibility.md`. Locked design decisions
in `docs/phase_zero_prerequisites.md` §3.5.

Plugins extend the compiler with new dialects, pragmas, targets, and
the like. Each plugin is a callable `register(compiler)` that mutates
the compiler's registries. Plugins are discovered via Python entry
points OR registered explicitly.

Entry-point groups (PR-1a, `docs/phase_decomposition_redesign.md`
§3.3.4): `srdatalog.dialects` for data-dialect / pragma / IIR
contributions, `srdatalog.targets` for render targets, and the legacy
`srdatalog.plugins` kept for one release as a back-compat shim.

Per the locked design (see `docs/phase_zero_prerequisites.md` §3.5):

  - `Compiler()` is empty by default — no auto-discovery.
  - `Compiler.with_default_plugins()` is the opt-in factory that walks
    `importlib.metadata.entry_points(group=...)` and loads every
    entry point.
  - Plugins declare their `provides` / `requires` / `replaces`
    metadata as function attributes on the `register` callable. The
    loader Kahn-topo-sorts by these attributes so a plugin that
    depends on another's dialect is loaded after it. Ties broken
    lexicographically by plugin name (deterministic).
  - Two plugins registering the same dialect name raises
    `PluginConflictError`. The escape hatch is to set
    `register.replaces = ("<name>", ...)` on the colliding plugin's
    `register` callable; that plugin then overwrites the prior
    registration.
  - A `register(compiler)` callable that raises propagates as
    `PluginLoadError(plugin_name, original_exception)`.
  - Cyclic `provides` / `requires` declarations raise
    `PluginCycleError`.

This module is part of `core/`. Per discipline rule D6
(`docs/code_discipline.md` §2), it does NOT import any concrete
dialect — it does not even import `Compiler` (the dependency runs
the other way: `core/dialect.py` imports from this module).
'''

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

# Note on D6 (`docs/code_discipline.md` §2): this module deliberately
# does NOT import any concrete dialect or even the `Compiler` class.
# `register_fn(compiler)` callbacks take an opaque `Any` here; the
# real type lives in `core/dialect.py`, which imports FROM this
# module (one-way dependency: dialect.py -> plugin.py).


# Public entry-point group names.
#
# PR-1a (`docs/phase_decomposition_redesign.md` §3.3.4) splits the
# single legacy `srdatalog.plugins` group into two semantic groups:
#
#   - `srdatalog.dialects` — data-dialect / pragma / IIR contributions
#     (every built-in lives here from PR-1a onward).
#   - `srdatalog.targets` — render-target contributions (CUDA / TBB /
#     SYCL / ...). Empty in PR-1a; populated in a later PR when the
#     CUDA target ships as a plugin.
#
# The legacy `srdatalog.plugins` group is kept for one release as a
# back-compat shim so external packages (e.g. the jaccard demo) that
# still declare entries under the old group continue to load — with a
# one-shot `DeprecationWarning` per process at
# `Compiler.with_default_plugins()` time.
DIALECT_ENTRY_POINT_GROUP = 'srdatalog.dialects'
TARGET_ENTRY_POINT_GROUP = 'srdatalog.targets'
ENTRY_POINT_GROUP = 'srdatalog.plugins'  # LEGACY — kept for back-compat


# -----------------------------------------------------------------------------
# Error classes
# -----------------------------------------------------------------------------


class PluginLoadError(RuntimeError):
  '''A plugin's `register(compiler)` callable raised an exception.

  Wraps the original exception so the caller can see both which
  plugin failed and why. The original exception is available via
  `__cause__` (set by `raise ... from e`).
  '''


class PluginConflictError(RuntimeError):
  '''A plugin tried to register something already registered by
  another plugin and did not declare `replaces=` for that name.

  The escape hatch: set `register.replaces = ("<name>", ...)` on the
  colliding plugin's `register` callable. This is opt-in per the
  locked design (see `docs/phase_zero_prerequisites.md` §5).
  '''


class PluginCycleError(RuntimeError):
  '''The `provides` / `requires` declarations across the discovered
  plugins form a cycle.

  Raised by the Kahn topo-sort in
  `Compiler.with_default_plugins()`.
  '''


# -----------------------------------------------------------------------------
# Resolved plugin metadata
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PluginInfo:
  '''Resolved metadata for a registered plugin.

  Fields:
    name      — plugin identifier (entry-point name OR an explicit
                name from a callable registration).
    provides  — dialect names this plugin contributes. Read from
                `register.provides`; defaults to `()`.
    requires  — dialect names this plugin needs to already be
                registered when its `register` runs. Read from
                `register.requires`; defaults to `()`.
    replaces  — dialect names this plugin overrides (escape hatch
                for the conflict check). Read from
                `register.replaces`; defaults to `()`.
  '''

  name: str
  provides: tuple[str, ...] = ()
  requires: tuple[str, ...] = ()
  replaces: tuple[str, ...] = ()


# -----------------------------------------------------------------------------
# Discovery + topo-sort
# -----------------------------------------------------------------------------


def _discover_entry_points(group: str = ENTRY_POINT_GROUP) -> list[Any]:
  '''Wraps `importlib.metadata.entry_points(group=group)`.

  Returns a list (not the underlying `EntryPoints` selectable view)
  so callers can sort / iterate without worrying about the
  importlib-metadata API churn between Python versions.
  '''
  return list(importlib.metadata.entry_points(group=group))


def _plugin_attr(register_fn: Callable[..., Any], name: str) -> tuple[str, ...]:
  '''Read a `(provides|requires|replaces)` tuple-attribute off a
  register callable. Missing attribute defaults to `()`.

  Raises `TypeError` if the attribute is present but not a tuple of
  strings — caught early so a typo (`register.requires = "foo"`,
  meaning a single-character iterable) becomes a loud failure
  instead of a silent miss.
  '''
  value = getattr(register_fn, name, ())
  if value == ():
    return ()
  if not isinstance(value, tuple) or not all(isinstance(s, str) for s in value):
    raise TypeError(f'plugin register.{name} must be a tuple[str, ...]; got {value!r}')
  return value


def _info_for(plugin_name: str, register_fn: Callable[..., Any]) -> PluginInfo:
  '''Build a `PluginInfo` from a register callable's attributes.'''
  return PluginInfo(
    name=plugin_name,
    provides=_plugin_attr(register_fn, 'provides'),
    requires=_plugin_attr(register_fn, 'requires'),
    replaces=_plugin_attr(register_fn, 'replaces'),
  )


def _topo_sort_plugins(infos: Iterable[PluginInfo]) -> list[str]:
  '''Kahn topo-sort by `provides` / `requires`. Stable on ties
  (lexicographic by plugin name).

  Returns the plugin names in load order. Raises `PluginCycleError`
  if the graph has a cycle.

  Edge semantics: if plugin B has `requires = ("foo",)` and plugin
  A has `provides = ("foo",)`, then A must load before B. The graph
  has an edge A -> B for every (provides, requires) match.

  A `requires` entry that no plugin provides is silently allowed —
  the plugin just loads in whatever order its name dictates and
  fails inside its own `register` if the dependency isn't satisfied
  another way. This matches the "loud failure on conflict, quiet
  failure on optional dependency" stance in
  `docs/phase_zero_prerequisites.md` §3.5.
  '''
  by_name: dict[str, PluginInfo] = {info.name: info for info in infos}

  # Map dialect-name -> plugin that provides it. Last writer wins for
  # the topo-sort (the conflict check happens later, during actual
  # registration in `Compiler.register_plugin`).
  provider: dict[str, str] = {}
  for info in by_name.values():
    for d in info.provides:
      provider[d] = info.name

  # Build edges: provider -> consumer.
  out_edges: dict[str, set[str]] = {n: set() for n in by_name}
  in_degree: dict[str, int] = dict.fromkeys(by_name, 0)
  for info in by_name.values():
    for need in info.requires:
      src = provider.get(need)
      if src is None or src == info.name:
        continue
      if info.name not in out_edges[src]:
        out_edges[src].add(info.name)
        in_degree[info.name] += 1

  # Kahn: pop the lex-smallest zero-in-degree node, repeat.
  ready = sorted(n for n, d in in_degree.items() if d == 0)
  order: list[str] = []
  while ready:
    n = ready.pop(0)
    order.append(n)
    new_ready = []
    for m in sorted(out_edges[n]):
      in_degree[m] -= 1
      if in_degree[m] == 0:
        new_ready.append(m)
    # Merge into `ready` while preserving lex order.
    ready = sorted(set(ready) | set(new_ready))

  if len(order) != len(by_name):
    remaining = sorted(set(by_name) - set(order))
    raise PluginCycleError(f'plugin provides/requires graph has a cycle involving: {remaining}')
  return order


__all__ = [
  'DIALECT_ENTRY_POINT_GROUP',
  'ENTRY_POINT_GROUP',
  'PluginConflictError',
  'PluginCycleError',
  'PluginInfo',
  'PluginLoadError',
  'TARGET_ENTRY_POINT_GROUP',
]
