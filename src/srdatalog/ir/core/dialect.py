'''Dialect base + Compiler registry.

A `Dialect` is a coherent set of types, ops, lowerings, rewrites, and
a verifier. Dialects register with a `Compiler` at init time, which
indexes them by name and exposes lookups for the pass driver.

The registry has no central enum of dialects — Property P1. Adding a
new dialect = constructing a `Dialect` and calling `register_dialect`.
No edits to this module.

Plugin discovery (per `docs/phase_e_plugin_extensibility.md` and the
locked design in `docs/phase_zero_prerequisites.md` §3.5):

  - `Compiler()` is empty. No auto-discovery.
  - `Compiler.with_default_plugins()` walks the
    `srdatalog.plugins` entry-point group, topo-sorts by each
    plugin's provides/requires attributes, and registers each in
    order.
  - `Compiler.register_plugin(plugin)` accepts a plugin name, a
    register callable, or an `EntryPoint`. Idempotent.
  - Conflict detection: a dialect of the same name registered by a
    different plugin raises `PluginConflictError` unless the second
    plugin sets `register.replaces = ("<name>", ...)`.

See `docs/ir_lowering_semantics.md`, section 19.
'''

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from srdatalog.ir.core.plugin import (
  ENTRY_POINT_GROUP,
  PluginConflictError,
  PluginInfo,
  PluginLoadError,
  _discover_all_groups,
  _discover_entry_points,
  _info_for,
  _topo_sort_plugins,
)


@dataclass
class Dialect:
  '''A registered dialect.

  Fields:
    name      — dialect identifier (e.g. "relation.sorted_array").
    types     — Type subclasses owned by this dialect.
    ops       — Op subclasses owned by this dialect.
    lowerings — Lowering rules emitted *out* of this dialect.
    rewrites  — Rewrite rules *within* this dialect.
    verifier  — optional callable that validates the dialect's IR
                shape; returns a list of VerificationError on failure.
  '''

  name: str
  types: list[type] = field(default_factory=list)
  ops: list[type] = field(default_factory=list)
  lowerings: list[Any] = field(default_factory=list)
  rewrites: list[Any] = field(default_factory=list)
  verifier: Callable[[Any], list[Any]] | None = None


# Sentinel used internally for `register_dialect` calls that happen
# OUTSIDE a `register_plugin` context (e.g. tests calling
# `compiler.register_dialect(...)` directly). Such registrations are
# attributed to the sentinel "owner" and conflict with everything that
# isn't the same sentinel. This keeps the conflict-detection path
# simple without forcing every direct-registration test to invent a
# fake plugin.
_DIRECT_OWNER = '<direct>'


class Compiler:
  '''Holds the registered dialects and tracks which plugin owns each.

  The registry is the single source of truth for what dialects exist.
  Lookups happen by name; cross-dialect lowerings are resolved by
  matching the source op kind against each dialect's `lowerings` list.

  Plugin tracking (per `docs/phase_e_plugin_extensibility.md`):

    - `_plugins_loaded` maps plugin name -> resolved `PluginInfo`.
    - `_dialects_by_plugin` maps dialect name -> plugin name that
      registered it. `register_dialect` consults this to detect
      conflicts (a different plugin trying to register the same
      dialect name without declaring `replaces=`).
    - `_active_plugin_context` is set during `register_plugin` so
      `register_dialect` knows who is calling. Outside that context,
      the owner is `_DIRECT_OWNER`.
  '''

  def __init__(self) -> None:
    self._dialects: dict[str, Dialect] = {}
    self._plugins_loaded: dict[str, PluginInfo] = {}
    self._dialects_by_plugin: dict[str, str] = {}
    # Set by `register_plugin` while a plugin's `register` callable
    # runs. `register_dialect` reads it to attribute the registration
    # to the right plugin.
    self._active_plugin_context: str | None = None
    self._active_plugin_replaces: tuple[str, ...] = ()

  # -- dialect registration --------------------------------------------------

  def register_dialect(self, d: Dialect) -> None:
    '''Register a dialect.

    If called from inside a `register_plugin` flow, the dialect is
    attributed to that plugin. Re-registering a name owned by a
    different plugin raises `PluginConflictError` unless the calling
    plugin declared the name in its `register.replaces` tuple.

    Direct calls (outside a plugin) collide with anything that isn't
    a prior direct call with the same name; `ValueError` is raised
    in that case for backwards-compat with the pre-plugin behavior.
    '''
    owner = self._active_plugin_context or _DIRECT_OWNER
    prev_owner = self._dialects_by_plugin.get(d.name)

    if prev_owner is None:
      self._dialects[d.name] = d
      self._dialects_by_plugin[d.name] = owner
      return

    # A prior registration exists. Decide whether to allow override.
    if owner == _DIRECT_OWNER and prev_owner == _DIRECT_OWNER:
      # Pre-plugin contract: re-registering a dialect by direct call
      # is a programmer error. Preserve the old `ValueError` so
      # existing tests / callers see the same behavior.
      raise ValueError(f'Dialect {d.name!r} already registered')

    if owner == prev_owner:
      # Same plugin re-registering the same dialect (e.g. the
      # idempotent re-load path). Treat as a no-op.
      return

    if d.name in self._active_plugin_replaces:
      # Escape hatch: this plugin declared `replaces=("<name>", ...)`
      # on its register fn. Override.
      self._dialects[d.name] = d
      self._dialects_by_plugin[d.name] = owner
      return

    raise PluginConflictError(
      f'plugin {owner!r} tried to register dialect {d.name!r}, which is already '
      f'registered by plugin {prev_owner!r}; declare '
      f'`register.replaces = ({d.name!r},)` on the new plugin to override'
    )

  def get_dialect(self, name: str) -> Dialect:
    '''Look up a registered dialect by name. Raises KeyError if missing.'''
    return self._dialects[name]

  @property
  def dialects(self) -> list[Dialect]:
    '''All registered dialects, in registration order.'''
    return list(self._dialects.values())

  def run(
    self,
    prog: Any,
    *,
    pipeline: list[Any],
    target: str = 'cuda',
  ) -> Any:
    '''Drive a list of `Pass` instances over `prog`. Returns the
    (possibly transformed) prog.

    Pre-flight: validate pipeline ordering. For each `Pass`, check
    that all `consumes` dialects are either registered with this
    Compiler OR produced by an earlier Pass in the list. Raises
    `PassOrderingError` on mismatch — at construction time, before
    any pass executes.

    Per `docs/compiler_redesign.md` §4 and the R4 research report:
    pipelines are data; ordering errors are caught up-front.

    PR-1 (per `docs/phase_decomposition_redesign.md` § 3.3.1): the
    `target` kwarg threads the active render target into the per-
    pass through-state when the state carries a `target` field
    (e.g. `KernelCtx`). This lets `RenderShim` dispatch the per-op
    `@register_render(target=T)` resolution without hardcoding the
    target. Default preserves the current single-target ('cuda')
    behavior.
    '''
    # Local import to avoid a circular at module-load time
    # (passes.py imports Compiler from this module).
    import dataclasses

    from srdatalog.ir.core.passes import PassOrderingError

    available: set[str] = {d.name for d in self.dialects}
    for i, p in enumerate(pipeline):
      for needed in p.consumes:
        if needed not in available:
          raise PassOrderingError(p.name, needed, i)
      available |= set(p.produces)

    # If the through-state dataclass exposes a `target` field, thread
    # the active target into it. This keeps the per-pass shims target-
    # parametric: each shim reads `state.target` to drive per-target
    # dispatch (e.g. RenderShim's @register_render lookup).
    if dataclasses.is_dataclass(prog) and not isinstance(prog, type):
      if any(f.name == 'target' for f in dataclasses.fields(prog)):
        current = getattr(prog, 'target', None)
        if current != target:
          prog = dataclasses.replace(prog, target=target)

    for p in pipeline:
      prog = p.apply(prog, self)
    return prog

  # -- plugin registration ---------------------------------------------------

  def register_plugin(self, plugin: Any) -> None:
    '''Register a plugin.

    `plugin` is one of:

      - `str` — entry-point name; resolved via
        `importlib.metadata.entry_points(group="srdatalog.plugins")`.
      - `Callable[[Compiler], None]` — a raw `register(compiler)`
        function. Its `__name__` is used as the plugin name (or
        the value of `register.plugin_name` if set).
      - An `EntryPoint`-shaped object (from
        `importlib.metadata.entry_points`) with `.name` and
        `.load()` — common case from
        `Compiler.with_default_plugins`.

    Idempotent: re-registering the same plugin name is a no-op.
    Raises `PluginLoadError` if the register callable raises (the
    original exception is chained via `__cause__`).
    Raises `PluginConflictError` via `register_dialect` if the
    plugin tries to register a dialect already owned by another
    plugin without declaring `replaces=`.
    '''
    plugin_name, register_fn = self._resolve_plugin(plugin)

    if plugin_name in self._plugins_loaded:
      # Already loaded — idempotent no-op.
      return

    info = _info_for(plugin_name, register_fn)

    # Set the active context so `register_dialect` (called from
    # within `register_fn`) attributes registrations to this plugin.
    prev_ctx = self._active_plugin_context
    prev_replaces = self._active_plugin_replaces
    self._active_plugin_context = plugin_name
    self._active_plugin_replaces = info.replaces
    try:
      try:
        register_fn(self)
      except PluginConflictError:
        # Conflict errors are part of the plugin API surface; surface
        # them directly without wrapping so callers can `except
        # PluginConflictError`.
        raise
      except Exception as e:
        raise PluginLoadError(f'plugin {plugin_name!r} register() raised: {e!r}') from e
    finally:
      self._active_plugin_context = prev_ctx
      self._active_plugin_replaces = prev_replaces

    self._plugins_loaded[plugin_name] = info

  @staticmethod
  def _resolve_plugin(plugin: Any) -> tuple[str, Callable[[Any], None]]:
    '''Coerce a plugin argument to `(name, register_fn)`.

    Recognizes:
      - `str` — entry-point name, looked up in the default group.
      - object with `.name` and `.load()` — `EntryPoint` shape.
      - callable — raw register function. Name comes from
        `register.plugin_name` if set, else `register.__name__`.
    '''
    if isinstance(plugin, str):
      # Look up by entry-point name across every registered group
      # (dialects + targets + legacy). PR-1 (per § 3.3.4) split the
      # legacy group; the resolver follows.
      for ep in _discover_all_groups():
        if ep.name == plugin:
          return ep.name, ep.load()
      raise LookupError(
        f'no plugin named {plugin!r} across entry-point groups '
        f"('srdatalog.dialects', 'srdatalog.targets', {ENTRY_POINT_GROUP!r})"
      )

    if hasattr(plugin, 'name') and hasattr(plugin, 'load'):
      # EntryPoint-shaped (importlib.metadata.EntryPoint or a test fake).
      return plugin.name, plugin.load()

    if callable(plugin):
      name = getattr(plugin, 'plugin_name', None) or getattr(plugin, '__name__', None)
      if not name:
        raise TypeError(
          'register_plugin: callable has no usable name; set '
          '`register.plugin_name` or pass a named function'
        )
      return name, plugin

    raise TypeError(f'register_plugin: unsupported plugin argument {plugin!r}')

  # -- discovery factory -----------------------------------------------------

  @classmethod
  def with_default_plugins(cls, *, group: str | None = None) -> Compiler:
    '''Construct a `Compiler` and auto-discover all entry-point plugins.

    Walks the production entry-point groups (per PR-1, per
    `docs/phase_decomposition_redesign.md` § 3.3.4):

      - `srdatalog.dialects` — data-side dialect contributions.
      - `srdatalog.targets`  — render-side target contributions.
      - `srdatalog.plugins`  — legacy group (kept for back-compat
        for one release cycle; emits `DeprecationWarning` when
        non-empty).

    Topo-sorts by each plugin's `provides` / `requires` attributes
    on the `register` callable (defaulting to `()` when absent),
    then calls `register_plugin` on each in order.

    The `group` keyword exists for test isolation — production code
    uses the default (walks all groups). Passing a specific group
    name walks only that group; passing `None` (the default) walks
    every registered group.

    See `docs/phase_e_plugin_extensibility.md` §2 and
    `docs/phase_decomposition_redesign.md` § 3.3.4.
    '''
    compiler = cls()

    # Discover, then resolve each EP to (name, register_fn) once so
    # we can both topo-sort and register without double-loading.
    eps = _discover_entry_points(group) if group is not None else _discover_all_groups()
    resolved: dict[str, tuple[Any, Callable[[Any], None]]] = {}
    infos: list[PluginInfo] = []
    for ep in eps:
      register_fn = ep.load()
      info = _info_for(ep.name, register_fn)
      resolved[ep.name] = (ep, register_fn)
      infos.append(info)

    order = _topo_sort_plugins(infos)

    for plugin_name in order:
      _ep, register_fn = resolved[plugin_name]
      # Pass the already-loaded callable directly; this avoids a
      # second `ep.load()` and makes the path identical regardless of
      # whether the EP loader is fast or slow.
      # Attach the EP name onto a wrapper-equivalent — we just call
      # register_plugin with the callable, but we need the name to be
      # `plugin_name`. Stamp it on so `_resolve_plugin` picks it up.
      try:
        register_fn.plugin_name = plugin_name  # type: ignore[attr-defined]
      except (AttributeError, TypeError):
        # Built-in / C-callable: fall back to wrapping in a thin
        # function that carries the metadata.
        register_fn = _named(plugin_name, register_fn)
      compiler.register_plugin(register_fn)

    return compiler


def _named(name: str, fn: Callable[[Any], None]) -> Callable[[Any], None]:
  '''Wrap `fn` so it has `plugin_name == name` and copies through
  the topo-sort attributes. Used by `with_default_plugins` when the
  raw callable is not a writable Python function.
  '''

  def _wrapper(compiler: Any) -> None:
    fn(compiler)

  _wrapper.__name__ = name
  _wrapper.plugin_name = name  # type: ignore[attr-defined]
  for attr in ('provides', 'requires', 'replaces'):
    if hasattr(fn, attr):
      setattr(_wrapper, attr, getattr(fn, attr))
  return _wrapper
