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
    `srdatalog.dialects` + `srdatalog.targets` entry-point groups
    plus the legacy `srdatalog.plugins` group (one-shot
    `DeprecationWarning` per process when the legacy group has
    entries; see PR-1a, `docs/phase_decomposition_redesign.md`
    §3.3.4). Topo-sorts by each plugin's provides/requires
    attributes and registers each in order. De-dupes by plugin name
    across groups.
  - `Compiler.register_plugin(plugin)` accepts a plugin name, a
    register callable, or an `EntryPoint`. Idempotent.
  - Conflict detection: a dialect of the same name registered by a
    different plugin raises `PluginConflictError` unless the second
    plugin sets `register.replaces = ("<name>", ...)`.

See `docs/ir_lowering_semantics.md`, section 19.
'''

from __future__ import annotations

import threading
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from srdatalog.ir.core.plugin import (
  DIALECT_ENTRY_POINT_GROUP,
  ENTRY_POINT_GROUP,
  TARGET_ENTRY_POINT_GROUP,
  PluginConflictError,
  PluginInfo,
  PluginLoadError,
  _discover_entry_points,
  _info_for,
  _topo_sort_plugins,
)
from srdatalog.ir.core.services import Services

if TYPE_CHECKING:
  from srdatalog.ir.core.attribute import Attribute
  from srdatalog.ir.core.ops import Op
  from srdatalog.ir.core.passes import Lowering, Pass
  from srdatalog.ir.core.pragma_plugin import PragmaPlugin, Render

# One-shot DeprecationWarning emission marker. PR-1a's spec
# (`docs/phase_decomposition_redesign.md` §3.3.4) requires the legacy
# `srdatalog.plugins` group to keep working for one release while
# emitting `DeprecationWarning` exactly once per process when it
# contains entries (so a noisy install with two legacy plugins still
# only warns once, not twice).
_LEGACY_GROUP_WARNED = False


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


# PR-P0 (spec § 3.2.1.2 Risk 4 fix): per-Compiler "current compiler"
# thread-local. Decorator sugar (@register_render / @lowering /
# @pragma_handler) consults this; when set, the decorator stages its
# registration into the per-Compiler registries; otherwise it falls
# back to the legacy module-global registries (back-compat shim,
# deleted in PR-P5).
#
# This dissolves the Risk-4 anti-pattern ("decorators wrap mutable
# module-global state") without requiring every existing decorator
# call site to grow a `compiler=` kwarg. The thread-local is set by
# `Compiler.register_pragma_plugin(...)` while importing / running
# the plugin's contribution code; outside that scope the decorators
# preserve their existing module-global behavior.
_current_compiler: threading.local = threading.local()


def _get_current_compiler() -> Compiler | None:
  '''Return the thread-local "current compiler", or None if not in a
  `_compiler_registration_scope`. Decorator sugar reads this to
  decide whether to stage into the per-Compiler registries or fall
  back to module-globals.
  '''
  return getattr(_current_compiler, 'compiler', None)


@contextmanager
def _compiler_registration_scope(compiler: Compiler) -> Iterator[None]:
  '''Set the thread-local "current compiler" for the duration of the
  with-block (PR-P0, spec § 3.2.1.2 Risk 4 fix).

  Used by `Compiler.register_pragma_plugin` to make decorators stage
  registrations into THIS compiler's per-Compiler registries instead
  of the legacy module-globals. Nested scopes are tolerated and
  restore the previous value on exit (sequential plugin loads from
  inside another plugin's `register` are legal).
  '''
  import contextlib

  prev = getattr(_current_compiler, 'compiler', None)
  _current_compiler.compiler = compiler
  try:
    yield
  finally:
    if prev is None:
      with contextlib.suppress(AttributeError):
        del _current_compiler.compiler
    else:
      _current_compiler.compiler = prev


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
    # Active render target for the in-flight `Compiler.run(...)` call.
    # Set on entry, restored on exit. Outside a `run` call this is the
    # process-wide default. PR-1c (Phase B2-1, foundation) introduces
    # this attr so downstream passes can read the target the caller
    # chose; the actual per-target dispatch logic lands in Phase B2-2.
    # See docs/phase_decomposition_redesign.md §3.3.1.
    self.active_target: str = 'cuda'

    # ----------------------------------------------------------------
    # PR-P0 per-Compiler registries (spec § 3.2.1.2 Risk 4 fix +
    # § 3.2.1.3). The legacy module-globals (`_PRAGMA_REGISTRY`,
    # `_PLUGIN_REGISTRY`, the cuda render registries) stay around as
    # back-compat shims through PR-P1..P4; PR-P5 deletes them. From
    # PR-P0 onward these per-Compiler dicts are the SOURCE OF TRUTH
    # for everything the decorator sugar stages while a
    # `_compiler_registration_scope(self)` is active.
    # ----------------------------------------------------------------
    self.ops: dict[str, type[Op]] = {}
    self.attributes: dict[str, type[Attribute]] = {}
    self.passes: list[Pass] = []  # topo-sorted; see register_pragma_plugin
    # Lowerings keyed on (source_op_type, target_op_type). Today's
    # `Lowering.matches` is `type` (un-bound, pre-existing field
    # shape from passes.py); the typed-by-op narrowing happens in
    # PR-P1..P4 as each plugin migrates. The dict is typed `type`,
    # not `type[Op]`, to stay compatible with the un-narrowed Lowering
    # shape while still encoding the (source, target) pair shape from
    # spec § 6.0.0.
    self.lowerings: dict[tuple[type, type], Lowering] = {}
    self.renders: dict[tuple[type, str], Render] = {}
    self.services: Services = Services()
    self.pragma_plugins: list[PragmaPlugin] = []

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
    pipeline: list[Any] | None = None,
    target: str = 'cuda',
  ) -> Any:
    '''Drive a list of `Pass` instances over `prog`. Returns the
    (possibly transformed) prog.

    PR-P0: `pipeline` is now optional. If omitted, the run uses
    `self.passes` (the topo-sorted per-Compiler pass list maintained
    by `register_pragma_plugin(...)`). Existing call sites that pass
    `pipeline=[...]` continue to work unchanged — the existing
    explicit-pipeline construction surface stays around through the
    per-pragma migration phases.

    Pre-flight: validate pipeline ordering. For each `Pass`, check
    that all `consumes` dialects are either registered with this
    Compiler OR produced by an earlier Pass in the list. Raises
    `PassOrderingError` on mismatch — at construction time, before
    any pass executes.

    `target` selects the render target for the run; passes that need
    it read `compiler.active_target` (set for the duration of the
    call, restored on exit). Defaults to ``'cuda'`` so all existing
    call sites are byte-equivalent. PR-1c (Phase B2-1, foundation)
    introduces the kwarg; actual per-target dispatch lands in Phase
    B2-2. See docs/phase_decomposition_redesign.md §3.3.1.

    Per `docs/compiler_redesign.md` §4 and the R4 research report:
    pipelines are data; ordering errors are caught up-front.
    '''
    # Local import to avoid a circular at module-load time
    # (passes.py imports Compiler from this module).
    from srdatalog.ir.core.passes import PassOrderingError

    pipeline_list: list[Any] = pipeline if pipeline is not None else list(self.passes)

    available: set[str] = {d.name for d in self.dialects}
    for i, p in enumerate(pipeline_list):
      for needed in p.consumes:
        if needed not in available:
          raise PassOrderingError(p.name, needed, i)
      available |= set(p.produces)

    prev_target = self.active_target
    self.active_target = target
    try:
      for p in pipeline_list:
        prog = p.apply(prog, self)
      return prog
    finally:
      self.active_target = prev_target

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

  # -- PR-P0: atomic pragma-plugin registration -----------------------------

  def register_pragma_plugin(self, plugin: PragmaPlugin) -> None:
    '''Atomically unpack a `PragmaPlugin` into the per-Compiler
    registries (PR-P0, spec § 3.2.1.3 / § 6.0.0).

    All-or-nothing transaction. Conflicts detected up front; if any
    is found, the per-Compiler registries are left UNCHANGED — no
    partial registration. Conflicts checked (spec § 6.0.0):

      - Op-name collision: `plugin.new_ops` contains an op whose
        `__name__` is already in `self.ops` under a different class.
      - Attribute-name collision: same shape on `plugin.new_attributes`
        vs `self.attributes`.
      - Render double-registration: `(op_type, target)` already in
        `self.renders`.
      - Pass-cycle: after appending `plugin.passes`, the LLVM-style
        topo-sort raises `PassCycleError`.
      - Missing required service: a type in
        `plugin.requires_services` is not registered with
        `self.services`.

    On success:

      - `self.ops` / `self.attributes` / `self.renders` populated
        with the plugin's contributions.
      - `self.lowerings` populated keyed on `(matches, target_op)`
        pairs. Today's `Lowering` carries only `matches` (source op
        type); the target op is read from a single produced op when
        unambiguous. Multi-target lowerings are a Phase-B2 concern
        outside PR-P0 scope; for now, single-target lowerings store
        under `(matches, matches)` if no `produces_ops` is declared.
      - `self.passes` extended with the plugin's passes, then
        topo-sorted via `topo_sort_passes(...)`.
      - `self.pragma_plugins` appended for introspection.

    See spec § 3.2.1.3 for the rationale ("one atomic registration
    that bundles ALL of [pragma_cls, ops, attributes, passes,
    lowerings, renders, services, dependencies]").
    '''
    # Local imports avoid the core-init cycle. pragma_plugin imports
    # passes which imports dialect (this module).
    from srdatalog.ir.core.passes import PassCycleError, topo_sort_passes
    from srdatalog.ir.core.pragma_plugin import (
      AttributeNameCollisionError,
      MissingRequiredServiceError,
      OpNameCollisionError,
      RenderDoubleRegistrationError,
    )

    # -------- Phase 1: conflict detection (no mutation) --------

    # Op-name collisions.
    for op_cls in plugin.new_ops:
      existing = self.ops.get(op_cls.__name__)
      if existing is not None and existing is not op_cls:
        raise OpNameCollisionError(
          f'PragmaPlugin {plugin.pragma_cls.__name__!r}: op name '
          f'{op_cls.__name__!r} already registered to {existing!r}.'
        )

    # Attribute-name collisions.
    for attr_cls in plugin.new_attributes:
      existing_attr = self.attributes.get(attr_cls.__name__)
      if existing_attr is not None and existing_attr is not attr_cls:
        raise AttributeNameCollisionError(
          f'PragmaPlugin {plugin.pragma_cls.__name__!r}: attribute name '
          f'{attr_cls.__name__!r} already registered to {existing_attr!r}.'
        )

    # Render double-registration.
    for target, render_tuple in plugin.renders.items():
      for r in render_tuple:
        key = (r.op_type, target)
        if key in self.renders:
          raise RenderDoubleRegistrationError(
            f'PragmaPlugin {plugin.pragma_cls.__name__!r}: render for '
            f'(op_type={r.op_type.__name__!r}, target={target!r}) is '
            f'already registered.'
          )

    # Missing required services.
    for svc_type in plugin.requires_services:
      if svc_type not in self.services:
        raise MissingRequiredServiceError(
          f'PragmaPlugin {plugin.pragma_cls.__name__!r}: requires service '
          f'of type {svc_type.__name__!r}, but none is registered with '
          f'this Compiler. Register the service before this plugin (e.g. '
          f'`compiler.services.register({svc_type.__name__}, <instance>)`).'
        )

    # Pass-cycle detection. Build the *proposed* combined pass list
    # and topo-sort it in a staging variable; commit only on success.
    proposed_passes = [*self.passes, *plugin.passes]
    try:
      sorted_passes = topo_sort_passes(proposed_passes)
    except PassCycleError as e:
      raise PassCycleError(e.cycle) from None

    # -------- Phase 2: commit (no early returns past this point) --------

    for op_cls in plugin.new_ops:
      self.ops[op_cls.__name__] = op_cls

    for attr_cls in plugin.new_attributes:
      self.attributes[attr_cls.__name__] = attr_cls

    for low in plugin.lowerings:
      # Single-target lowering: key on (matches, matches) when no
      # explicit target-op is known. Multi-target dispatching is
      # Phase B2 work. The key shape mirrors spec § 6.0.0 's
      # `lowerings: dict[tuple[type[Op], type[Op]], Lowering]`.
      self.lowerings[(low.matches, low.matches)] = low

    for target, render_tuple in plugin.renders.items():
      for r in render_tuple:
        self.renders[(r.op_type, target)] = r

    self.passes = sorted_passes
    self.pragma_plugins.append(plugin)

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
      # Look up by entry-point name. PR-1a (§3.3.4) splits the legacy
      # `srdatalog.plugins` group into `srdatalog.dialects` +
      # `srdatalog.targets`; consult both new groups first, then the
      # legacy group, mirroring the search order of
      # `with_default_plugins()`.
      for g in (DIALECT_ENTRY_POINT_GROUP, TARGET_ENTRY_POINT_GROUP, ENTRY_POINT_GROUP):
        for ep in _discover_entry_points(g):
          if ep.name == plugin:
            return ep.name, ep.load()
      raise LookupError(
        f'no plugin named {plugin!r} in entry-point groups '
        f'{DIALECT_ENTRY_POINT_GROUP!r}, {TARGET_ENTRY_POINT_GROUP!r}, '
        f'or {ENTRY_POINT_GROUP!r}'
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

    Production form (`group=None`): walks all three entry-point
    groups in order — `srdatalog.dialects` (primary),
    `srdatalog.targets` (currently empty; populated when a target
    plugin ships), then the legacy `srdatalog.plugins` group. If the
    legacy group contains entries, a one-shot `DeprecationWarning`
    fires per process (see PR-1a, `docs/phase_decomposition_redesign.md`
    §3.3.4). De-dupes by plugin name across groups: a plugin that
    appears in both the legacy group and one of the new groups is
    loaded once, from the first new group it appears in.

    Test-isolation form (`group=<str>`): walks only the named group.
    Used by `tests/test_core_plugin.py` to install synthetic plugins
    under a sandbox group.
    '''
    compiler = cls()

    groups: list[str]
    if group is None:
      groups = [
        DIALECT_ENTRY_POINT_GROUP,
        TARGET_ENTRY_POINT_GROUP,
        ENTRY_POINT_GROUP,
      ]
    else:
      groups = [group]

    # Discover across all groups, de-duping by plugin name. The first
    # group that yields a given name wins — so a plugin redundantly
    # listed in both the legacy group and a new group is loaded once.
    resolved: dict[str, tuple[Any, Callable[[Any], None]]] = {}
    infos: list[PluginInfo] = []
    legacy_eps: list[Any] = []
    for g in groups:
      eps = _discover_entry_points(g)
      if g == ENTRY_POINT_GROUP and group is None:
        legacy_eps = list(eps)
      for ep in eps:
        if ep.name in resolved:
          # Already discovered under an earlier (new) group; skip.
          continue
        register_fn = ep.load()
        info = _info_for(ep.name, register_fn)
        resolved[ep.name] = (ep, register_fn)
        infos.append(info)

    # Emit the legacy-group DeprecationWarning at most once per
    # process when the legacy group has entries. Suppressed under
    # the test-isolation form — those tests install fake plugins
    # under custom groups and shouldn't trigger the legacy warning.
    if group is None and legacy_eps:
      global _LEGACY_GROUP_WARNED
      if not _LEGACY_GROUP_WARNED:
        _LEGACY_GROUP_WARNED = True
        warnings.warn(
          f'entry-point group {ENTRY_POINT_GROUP!r} is deprecated; '
          f'move plugin registrations to {DIALECT_ENTRY_POINT_GROUP!r} '
          f'(data dialects) or {TARGET_ENTRY_POINT_GROUP!r} (render '
          f'targets). See docs/phase_decomposition_redesign.md §3.3.4.',
          DeprecationWarning,
          stacklevel=2,
        )

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
