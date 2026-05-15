'''Tests for the plugin discovery + registration infrastructure in
`core/plugin.py` and the `Compiler.register_plugin` /
`Compiler.with_default_plugins` extensions in `core/dialect.py`.

Spec: `docs/phase_e_plugin_extensibility.md`. Locked design in
`docs/phase_zero_prerequisites.md` §3.5.

All tests are self-contained: they construct synthetic `Dialect` and
`register` callables, never importing any real dialect (which would
violate the discipline boundary D6 / D11 in `docs/code_discipline.md`).

Entry-point discovery is exercised by monkeypatching
`importlib.metadata.entry_points` with a fake `EntryPoint`-shaped
object that exposes `.name` and `.load()`.
'''

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from srdatalog.ir.core import (
  Compiler,
  Dialect,
  PluginConflictError,
  PluginCycleError,
  PluginInfo,
  PluginLoadError,
)
from srdatalog.ir.core import (
  plugin as plugin_mod,
)

# -----------------------------------------------------------------------------
# Fakes
# -----------------------------------------------------------------------------


class FakeEntryPoint:
  '''Minimal stand-in for `importlib.metadata.EntryPoint`.

  Real `EntryPoint` instances are immutable namedtuples; the only
  surface our code uses is `.name` and `.load()`, so this fake is
  enough.
  '''

  def __init__(self, name: str, register_fn: Callable[[Compiler], None]) -> None:
    self.name = name
    self._register_fn = register_fn

  def load(self) -> Callable[[Compiler], None]:
    return self._register_fn


def _install_entry_points(
  monkeypatch: pytest.MonkeyPatch,
  group_to_eps: dict[str, list[FakeEntryPoint]],
) -> None:
  '''Patch `importlib.metadata.entry_points` for the duration of a
  test. The patched function returns the list registered for the
  requested group, or `[]` if the group is unknown.
  '''

  def _fake_entry_points(*, group: str) -> list[FakeEntryPoint]:
    return list(group_to_eps.get(group, []))

  # Patch the module that `core/plugin.py` actually calls into.
  monkeypatch.setattr(plugin_mod.importlib.metadata, 'entry_points', _fake_entry_points)


def _make_register(
  *,
  dialects: tuple[str, ...] = (),
  provides: tuple[str, ...] = (),
  requires: tuple[str, ...] = (),
  replaces: tuple[str, ...] = (),
  raises: BaseException | None = None,
  side_effect: Callable[[Compiler], None] | None = None,
) -> Callable[[Compiler], None]:
  '''Build a synthetic `register(compiler)` callable.

  The returned callable, when invoked with a `Compiler`, registers a
  `Dialect` for each name in `dialects` (no ops / lowerings — the
  point is to exercise the registry plumbing). The function carries
  the `provides` / `requires` / `replaces` attributes that the
  topo-sort and conflict-detection paths read.
  '''

  def register(compiler: Compiler) -> None:
    if raises is not None:
      raise raises
    for name in dialects:
      compiler.register_dialect(Dialect(name=name))
    if side_effect is not None:
      side_effect(compiler)

  register.provides = provides  # type: ignore[attr-defined]
  register.requires = requires  # type: ignore[attr-defined]
  register.replaces = replaces  # type: ignore[attr-defined]
  return register


# -----------------------------------------------------------------------------
# 1. Compiler() is empty by default
# -----------------------------------------------------------------------------


def test_compiler_is_empty_by_default() -> None:
  '''`Compiler()` performs no auto-discovery; both registries are
  empty until something is explicitly registered.'''
  c = Compiler()
  assert c.dialects == []
  assert c._plugins_loaded == {}


# -----------------------------------------------------------------------------
# 2. register_plugin(callable) — direct callable + idempotency
# -----------------------------------------------------------------------------


def test_register_plugin_callable_mutates_compiler() -> None:
  '''A synthetic `register` fn passed directly to `register_plugin`
  fires and mutates the compiler.'''
  c = Compiler()
  reg = _make_register(dialects=('synthetic_a',))
  c.register_plugin(reg)

  assert c.get_dialect('synthetic_a').name == 'synthetic_a'
  assert 'register' in c._plugins_loaded  # callable's __name__


def test_register_plugin_callable_uses_plugin_name_attr() -> None:
  '''If the callable carries `plugin_name`, that beats `__name__`.'''
  c = Compiler()
  reg = _make_register(dialects=('a',))
  reg.plugin_name = 'aliased_plugin'  # type: ignore[attr-defined]
  c.register_plugin(reg)
  assert 'aliased_plugin' in c._plugins_loaded


def test_register_plugin_is_idempotent() -> None:
  '''Re-registering the same plugin name a second time is a no-op
  (the register fn does NOT fire twice).'''
  c = Compiler()

  call_count = {'n': 0}

  def side_effect(_compiler: Compiler) -> None:
    call_count['n'] += 1

  reg = _make_register(side_effect=side_effect)
  reg.plugin_name = 'idem'  # type: ignore[attr-defined]

  c.register_plugin(reg)
  c.register_plugin(reg)

  assert call_count['n'] == 1


# -----------------------------------------------------------------------------
# 3. register_plugin(name) — entry-point name lookup
# -----------------------------------------------------------------------------


def test_register_plugin_by_name_uses_entry_point_lookup(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''Passing a string looks the name up in the default entry-point
  group via `importlib.metadata.entry_points`.'''
  reg = _make_register(dialects=('looked_up',))
  ep = FakeEntryPoint('looked_up_plugin', reg)
  _install_entry_points(monkeypatch, {plugin_mod.ENTRY_POINT_GROUP: [ep]})

  c = Compiler()
  c.register_plugin('looked_up_plugin')

  assert c.get_dialect('looked_up').name == 'looked_up'
  assert 'looked_up_plugin' in c._plugins_loaded


def test_register_plugin_by_unknown_name_raises(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''A name not present in the entry-point group raises `LookupError`.'''
  _install_entry_points(monkeypatch, {})
  c = Compiler()
  with pytest.raises(LookupError):
    c.register_plugin('nonexistent_plugin')


# -----------------------------------------------------------------------------
# 4. with_default_plugins — uses a custom group for test isolation
# -----------------------------------------------------------------------------


def test_with_default_plugins_loads_all_eps_in_group(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''`with_default_plugins(group=...)` walks the named entry-point
  group and registers every plugin found.'''
  reg_a = _make_register(dialects=('a',))
  reg_b = _make_register(dialects=('b',))
  custom_group = 'srdatalog.plugins.test_isolated_4'
  _install_entry_points(
    monkeypatch,
    {custom_group: [FakeEntryPoint('plug_a', reg_a), FakeEntryPoint('plug_b', reg_b)]},
  )

  c = Compiler.with_default_plugins(group=custom_group)

  names = {d.name for d in c.dialects}
  assert names == {'a', 'b'}
  assert set(c._plugins_loaded) == {'plug_a', 'plug_b'}


# -----------------------------------------------------------------------------
# 5. provides/requires topo-sort
# -----------------------------------------------------------------------------


def test_provides_requires_topo_sort_orders_dependencies_first(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''Plugin B with `requires = ("foo",)` loads after plugin A with
  `provides = ("foo",)`, regardless of entry-point declaration order.'''
  load_order: list[str] = []

  def fn_a(_c: Compiler) -> None:
    load_order.append('a')

  def fn_b(_c: Compiler) -> None:
    load_order.append('b')

  fn_a.provides = ('foo',)  # type: ignore[attr-defined]
  fn_b.requires = ('foo',)  # type: ignore[attr-defined]

  custom_group = 'srdatalog.plugins.test_isolated_5'
  # Deliberately put B *before* A in the EP listing — topo-sort
  # should still load A first.
  _install_entry_points(
    monkeypatch,
    {custom_group: [FakeEntryPoint('plug_b', fn_b), FakeEntryPoint('plug_a', fn_a)]},
  )

  Compiler.with_default_plugins(group=custom_group)

  assert load_order == ['a', 'b']


def test_topo_sort_ties_broken_lex_by_name(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''When two plugins have no provides/requires relationship, they
  load in lexicographic name order (deterministic).'''
  load_order: list[str] = []

  def make(name: str) -> Callable[[Compiler], None]:
    def fn(_c: Compiler) -> None:
      load_order.append(name)

    return fn

  custom_group = 'srdatalog.plugins.test_isolated_5b'
  _install_entry_points(
    monkeypatch,
    {
      custom_group: [
        FakeEntryPoint('zebra', make('zebra')),
        FakeEntryPoint('apple', make('apple')),
        FakeEntryPoint('mango', make('mango')),
      ]
    },
  )

  Compiler.with_default_plugins(group=custom_group)

  assert load_order == ['apple', 'mango', 'zebra']


# -----------------------------------------------------------------------------
# 6. PluginCycleError on cyclic provides/requires
# -----------------------------------------------------------------------------


def test_cyclic_requires_raises_plugin_cycle_error(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''Plugin A provides `x` and requires `y`; plugin B provides `y`
  and requires `x`. Topo-sort raises `PluginCycleError`.'''

  def fn_a(_c: Compiler) -> None:
    pass

  def fn_b(_c: Compiler) -> None:
    pass

  fn_a.provides = ('x',)  # type: ignore[attr-defined]
  fn_a.requires = ('y',)  # type: ignore[attr-defined]
  fn_b.provides = ('y',)  # type: ignore[attr-defined]
  fn_b.requires = ('x',)  # type: ignore[attr-defined]

  custom_group = 'srdatalog.plugins.test_isolated_6'
  _install_entry_points(
    monkeypatch,
    {custom_group: [FakeEntryPoint('plug_a', fn_a), FakeEntryPoint('plug_b', fn_b)]},
  )

  with pytest.raises(PluginCycleError):
    Compiler.with_default_plugins(group=custom_group)


# -----------------------------------------------------------------------------
# 7. PluginConflictError when two plugins register same dialect name
# -----------------------------------------------------------------------------


def test_conflicting_dialect_registration_raises(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''Two plugins both registering a dialect with the same name —
  the second one (without `replaces=`) raises `PluginConflictError`.'''
  reg_a = _make_register(dialects=('shared_dialect',))
  reg_b = _make_register(dialects=('shared_dialect',))

  custom_group = 'srdatalog.plugins.test_isolated_7'
  _install_entry_points(
    monkeypatch,
    {custom_group: [FakeEntryPoint('plug_a', reg_a), FakeEntryPoint('plug_b', reg_b)]},
  )

  with pytest.raises(PluginConflictError):
    Compiler.with_default_plugins(group=custom_group)


# -----------------------------------------------------------------------------
# 8. replaces= escape hatch resolves the conflict
# -----------------------------------------------------------------------------


def test_replaces_escape_hatch_resolves_conflict(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''When the second plugin declares `replaces=("shared",)`, its
  registration overrides the first instead of raising.'''
  reg_a = _make_register(dialects=('shared',))
  reg_b = _make_register(dialects=('shared',), replaces=('shared',))

  custom_group = 'srdatalog.plugins.test_isolated_8'
  _install_entry_points(
    monkeypatch,
    {custom_group: [FakeEntryPoint('plug_a', reg_a), FakeEntryPoint('plug_b', reg_b)]},
  )

  c = Compiler.with_default_plugins(group=custom_group)

  # Both plugins are recorded as loaded.
  assert set(c._plugins_loaded) == {'plug_a', 'plug_b'}
  # The dialect exists, owned by plug_b (the replacing plugin).
  assert c.get_dialect('shared').name == 'shared'
  assert c._dialects_by_plugin['shared'] == 'plug_b'


# -----------------------------------------------------------------------------
# 9. PluginLoadError wraps register-fn exceptions
# -----------------------------------------------------------------------------


def test_register_fn_raising_propagates_as_plugin_load_error() -> None:
  '''A register fn that raises is caught and re-raised as
  `PluginLoadError`, with the original exception attached as
  `__cause__`.'''
  c = Compiler()

  class _Boom(RuntimeError):
    pass

  reg = _make_register(raises=_Boom('synthetic failure'))
  reg.plugin_name = 'failing_plugin'  # type: ignore[attr-defined]

  with pytest.raises(PluginLoadError) as ei:
    c.register_plugin(reg)

  assert isinstance(ei.value.__cause__, _Boom)
  assert 'failing_plugin' in str(ei.value)
  # Failed plugin is NOT recorded as loaded.
  assert 'failing_plugin' not in c._plugins_loaded


# -----------------------------------------------------------------------------
# 10. Default group is `srdatalog.plugins`; override accepted
# -----------------------------------------------------------------------------


def test_default_group_is_srdatalog_plugins() -> None:
  '''The constant the loader uses is `srdatalog.plugins`, matching
  the spec in `docs/phase_e_plugin_extensibility.md` §2.'''
  assert plugin_mod.ENTRY_POINT_GROUP == 'srdatalog.plugins'


def test_with_default_plugins_default_group_uses_srdatalog_plugins(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''Calling `with_default_plugins()` with no `group=` argument
  walks the `srdatalog.plugins` group.'''
  reg = _make_register(dialects=('default_group_dialect',))
  ep = FakeEntryPoint('default_plug', reg)
  _install_entry_points(monkeypatch, {plugin_mod.ENTRY_POINT_GROUP: [ep]})

  c = Compiler.with_default_plugins()
  assert c.get_dialect('default_group_dialect').name == 'default_group_dialect'


# -----------------------------------------------------------------------------
# Bonus: PluginInfo records the resolved metadata
# -----------------------------------------------------------------------------


def test_plugin_info_records_provides_requires_replaces() -> None:
  '''`Compiler._plugins_loaded[name]` is a `PluginInfo` carrying the
  plugin's metadata as resolved from the register callable.'''
  c = Compiler()
  reg = _make_register(dialects=('m',), provides=('m',), requires=(), replaces=())
  reg.plugin_name = 'metadata_plugin'  # type: ignore[attr-defined]
  c.register_plugin(reg)

  info = c._plugins_loaded['metadata_plugin']
  assert isinstance(info, PluginInfo)
  assert info.name == 'metadata_plugin'
  assert info.provides == ('m',)
  assert info.requires == ()
  assert info.replaces == ()


def test_register_plugin_with_invalid_attr_raises_typeerror() -> None:
  '''A non-tuple `register.provides` (e.g. a bare string) is caught
  early as `TypeError` rather than misinterpreted as a 1-char tuple.'''
  c = Compiler()

  def reg(_c: Compiler) -> None:
    pass

  reg.provides = 'oops_a_string'  # type: ignore[attr-defined]
  reg.plugin_name = 'bad_attrs'  # type: ignore[attr-defined]

  with pytest.raises(TypeError):
    c.register_plugin(reg)


# -----------------------------------------------------------------------------
# Direct register_dialect still raises ValueError on duplicate
# (preserving pre-plugin behavior)
# -----------------------------------------------------------------------------


def test_direct_register_dialect_duplicate_still_raises_valueerror() -> None:
  '''Direct (non-plugin) calls to `register_dialect` for a name
  already registered by a direct call still raise `ValueError`,
  preserving the pre-plugin contract for tests / callers that don't
  go through `register_plugin`.'''
  c = Compiler()
  c.register_dialect(Dialect(name='direct'))
  with pytest.raises(ValueError):
    c.register_dialect(Dialect(name='direct'))


def test_entry_point_argument_form_works(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''Passing a `FakeEntryPoint` (EP-shaped object with `.name` and
  `.load`) to `register_plugin` directly works the same as a name
  lookup.'''
  reg = _make_register(dialects=('ep_form',))
  ep = FakeEntryPoint('ep_form_plugin', reg)
  c = Compiler()
  c.register_plugin(ep)
  assert c.get_dialect('ep_form').name == 'ep_form'
  assert 'ep_form_plugin' in c._plugins_loaded


# Marker so the file is non-empty even when collection-only:
_: Any = None
