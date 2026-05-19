'''PR-1a tests — plugin entry-point group split:
`srdatalog.plugins` -> `srdatalog.dialects` + `srdatalog.targets`.

Spec: `docs/phase_decomposition_redesign.md` §3.3.4.

The four assertions in this file:

  1. `_discover_entry_points(DIALECT_ENTRY_POINT_GROUP)` returns the
     6 built-in dialect plugins shipped under
     `[project.entry-points."srdatalog.dialects"]` in `pyproject.toml`.
  2. `_discover_entry_points(TARGET_ENTRY_POINT_GROUP)` returns the
     empty list — PR-1a does not ship any built-in target plugin; that
     lands in a later PR when the CUDA target plugin ships.
  3. `_discover_entry_points(ENTRY_POINT_GROUP)` returns whatever
     external packages have declared under the legacy group. With
     no jaccard-style external plugin installed in the test env, the
     list is empty; we don't pin a length here, just that the call
     succeeds and the legacy constant is still wired.
  4. `Compiler.with_default_plugins()` (no `group=`) walks ALL THREE
     groups — verified by patching the entry-point lookup to install
     fake plugins under each group and asserting all three load. The
     same test asserts that the one-shot DeprecationWarning fires
     exactly once per process when the legacy group has entries.

These tests use the same `FakeEntryPoint` + monkeypatch pattern as
`tests/test_core_plugin.py` for full control over what each group
returns — without depending on what's installed in the host env.
'''

from __future__ import annotations

import importlib.metadata
import warnings
from collections.abc import Callable
from typing import Any

import pytest

from srdatalog.ir.core import Compiler, Dialect
from srdatalog.ir.core import (
  plugin as plugin_mod,
)
from srdatalog.ir.core.plugin import (
  DIALECT_ENTRY_POINT_GROUP,
  ENTRY_POINT_GROUP,
  TARGET_ENTRY_POINT_GROUP,
  _discover_entry_points,
)

# -----------------------------------------------------------------------------
# Built-in dialect plugin names from pyproject.toml.
# -----------------------------------------------------------------------------

BUILTIN_DIALECT_PLUGIN_NAMES: frozenset[str] = frozenset(
  {
    'sorted_array',
    'iir_cf',
    'd2l',
    'parallel_data',
    'parallel_atomic_ws',
    'parallel_block_group',
  }
)


# -----------------------------------------------------------------------------
# Fake entry-point harness (same shape as tests/test_core_plugin.py).
# -----------------------------------------------------------------------------


class FakeEntryPoint:
  '''Minimal stand-in for `importlib.metadata.EntryPoint` — only `.name`
  and `.load()` are exercised by the plugin loader. Mirrors the fake
  used by `tests/test_core_plugin.py` to avoid coupling to the live
  importlib.metadata API.
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
  '''Patch `importlib.metadata.entry_points` so the plugin loader sees
  only the fakes registered here, for each group it asks about.'''

  def _fake_entry_points(*, group: str) -> list[FakeEntryPoint]:
    return list(group_to_eps.get(group, []))

  monkeypatch.setattr(plugin_mod.importlib.metadata, 'entry_points', _fake_entry_points)


def _make_register(dialect_name: str) -> Callable[[Compiler], None]:
  '''Build a synthetic `register(compiler)` that contributes one
  Dialect named `dialect_name`. Carries no provides/requires/replaces
  metadata; topo-sort orders it lex by plugin name.'''

  def register(compiler: Compiler) -> None:
    compiler.register_dialect(Dialect(name=dialect_name))

  return register


@pytest.fixture(autouse=True)
def _reset_legacy_warned_marker() -> Any:
  '''Reset the one-shot DeprecationWarning marker between tests so each
  test sees a fresh "haven't warned yet" state. Without this, the
  first test that triggers the warning would silence every subsequent
  test's warning assertion.'''
  from srdatalog.ir.core import dialect as dialect_mod

  prev = dialect_mod._LEGACY_GROUP_WARNED
  dialect_mod._LEGACY_GROUP_WARNED = False
  yield
  dialect_mod._LEGACY_GROUP_WARNED = prev


# -----------------------------------------------------------------------------
# 1. The 6 built-ins are declared under `srdatalog.dialects`
# -----------------------------------------------------------------------------


def test_dialect_group_lists_the_six_builtin_plugins() -> None:
  '''The live (un-monkeypatched) `srdatalog.dialects` group contains
  the 6 built-in dialect plugins shipped in this repo's pyproject.toml.

  If this test fails with "no plugins loaded" or a subset of the
  expected names, the most likely cause is that the editable install's
  `.egg-info` is stale — run
  `pip install -e . --no-deps --force-reinstall` to refresh the
  entry-point metadata. This is the same install-state warning every
  per-dialect `_as_plugin` test carries.
  '''
  eps = _discover_entry_points(DIALECT_ENTRY_POINT_GROUP)
  names = {ep.name for ep in eps}
  missing = BUILTIN_DIALECT_PLUGIN_NAMES - names
  assert not missing, (
    f'missing expected built-in dialect plugins in '
    f'{DIALECT_ENTRY_POINT_GROUP!r}: {sorted(missing)!r}. '
    f'Got: {sorted(names)!r}. If empty, refresh the editable install: '
    f'`pip install -e . --no-deps --force-reinstall`.'
  )


# -----------------------------------------------------------------------------
# 2. The targets group is empty in PR-1a
# -----------------------------------------------------------------------------


def test_target_group_is_empty_in_pr_1a() -> None:
  '''PR-1a creates the `srdatalog.targets` group but ships no
  built-in target plugin under it — the CUDA target lands as a plugin
  in a later PR. External plugin packages MAY declare entries here.

  We assert the BUILT-IN target plugin set is empty by filtering out
  any entry-point whose value path starts with `srdatalog.` (our own
  package) — leaving room for an externally-installed target plugin
  (cpu_tbb / sycl / ...) without flaking this test.
  '''
  eps = _discover_entry_points(TARGET_ENTRY_POINT_GROUP)
  builtin_target_eps = [ep for ep in eps if getattr(ep, 'value', '').startswith('srdatalog.')]
  assert builtin_target_eps == [], (
    f'PR-1a should ship NO built-in target plugins; got {builtin_target_eps!r}'
  )


# -----------------------------------------------------------------------------
# 3. Legacy group is still walkable; no built-ins remain there
# -----------------------------------------------------------------------------


def test_legacy_group_contains_no_builtins() -> None:
  '''Post-PR-1a, NONE of the 6 built-in dialect plugins should remain
  in the legacy `srdatalog.plugins` group — they all moved to
  `srdatalog.dialects`. An external plugin (e.g. the jaccard demo)
  may still appear here; we explicitly assert only the BUILT-IN
  names are absent.

  This is the regression guard for the migration: if a future
  refactor accidentally re-adds (or fails to delete) a built-in
  entry in the legacy group, the entry would also remain reachable
  via the new group and double-register at runtime (no-op due to
  idempotency, but still meaningful as a contract violation).
  '''
  eps = _discover_entry_points(ENTRY_POINT_GROUP)
  legacy_builtin_names = {ep.name for ep in eps} & BUILTIN_DIALECT_PLUGIN_NAMES
  assert legacy_builtin_names == set(), (
    f'built-in plugins still declared under legacy group '
    f'{ENTRY_POINT_GROUP!r}: {sorted(legacy_builtin_names)!r}. '
    f'They must move to {DIALECT_ENTRY_POINT_GROUP!r}.'
  )


def test_discover_entry_points_legacy_group_call_succeeds() -> None:
  '''The legacy group is still a valid query target for
  `_discover_entry_points`. The return list may be empty (no
  external legacy-group plugin installed) or non-empty (e.g. the
  jaccard demo is `pip install -e`d); we don't pin length.
  '''
  eps = _discover_entry_points(ENTRY_POINT_GROUP)
  assert isinstance(eps, list)


# -----------------------------------------------------------------------------
# 4. with_default_plugins walks all three groups + de-dupes by name
# -----------------------------------------------------------------------------


def test_with_default_plugins_walks_all_three_groups(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''`Compiler.with_default_plugins()` with no `group=` argument walks
  `srdatalog.dialects` AND `srdatalog.targets` AND the legacy
  `srdatalog.plugins` — verified by installing one fake plugin per
  group and asserting all three show up in `_plugins_loaded`.
  '''
  dialect_reg = _make_register('fake_dialect_thing')
  target_reg = _make_register('fake_target_thing')
  legacy_reg = _make_register('fake_legacy_thing')

  _install_entry_points(
    monkeypatch,
    {
      DIALECT_ENTRY_POINT_GROUP: [FakeEntryPoint('plug_dialect', dialect_reg)],
      TARGET_ENTRY_POINT_GROUP: [FakeEntryPoint('plug_target', target_reg)],
      ENTRY_POINT_GROUP: [FakeEntryPoint('plug_legacy', legacy_reg)],
    },
  )

  with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    c = Compiler.with_default_plugins()

  assert set(c._plugins_loaded) == {'plug_dialect', 'plug_target', 'plug_legacy'}
  assert {d.name for d in c.dialects} == {
    'fake_dialect_thing',
    'fake_target_thing',
    'fake_legacy_thing',
  }


def test_with_default_plugins_dedupes_by_plugin_name_across_groups(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''If the same plugin name appears under BOTH the new
  `srdatalog.dialects` group and the legacy `srdatalog.plugins`
  group (e.g. a package that updated for PR-1a but kept the legacy
  entry around during the transition), it loads exactly once — from
  the new group. De-duping prevents `PluginConflictError` on the
  dialect re-registration that the second copy would trigger.
  '''
  # Both groups carry a plugin named `dup_plug`, but with DIFFERENT
  # register callables (contributing dialects of different names so we
  # can tell which one ran). The new-group copy must win.
  new_group_reg = _make_register('from_new_group')
  legacy_reg = _make_register('from_legacy_group')

  _install_entry_points(
    monkeypatch,
    {
      DIALECT_ENTRY_POINT_GROUP: [FakeEntryPoint('dup_plug', new_group_reg)],
      ENTRY_POINT_GROUP: [FakeEntryPoint('dup_plug', legacy_reg)],
    },
  )

  with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    c = Compiler.with_default_plugins()

  assert set(c._plugins_loaded) == {'dup_plug'}
  assert {d.name for d in c.dialects} == {'from_new_group'}


# -----------------------------------------------------------------------------
# 5. DeprecationWarning fires exactly once when legacy group has entries
# -----------------------------------------------------------------------------


def test_legacy_group_emits_deprecation_warning_when_nonempty(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''When the legacy `srdatalog.plugins` group contains entries,
  `Compiler.with_default_plugins()` emits exactly one
  `DeprecationWarning` per process. PR-1a (§3.3.4) requires both
  the warning text and the once-per-process emission policy so a
  noisy environment with 5 legacy plugins still produces 1 warning,
  not 5.
  '''
  legacy_reg = _make_register('legacy_dialect_thing')
  _install_entry_points(
    monkeypatch,
    {ENTRY_POINT_GROUP: [FakeEntryPoint('legacy_plug', legacy_reg)]},
  )

  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    Compiler.with_default_plugins()
    Compiler.with_default_plugins()  # second call must NOT re-warn

  dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
  assert len(dep) == 1, f'expected exactly one DeprecationWarning, got {len(dep)}'
  msg = str(dep[0].message)
  assert ENTRY_POINT_GROUP in msg
  assert DIALECT_ENTRY_POINT_GROUP in msg
  assert TARGET_ENTRY_POINT_GROUP in msg
  assert 'deprecated' in msg.lower()


def test_legacy_group_emits_no_warning_when_empty(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''When the legacy group has no entries, no `DeprecationWarning`
  fires — the warning is gated on the presence of legacy-group
  entries so a clean install stays quiet.
  '''
  # New group only; legacy group is implicitly empty.
  _install_entry_points(
    monkeypatch,
    {DIALECT_ENTRY_POINT_GROUP: [FakeEntryPoint('only_new', _make_register('only_new_d'))]},
  )

  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    Compiler.with_default_plugins()

  dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
  assert dep == [], f'unexpected DeprecationWarning(s) when legacy group is empty: {dep!r}'


# -----------------------------------------------------------------------------
# 6. importlib.metadata.entry_points still works on the live env
# -----------------------------------------------------------------------------


def test_importlib_metadata_returns_the_dialect_group_entries() -> None:
  '''Sanity: bypass our wrapper and confirm `importlib.metadata` itself
  lists the built-in dialects under `srdatalog.dialects`. This pins
  the contract that the entry-point declaration in `pyproject.toml`
  actually reaches the installed metadata after
  `pip install -e .` — independent of our discovery helper.
  '''
  eps = importlib.metadata.entry_points(group=DIALECT_ENTRY_POINT_GROUP)
  names = {ep.name for ep in eps}
  missing = BUILTIN_DIALECT_PLUGIN_NAMES - names
  assert not missing, (
    f'missing entries in importlib.metadata view of {DIALECT_ENTRY_POINT_GROUP!r}: '
    f'{sorted(missing)!r}. Refresh with '
    f'`pip install -e . --no-deps --force-reinstall`.'
  )
