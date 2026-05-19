'''Plugin entry-point group split — PR-1 foundation tests.

Per `docs/phase_decomposition_redesign.md` § 3.3.4 (Wave B2-1).

PR-1 splits the single legacy `srdatalog.plugins` entry-point group
into two production groups:

  - `srdatalog.dialects` — data-side dialect contributions.
  - `srdatalog.targets`  — render-side target contributions.

The legacy group is kept for one release cycle (with a
`DeprecationWarning`). External plugins (e.g. the jaccard demo)
discoverable under the legacy group continue to work.

These tests pin the split contract: the discovery helper walks all
three groups; the deprecation warning fires once per process on
legacy-group use; the constants are exported.
'''

from __future__ import annotations

import importlib.metadata
import warnings
from typing import Any

import pytest

from srdatalog.ir.core.plugin import (
  DIALECT_ENTRY_POINT_GROUP,
  ENTRY_POINT_GROUP,
  TARGET_ENTRY_POINT_GROUP,
  _discover_all_groups,
  _discover_entry_points,
)


def test_entry_point_group_constants_are_distinct() -> None:
  '''Three distinct group names per § 3.3.4.'''
  assert DIALECT_ENTRY_POINT_GROUP == 'srdatalog.dialects'
  assert TARGET_ENTRY_POINT_GROUP == 'srdatalog.targets'
  assert ENTRY_POINT_GROUP == 'srdatalog.plugins'  # legacy
  assert len({DIALECT_ENTRY_POINT_GROUP, TARGET_ENTRY_POINT_GROUP, ENTRY_POINT_GROUP}) == 3


def test_built_in_dialects_register_under_dialects_group() -> None:
  '''The 6 built-in dialects (d2l, iir_cf, sorted_array,
  parallel_data, parallel_atomic_ws, parallel_block_group) all live
  in the new `srdatalog.dialects` group post-PR-1. The legacy group
  has only external plugins (today: the jaccard demo).
  '''
  eps = list(importlib.metadata.entry_points(group=DIALECT_ENTRY_POINT_GROUP))
  names = {ep.name for ep in eps}
  for builtin in (
    'd2l',
    'iir_cf',
    'sorted_array',
    'parallel_data',
    'parallel_atomic_ws',
    'parallel_block_group',
  ):
    assert builtin in names, (
      f'Built-in dialect {builtin!r} should be discoverable under '
      f'{DIALECT_ENTRY_POINT_GROUP!r}; found: {sorted(names)}'
    )


def test_targets_group_exists_but_may_be_empty() -> None:
  '''The `srdatalog.targets` group exists post-PR-1 so external
  packages can register render targets. No built-in entries today
  (the CUDA target is still wired via the dialect registrations).
  '''
  eps = list(importlib.metadata.entry_points(group=TARGET_ENTRY_POINT_GROUP))
  # No assertion on count — just that the group is queryable.
  assert isinstance(eps, list)


def test_discover_all_groups_returns_union() -> None:
  '''The `_discover_all_groups` helper walks all three groups and
  returns a deduplicated list of entry points.'''
  all_eps = _discover_all_groups()
  dialect_eps = list(importlib.metadata.entry_points(group=DIALECT_ENTRY_POINT_GROUP))
  all_names = {ep.name for ep in all_eps}
  dialect_names = {ep.name for ep in dialect_eps}
  # Every dialect should appear in the union.
  assert dialect_names.issubset(all_names), (
    f'dialect entries missing from _discover_all_groups(): {sorted(dialect_names - all_names)}'
  )


def test_legacy_group_walk_emits_deprecation_warning() -> None:
  '''Walking the legacy `srdatalog.plugins` group emits a
  `DeprecationWarning` exactly once per process. The check is
  smoke-test only: we re-arm the warning by resetting the flag,
  call the walker, and confirm the warning fires.
  '''
  # Reset the per-process flag so the warning re-arms for this test.
  import srdatalog.ir.core.plugin as plugin_mod

  plugin_mod._LEGACY_GROUP_WARNED = False

  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    # Only emits the warning if the legacy group has entries.
    eps = _discover_entry_points(ENTRY_POINT_GROUP)
    if eps:
      assert any(
        issubclass(w.category, DeprecationWarning) and 'srdatalog.plugins' in str(w.message)
        for w in caught
      ), (
        f'expected DeprecationWarning naming srdatalog.plugins; '
        f'got: {[(w.category.__name__, str(w.message)) for w in caught]}'
      )


def test_legacy_group_warning_fires_only_once_per_process() -> None:
  '''After the first walk, subsequent walks DO NOT re-warn.'''
  import srdatalog.ir.core.plugin as plugin_mod

  plugin_mod._LEGACY_GROUP_WARNED = False
  # Prime the warning (first walk).
  with warnings.catch_warnings(record=True):
    warnings.simplefilter('always')
    _discover_entry_points(ENTRY_POINT_GROUP)

  # Second walk: no new DeprecationWarning.
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    _discover_entry_points(ENTRY_POINT_GROUP)
    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deprecation, (
      f'unexpected DeprecationWarning on second walk: {[str(w.message) for w in deprecation]}'
    )


def test_discover_entry_points_on_dialects_group_emits_no_warning() -> None:
  '''Walking the new groups does NOT trigger the legacy warning.'''
  import srdatalog.ir.core.plugin as plugin_mod

  plugin_mod._LEGACY_GROUP_WARNED = False
  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    _discover_entry_points(DIALECT_ENTRY_POINT_GROUP)
    _discover_entry_points(TARGET_ENTRY_POINT_GROUP)
    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deprecation, (
      f'new groups must not emit DeprecationWarning; got: {[str(w.message) for w in deprecation]}'
    )


# -----------------------------------------------------------------------------
# with_default_plugins integration: synthetic plugin registrations
# -----------------------------------------------------------------------------
#
# These tests use a synthetic EntryPoint-shaped fixture so they don't
# depend on the local install's actual entry points (which can vary
# across environments and would couple this test to the jaccard demo).


class _FakeEntryPoint:
  '''Test-only stand-in for `importlib.metadata.EntryPoint`.'''

  def __init__(self, name: str, register_fn: Any) -> None:
    self.name = name
    self._register_fn = register_fn

  def load(self) -> Any:
    return self._register_fn


def test_compiler_register_plugin_accepts_entry_point_shape_from_targets_group(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  '''Synthetic test: register a fake target plugin via the
  `srdatalog.targets` group's discovery path. The Compiler should
  load it just like a dialect plugin.'''
  from srdatalog.ir.core.dialect import Compiler

  load_calls: list[Any] = []

  def _fake_register(compiler: Any) -> None:
    load_calls.append(compiler)

  fake_ep = _FakeEntryPoint('my_cpu_tbb_target', _fake_register)

  c = Compiler()
  c.register_plugin(fake_ep)
  assert load_calls == [c]
  assert 'my_cpu_tbb_target' in c._plugins_loaded
