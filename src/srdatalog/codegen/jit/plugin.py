'''Index-type plugin registry + default DSAI (DeviceSortedArrayIndex) hooks.

Port of src/srdatalog/index_plugin.nim. Every C++ expression the JIT
backend emits for handle/view operations dispatches through an
IndexPlugin so custom index types (e.g., Device2LevelIndex used by
polonius_test) can override behavior. The default plugin matches
DSAI and is what every existing integration fixture uses.
'''
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class PrefixMode(Enum):
  '''How a prefix() / prefix_lower_bound() call is dispatched.'''
  COOPERATIVE = "cooperative"  # warp-wide: .prefix(key, tile, view)
  SEQUENTIAL = "sequential"    # per-thread: .prefix_seq(key, view)


# -----------------------------------------------------------------------------
# IndexPlugin
# -----------------------------------------------------------------------------

@dataclass
class IndexPlugin:
  '''Codegen hooks for one index family. Each field is a callable that
  emits a C++ expression string. Plugins register themselves by C++ type
  (`cpp_type`) and the resolver falls back to the default (DSAI) when
  no plugin is registered for a given type.
  '''
  name: str
  cpp_type: str
  cpp_headers: list[str] = field(default_factory=list)

  # Expression-level hooks — each returns a C++ expression string.
  gen_root_handle: Callable[[str], str] = lambda v: ""
  gen_prefix: Callable[[str, str, str, PrefixMode], str] = lambda h, k, v, m: ""
  gen_prefix_lower_bound: Callable[[str, str, str, PrefixMode], str] = (
    lambda h, k, v, m: ""
  )
  gen_degree: Callable[[str], str] = lambda h: ""
  gen_valid: Callable[[str], str] = lambda h: ""
  gen_get_value_at: Callable[[str, str, str], str] = lambda h, v, i: ""
  gen_get_value: Callable[[str, int, str], str] = lambda v, c, p: ""
  gen_child: Callable[[str, str], str] = lambda h, i: ""
  gen_child_range: Callable[[str, str, str, str, str], str] = (
    lambda h, p, k, t, v: ""
  )
  gen_iterators: Callable[[str, str], str] = lambda h, v: ""

  # View-level hooks.
  view_count: Callable[[str], int] = lambda version: 1
  gen_host_view_setup: Callable[[str, str], list[str]] = (
    lambda idx_expr, version: [f"{idx_expr}.view()"]
  )


# -----------------------------------------------------------------------------
# Default DSAI hooks (plain, non-lambda functions for readability)
# -----------------------------------------------------------------------------

def _default_gen_root_handle(view_var: str) -> str:
  return f"HandleType(0, {view_var}.num_rows_, 0)"


def _default_gen_prefix(handle: str, key: str, view_var: str, mode: PrefixMode) -> str:
  if mode is PrefixMode.COOPERATIVE:
    return f"{handle}.prefix({key}, tile, {view_var})"
  return f"{handle}.prefix_seq({key}, {view_var})"


def _default_gen_prefix_lower_bound(
  handle: str, key: str, view_var: str, mode: PrefixMode,
) -> str:
  if mode is PrefixMode.COOPERATIVE:
    return f"{handle}.prefix_lower_bound({key}, tile, {view_var})"
  return f"{handle}.prefix_lower_bound_seq({key}, {view_var})"


def _default_gen_degree(handle: str) -> str:
  return f"{handle}.degree()"


def _default_gen_valid(handle: str) -> str:
  return f"{handle}.valid()"


def _default_gen_get_value_at(handle: str, view_var: str, idx: str) -> str:
  return f"{handle}.get_value_at({view_var}, {idx})"


def _default_gen_get_value(view_var: str, col: int, pos: str) -> str:
  return f"{view_var}.get_value({col}, {pos})"


def _default_gen_child(handle: str, idx: str) -> str:
  return f"{handle}.child({idx})"


def _default_gen_child_range(handle: str, pos: str, key: str, tile: str, view_var: str) -> str:
  return f"{handle}.child_range({pos}, {key}, tile, {view_var})"


def _default_gen_iterators(handle: str, view_var: str) -> str:
  return f"{handle}.iterators({view_var})"


def _default_view_count(version: str) -> int:
  return 1


def _default_gen_host_view_setup(idx_expr: str, version: str) -> list[str]:
  return [f"{idx_expr}.view()"]


def new_default_plugin() -> IndexPlugin:
  '''Baseline DSAI plugin — matches every built-in integration fixture.'''
  return IndexPlugin(
    name="DeviceSortedArrayIndex",
    cpp_type="SRDatalog::GPU::DeviceSortedArrayIndex",
    cpp_headers=["gpu/device_sorted_array_index.h"],
    gen_root_handle=_default_gen_root_handle,
    gen_prefix=_default_gen_prefix,
    gen_prefix_lower_bound=_default_gen_prefix_lower_bound,
    gen_degree=_default_gen_degree,
    gen_valid=_default_gen_valid,
    gen_get_value_at=_default_gen_get_value_at,
    gen_get_value=_default_gen_get_value,
    gen_child=_default_gen_child,
    gen_child_range=_default_gen_child_range,
    gen_iterators=_default_gen_iterators,
    view_count=_default_view_count,
    gen_host_view_setup=_default_gen_host_view_setup,
  )


# -----------------------------------------------------------------------------
# Global registry
# -----------------------------------------------------------------------------

_PLUGIN_REGISTRY: dict[str, IndexPlugin] = {}
_DEFAULT_PLUGIN: IndexPlugin = new_default_plugin()


def register_index_plugin(plugin: IndexPlugin) -> None:
  '''Register a plugin by its `cpp_type` string. Called from each index
  module at import time (mirroring Nim's compile-time registration).
  '''
  _PLUGIN_REGISTRY[plugin.cpp_type] = plugin


def resolve_plugin(index_type: str) -> IndexPlugin:
  '''Look up plugin by C++ type; empty string / unknown type returns the
  default (DSAI). Substring match lets partial type strings resolve, matching
  Nim's resolvePlugin behavior.
  '''
  if not index_type:
    return _DEFAULT_PLUGIN
  if index_type in _PLUGIN_REGISTRY:
    return _PLUGIN_REGISTRY[index_type]
  for key, plugin in _PLUGIN_REGISTRY.items():
    if key in index_type or index_type in key:
      return plugin
  return _DEFAULT_PLUGIN


def get_extra_headers_for_types(index_types: list[str]) -> list[str]:
  '''Collect unique C++ headers declared by the plugins resolving from
  `index_types`. Empty strings are skipped.'''
  out: list[str] = []
  for t in index_types:
    if not t:
      continue
    for h in resolve_plugin(t).cpp_headers:
      if h not in out:
        out.append(h)
  return out


# -----------------------------------------------------------------------------
# Plugin-dispatched expression wrappers (the public API every other
# codegen module calls into). `index_type=""` always hits the DSAI default.
# -----------------------------------------------------------------------------

def plugin_gen_root_handle(view_var: str, index_type: str = "") -> str:
  return resolve_plugin(index_type).gen_root_handle(view_var)


def plugin_gen_prefix(
  handle: str, key: str, view_var: str, mode: PrefixMode, index_type: str = "",
) -> str:
  return resolve_plugin(index_type).gen_prefix(handle, key, view_var, mode)


def plugin_gen_prefix_lower_bound(
  handle: str, key: str, view_var: str, mode: PrefixMode, index_type: str = "",
) -> str:
  return resolve_plugin(index_type).gen_prefix_lower_bound(
    handle, key, view_var, mode,
  )


def plugin_gen_degree(handle: str, index_type: str = "") -> str:
  return resolve_plugin(index_type).gen_degree(handle)


def plugin_gen_valid(handle: str, index_type: str = "") -> str:
  return resolve_plugin(index_type).gen_valid(handle)


def plugin_gen_get_value_at(handle: str, view_var: str, idx: str, index_type: str = "") -> str:
  return resolve_plugin(index_type).gen_get_value_at(handle, view_var, idx)


def plugin_gen_get_value(view_var: str, col: int, pos: str, index_type: str = "") -> str:
  return resolve_plugin(index_type).gen_get_value(view_var, col, pos)


def plugin_gen_child(handle: str, idx: str, index_type: str = "") -> str:
  return resolve_plugin(index_type).gen_child(handle, idx)


def plugin_gen_child_range(
  handle: str, pos: str, key: str, tile: str, view_var: str, index_type: str = "",
) -> str:
  return resolve_plugin(index_type).gen_child_range(handle, pos, key, tile, view_var)


def plugin_gen_iterators(handle: str, view_var: str, index_type: str = "") -> str:
  return resolve_plugin(index_type).gen_iterators(handle, view_var)


def plugin_view_count(version: str, index_type: str = "") -> int:
  return resolve_plugin(index_type).view_count(version)


def plugin_gen_host_view_setup(
  idx_expr: str, version: str, index_type: str = "",
) -> list[str]:
  return resolve_plugin(index_type).gen_host_view_setup(idx_expr, version)


# -----------------------------------------------------------------------------
# Chained prefix calls — handles variable-keyword sanitization + modes
# -----------------------------------------------------------------------------

def _sanitize_cpp_kw(name: str) -> str:
  '''Minimal sanitization — full keyword list lives in context.py; this
  shim is here to keep plugin.py self-contained for tests. context.py's
  sanitize_var_name is authoritative when imported together.'''
  _reserved = {"int", "float", "double", "char", "bool", "long", "short", "void",
                "for", "if", "else", "while", "return", "class", "struct", "union",
                "template", "typename", "namespace", "new", "delete", "this"}
  return name + "_val" if name in _reserved else name


def plugin_chained_prefix_calls(
  parent_handle: str,
  prefix_vars: list[str],
  view_var: str,
  cartesian_bound_vars: list[str] | None = None,
  scalar_mode: bool = False,
  index_type: str = "",
) -> str:
  '''Emit chained .prefix(key, ...) calls. For each `v` in `prefix_vars`:
    - Sequential mode (per-thread), or cartesian_bound_vars containing v:
        use `.prefix_seq(v, view)`
    - Else cooperative: `.prefix(v, tile, view)`

  Matches Nim's pluginChainedPrefixCalls exactly.
  '''
  if cartesian_bound_vars is None:
    cartesian_bound_vars = []
  if not prefix_vars:
    return parent_handle

  plugin = resolve_plugin(index_type)
  result = parent_handle
  for v in prefix_vars:
    sanitized = _sanitize_cpp_kw(v)
    in_cartesian = sanitized in cartesian_bound_vars or v in cartesian_bound_vars
    mode = (
      PrefixMode.SEQUENTIAL if (scalar_mode or in_cartesian)
      else PrefixMode.COOPERATIVE
    )
    result = plugin.gen_prefix(result, sanitized, view_var, mode)
  return result


def plugin_chained_prefix_with_last_lower_bound(
  parent_handle: str,
  prefix_vars: list[str],
  view_var: str,
  cartesian_bound_vars: list[str] | None = None,
  scalar_mode: bool = False,
  index_type: str = "",
) -> str:
  '''Like `plugin_chained_prefix_calls` but the LAST key uses
  `prefix_lower_bound` instead of `prefix`. Used for range intersection.
  '''
  if cartesian_bound_vars is None:
    cartesian_bound_vars = []
  if not prefix_vars:
    return parent_handle

  plugin = resolve_plugin(index_type)
  result = parent_handle
  last_idx = len(prefix_vars) - 1
  for i, v in enumerate(prefix_vars):
    sanitized = _sanitize_cpp_kw(v)
    in_cartesian = sanitized in cartesian_bound_vars or v in cartesian_bound_vars
    mode = (
      PrefixMode.SEQUENTIAL if (scalar_mode or in_cartesian)
      else PrefixMode.COOPERATIVE
    )
    if i == last_idx:
      result = plugin.gen_prefix_lower_bound(result, sanitized, view_var, mode)
    else:
      result = plugin.gen_prefix(result, sanitized, view_var, mode)
  return result
