'''Import a user's .py program file and find its Program instance.

The auto-generated files produced by `tools/nim_to_dsl.py` all follow
the same shape:

    def build_<name>_program(meta: dict[str, int] | None = None) -> Program:
        ...

Users can also write programs that just construct a Program at module
top level. This module handles both shapes and plumbs optional
`meta.json` + `entry` overrides through.
'''

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from srdatalog.dsl import Program


class ProgramDiscoveryError(RuntimeError):
  '''Raised when `load_program` can't find or build a Program.'''


def load_program(
  path: str | Path,
  *,
  entry: str | None = None,
  meta: dict | str | Path | None = None,
) -> Program:
  '''Import the .py file, discover and invoke a Program builder.

  Discovery order (stops at first match):
    1. `entry` arg — explicit function name; takes precedence.
    2. A top-level callable matching `build_*_program`.
    3. A module-level `Program` instance (non-callable).

  `meta` — dataset_const metadata. Passed to the builder if the builder
  accepts a positional arg. Can be:
    - dict: passed as-is
    - str / Path: loaded as JSON
    - None: builder called with no args
  '''
  from srdatalog.dsl import Program  # local import to avoid cycles

  path = Path(path).resolve()
  if not path.exists():
    raise ProgramDiscoveryError(f"source file not found: {path}")
  module = _import_file(path)

  # 1. Explicit entry
  if entry is not None:
    if not hasattr(module, entry):
      raise ProgramDiscoveryError(f"{path}: no attribute {entry!r}")
    candidate = getattr(module, entry)
  else:
    # 2. Auto-discovered build_*_program callable
    candidate = None
    for name in dir(module):
      if name.startswith("build_") and name.endswith("_program"):
        fn = getattr(module, name)
        if callable(fn):
          candidate = fn
          break
    # 3. Module-level Program instance
    if candidate is None:
      for name in dir(module):
        if name.startswith("_"):
          continue
        val = getattr(module, name)
        if isinstance(val, Program):
          return val
      raise ProgramDiscoveryError(
        f"{path}: no Program found. Expected `build_<name>_program()` or a top-level Program."
      )

  if not callable(candidate):
    if isinstance(candidate, Program):
      return candidate
    raise ProgramDiscoveryError(
      f"{path}: {entry or candidate.__name__!r} is not callable and not a Program"
    )

  meta_dict = _resolve_meta(meta)
  result = _invoke_builder(candidate, meta_dict)
  if not isinstance(result, Program):
    raise ProgramDiscoveryError(f"{path}: builder returned {type(result).__name__}, not Program")
  return result


def _import_file(path: Path):
  # Use a sanitized module name so repeated loads don't collide in sys.modules.
  mod_name = f"_srdatalog_viz_user_{path.stem}_{abs(hash(str(path))):x}"
  spec = importlib.util.spec_from_file_location(mod_name, path)
  if spec is None or spec.loader is None:
    raise ProgramDiscoveryError(f"could not create import spec for {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[mod_name] = module
  # Let the user's file import relative to its directory (doop_run.py style).
  sys.path.insert(0, str(path.parent))
  try:
    spec.loader.exec_module(module)
  finally:
    if sys.path and sys.path[0] == str(path.parent):
      sys.path.pop(0)
  return module


def _resolve_meta(meta):
  if meta is None:
    return None
  if isinstance(meta, dict):
    return meta
  return json.loads(Path(meta).read_text())


def _invoke_builder(fn, meta_dict):
  '''Call fn(). If it accepts a positional arg and meta is provided,
  pass meta. If fn needs a required arg and meta is None, raise
  a helpful error rather than TypeError-ing inside the user code.'''
  sig = inspect.signature(fn)
  positional = [
    p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
  ]
  required = [p for p in positional if p.default is inspect.Parameter.empty]
  if required:
    if meta_dict is None:
      raise ProgramDiscoveryError(
        f"{fn.__name__} requires {len(required)} arg(s) — pass --meta <json>"
      )
    return fn(meta_dict)
  if positional and meta_dict is not None:
    return fn(meta_dict)
  return fn()
